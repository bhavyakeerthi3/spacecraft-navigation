"""
python/experiments/phase4_bias.py

Phase 4C: Accelerometer bias estimation demonstration.

SCENARIO:
  - 500 km circular LEO, i=51.6 deg, ~1 orbit
  - True accel bias: ba_true = [1e-4, -5e-5, 2e-5] m/s^2
  - Filter initial bias: ba_hat_0 = [0, 0, 0]
  - IMU at 10 Hz, GNSS at 1 Hz
  - GNSS noise: sigma_pos=3 m, sigma_vel=0.03 m/s

WHAT WE SHOW:
  The filter must estimate the bias state from GNSS position/velocity
  measurements. This is possible because:
    1. Biased IMU gives wrong velocity prediction
    2. GNSS velocity innovation feeds back through K[b,v] to update b_a
    3. The Phi cross-covariance builds up over many orbital periods

  We plot: true bias, estimated bias, +/-1-sigma covariance band, error norm.
  We do NOT just show position improvement -- we show the bias state directly.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from python.dynamics.constants import MU_EARTH_M3S2, RE_M
from python.dynamics.two_body import circular_speed, orbital_period, gravity_acceleration
from python.dynamics.propagation import circular_orbit_initial_condition, propagate_truth
from python.dynamics.forces import ForceModel
from python.navigation.ekf_9state import NineStateEKF, propagate_9state_rk4
from python.navigation.orbit_model import build_P0


def _build_stream(seed):
    return np.random.default_rng(seed)


def main():
    print("=" * 60)
    print("Phase 4C: Bias Estimation")
    print("=" * 60)
    os.makedirs("results/figures", exist_ok=True)

    # ---------------------------------------------------------------
    # Orbital setup
    # ---------------------------------------------------------------
    MU      = MU_EARTH_M3S2
    RE      = RE_M
    alt     = 500_000.0
    r_orb   = RE + alt
    v_circ  = circular_speed(r_orb, MU)
    T_orb   = orbital_period(r_orb, MU)
    t_end   = T_orb          # ~5677 s (one orbit)
    dt_imu  = 0.1            # IMU at 10 Hz
    dt_gnss = 1.0            # GNSS at 1 Hz

    print(f"  Altitude : {alt/1e3:.0f} km")
    print(f"  Period   : {T_orb:.0f} s ({T_orb/3600:.2f} h)")

    # ---------------------------------------------------------------
    # True bias (expressed in body frame; identity attitude assumed)
    # ---------------------------------------------------------------
    ba_true = np.array([1.0e-4, -5.0e-5, 2.0e-5])   # [m/s^2]
    ba_hat0 = np.zeros(3)                              # Filter starts wrong

    print(f"  True bias: {ba_true} m/s^2")
    print(f"  Init bias: {ba_hat0} m/s^2")

    # ---------------------------------------------------------------
    # Initial state
    # ---------------------------------------------------------------
    state0 = circular_orbit_initial_condition(altitude_m=alt, inclination_deg=51.6)
    r0, v0 = state0[:3], state0[3:]
    fm = ForceModel(mu=MU)
    t_eval = np.arange(0.0, t_end + dt_imu, dt_imu)
    print("\n[1] Propagating truth trajectory ...")
    times, states = propagate_truth(
        state0, (0.0, t_end), fm, C_IB_func=lambda t: np.eye(3),
        rtol=1e-12, atol=1e-12, t_eval=t_eval,
    )
    print(f"    {len(times)} truth epochs, dt={dt_imu}s")

    # ---------------------------------------------------------------
    # Generate IMU packets
    # The spacecraft is in free-fall. The accelerometer measures:
    #   f_m = C_BI(a_nongrav) + ba_true + noise = 0 + ba_true + noise
    # Here we use ZERO nongrav (coast), so f_m ~ ba_true + noise.
    # ---------------------------------------------------------------
    rng_imu = _build_stream(42)
    Qa      = 1.0e-10    # (1e-5 m/s^2)^2/Hz accel noise PSD
    sigma_a = np.sqrt(Qa / dt_imu)  # per-sample std

    from python.records import MeasurementPacket
    imu_packets = []
    for i, t_s in enumerate(t_eval[1:], 1):
        f_m = ba_true + rng_imu.standard_normal(3) * sigma_a
        pkt = MeasurementPacket(
            sensor_id="imu_accel_0", sequence_num=i,
            sample_time_s=float(t_s), delivery_time_s=float(t_s),
            value=f_m,
            declared_covariance=np.eye(3) * Qa / dt_imu,
            frame="body", units="m/s^2", is_valid=True,
        )
        imu_packets.append(pkt)

    print(f"    {len(imu_packets)} IMU packets")

    # ---------------------------------------------------------------
    # Generate GNSS packets from truth trajectory
    # ---------------------------------------------------------------
    rng_gnss  = _build_stream(43)
    sigma_pos = 3.0
    sigma_vel = 0.03
    R_gnss    = np.diag([sigma_pos**2]*3 + [sigma_vel**2]*3)

    t_gnss_times = np.arange(dt_gnss, t_end + dt_gnss, dt_gnss)

    # Propagate truth forward with bias as a non-grav force
    # (we do a separate 9-state truth for the biased trajectory)
    x_truth = np.concatenate([r0, v0, ba_true])
    truth_states_imu = [x_truth.copy()]
    for i in range(1, len(t_eval)):
        dt = float(t_eval[i] - t_eval[i-1])
        x_truth = propagate_9state_rk4(x_truth, ba_true, np.eye(3), dt, MU)
        truth_states_imu.append(x_truth.copy())
    truth_arr = np.array(truth_states_imu)   # (N, 9)

    gnss_packets = []
    for gnss_seq, t_g in enumerate(t_gnss_times):
        idx = np.argmin(np.abs(t_eval - t_g))
        r_t = truth_arr[idx, :3]
        v_t = truth_arr[idx, 3:6]
        noise = rng_gnss.standard_normal(6) * np.array([sigma_pos]*3 + [sigma_vel]*3)
        z = np.concatenate([r_t, v_t]) + noise
        pkt = MeasurementPacket(
            sensor_id="gnss_0", sequence_num=gnss_seq+1,
            sample_time_s=float(t_g), delivery_time_s=float(t_g),
            value=z, declared_covariance=R_gnss.copy(),
            frame="ECI", units="m,m/s", is_valid=True,
        )
        gnss_packets.append(pkt)

    print(f"    {len(gnss_packets)} GNSS packets")

    # ---------------------------------------------------------------
    # Run 9-state EKF
    # ---------------------------------------------------------------
    Qb  = 1.0e-20   # Nearly constant bias
    x0_filter = np.concatenate([r0, v0, ba_hat0])
    P0  = build_P0(pos_sigma_m=10.0, vel_sigma_ms=0.1, bias_sigma_ms2=5e-4)

    ekf = NineStateEKF(
        x0=x0_filter, P0=P0,
        Qa=Qa, Qb=Qb, R_gnss=R_gnss, mu=MU, t0=0.0,
    )

    print("\n[2] Running 9-state EKF ...")
    t_nav = np.arange(dt_imu, t_end + dt_imu, dt_imu)
    ekf.run(t_nav=t_nav, imu_packets=imu_packets, gnss_packets=gnss_packets)

    # ---------------------------------------------------------------
    # Extract bias history from records
    # ---------------------------------------------------------------
    rec_t    = np.array([r.t_s for r in ekf.records])
    rec_bx   = np.array([r.x_hat[6] for r in ekf.records])
    rec_by   = np.array([r.x_hat[7] for r in ekf.records])
    rec_bz   = np.array([r.x_hat[8] for r in ekf.records])
    rec_sbx  = np.array([np.sqrt(r.P[6, 6]) for r in ekf.records])
    rec_sby  = np.array([np.sqrt(r.P[7, 7]) for r in ekf.records])
    rec_sbz  = np.array([np.sqrt(r.P[8, 8]) for r in ekf.records])

    err_x = rec_bx - ba_true[0]
    err_y = rec_by - ba_true[1]
    err_z = rec_bz - ba_true[2]
    err_norm = np.sqrt(err_x**2 + err_y**2 + err_z**2)

    # ---------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------
    initial_err = np.linalg.norm(ba_hat0 - ba_true)
    final_err   = err_norm[-1]
    final_bias  = ekf.x[6:9]

    print(f"\n  Initial bias error   : {initial_err:.4e} m/s^2")
    print(f"  Final   bias error   : {final_err:.4e} m/s^2")
    print(f"  Reduction            : {100*(1 - final_err/initial_err):.1f}%")
    print(f"  True bias            : {ba_true}")
    print(f"  Estimated bias       : {final_bias}")
    print(f"  Final 1-sigma (X,Y,Z): ({ekf.P[6,6]**0.5:.2e}, {ekf.P[7,7]**0.5:.2e}, {ekf.P[8,8]**0.5:.2e})")

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase 4C: Accelerometer Bias Estimation (9-state EKF)\n"
                 f"True bias: [{ba_true[0]:.0e}, {ba_true[1]:.0e}, {ba_true[2]:.0e}] m/s^2 "
                 f"| Filter init: [0, 0, 0]",
                 fontsize=12, fontweight="bold")

    t_orb = rec_t / T_orb

    def _bias_panel(ax, t, est, sigma, true_val, label, color):
        ax.fill_between(t, (est - sigma)*1e6, (est + sigma)*1e6,
                        alpha=0.25, color=color, label="+/-1-sigma")
        ax.axhline(true_val*1e6, color="red", ls="--", lw=1.5, label="True bias")
        ax.plot(t, est*1e6, color=color, lw=0.8, label="Estimated")
        ax.set_ylabel(f"Bias {label} [micro-m/s^2]")
        ax.set_xlabel("Time [orbits]")
        ax.legend(fontsize=8)
        ax.set_title(f"Accelerometer Bias {label}")
        ax.grid(True, alpha=0.3)

    _bias_panel(axes[0, 0], t_orb, rec_bx, rec_sbx, ba_true[0], "X", "steelblue")
    _bias_panel(axes[0, 1], t_orb, rec_by, rec_sby, ba_true[1], "Y", "seagreen")
    _bias_panel(axes[1, 0], t_orb, rec_bz, rec_sbz, ba_true[2], "Z", "darkorchid")

    axes[1, 1].semilogy(t_orb, err_norm*1e6, color="firebrick", lw=1.0)
    axes[1, 1].axhline(initial_err*1e6, color="gray", ls=":", label="Initial error")
    axes[1, 1].set_xlabel("Time [orbits]")
    axes[1, 1].set_ylabel("Bias error norm [micro-m/s^2]")
    axes[1, 1].set_title("Bias Estimation Error Norm (log scale)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join("results", "figures", "phase4_bias.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure: {fig_path}")
    print("Done.")


if __name__ == "__main__":
    main()
