"""
python/experiments/phase4_outage.py

Phase 4E+4F: GNSS Outage Analysis + Analytic Covariance Sanity Check.

SCENARIO:
  - 500 km circular LEO, i=51.6 deg
  - GNSS nominal, then OUTAGE from t=1000s to t=1300s (300 s)
  - IMU keeps running during outage
  - GNSS resumes after outage

WHAT WE SHOW:
  1. Position error + 3-sigma envelope (grows during outage)
  2. Velocity error + 3-sigma
  3. Bias estimate X axis during and after outage
  4. P[r,r] trace over time with outage shaded

ANALYTIC SANITY CHECK (Phase 4F):
  Simplified analytic bound for position uncertainty during outage:
    sigma_pos_simple(t) = sigma_ba * (t - t_outage_start)^2 / 2
  where sigma_ba is the RMS of the bias uncertainty at outage start.

  This is a FIRST-ORDER APPROXIMATION. It ignores:
    - Initial velocity uncertainty (dominant at short times)
    - Gravity coupling in Phi
    - Cross-covariance terms
    - Qd accumulation during outage
  The EKF covariance includes all of these, so it will generally
  differ from the simplified bound. The check verifies that the
  EKF covariance is in the same ballpark (not diverged, not
  implausibly tight), NOT that it exactly matches.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from python.dynamics.constants import MU_EARTH_M3S2, RE_M
from python.dynamics.two_body import circular_speed, orbital_period
from python.dynamics.propagation import circular_orbit_initial_condition
from python.navigation.ekf_9state import NineStateEKF, propagate_9state_rk4
from python.navigation.orbit_model import build_P0
from python.records import MeasurementPacket


def main():
    print("=" * 65)
    print("Phase 4E+4F: GNSS Outage + Analytic Sanity Check")
    print("=" * 65)
    os.makedirs("results/figures", exist_ok=True)

    MU = MU_EARTH_M3S2
    RE = RE_M
    alt = 500_000.0
    r_orb = RE + alt
    v_circ = circular_speed(r_orb, MU)
    T_orb = orbital_period(r_orb, MU)

    # Scenario timing
    t_end   = 2000.0    # 2000 s scenario (covers outage + recovery)
    dt_imu  = 0.1       # IMU at 10 Hz
    dt_gnss = 1.0       # GNSS at 1 Hz
    t_out_start = 1000.0
    t_out_end   = 1300.0

    sigma_pos = 3.0
    sigma_vel = 0.03
    Qa = 1.0e-10
    Qb = 1.0e-14
    ba_true = np.array([1.0e-4, -5.0e-5, 2.0e-5])

    print(f"\n  Outage window: {t_out_start:.0f}s to {t_out_end:.0f}s ({t_out_end-t_out_start:.0f}s)")

    state0 = circular_orbit_initial_condition(altitude_m=alt, inclination_deg=51.6)
    r0, v0 = state0[:3], state0[3:]

    # Truth trajectory with bias
    t_eval = np.arange(0.0, t_end + dt_imu, dt_imu)
    x_truth = np.concatenate([r0, v0, ba_true])
    truth_states = [x_truth.copy()]
    for i in range(1, len(t_eval)):
        dt = float(t_eval[i] - t_eval[i-1])
        x_truth = propagate_9state_rk4(x_truth, ba_true, np.eye(3), dt, MU)
        truth_states.append(x_truth.copy())
    truth_arr = np.array(truth_states)

    # IMU packets (entire run)
    rng_imu = np.random.default_rng(42)
    sigma_a = np.sqrt(Qa / dt_imu)
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

    # GNSS packets with outage window
    rng_gnss = np.random.default_rng(99)
    R_gnss = np.diag([sigma_pos**2]*3 + [sigma_vel**2]*3)
    t_gnss_all = np.arange(dt_gnss, t_end + dt_gnss, dt_gnss)
    gnss_packets = []
    for i, tg in enumerate(t_gnss_all):
        idx = np.argmin(np.abs(t_eval - tg))
        r_t = truth_arr[idx, :3]
        v_t = truth_arr[idx, 3:6]
        is_valid = not (t_out_start <= tg <= t_out_end)
        noise = rng_gnss.standard_normal(6) * np.array([sigma_pos]*3 + [sigma_vel]*3)
        z = np.concatenate([r_t, v_t]) + noise
        pkt = MeasurementPacket(
            sensor_id="gnss_0", sequence_num=i+1,
            sample_time_s=float(tg), delivery_time_s=float(tg),
            value=z, declared_covariance=R_gnss.copy(),
            frame="ECI", units="m,m/s", is_valid=is_valid,
        )
        gnss_packets.append(pkt)

    n_outage = sum(1 for p in gnss_packets if not p.is_valid)
    print(f"  GNSS outage packets: {n_outage}")

    # Run 9-state EKF
    print("\n[Running 9-state EKF with outage ...]")
    P0 = build_P0(pos_sigma_m=10.0, vel_sigma_ms=0.1, bias_sigma_ms2=5e-4)
    x0_filter = np.concatenate([r0, v0, np.zeros(3)])

    ekf = NineStateEKF(
        x0=x0_filter, P0=P0, Qa=Qa, Qb=Qb, R_gnss=R_gnss, mu=MU, t0=0.0,
    )
    t_nav = np.arange(dt_imu, t_end + dt_imu, dt_imu)
    ekf.run(t_nav=t_nav, imu_packets=imu_packets, gnss_packets=gnss_packets)

    # Extract records
    rec_t   = np.array([r.t_s for r in ekf.records])
    rec_r   = np.array([r.x_hat[:3] for r in ekf.records])
    rec_v   = np.array([r.x_hat[3:6] for r in ekf.records])
    rec_b   = np.array([r.x_hat[6:9] for r in ekf.records])
    rec_P   = [r.P for r in ekf.records]
    rec_sr  = np.array([np.sqrt(np.diag(r.P)[:3]) for r in ekf.records])  # pos sigma each axis
    rec_sv  = np.array([np.sqrt(np.diag(r.P)[3:6]) for r in ekf.records])
    rec_sb  = np.array([np.sqrt(np.diag(r.P)[6:9]) for r in ekf.records])
    rec_Ptr = np.array([np.trace(r.P[:3,:3]) for r in ekf.records])

    # Truth at record epochs
    truth_r_rec = np.array([truth_arr[np.argmin(np.abs(t_eval - t)), :3] for t in rec_t])
    truth_v_rec = np.array([truth_arr[np.argmin(np.abs(t_eval - t)), 3:6] for t in rec_t])

    err_r = rec_r - truth_r_rec     # (N, 3)
    err_v = rec_v - truth_v_rec
    err_b = rec_b - ba_true          # bias error vs truth

    # Outage statistics
    in_outage = (rec_t >= t_out_start) & (rec_t <= t_out_end)
    after_out  = rec_t > t_out_end
    before_out = rec_t < t_out_start

    # Covariance at outage start
    idx_out = np.argmin(np.abs(rec_t - t_out_start))
    P_at_outage_start = rec_P[idx_out]
    sigma_ba_start = np.sqrt(np.trace(P_at_outage_start[6:9, 6:9]) / 3)
    sigma_vel_start = np.sqrt(np.trace(P_at_outage_start[3:6, 3:6]) / 3)
    sigma_pos_start = np.sqrt(np.trace(P_at_outage_start[:3, :3]) / 3)

    print(f"\n  At outage start t={t_out_start:.0f}s:")
    print(f"    sigma_pos = {sigma_pos_start:.4f} m")
    print(f"    sigma_vel = {sigma_vel_start:.6f} m/s")
    print(f"    sigma_ba  = {sigma_ba_start:.4e} m/s^2")

    # Analytic bound during outage (Phase 4F)
    t_out_duration = rec_t[in_outage] - t_out_start
    sigma_analytic_pos = sigma_vel_start * t_out_duration + 0.5 * sigma_ba_start * t_out_duration**2
    ekf_pos_sigma_outage = np.sqrt(rec_Ptr[in_outage] / 3)

    print(f"\n  Analytic sanity check (Phase 4F):")
    print(f"    At t=300s into outage:")
    if len(t_out_duration) > 0:
        idx_dur = min(len(t_out_duration)-1, int(300/dt_imu))
        print(f"      Simplified bound  : {sigma_analytic_pos[idx_dur]:.2f} m")
        print(f"      EKF 1-sigma_pos   : {ekf_pos_sigma_outage[idx_dur]:.2f} m")
    print(f"  (Bound assumes: initial vel. unc. + bias unc. grows as dt and dt^2)")
    print(f"  (EKF includes: full Phi coupling, Q_d accumulation, cross-terms)")

    # After recovery
    idx_post_out = np.where(after_out)[0]
    if len(idx_post_out) > 10:
        pos_err_post = np.sqrt(np.sum(err_r[idx_post_out]**2, axis=1))
        print(f"\n  After GNSS recovery:")
        print(f"    Final pos RMSE (last 100 epochs): {np.sqrt(np.mean(pos_err_post[-100:]**2)):.3f} m")
        print(f"    Final P[r,r] trace: {rec_Ptr[idx_post_out[-1]]:.4f} m^2")

    # ---------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle("Phase 4E/4F: GNSS Outage Analysis (9-state EKF)\n"
                 f"Outage: {t_out_start:.0f}s to {t_out_end:.0f}s | "
                 f"ba_true=[1e-4,-5e-5,2e-5] m/s^2",
                 fontsize=11, fontweight="bold")

    def shade_outage(ax):
        ax.axvspan(t_out_start, t_out_end, alpha=0.15, color="red", label="GNSS outage")

    # Row 1: position error X + 3-sigma
    ax = axes[0, 0]
    ax.plot(rec_t, err_r[:, 0], color="steelblue", lw=0.6, label="Pos error X")
    ax.fill_between(rec_t, -3*rec_sr[:, 0], 3*rec_sr[:, 0],
                    alpha=0.25, color="steelblue", label="+/-3-sigma")
    shade_outage(ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Error X [m]")
    ax.set_title("Position Error X + 3-sigma Envelope")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(rec_t, err_r[:, 1], color="seagreen", lw=0.6, label="Pos error Y")
    ax.fill_between(rec_t, -3*rec_sr[:, 1], 3*rec_sr[:, 1],
                    alpha=0.25, color="seagreen", label="+/-3-sigma")
    shade_outage(ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Error Y [m]")
    ax.set_title("Position Error Y + 3-sigma Envelope")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 2: velocity error X
    ax = axes[1, 0]
    ax.plot(rec_t, err_v[:, 0] * 1e3, color="darkorange", lw=0.6, label="Vel error X")
    ax.fill_between(rec_t, -3*rec_sv[:, 0]*1e3, 3*rec_sv[:, 0]*1e3,
                    alpha=0.25, color="darkorange", label="+/-3-sigma")
    shade_outage(ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity Error X [mm/s]")
    ax.set_title("Velocity Error X + 3-sigma Envelope")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 2: bias estimate X
    ax = axes[1, 1]
    ax.plot(rec_t, rec_b[:, 0]*1e6, color="darkorchid", lw=0.6, label="Bias estimate X")
    ax.fill_between(rec_t,
                    (rec_b[:, 0] - 3*rec_sb[:, 0])*1e6,
                    (rec_b[:, 0] + 3*rec_sb[:, 0])*1e6,
                    alpha=0.25, color="darkorchid", label="+/-3-sigma")
    ax.axhline(ba_true[0]*1e6, color="red", ls="--", lw=1.2, label="True bias X")
    shade_outage(ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Bias X [micro-m/s^2]")
    ax.set_title("Accelerometer Bias X Estimate + 3-sigma")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 3: Covariance trace
    ax = axes[2, 0]
    ax.semilogy(rec_t, rec_Ptr, color="royalblue", lw=0.8, label="trace(P_pos)")
    shade_outage(ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Covariance Trace [m^2] (log)")
    ax.set_title("Position Covariance Trace -- Outage Highlighted")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 3: Analytic sanity check (Phase 4F)
    ax = axes[2, 1]
    t_out_dur_plot = np.linspace(0, t_out_end - t_out_start, 300)
    sigma_ana_plt2 = sigma_vel_start * t_out_dur_plot + 0.5 * sigma_ba_start * t_out_dur_plot**2

    ax.plot(t_out_dur_plot, sigma_ana_plt2, color="red", ls="--", lw=1.5,
            label=f"Analytic: sigma_v*dt + sigma_ba*dt^2/2")
    if len(t_out_duration) > 0:
        ax.plot(t_out_duration, ekf_pos_sigma_outage, color="steelblue", lw=1.2,
                label="EKF: sqrt(trace(P_pos)/3)")
    ax.set_xlabel("Time since outage start [s]")
    ax.set_ylabel("1-sigma position uncertainty [m]")
    ax.set_title("Phase 4F: Analytic Sanity Check\n"
                 "(Simplified bound vs EKF covariance during outage)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Annotation explaining difference
    ax.text(0.05, 0.6,
            "Analytic bound: zero P_init,\nno gravity coupling.\n"
            "EKF includes all terms.",
            transform=ax.transAxes, fontsize=8,
            va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                                edgecolor="gray", alpha=0.8))

    plt.tight_layout()
    fig_path = os.path.join("results", "figures", "phase4_outage.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure: {fig_path}")
    print("Done.")


if __name__ == "__main__":
    main()
