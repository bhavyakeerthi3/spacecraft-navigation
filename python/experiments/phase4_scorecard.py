"""
python/experiments/phase4_scorecard.py

Phase 4D: Fair B1/B2/C1 scorecard comparison.

IDENTICAL across all estimators:
  - Truth trajectory (same seed, DOP853)
  - GNSS measurements (same seed, same noise realisation)
  - Duration: ~1 orbit
  - Initial conditions: 500 km circular, i=51.6 deg

ESTIMATORS:
  B1 -- Raw GNSS passthrough (position/velocity from GNSS directly)
  B2 -- 6-state orbital EKF (GNSS updates, orbital dynamics model)
  C1 -- 9-state GNSS/IMU EKF (IMU mechanization + bias estimation)

METRICS:
  Pos RMSE, Vel RMSE, Bias RMSE (C1 only), 3-sigma coverage, E[NIS], E[NEES_pos]

NOTE on E[NEES]:
  NEES = (x_hat - x_true)^T P^{-1} (x_hat - x_true)
  Expected value = state_dim for a consistent filter.
  E[NEES_pos] uses only the position sub-block for fair comparison.
  A single run gives noisy NEES. Monte Carlo would tighten bounds.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from python.dynamics.constants import MU_EARTH_M3S2, RE_M
from python.dynamics.two_body import circular_speed, orbital_period
from python.dynamics.propagation import circular_orbit_initial_condition, propagate_truth
from python.dynamics.forces import ForceModel
from python.navigation.ekf_9state import NineStateEKF, propagate_9state_rk4
from python.navigation.orbit_model import build_P0, build_P0_6state, H_GNSS_9
from python.navigation.baselines import OrbitalEKF6State
from python.records import MeasurementPacket


def _build_gnss_packets(t_gnss, truth_arr, t_truth, sigma_pos, sigma_vel, seed):
    rng = np.random.default_rng(seed)
    R = np.diag([sigma_pos**2]*3 + [sigma_vel**2]*3)
    packets = []
    for i, tg in enumerate(t_gnss):
        idx = np.argmin(np.abs(t_truth - tg))
        r_t = truth_arr[idx, :3]
        v_t = truth_arr[idx, 3:6]
        noise = rng.standard_normal(6) * np.array([sigma_pos]*3 + [sigma_vel]*3)
        z = np.concatenate([r_t, v_t]) + noise
        pkt = MeasurementPacket(
            sensor_id="gnss_0", sequence_num=i+1,
            sample_time_s=float(tg), delivery_time_s=float(tg),
            value=z, declared_covariance=R.copy(),
            frame="ECI", units="m,m/s", is_valid=True,
        )
        packets.append(pkt)
    return packets, R


def main():
    print("=" * 70)
    print("Phase 4D: Fair B1 / B2 / C1 Scorecard Comparison")
    print("=" * 70)
    os.makedirs("results/figures", exist_ok=True)

    MU = MU_EARTH_M3S2
    RE = RE_M
    alt = 500_000.0
    r_orb = RE + alt
    v_circ = circular_speed(r_orb, MU)
    T_orb = orbital_period(r_orb, MU)
    t_end = T_orb

    # Sensor parameters (identical for all estimators)
    sigma_pos = 3.0
    sigma_vel = 0.03
    Qa = 1.0e-10    # Accel noise PSD [m^2/s^3]
    Qb = 1.0e-14    # Bias walk PSD [m^2/s^5]
    dt_imu  = 0.1
    dt_gnss = 1.0
    ba_true = np.array([1.0e-4, -5.0e-5, 2.0e-5])   # True accel bias

    state0 = circular_orbit_initial_condition(altitude_m=alt, inclination_deg=51.6)
    r0, v0 = state0[:3], state0[3:]

    # ---------------------------------------------------------------
    # 1. Truth trajectory (biased -- ba_true acts as nongrav force)
    # ---------------------------------------------------------------
    print("\n[1] Propagating truth trajectory ...")
    t_eval = np.arange(0.0, t_end + dt_imu, dt_imu)
    x_truth = np.concatenate([r0, v0, ba_true])
    truth_states = [x_truth.copy()]
    for i in range(1, len(t_eval)):
        dt = float(t_eval[i] - t_eval[i-1])
        x_truth = propagate_9state_rk4(x_truth, ba_true, np.eye(3), dt, MU)
        truth_states.append(x_truth.copy())
    truth_arr = np.array(truth_states)   # (N, 9) -- last 3 are ba_true (constant)
    print(f"    {len(t_eval)} epochs")

    # ---------------------------------------------------------------
    # 2. GNSS packets (shared across B1, B2, C1)
    # ---------------------------------------------------------------
    print("\n[2] Generating GNSS measurements ...")
    t_gnss = np.arange(dt_gnss, t_end + dt_gnss, dt_gnss)
    gnss_packets, R_gnss = _build_gnss_packets(
        t_gnss, truth_arr, t_eval, sigma_pos, sigma_vel, seed=42
    )
    print(f"    {len(gnss_packets)} packets")

    # ---------------------------------------------------------------
    # 3. IMU packets (for C1 only)
    # ---------------------------------------------------------------
    rng_imu = np.random.default_rng(99)
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
    print(f"    {len(imu_packets)} IMU packets")

    # ---------------------------------------------------------------
    # 4. Align truth at evaluation epochs
    # ---------------------------------------------------------------
    # Use GNSS sample times as eval epochs for fairness
    t_eval_score = t_gnss.copy()
    truth_r_eval = np.array([
        truth_arr[np.argmin(np.abs(t_eval - t)), :3] for t in t_eval_score
    ])
    truth_v_eval = np.array([
        truth_arr[np.argmin(np.abs(t_eval - t)), 3:6] for t in t_eval_score
    ])
    truth_b_eval = np.tile(ba_true, (len(t_eval_score), 1))

    # ---------------------------------------------------------------
    # B1: Raw GNSS
    # ---------------------------------------------------------------
    print("\n[3] Running B1 - Raw GNSS ...")
    b1_pos = np.array([p.value[:3] for p in gnss_packets])
    b1_vel = np.array([p.value[3:6] for p in gnss_packets])
    pos_err_b1 = b1_pos - truth_r_eval
    vel_err_b1 = b1_vel - truth_v_eval
    rmse_pos_b1 = float(np.sqrt(np.mean(np.sum(pos_err_b1**2, axis=1))))
    rmse_vel_b1 = float(np.sqrt(np.mean(np.sum(vel_err_b1**2, axis=1))))
    print(f"    Pos RMSE: {rmse_pos_b1:.3f} m  Vel RMSE: {rmse_vel_b1:.5f} m/s")

    # ---------------------------------------------------------------
    # B2: 6-state orbital EKF
    # ---------------------------------------------------------------
    print("\n[4] Running B2 - 6-state orbital EKF ...")
    P0_6 = build_P0_6state(pos_sigma_m=30.0, vel_sigma_ms=0.3)
    x0_6 = np.concatenate([r0, v0])
    R_gnss = np.diag([sigma_pos**2]*3 + [sigma_vel**2]*3)
    ekf6 = OrbitalEKF6State(
        x0=x0_6, P0=P0_6, Q_pos_m2s3=Qa, Q_vel_m2s3=Qa,
        R=R_gnss, dt_nav=dt_gnss, mu=MU,
    )
    b2_records = ekf6.run(t_gnss, gnss_packets)
    # Align B2 estimates to evaluation epochs
    b2_t = np.array([r.t_s for r in b2_records])
    b2_pos = np.array([r.x_hat[:3] for r in b2_records])
    b2_vel = np.array([r.x_hat[3:6] for r in b2_records])
    b2_P   = np.array([r.P for r in b2_records])

    # Interpolate to t_eval_score epochs
    b2_pos_eval = np.array([b2_pos[np.argmin(np.abs(b2_t - t))] for t in t_eval_score])
    b2_vel_eval = np.array([b2_vel[np.argmin(np.abs(b2_t - t))] for t in t_eval_score])
    b2_P_eval   = np.array([b2_P[np.argmin(np.abs(b2_t - t))] for t in t_eval_score])

    pos_err_b2 = b2_pos_eval - truth_r_eval
    vel_err_b2 = b2_vel_eval - truth_v_eval
    rmse_pos_b2 = float(np.sqrt(np.mean(np.sum(pos_err_b2**2, axis=1))))
    rmse_vel_b2 = float(np.sqrt(np.mean(np.sum(vel_err_b2**2, axis=1))))

    # Coverage: fraction of epochs where |pos_err| < 3*sigma
    b2_sigma_pos = np.sqrt(np.array([np.trace(P[:3,:3])/3 for P in b2_P_eval]))
    b2_pos_norm  = np.sqrt(np.sum(pos_err_b2**2, axis=1))
    b2_cov_rate  = float(np.mean(b2_pos_norm < 3 * b2_sigma_pos))

    # NIS from records
    b2_nis = [r.NIS for r in b2_records if r.NIS is not None and r.gate_accepted]
    mean_nis_b2 = float(np.mean(b2_nis)) if b2_nis else float("nan")

    # NEES_pos
    b2_nees = []
    for i, t in enumerate(t_eval_score):
        idx = np.argmin(np.abs(b2_t - t))
        e = pos_err_b2[i]
        P3 = b2_P_eval[i][:3, :3]
        try:
            b2_nees.append(float(e @ np.linalg.solve(P3, e)))
        except Exception:
            pass
    mean_nees_b2 = float(np.mean(b2_nees)) if b2_nees else float("nan")

    print(f"    Pos RMSE: {rmse_pos_b2:.3f} m  Vel RMSE: {rmse_vel_b2:.5f} m/s")
    print(f"    3-sig coverage: {b2_cov_rate*100:.1f}%  E[NIS]: {mean_nis_b2:.2f}  E[NEES_pos]: {mean_nees_b2:.2f}")

    # ---------------------------------------------------------------
    # C1: 9-state GNSS/IMU EKF
    # ---------------------------------------------------------------
    print("\n[5] Running C1 - 9-state GNSS/IMU EKF ...")
    P0_9 = build_P0(pos_sigma_m=30.0, vel_sigma_ms=0.3, bias_sigma_ms2=5e-4)
    x0_9 = np.concatenate([r0, v0, np.zeros(3)])   # wrong initial bias

    ekf = NineStateEKF(
        x0=x0_9, P0=P0_9, Qa=Qa, Qb=Qb, R_gnss=R_gnss, mu=MU, t0=0.0,
    )
    t_nav = np.arange(dt_imu, t_end + dt_imu, dt_imu)
    ekf.run(t_nav=t_nav, imu_packets=imu_packets, gnss_packets=gnss_packets)

    c1_records = ekf.records
    c1_t   = np.array([r.t_s for r in c1_records])
    c1_pos = np.array([r.x_hat[:3] for r in c1_records])
    c1_vel = np.array([r.x_hat[3:6] for r in c1_records])
    c1_bia = np.array([r.x_hat[6:9] for r in c1_records])
    c1_P   = np.array([r.P for r in c1_records])

    # Align to eval epochs
    c1_pos_eval = np.array([c1_pos[np.argmin(np.abs(c1_t - t))] for t in t_eval_score])
    c1_vel_eval = np.array([c1_vel[np.argmin(np.abs(c1_t - t))] for t in t_eval_score])
    c1_bia_eval = np.array([c1_bia[np.argmin(np.abs(c1_t - t))] for t in t_eval_score])
    c1_P_eval   = np.array([c1_P[np.argmin(np.abs(c1_t - t))] for t in t_eval_score])

    pos_err_c1 = c1_pos_eval - truth_r_eval
    vel_err_c1 = c1_vel_eval - truth_v_eval
    bia_err_c1 = c1_bia_eval - truth_b_eval
    rmse_pos_c1 = float(np.sqrt(np.mean(np.sum(pos_err_c1**2, axis=1))))
    rmse_vel_c1 = float(np.sqrt(np.mean(np.sum(vel_err_c1**2, axis=1))))
    rmse_bia_c1 = float(np.sqrt(np.mean(np.sum(bia_err_c1**2, axis=1))))

    c1_sigma_pos = np.sqrt(np.array([np.trace(P[:3,:3])/3 for P in c1_P_eval]))
    c1_pos_norm  = np.sqrt(np.sum(pos_err_c1**2, axis=1))
    c1_cov_rate  = float(np.mean(c1_pos_norm < 3 * c1_sigma_pos))

    c1_nis = [r.NIS for r in c1_records if r.NIS is not None and r.gate_accepted]
    mean_nis_c1 = float(np.mean(c1_nis)) if c1_nis else float("nan")

    c1_nees = []
    for i, t in enumerate(t_eval_score):
        idx = np.argmin(np.abs(c1_t - t))
        e = pos_err_c1[i]
        P3 = c1_P_eval[i][:3, :3]
        try:
            c1_nees.append(float(e @ np.linalg.solve(P3, e)))
        except Exception:
            pass
    mean_nees_c1 = float(np.mean(c1_nees)) if c1_nees else float("nan")

    print(f"    Pos RMSE: {rmse_pos_c1:.3f} m  Vel RMSE: {rmse_vel_c1:.5f} m/s")
    print(f"    Bias RMSE: {rmse_bia_c1:.4e} m/s^2")
    print(f"    3-sig coverage: {c1_cov_rate*100:.1f}%  E[NIS]: {mean_nis_c1:.2f}  E[NEES_pos]: {mean_nees_c1:.2f}")

    # ---------------------------------------------------------------
    # Print scorecard
    # ---------------------------------------------------------------
    sep = "-" * 90
    print(f"\n{sep}")
    print(f"{'Estimator':<18}{'Pos RMSE':>12}{'Vel RMSE':>14}{'Bias RMSE':>14}{'3sig Cov':>10}{'E[NIS]':>10}{'E[NEES_pos]':>14}")
    print(sep)
    print(f"{'B1 Raw GNSS':<18}{rmse_pos_b1:>11.3f}m{rmse_vel_b1:>12.5f}m/s{'N/A':>15}{'N/A':>10}{'N/A':>10}{'N/A':>14}")
    print(f"{'B2 6-state EKF':<18}{rmse_pos_b2:>11.3f}m{rmse_vel_b2:>12.5f}m/s{'N/A':>15}{b2_cov_rate*100:>9.1f}%{mean_nis_b2:>10.2f}{mean_nees_b2:>14.2f}")
    print(f"{'C1 9-state EKF':<18}{rmse_pos_c1:>11.3f}m{rmse_vel_c1:>12.5f}m/s{rmse_bia_c1:>12.2e}{c1_cov_rate*100:>9.1f}%{mean_nis_c1:>10.2f}{mean_nees_c1:>14.2f}")
    print(sep)
    print(f"\nNote: E[NEES_pos] ~ 3 expected for consistent 3-state position sub-block.")
    print(f"      E[NIS]       ~ 6 expected for consistent 6-DOF GNSS update.")
    print(f"      Single-run values have high variance -- Monte Carlo needed for tight bounds.")

    # ---------------------------------------------------------------
    # Figure: position error comparison
    # ---------------------------------------------------------------
    t_orb = t_eval_score / T_orb
    pos_err_b1_norm = np.sqrt(np.sum((b1_pos - truth_r_eval)**2, axis=1))
    pos_err_b2_norm = np.sqrt(np.sum(pos_err_b2**2, axis=1))
    pos_err_c1_norm = np.sqrt(np.sum(pos_err_c1**2, axis=1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase 4D: B1 / B2 / C1 Scorecard Comparison\n"
                 f"500 km LEO, ~1 orbit, GNSS sigma_pos={sigma_pos}m, ba_true=[1e-4,-5e-5,2e-5] m/s^2",
                 fontsize=11, fontweight="bold")

    # Top-left: position error (log)
    ax = axes[0, 0]
    ax.semilogy(t_orb, pos_err_b1_norm, color="salmon", lw=0.7, alpha=0.8, label="B1 Raw GNSS")
    ax.semilogy(t_orb, pos_err_b2_norm, color="steelblue", lw=0.8, label=f"B2 6-state EKF")
    ax.semilogy(t_orb, pos_err_c1_norm, color="seagreen", lw=0.8, label=f"C1 9-state EKF")
    ax.set_xlabel("Time [orbits]")
    ax.set_ylabel("Position Error [m] (log)")
    ax.set_title("Position Error History (log scale)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: velocity error
    vel_err_b1_norm = np.sqrt(np.sum((b1_vel - truth_v_eval)**2, axis=1))
    vel_err_b2_norm = np.sqrt(np.sum(vel_err_b2**2, axis=1))
    vel_err_c1_norm = np.sqrt(np.sum(vel_err_c1**2, axis=1))

    ax = axes[0, 1]
    ax.semilogy(t_orb, vel_err_b1_norm, color="salmon", lw=0.7, alpha=0.8, label="B1")
    ax.semilogy(t_orb, vel_err_b2_norm, color="steelblue", lw=0.8, label="B2")
    ax.semilogy(t_orb, vel_err_c1_norm, color="seagreen", lw=0.8, label="C1")
    ax.set_xlabel("Time [orbits]")
    ax.set_ylabel("Velocity Error [m/s] (log)")
    ax.set_title("Velocity Error History (log scale)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: bias estimation (C1)
    bia_err_norm_c1 = np.sqrt(np.sum(bia_err_c1**2, axis=1))
    c1_bia_sigma = np.sqrt(np.array([np.trace(P[6:9,6:9])/3 for P in c1_P_eval]))
    ax = axes[1, 0]
    ax.semilogy(t_orb, bia_err_norm_c1 * 1e6, color="darkorchid", lw=0.8, label="C1 bias error")
    ax.set_xlabel("Time [orbits]")
    ax.set_ylabel("Bias Error Norm [micro-m/s^2]")
    ax.set_title("C1 Accelerometer Bias Estimation Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: scorecard table
    ax = axes[1, 1]
    ax.axis("off")
    table_data = [
        ["Estimator", "Pos RMSE", "Vel RMSE", "3sig Cov", "E[NIS]", "E[NEES]"],
        ["B1 Raw GNSS", f"{rmse_pos_b1:.2f}m", f"{rmse_vel_b1:.4f}m/s", "N/A", "N/A", "N/A"],
        ["B2 6-state EKF", f"{rmse_pos_b2:.3f}m", f"{rmse_vel_b2:.5f}m/s",
         f"{b2_cov_rate*100:.1f}%", f"{mean_nis_b2:.1f}", f"{mean_nees_b2:.2f}"],
        ["C1 9-state EKF", f"{rmse_pos_c1:.3f}m", f"{rmse_vel_c1:.5f}m/s",
         f"{c1_cov_rate*100:.1f}%", f"{mean_nis_c1:.1f}", f"{mean_nees_c1:.2f}"],
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.5)
    ax.set_title("Scorecard Summary", fontweight="bold")

    plt.tight_layout()
    fig_path = os.path.join("results", "figures", "phase4_scorecard.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure: {fig_path}")
    print("Done.")


if __name__ == "__main__":
    main()
