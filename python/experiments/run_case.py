"""
python/experiments/run_case.py

Single scenario orchestration — Phase 2 and Phase 3.

This script runs one complete simulation scenario from config:
  1. Generate truth trajectory (DOP853)
  2. Simulate GNSS measurements (Phase 2)
  3. Run baseline estimators (B1 raw GNSS, B2 6-state EKF)
  4. Run 9-state GNSS/IMU EKF (Phase 4 — stub for now)
  5. Evaluate all estimators and print scorecard
  6. Generate comparison figures

Usage:
  python python/experiments/run_case.py                         [nominal scenario]
  python python/experiments/run_case.py --config configs/outage.yaml [outage scenario]
  python python/experiments/run_case.py --scenario gnss_outage

Output:
  results/figures/case_<scenario>.png
  results/manifests/case_<scenario>.json
  results/tables/scorecard_<scenario>.csv
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Allow running without package install
_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))

from python.config import load_config
from python.random_streams import StreamFactory
from python.records import TruthState
from python.dynamics.propagation import (
    circular_orbit_initial_condition,
    propagate_truth,
)
from python.dynamics.frames import identity_C_IB
from python.dynamics.forces import ForceModel
from python.dynamics.constants import MU_EARTH_M3S2, M2KM
from python.sensors.gnss import GNSSSimulator
from python.sensors.timing import build_measurement_bus
from python.navigation.baselines import RawGNSSBaseline, OrbitalEKF6State
from python.navigation.orbit_model import build_P0_6state
from python.navigation.evaluation import (
    join_truth_estimates,
    compute_scorecard,
    print_scorecard,
)


# -----------------------------------------------------------------------
# Simulation orchestrator
# -----------------------------------------------------------------------

def run_scenario(config_path: str | Path) -> dict:
    """
    Run one complete simulation scenario.

    Returns
    -------
    results : dict with all output arrays, metrics, and records
    """
    cfg    = load_config(config_path)
    factory = StreamFactory(cfg.master_seed, cfg.streams)

    print(f"\nScenario : {cfg.scenario} (hash={cfg.config_hash})")
    print(f"Altitude : {cfg.altitude_m/1e3:.0f} km, i={np.degrees(cfg.inclination_rad):.1f}°")
    print(f"Duration : {cfg.t_end_s:.0f} s ({cfg.t_end_s/cfg.period_s:.2f} orbits)")

    # ---------------------------------------------------------------
    # 1. Truth trajectory (DOP853)
    # ---------------------------------------------------------------
    print("\n[1] Propagating truth trajectory ...")
    state0 = circular_orbit_initial_condition(
        altitude_m=cfg.altitude_m,
        inclination_deg=np.degrees(cfg.inclination_rad),
    )

    force_model = ForceModel(mu=cfg.mu_m3s2)

    # Evaluate at both truth-dense and navigation epochs
    t_nav   = np.arange(cfg.t_start_s, cfg.t_end_s + cfg.dt_nav_s, cfg.dt_nav_s)
    n_dense = max(1000, len(t_nav) * 2)
    t_dense = np.linspace(cfg.t_start_s, cfg.t_end_s, n_dense)
    t_eval  = np.union1d(t_dense, t_nav)

    t_truth_out, states_truth = propagate_truth(
        state0, (cfg.t_start_s, cfg.t_end_s),
        force_model, identity_C_IB,
        rtol=cfg.truth_rtol, atol=cfg.truth_atol,
        t_eval=t_eval,
    )

    # Build TruthState record list at navigation epochs
    truth_records: list[TruthState] = []
    for ti, si in zip(t_truth_out, states_truth):
        truth_records.append(TruthState(
            t_s=ti,
            r_I=si[:3].copy(),
            v_I=si[3:6].copy(),
            q_IB=None,
            C_IB=identity_C_IB(ti),
            accel_bias_B=np.zeros(3),
            gyro_bias_B=np.zeros(3),
            specific_force_B=np.zeros(3),   # coast: IMU reads zero
            is_thrusting=False,
        ))

    print(f"   Truth: {len(t_truth_out)} epochs, "
          f"r_init={np.linalg.norm(state0[:3])/1e3:.2f} km")

    # ---------------------------------------------------------------
    # 2. GNSS measurement simulation (Phase 2)
    # ---------------------------------------------------------------
    print("\n[2] Simulating GNSS measurements ...")
    gnss_sim = GNSSSimulator(cfg.gnss, factory)
    gnss_packets = gnss_sim.simulate(t_truth_out, states_truth)

    print(f"   GNSS packets: {len(gnss_packets)} "
          f"(rate={cfg.gnss.rate_hz} Hz, "
          f"outages={len(cfg.gnss.outage_windows)})")

    # Separate the delivery bus
    bus = build_measurement_bus(gnss_packets)
    gnss_sorted = bus.as_sorted_list()

    # Quick noise statistics
    if len(gnss_sorted) > 10:
        z_vals = np.array([p.value for p in gnss_sorted])
        t_vals = np.array([p.sample_time_s for p in gnss_sorted])
        # Interpolate truth at GNSS epochs
        r_true_at_gnss = np.column_stack([
            np.interp(t_vals, t_truth_out, states_truth[:, j]) for j in range(3)
        ])
        v_true_at_gnss = np.column_stack([
            np.interp(t_vals, t_truth_out, states_truth[:, j]) for j in range(3, 6)
        ])
        pos_residuals = z_vals[:, :3] - r_true_at_gnss
        vel_residuals = z_vals[:, 3:6] - v_true_at_gnss
        sigma_pos_declared = np.sqrt(cfg.gnss.pos_noise_1sigma_m**2 +
                                     cfg.gnss.gm_sigma_pos_m**2 * int(cfg.gnss.gm_bias_enabled) +
                                     cfg.gnss.bias_pos_m[0]**2 * int(cfg.gnss.bias_enabled))
        print(f"   Pos residual std: {np.std(pos_residuals):.2f} m "
              f"(declared sigma~{cfg.gnss.pos_noise_1sigma_m:.2f} m white noise)")
        print(f"   Vel residual std: {np.std(vel_residuals):.4f} m/s")
    else:
        pos_residuals = vel_residuals = np.array([])

    # ---------------------------------------------------------------
    # 3. B1 — Raw GNSS baseline
    # ---------------------------------------------------------------
    print("\n[3] Running B1 — Raw GNSS baseline ...")
    b1 = RawGNSSBaseline(R_gnss=gnss_sim.R_declared)
    b1_records = b1.run(gnss_sorted)
    print(f"   B1 estimates: {len(b1_records)}")

    # ---------------------------------------------------------------
    # 4. B2 — 6-state orbital EKF
    # ---------------------------------------------------------------
    print("\n[4] Running B2 — 6-state orbital EKF ...")
    # Initialize at truth (perturbed by GNSS noise of first packet)
    r0_est = state0[:3].copy()
    v0_est = state0[3:6].copy()
    if gnss_sorted:
        first_z = gnss_sorted[0].value
        r0_est = first_z[:3]
        v0_est = first_z[3:6]

    P0_6 = build_P0_6state(
        pos_sigma_m=cfg.filter.init_pos_1sigma_m,
        vel_sigma_ms=cfg.filter.init_vel_1sigma_ms,
    )
    x0_6 = np.concatenate([r0_est, v0_est])

    b2 = OrbitalEKF6State(
        x0=x0_6,
        P0=P0_6,
        Q_pos_m2s3=1e-12,        # Very small — nearly Keplerian
        Q_vel_m2s3=cfg.filter.Qa_m2s3,
        R=gnss_sim.R_declared,
        dt_nav=cfg.dt_nav_s,
        mu=cfg.mu_m3s2,
        gate_prob=cfg.filter.gate_probability,
    )
    b2_records = b2.run(t_nav, gnss_sorted)
    posterior_b2 = [r for r in b2_records if r.update_type == "posterior"]
    print(f"   B2 estimate epochs: {len(posterior_b2)}")

    # ---------------------------------------------------------------
    # 5. Evaluation
    # ---------------------------------------------------------------
    print("\n[5] Evaluating estimators ...")

    # Filter truth records to navigation epochs only
    t_nav_set = set(np.round(t_nav, 3))
    truth_nav = [
        tr for tr in truth_records
        if round(tr.t_s, 3) in t_nav_set or
           any(abs(tr.t_s - tn) < 0.1 for tn in t_nav[:10])
    ]
    # Simpler: use all truth records (join_truth_estimates matches by nearest)
    eval_b1 = join_truth_estimates(truth_records, b1_records, tolerance_s=0.6)
    eval_b2 = join_truth_estimates(truth_records, posterior_b2, tolerance_s=0.6)

    score_b1 = compute_scorecard(eval_b1, "B1 Raw GNSS")
    score_b2 = compute_scorecard(eval_b2, "B2 6-state EKF", posterior_b2)

    print()
    print_scorecard([score_b1, score_b2])

    # ---------------------------------------------------------------
    # 6. Figures
    # ---------------------------------------------------------------
    print("\n[6] Generating figures ...")
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    cfg.manifests_dir.mkdir(parents=True, exist_ok=True)

    fig_path = _plot_case(
        cfg, t_truth_out, states_truth,
        gnss_sorted, pos_residuals,
        b1_records, posterior_b2,
        eval_b1, eval_b2,
        score_b1, score_b2,
    )

    # ---------------------------------------------------------------
    # 7. Manifest
    # ---------------------------------------------------------------
    manifest = {
        "scenario":   cfg.scenario,
        "hash":       cfg.config_hash,
        "timestamp":  datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "duration_s": cfg.t_end_s,
        "n_orbits":   cfg.t_end_s / cfg.period_s,
        "gnss_packets": len(gnss_packets),
        "scores": {
            "B1_pos_rmse_m":         score_b1.pos_rmse_m,
            "B1_vel_rmse_ms":        score_b1.vel_rmse_ms,
            "B2_pos_rmse_m":         score_b2.pos_rmse_m,
            "B2_vel_rmse_ms":        score_b2.vel_rmse_ms,
            "B2_mean_NEES_pos":      score_b2.mean_NEES_pos,
            "B2_mean_NIS":           score_b2.mean_NIS,
        }
    }
    mfest_path = cfg.manifests_dir / f"case_{cfg.scenario}.json"
    with open(mfest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"   Manifest: {mfest_path}")

    return dict(
        cfg=cfg,
        t_truth=t_truth_out,
        states_truth=states_truth,
        gnss_packets=gnss_packets,
        eval_b1=eval_b1,
        eval_b2=eval_b2,
        score_b1=score_b1,
        score_b2=score_b2,
    )


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

COLORS = {
    "truth":   "#2E86AB",
    "gnss":    "#F18F01",
    "b1":      "#E84855",
    "b2":      "#3BB273",
    "3sigma":  "#BBBBBB",
    "grid":    "#E8E8E8",
}


def _set_style() -> None:
    plt.rcParams.update({
        "font.family":   "DejaVu Sans",
        "font.size":     10,
        "axes.grid":     True,
        "grid.color":    COLORS["grid"],
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "legend.framealpha": 0.85,
        "savefig.dpi":   300,
        "savefig.bbox":  "tight",
    })


def _plot_case(
    cfg, t_truth, states_truth,
    gnss_sorted, pos_residuals,
    b1_records, b2_records,
    eval_b1, eval_b2,
    score_b1, score_b2,
) -> Path:
    """Generate a 6-panel case comparison figure."""
    _set_style()

    T = cfg.period_s
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        f"Scenario: {cfg.scenario}  |  "
        f"h=500km, i=51.6°, T={T:.0f}s, "
        f"GNSS σ_pos={cfg.gnss.pos_noise_1sigma_m:.0f}m",
        fontsize=13, fontweight="bold",
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.38)

    # ---- Panel 1: 3D orbit ----
    ax = fig.add_subplot(gs[0:2, 0], projection="3d")
    r_km = states_truth[:, :3] * M2KM
    ax.plot(r_km[:, 0], r_km[:, 1], r_km[:, 2],
            color=COLORS["truth"], lw=1.2, label="Truth")
    if b2_records:
        b2_r = np.array([r.x_hat[:3] for r in b2_records]) * M2KM
        ax.plot(b2_r[:, 0], b2_r[:, 1], b2_r[:, 2],
                color=COLORS["b2"], lw=0.8, alpha=0.7, label="B2 EKF")
    ax.set_xlabel("X [km]"); ax.set_ylabel("Y [km]"); ax.set_zlabel("Z [km]")
    ax.set_title("3D Orbit (ECI)")
    ax.legend(loc="upper right")

    # ---- Panel 2: GNSS position residuals ----
    ax2 = fig.add_subplot(gs[0, 1])
    if len(pos_residuals) > 1:
        t_gnss_arr = np.array([p.sample_time_s for p in gnss_sorted]) / T
        labels = ["R (X)", "T (Y)", "N (Z)"]
        clrs   = ["#E84855", "#3BB273", "#2E86AB"]
        for j in range(3):
            ax2.plot(t_gnss_arr, pos_residuals[:, j], color=clrs[j],
                     lw=0.8, alpha=0.85, label=labels[j])
        ax2.axhline(3 * cfg.gnss.pos_noise_1sigma_m, color="gray",
                    ls=":", lw=1.1, label="±3σ declared")
        ax2.axhline(-3 * cfg.gnss.pos_noise_1sigma_m, color="gray", ls=":", lw=1.1)
    ax2.set_xlabel("Time [orbits]")
    ax2.set_ylabel("Residual [m]")
    ax2.set_title("GNSS Position Residuals (z − truth)")
    ax2.legend(ncol=2)

    # ---- Panel 3: B1 vs B2 position error ----
    ax3 = fig.add_subplot(gs[0, 2])
    if eval_b1:
        t_b1 = [r.t_s / T for r in eval_b1]
        e_b1 = [np.linalg.norm(r.pos_error_I) for r in eval_b1]
        ax3.semilogy(t_b1, e_b1, color=COLORS["b1"], lw=0.9, alpha=0.8, label="B1 Raw GNSS")
    if eval_b2:
        t_b2 = [r.t_s / T for r in eval_b2]
        e_b2 = [np.linalg.norm(r.pos_error_I) for r in eval_b2]
        ax3.semilogy(t_b2, e_b2, color=COLORS["b2"], lw=1.2, label="B2 6-state EKF")
    ax3.set_xlabel("Time [orbits]")
    ax3.set_ylabel("Position error [m]")
    ax3.set_title("Position Error History")
    ax3.legend()

    # ---- Panel 4: B2 EKF 3σ bounds (position) ----
    ax4 = fig.add_subplot(gs[1, 1])
    if eval_b2:
        t_b2 = np.array([r.t_s / T for r in eval_b2])
        err_r = np.array([r.pos_error_I[0] if r.pos_error_I is not None else np.nan
                          for r in eval_b2])
        sigma_r = np.array([r.estimate.pos_std()[0] for r in eval_b2])
        ax4.plot(t_b2, err_r, color=COLORS["b2"], lw=1.0, label="Error (X)")
        ax4.fill_between(t_b2, -3*sigma_r, 3*sigma_r,
                         alpha=0.25, color=COLORS["b2"], label="±3σ EKF")
    ax4.set_xlabel("Time [orbits]")
    ax4.set_ylabel("Pos error X [m]")
    ax4.set_title("B2 EKF: Error vs 3σ Bounds (X)")
    ax4.legend()

    # ---- Panel 5: B2 NIS ----
    ax5 = fig.add_subplot(gs[1, 2])
    post_b2_with_nis = [r for r in b2_records if r.NIS is not None and r.gate_accepted]
    if post_b2_with_nis:
        t_nis = np.array([r.t_s / T for r in post_b2_with_nis])
        nis   = np.array([r.NIS for r in post_b2_with_nis])
        ax5.plot(t_nis, nis, color=COLORS["b2"], lw=0.8, alpha=0.8, label="NIS")
        ax5.axhline(6.0, color="orange", ls="--", lw=1.2, label="E[NIS]=6 (consistent)")
        ax5.axhline(20.515, color="red", ls=":", lw=1.2, label="χ²(6,0.997)=20.5")
        ax5.set_ylim(0, min(40, np.percentile(nis, 99) * 1.5))
    ax5.set_xlabel("Time [orbits]")
    ax5.set_ylabel("NIS")
    ax5.set_title("B2 Normalized Innovation Squared")
    ax5.legend()

    # ---- Panel 6: Scorecard text box ----
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis("off")
    table_data = [
        ["Estimator", "Pos RMSE", "Vel RMSE", "3σ Coverage", "E[NEES_pos]", "E[NIS]"],
        ["B1 Raw GNSS",
         f"{score_b1.pos_rmse_m:.2f} m",
         f"{score_b1.vel_rmse_ms:.4f} m/s",
         f"{score_b1.pos_3sigma_coverage:.1%}",
         f"{score_b1.mean_NEES_pos:.2f}",
         "N/A"],
        ["B2 6-state EKF",
         f"{score_b2.pos_rmse_m:.2f} m",
         f"{score_b2.vel_rmse_ms:.4f} m/s",
         f"{score_b2.pos_3sigma_coverage:.1%}",
         f"{score_b2.mean_NEES_pos:.2f}",
         f"{score_b2.mean_NIS:.2f}" if score_b2.mean_NIS else "N/A"],
    ]
    tbl = ax6.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2E86AB")
            cell.set_text_props(color="white", fontweight="bold")

    out_path = cfg.figures_dir / f"case_{cfg.scenario}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   Figure: {out_path}")
    return out_path


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run spacecraft navigation case")
    parser.add_argument("--config",
                        default=str(_root / "configs" / "nominal.yaml"),
                        help="Path to YAML config file")
    args = parser.parse_args()

    results = run_scenario(args.config)
    print("\nDone.")


if __name__ == "__main__":
    main()
