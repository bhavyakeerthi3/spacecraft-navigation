"""
python/experiments/orbit_baseline.py

Phase 1 review gate experiment: two-body orbit propagation and verification.

This script:
  1. Computes the circular orbit initial condition
  2. Propagates with DOP853 over one full orbital period
  3. Computes orbital invariants (energy, angular momentum) over time
  4. Compares with the analytic circular orbit solution
  5. Generates the Phase 1 review figure: orbit_verification.png

EXPECTED RESULTS (before running — replace with actual values):
  - Position error at end of orbit: < 0.1 m
  - Energy drift: < 1e-9 relative
  - Angular momentum drift: < 1e-9 relative
  - Figure: 3-panel plot showing 3D orbit, invariant drift, and position error

RUN:
  python -m spacecraft_nav.experiments.orbit_baseline
  OR
  python python/experiments/orbit_baseline.py
"""

from __future__ import annotations

import os
import sys
import json
import hashlib
import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for reproducible output
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Allow running without pip install (direct invocation)
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from python.dynamics.constants import (
    MU_EARTH_M3S2, RE_M, M2KM
)
from python.dynamics.two_body import (
    orbital_period, circular_speed
)
from python.dynamics.forces import ForceModel
from python.dynamics.propagation import (
    circular_orbit_initial_condition,
    propagate_truth,
)
from python.dynamics.frames import (
    identity_C_IB,
    rtn_from_state,
)


# -----------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------

STYLE = {
    "truth_color":   "#2E86AB",   # Steel blue
    "analytic_color": "#E84855",  # Crimson
    "energy_color":  "#A23B72",   # Purple
    "hmag_color":    "#F18F01",   # Amber
    "error_color":   "#C73E1D",   # Red
    "bg_color":      "#FAFAFA",
    "grid_color":    "#E0E0E0",
    "font_family":   "DejaVu Sans",
}


def _set_style() -> None:
    """Apply consistent style for publication-quality figures."""
    plt.rcParams.update({
        "font.family":       STYLE["font_family"],
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "axes.grid":         True,
        "axes.facecolor":    STYLE["bg_color"],
        "figure.facecolor":  "white",
        "grid.color":        STYLE["grid_color"],
        "grid.linewidth":    0.6,
        "lines.linewidth":   1.8,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "legend.framealpha": 0.9,
        "legend.fontsize":   9,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
    })


# -----------------------------------------------------------------------
# Verification computation
# -----------------------------------------------------------------------

def compute_orbit_verification(
    altitude_m: float = 500_000.0,
    inclination_deg: float = 51.6,
    n_eval_points: int = 500,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> dict:
    """
    Propagate a circular orbit and compute verification statistics.

    Returns
    -------
    results : dict with keys:
        t, states, period_s, analytic_r, analytic_v,
        energy, h_mag, pos_err_m, vel_err_ms,
        energy_drift_rel, hmag_drift_rel,
        state0
    """
    MU = MU_EARTH_M3S2
    RE = RE_M

    state0 = circular_orbit_initial_condition(
        altitude_m=altitude_m,
        inclination_deg=inclination_deg,
    )
    r0 = state0[:3]
    v0 = state0[3:6]
    r_mag = np.linalg.norm(r0)
    T     = orbital_period(r_mag, MU)

    t_eval = np.linspace(0.0, T, n_eval_points)
    force_model = ForceModel(mu=MU)

    t_out, states = propagate_truth(
        state0, (0.0, T), force_model,
        identity_C_IB,
        rtol=rtol, atol=atol,
        t_eval=t_eval,
    )

    # Orbital invariants
    energy  = np.array([
        0.5 * np.dot(s[3:6], s[3:6]) - MU / np.linalg.norm(s[:3])
        for s in states
    ])
    h_vecs  = np.array([np.cross(s[:3], s[3:6]) for s in states])
    h_mags  = np.linalg.norm(h_vecs, axis=1)

    E0     = energy[0]
    h0_mag = h_mags[0]
    energy_drift_rel = np.abs((energy - E0) / abs(E0))
    hmag_drift_rel   = np.abs((h_mags - h0_mag) / h0_mag)

    # Analytic circular orbit
    omega = circular_speed(r_mag, MU) / r_mag
    r_hat = r0 / r_mag
    h_vec = np.cross(r0, v0)
    h_hat = h_vec / np.linalg.norm(h_vec)
    t_hat = np.cross(h_hat, r_hat)

    analytic_r = np.array([
        r_mag * (np.cos(omega * t) * r_hat + np.sin(omega * t) * t_hat)
        for t in t_out
    ])
    analytic_v = np.array([
        r_mag * omega * (-np.sin(omega * t) * r_hat + np.cos(omega * t) * t_hat)
        for t in t_out
    ])

    pos_err_m  = np.linalg.norm(states[:, :3] - analytic_r, axis=1)
    vel_err_ms = np.linalg.norm(states[:, 3:6] - analytic_v, axis=1)

    return dict(
        t=t_out,
        states=states,
        period_s=T,
        analytic_r=analytic_r,
        analytic_v=analytic_v,
        energy=energy,
        h_mag=h_mags,
        pos_err_m=pos_err_m,
        vel_err_ms=vel_err_ms,
        energy_drift_rel=energy_drift_rel,
        hmag_drift_rel=hmag_drift_rel,
        state0=state0,
        r_orbit_m=r_mag,
        altitude_m=altitude_m,
        inclination_deg=inclination_deg,
    )


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def plot_orbit_verification(results: dict, out_dir: Path) -> Path:
    """
    Generate Phase 1 orbit verification figure.

    Panel 1: 3D orbit trajectory (propagated vs analytic)
    Panel 2: Orbital invariant drift (energy + h magnitude)
    Panel 3: Position and velocity error vs analytic
    Panel 4: Radial, along-track, cross-track position error
    """
    _set_style()

    t      = results["t"]
    T      = results["period_s"]
    t_norm = t / T  # normalized time [0, 1] in orbits
    states = results["states"]
    pos_err = results["pos_err_m"]
    vel_err = results["vel_err_ms"]
    E_drift = results["energy_drift_rel"]
    h_drift = results["hmag_drift_rel"]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Phase 1 — Two-Body Orbit Verification\n"
        f"Circular LEO, h = {results['altitude_m']/1e3:.0f} km, "
        f"i = {results['inclination_deg']:.1f}°, T = {T:.1f} s",
        fontsize=14, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    # --- Panel 1: 3D orbit ---
    ax_3d = fig.add_subplot(gs[0:2, 0], projection="3d")
    r_km = states[:, :3] * M2KM
    a_km = results["analytic_r"] * M2KM
    ax_3d.plot(r_km[:, 0], r_km[:, 1], r_km[:, 2],
               color=STYLE["truth_color"], lw=1.5, label="DOP853 truth")
    ax_3d.plot(a_km[:, 0], a_km[:, 1], a_km[:, 2],
               color=STYLE["analytic_color"], lw=1.0, linestyle="--",
               alpha=0.7, label="Analytic")
    ax_3d.scatter(*r_km[0], color="green", s=50, zorder=5, label="t=0")
    ax_3d.scatter(*r_km[-1], color="red",  s=50, zorder=5, label="t=T")
    ax_3d.set_xlabel("X [km]", labelpad=6)
    ax_3d.set_ylabel("Y [km]", labelpad=6)
    ax_3d.set_zlabel("Z [km]", labelpad=6)
    ax_3d.set_title("3D Orbit (ECI)", pad=10)
    ax_3d.legend(loc="upper right", fontsize=8)

    # --- Panel 2: Energy drift ---
    ax_E = fig.add_subplot(gs[0, 1])
    ax_E.semilogy(t_norm, E_drift + 1e-20,
                  color=STYLE["energy_color"], lw=1.8)
    ax_E.set_xlabel("Time [orbits]")
    ax_E.set_ylabel("Relative energy drift")
    ax_E.set_title("Specific Orbital Energy Conservation")
    ax_E.axhline(1e-9, color="gray", linestyle=":", lw=1.2, label="Threshold 1e-9")
    ax_E.legend()
    ax_E.set_xlim(0, 1)

    # --- Panel 3: Angular momentum drift ---
    ax_h = fig.add_subplot(gs[0, 2])
    ax_h.semilogy(t_norm, h_drift + 1e-20,
                  color=STYLE["hmag_color"], lw=1.8)
    ax_h.set_xlabel("Time [orbits]")
    ax_h.set_ylabel("Relative |h| drift")
    ax_h.set_title("Angular Momentum Conservation")
    ax_h.axhline(1e-9, color="gray", linestyle=":", lw=1.2, label="Threshold 1e-9")
    ax_h.legend()
    ax_h.set_xlim(0, 1)

    # --- Panel 4: Position error vs analytic ---
    ax_perr = fig.add_subplot(gs[1, 1])
    ax_perr.semilogy(t_norm, pos_err + 1e-6,
                     color=STYLE["error_color"], lw=1.8, label="3D pos error")
    ax_perr.axhline(0.1, color="gray", linestyle=":", lw=1.2, label="Threshold 0.1 m")
    ax_perr.set_xlabel("Time [orbits]")
    ax_perr.set_ylabel("Position error [m]")
    ax_perr.set_title("Position Error vs Analytic Circular Orbit")
    ax_perr.legend()
    ax_perr.set_xlim(0, 1)

    # --- Panel 5: Velocity error vs analytic ---
    ax_verr = fig.add_subplot(gs[1, 2])
    ax_verr.semilogy(t_norm, vel_err + 1e-9,
                     color=STYLE["truth_color"], lw=1.8, label="3D vel error")
    ax_verr.axhline(1e-4, color="gray", linestyle=":", lw=1.2, label="Threshold 1e-4 m/s")
    ax_verr.set_xlabel("Time [orbits]")
    ax_verr.set_ylabel("Velocity error [m/s]")
    ax_verr.set_title("Velocity Error vs Analytic Circular Orbit")
    ax_verr.legend()
    ax_verr.set_xlim(0, 1)

    out_path = out_dir / "orbit_verification.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")
    return out_path


# -----------------------------------------------------------------------
# Manifest
# -----------------------------------------------------------------------

def write_manifest(results: dict, out_dir: Path) -> Path:
    """Save a JSON manifest with configuration and summary statistics."""
    manifest = {
        "experiment":  "orbit_baseline",
        "phase":       1,
        "timestamp":   datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": {
            "altitude_m":       results["altitude_m"],
            "inclination_deg":  results["inclination_deg"],
            "mu_m3s2":          MU_EARTH_M3S2,
            "Re_m":             RE_M,
            "period_s":         results["period_s"],
            "n_eval_points":    len(results["t"]),
            "integrator":       "DOP853",
            "rtol":             1e-12,
            "atol":             1e-12,
        },
        "summary": {
            "max_pos_error_m":         float(np.max(results["pos_err_m"])),
            "final_pos_error_m":       float(results["pos_err_m"][-1]),
            "max_vel_error_ms":        float(np.max(results["vel_err_ms"])),
            "final_vel_error_ms":      float(results["vel_err_ms"][-1]),
            "max_energy_drift_rel":    float(np.max(results["energy_drift_rel"])),
            "max_hmag_drift_rel":      float(np.max(results["hmag_drift_rel"])),
        },
        "thresholds": {
            "pos_error_m":       0.1,
            "vel_error_ms":      1e-4,
            "energy_drift_rel":  1e-9,
            "hmag_drift_rel":    1e-9,
        },
        "passed": {
            "pos_error":    float(results["pos_err_m"][-1]) < 0.1,
            "vel_error":    float(results["vel_err_ms"][-1]) < 1e-4,
            "energy_drift": float(np.max(results["energy_drift_rel"])) < 1e-9,
            "hmag_drift":   float(np.max(results["hmag_drift_rel"])) < 1e-9,
        },
    }

    out_path = out_dir / "orbit_baseline_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return out_path


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    """Entry point for the orbit baseline experiment."""
    # Resolve output directories relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent
    fig_dir = project_root / "results" / "figures"
    manifest_dir = project_root / "results" / "manifests"
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 1 — Orbit Baseline Experiment")
    print("=" * 60)
    print(f"Altitude:     500 km circular")
    print(f"Inclination:  51.6°")
    print(f"Propagator:   DOP853 (rtol=atol=1e-12)")
    print()

    results = compute_orbit_verification()
    T = results["period_s"]

    # Print summary
    print(f"Orbital period:           {T:.2f} s ({T/60:.2f} min)")
    print(f"Circular speed:           {circular_speed(results['r_orbit_m']):.2f} m/s")
    print()
    print("Verification results:")
    print(f"  Final position error:   {results['pos_err_m'][-1]:.4e} m   "
          f"({'PASS' if results['pos_err_m'][-1] < 0.1 else 'FAIL'}, threshold 0.1 m)")
    print(f"  Final velocity error:   {results['vel_err_ms'][-1]:.4e} m/s  "
          f"({'PASS' if results['vel_err_ms'][-1] < 1e-4 else 'FAIL'}, threshold 1e-4 m/s)")
    print(f"  Max energy drift:       {np.max(results['energy_drift_rel']):.4e}  "
          f"({'PASS' if np.max(results['energy_drift_rel']) < 1e-9 else 'FAIL'}, threshold 1e-9)")
    print(f"  Max h-mag drift:        {np.max(results['hmag_drift_rel']):.4e}  "
          f"({'PASS' if np.max(results['hmag_drift_rel']) < 1e-9 else 'FAIL'}, threshold 1e-9)")
    print()

    # Generate figure
    fig_path = plot_orbit_verification(results, fig_dir)

    # Write manifest
    mfest_path = write_manifest(results, manifest_dir)
    print(f"Manifest saved: {mfest_path}")

    all_passed = all([
        results["pos_err_m"][-1] < 0.1,
        results["vel_err_ms"][-1] < 1e-4,
        np.max(results["energy_drift_rel"]) < 1e-9,
        np.max(results["hmag_drift_rel"]) < 1e-9,
    ])
    print()
    print("=" * 60)
    print(f"Phase 1 Review Gate: {'ALL CHECKS PASSED [OK]' if all_passed else 'SOME CHECKS FAILED [FAIL]'}")
    print("=" * 60)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
