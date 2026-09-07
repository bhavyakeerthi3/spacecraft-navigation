# Autonomous Spacecraft Navigation — GNSS/INS Sensor Fusion and EKF

> **Status: Phase 0–1 complete** — Two-body truth propagator verified.  
> Phase 2 (GNSS simulator + 6-state EKF) in progress.

---

## Problem Statement

This project builds a **reproducible, loosely coupled spacecraft navigation simulation for low Earth orbit (LEO)**, featuring a manually implemented Extended Kalman Filter (EKF) for GNSS/IMU sensor fusion. It directly targets the navigation engineering competencies required for spacecraft GNC roles.

The engineering question: **Under which sensor, maneuver, timing, and GNSS-availability conditions does IMU-aided orbital estimation improve accuracy and uncertainty calibration?**

> This is a simulation-only project. It models a spacecraft navigation algorithm, not a specific flight system. No real spacecraft data is used.

---

## Architecture

```
Versioned config + reproducible seeds
         │
         ▼
Truth: orbital dynamics + force schedule + sensor biases
         │
    ┌────┴────┐
    ▼         ▼
  GNSS       IMU (accelerometer + gyro)
  simulator  simulator
    │         │
    └────┬────┘
         ▼
  Timestamped measurement bus
         │
    ┌────┴──────────────┐
    ▼                   ▼
  9-state orbital EKF   Baselines:
  (position, velocity,   A: Ground truth
   accel bias)           B: Raw GNSS
                         C: Inertial dead reckoning
                         D: This EKF
                         E: 6-state GNSS+dynamics EKF
         │
         ▼
  Evaluator (joins truth + estimates — only consumer of truth)
         │
         ▼
  Error analysis, NEES/NIS, observability, Monte Carlo, figures
         │
         ▼
  Frozen replay → C++ validation → MATLAB comparison
```

**Three strict boundaries:**
- **Simulation** — may access true states and fault labels
- **Navigation** — receives only allowed measurements, their timestamps, declared covariances, and the model
- **Evaluation** — joins truth and estimates for scoring; never feeds back into navigation

---

## Quick Start

```bash
# 1. Clone and create environment
git clone <repo-url>
cd spacecraft-navigation
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 2. Install in development mode
pip install -e ".[dev]"

# 3. Run orbit verification (Phase 1 review gate)
python -m spacecraft_nav.experiments.orbit_baseline

# 4. Run tests
pytest tests/ -v

# 5. View results
# Figures saved to results/figures/
```

---

## Repository Structure

```
spacecraft-navigation/
├── configs/                  # All scenario configurations (YAML)
│   ├── nominal.yaml          # Baseline LEO scenario
│   ├── outage.yaml           # GNSS outage experiment
│   ├── maneuver.yaml         # Finite burn scenario
│   └── robustness.yaml       # Bias/outlier/latency stress tests
├── python/                   # Main Python package (spacecraft_nav)
│   ├── config.py             # Config loader + validated dataclass
│   ├── records.py            # Typed data records (truth, meas, estimate)
│   ├── random_streams.py     # Independent reproducible RNG per error source
│   ├── dynamics/
│   │   ├── constants.py      # SI constants with provenance
│   │   ├── two_body.py       # Gravity acceleration + Jacobian
│   │   ├── forces.py         # Gravitational/non-gravitational interfaces
│   │   ├── propagation.py    # Truth DOP853 + navigation RK4
│   │   └── frames.py         # ECI/RTN frame transforms
│   ├── sensors/
│   │   ├── gnss.py           # GNSS position/velocity measurement sim
│   │   ├── imu.py            # Accelerometer + gyro error model
│   │   └── timing.py         # Sample/delivery scheduling
│   ├── navigation/
│   │   ├── orbit_model.py    # 6/9-state f, F, H with attitude input
│   │   ├── discretization.py # Φ/B propagation + discrete Q
│   │   ├── ekf.py            # From-scratch EKF (predict/update)
│   │   └── baselines.py      # Raw GNSS, dead reckoning, model-only filter
│   ├── experiments/
│   │   ├── orbit_baseline.py # Phase 1 gate: orbit verification
│   │   ├── run_case.py       # Single scenario orchestration
│   │   └── monte_carlo.py    # 100-run ensemble
│   └── visualization/
│       ├── metrics.py        # RMSE, NEES/NIS, coverage statistics
│       └── plots.py          # Publication-quality figure functions
├── tests/                    # pytest unit tests
│   ├── test_dynamics.py      # Analytic orbit, conservation, Jacobians
│   ├── test_sensors.py       # Sensor statistics and reproducibility
│   ├── test_ekf.py           # Linear-limit update, Joseph covariance
│   └── fixtures/             # Small immutable reference cases
├── cpp/                      # C++ navigation core (Phase 8)
├── matlab/                   # MATLAB validation (Phase 8)
├── docs/
│   ├── architecture.md       # Boundaries, interfaces, failure paths
│   ├── mathematics.md        # Full derivations and conventions
│   └── design_decisions.md   # Assumptions and interview defense
└── results/
    ├── figures/              # Generated plots
    ├── tables/               # Metrics CSVs
    └── manifests/            # Config + seed + software version records
```

---

## Technical Scope and Assumptions

| Item | Choice | Reason |
|---|---|---|
| Orbit | Circular 500 km, 51.6° inclination | Concrete LEO; not a specific mission |
| Central body | Spherical Earth, μ = 3.986004418×10¹⁴ m³/s² | Point-mass; J2 is a later extension |
| Duration | 5700 s (~1 orbit) | Covers acquisition, outage, recovery |
| Inertial frame | Idealized ECI (not full GCRF/ITRF) | Sufficient for simulation scope |
| IMU | Accelerometer measures **specific force only** (not gravity) | Correct spacecraft physics |
| GNSS | Solution-level position/velocity (not pseudoranges) | Loosely coupled architecture |
| Attitude | Known externally (Phase 1–7); separate MEKF in Phase 9 | Cleanly isolates orbit filter |

**Critical physics note:** In free-fall orbit, a perfect accelerometer reads zero because gravity is not measured by an inertial sensor. Thrust, drag, and solar radiation pressure appear in the accelerometer signal. Gravity appears only in the propagation model.

---

## Phases and Execution Status

| Phase | Description | Status |
|---|---|---|
| 0 | Repository scaffold, config schema, unit contracts | ✅ Complete |
| 1 | Two-body truth, gravity Jacobian, conservation tests | ✅ Complete |
| 2 | GNSS simulator, raw-GNSS baseline | 🔲 Next |
| 3 | 6-state EKF, diagnostics | 🔲 Planned |
| 4 | IMU model, dead reckoning, 9-state EKF | 🔲 Planned |
| 5 | Error metrics, NEES/NIS, 100-run nominal campaign | 🔲 Planned |
| 6 | Outage/burn/calibration/outlier/latency experiments | 🔲 Planned |
| 7 | Observability analysis | 🔲 Planned |
| 8 | C++ host replay, MATLAB comparison | 🔲 Planned |
| 9 | Attitude MEKF extension | 🔲 Planned |
| 10 | Relative navigation extension | 🔲 Planned |
| 11 | Final reproducibility and publication | 🔲 Planned |

---

## Limitations (current)

- Two-body only (no J2, drag, SRP, third bodies)
- Idealized ECI frame (no Earth rotation, no leap seconds, no relativistic correction)
- Attitude assumed known and exact (MEKF is a separate extension)
- GNSS modeled as position/velocity solution (not pseudorange-level)
- IMU at spacecraft centre of mass (no lever arm)
- No flight qualification, no real spacecraft data, no HIL

---

## Differentiators vs Generic Student Projects

1. **Correct spacecraft physics** — gravity/specific-force separation; no double-counting
2. **Rigorous uncertainty accounting** — NEES/NIS with correct dimensions; failures retained
3. **Honest baseline comparison** — shows when IMU aiding wins *and* loses
4. **Computational observability** — SVD-based; not just a theory paragraph
5. **C++ parity** — algorithm portability demonstrated via frozen replay
6. **Separate verified extensions** — attitude MEKF and relative navigation, each independently tested

---

## License

MIT — see `LICENSE`.

---

*Blueprint: [`spacecraft-navigation-blueprint.md`](../spacecraft-navigation-blueprint.md)*
