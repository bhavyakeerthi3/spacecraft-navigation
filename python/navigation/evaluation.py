"""
python/navigation/evaluation.py

Navigation filter evaluation and scoring.

METRICS COMPUTED:
  1. RMSE (Root Mean Square Error) — scalar error summary
     RMSE_pos = sqrt( mean( ||r_hat_k - r_true_k||² ) )

  2. 3σ Consistency — filter declared uncertainty vs actual error
     Consistent if actual error stays within 3×declared σ > ~99.7% of time.
     A consistent filter is neither overconfident (error > 3σ too often)
     nor underconfident (error << σ always, wasteful).

  3. NEES (Normalized Estimation Error Squared)
     NEES_k = (x_k - x_hat_k)ᵀ P_k⁻¹ (x_k - x_hat_k)
     E[NEES_k] = n (state dimension) for consistent filter.
     Time-averaged NEES should equal n.

     WHY NEES MATTERS:
       NEES tests BOTH accuracy AND covariance calibration together.
       An accurate but overconfident filter has NEES >> n.
       An inaccurate but underconfident filter has NEES << n.
       Only a consistent filter has NEES ≈ n.

  4. NIS (Normalized Innovation Squared)
     NIS_k = νᵀ S_k⁻¹ ν
     E[NIS_k] = m (measurement dimension) for consistent filter.
     Computed INSIDE the EKF update; stored in EstimateRecord.

  5. Empirical GNSS noise verification
     Test that the measurement noise is statistically consistent
     with the declared R: (z - h(x_true)) / σ ~ N(0,1).

STATISTICAL TESTS:
  NEES chi-squared test (Monte Carlo):
    Over N runs, E[NEES] should be in [n·(1-ε), n·(1+ε)] for small ε.
    Specifically, N·NEES / n ~ F(n, N·n) — an F-distribution test.
    This is the standard aerospace Monte Carlo consistency check.

REFERENCE:
  Li, X.R. and Jilkov, V.P. (2001). "A Survey of Maneuvering Target
  Tracking" — standard NEES/NIS definitions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional
from dataclasses import dataclass

from ..records import TruthState, EstimateRecord, EvaluationRecord
from ..dynamics.frames import C_rtn_from_eci


@dataclass
class ScorecardRow:
    """One row of the navigation error scorecard."""
    label: str
    pos_rmse_m: float
    vel_rmse_ms: float
    pos_3sigma_coverage: float   # fraction of epochs within declared 3σ
    vel_3sigma_coverage: float
    mean_NEES_pos: float         # E[NEES_pos] — should equal 3 for consistent filter
    mean_NIS: Optional[float]    # E[NIS] — should equal measurement dim


def join_truth_estimates(
    truth_records: list[TruthState],
    estimate_records: list[EstimateRecord],
    tolerance_s: float = 0.05,
) -> list[EvaluationRecord]:
    """
    Join truth and estimate records at matched epochs.

    Matching is by nearest truth epoch within tolerance_s.
    Only POSTERIOR estimate records are joined (prior epochs skipped).

    Parameters
    ----------
    truth_records : list of TruthState
    estimate_records : list of EstimateRecord
    tolerance_s : float — max time gap for a match [s]

    Returns
    -------
    joined : list of EvaluationRecord (with error fields filled in)
    """
    # Index truth by time
    t_truth = np.array([tr.t_s for tr in truth_records])

    joined = []
    for est in estimate_records:
        if est.update_type not in ("posterior", "predict"):
            continue  # Skip pure prior records

        # Find nearest truth epoch
        idx = np.argmin(np.abs(t_truth - est.t_s))
        dt  = abs(t_truth[idx] - est.t_s)
        if dt > tolerance_s:
            continue  # No match within tolerance

        truth = truth_records[idx]
        n = len(est.x_hat)

        # Position/velocity error in ECI
        pos_err_I = est.x_hat[:3] - truth.r_I
        vel_err_I = est.x_hat[3:6] - truth.v_I

        # Error in RTN frame
        try:
            C_RI = C_rtn_from_eci(truth.r_I, truth.v_I)
            pos_err_RTN = C_RI @ pos_err_I
            vel_err_RTN = C_RI @ vel_err_I
        except ValueError:
            pos_err_RTN = None
            vel_err_RTN = None

        # Bias error (9-state only)
        bias_err = None
        if n >= 9 and hasattr(truth, 'accel_bias_B'):
            bias_err = est.x_hat[6:9] - truth.accel_bias_B

        # NEES (full state if possible)
        NEES = None
        NEES_pos = None
        try:
            # Full state NEES (if state includes bias and truth has it)
            if n == 9 and bias_err is not None:
                full_err = np.concatenate([pos_err_I, vel_err_I, bias_err])
                P_inv = np.linalg.inv(est.P + 1e-20 * np.eye(n))
                NEES = float(full_err @ P_inv @ full_err)

            # Position-only NEES (3-state)
            P_pos = est.P[:3, :3]
            P_pos_inv = np.linalg.inv(P_pos + 1e-20 * np.eye(3))
            NEES_pos = float(pos_err_I @ P_pos_inv @ pos_err_I)
        except np.linalg.LinAlgError:
            pass

        record = EvaluationRecord(
            t_s=est.t_s,
            truth=truth,
            estimate=est,
            pos_error_I=pos_err_I,
            vel_error_I=vel_err_I,
            bias_error_B=bias_err,
            NEES=NEES,
            NEES_pos=NEES_pos,
            pos_error_RTN=pos_err_RTN,
            vel_error_RTN=vel_err_RTN,
        )
        joined.append(record)

    return joined


def compute_scorecard(
    eval_records: list[EvaluationRecord],
    label: str,
    estimate_records: Optional[list[EstimateRecord]] = None,
) -> ScorecardRow:
    """
    Compute navigation error scorecard from evaluation records.

    Parameters
    ----------
    eval_records : list of EvaluationRecord
    label : str — estimator label
    estimate_records : optional list for NIS computation

    Returns
    -------
    row : ScorecardRow
    """
    if not eval_records:
        return ScorecardRow(
            label=label,
            pos_rmse_m=np.nan, vel_rmse_ms=np.nan,
            pos_3sigma_coverage=np.nan, vel_3sigma_coverage=np.nan,
            mean_NEES_pos=np.nan, mean_NIS=None,
        )

    pos_errors = np.array([
        np.linalg.norm(r.pos_error_I) for r in eval_records
        if r.pos_error_I is not None
    ])
    vel_errors = np.array([
        np.linalg.norm(r.vel_error_I) for r in eval_records
        if r.vel_error_I is not None
    ])

    pos_rmse = float(np.sqrt(np.mean(pos_errors**2))) if len(pos_errors) else np.nan
    vel_rmse = float(np.sqrt(np.mean(vel_errors**2))) if len(vel_errors) else np.nan

    # 3σ consistency check on position
    n_within_3sigma_pos = 0
    n_within_3sigma_vel = 0
    n_total = 0
    for r in eval_records:
        if r.pos_error_I is None:
            continue
        n_total += 1
        P = r.estimate.P
        pos_3sigma = 3.0 * np.sqrt(np.diag(P[:3, :3]))
        vel_3sigma = 3.0 * np.sqrt(np.diag(P[3:6, 3:6]))

        if np.all(np.abs(r.pos_error_I) <= pos_3sigma):
            n_within_3sigma_pos += 1
        if r.vel_error_I is not None and np.all(np.abs(r.vel_error_I) <= vel_3sigma):
            n_within_3sigma_vel += 1

    pos_coverage = n_within_3sigma_pos / n_total if n_total > 0 else np.nan
    vel_coverage = n_within_3sigma_vel / n_total if n_total > 0 else np.nan

    # NEES
    nees_vals = [r.NEES_pos for r in eval_records if r.NEES_pos is not None]
    mean_nees = float(np.mean(nees_vals)) if nees_vals else np.nan

    # NIS
    mean_nis = None
    if estimate_records:
        nis_vals = [r.NIS for r in estimate_records
                    if r.NIS is not None and r.gate_accepted]
        if nis_vals:
            mean_nis = float(np.mean(nis_vals))

    return ScorecardRow(
        label=label,
        pos_rmse_m=pos_rmse,
        vel_rmse_ms=vel_rmse,
        pos_3sigma_coverage=pos_coverage,
        vel_3sigma_coverage=vel_coverage,
        mean_NEES_pos=mean_nees,
        mean_NIS=mean_nis,
    )


def print_scorecard(rows: list[ScorecardRow]) -> None:
    """Print a formatted scorecard table to stdout."""
    header = (
        f"{'Estimator':<22} {'Pos RMSE':>10} {'Vel RMSE':>12} "
        f"{'3s Cov(pos)':>12} {'E[NEES_pos]':>12} {'E[NIS]':>10}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        nis_str = f"{row.mean_NIS:>10.3f}" if row.mean_NIS is not None else f"{'N/A':>10}"
        print(
            f"{row.label:<22} "
            f"{row.pos_rmse_m:>9.3f}m "
            f"{row.vel_rmse_ms:>10.4f}m/s "
            f"{row.pos_3sigma_coverage:>11.1%} "
            f"{row.mean_NEES_pos:>12.2f} "
            f"{nis_str}"
        )
    print(sep)
    print("(Consistent filter: E[NEES_pos] ~ 3, E[NIS] ~ measurement_dim)")
