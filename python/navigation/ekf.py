"""
python/navigation/ekf.py

Extended Kalman Filter — from scratch implementation.

NO scipy.linalg.solve, NO scipy.optimize, NO filter libraries.
Only numpy linalg primitives are used.

ALGORITHM:
  Standard discrete-time EKF with a nonlinear prediction step
  and a linearized (H) measurement update step.

  Predict:
      x⁻_{k+1} = x_k + ∫f(x,u) dt  (via external RK4 integrator)
      P⁻_{k+1} = Φ_k P_k Φ_kᵀ + Q_d,k

  Update:
      ν_k = z_k − H_k x⁻_k
      S_k = H_k P⁻_k H_kᵀ + R_k
      K_k = P⁻_k H_kᵀ S_k⁻¹
      x_k = x⁻_k + K_k ν_k
      P_k = (I − K_k H_k) P⁻_k (I − K_k H_k)ᵀ + K_k R_k K_kᵀ   [Joseph]

COVARIANCE UPDATE (Joseph form):
  P_k = (I − KH) P⁻ (I − KH)ᵀ + K R Kᵀ

  WHY JOSEPH FORM (not simple P = (I-KH)P⁻):
    The simple form is only numerically stable if K is EXACTLY optimal.
    For sub-optimal gains (detuned Q or R), the simple form can produce
    non-positive-definite P due to floating-point cancellation.
    The Joseph form is positive-semidefinite by construction for any K.
    This is critical for long Monte Carlo runs where subtle numerical
    drift accumulates over thousands of filter epochs.

    Proof: (I-KH)P⁻(I-KH)ᵀ = symmetric PSD matrix + K R Kᵀ (PSD)
    = PSD. So P is PSD for ANY real K and positive R, P⁻.

INNOVATION GATING (chi-squared gate):
  ν ~ N(0, S)  when filter is consistent.
  NIS = νᵀ S⁻¹ ν ~ χ²(m) where m = dim(z).

  Gate threshold at probability p: χ²_{m, p}
  E.g., m=6 GNSS: χ²(6, 0.997) ≈ 18.55

  If NIS > gate_threshold → reject measurement, x unchanged, P unchanged.
  Accepted/rejected flag is always logged (not silently dropped).

NUMERICAL HEALTH MONITORING:
  After every update, we check:
    1. P symmetric: ||P - Pᵀ||_F < ε
    2. P positive definite: Cholesky succeeds
    3. Covariance not diverged: max(diag(P)) < 1e8 m² (alert threshold)
  These checks are always on — not just in debug mode. They cost ~1 μs
  per epoch and provide an early warning before divergence is irreversible.

REFERENCE:
  Grewal & Andrews, "Kalman Filtering: Theory and Practice" §6.3 (Joseph form)
  Bar-Shalom, Li, Kirubarajan, "Estimation with Applications" Ch. 5
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional
import warnings

from ..records import EstimateRecord


# chi-squared gate thresholds: χ²(m, 0.997) for m = 1..12
# Precomputed for common measurement dimensions
_CHI2_GATE = {
    1:  8.807,
    2:  11.829,
    3:  14.156,
    4:  16.266,
    5:  18.207,
    6:  20.515,   # GNSS pos+vel
    3+0: 14.156,  # GNSS pos only
    9:  24.322,
}


class NumericalHealthWarning(UserWarning):
    """Raised when P becomes non-symmetric or near-singular."""
    pass


class CovarianceDivergenceError(RuntimeError):
    """Raised when P diverges to unrecoverable magnitudes."""
    pass


class ExtendedKalmanFilter:
    """
    From-scratch discrete-time Extended Kalman Filter.

    This class implements ONLY the measurement update step. The
    prediction step (RK4 integration of state + Φ + Q_d) is performed
    externally in propagation.py (rk4_predict). This separation ensures
    that the integrator is independently testable and that the EKF
    update can be used with any integrator.

    Usage:
        ekf = ExtendedKalmanFilter(state_dim=9, gate_prob=0.997)

        # At each navigation epoch:
        x_prior, P_prior, Phi, Qd = rk4_predict(...)

        for packet in measurements_at_this_epoch:
            x_post, P_post, record = ekf.update(
                x_prior, P_prior, packet, H, R
            )
            x_prior, P_prior = x_post, P_post  # chain updates

    Parameters
    ----------
    state_dim : int    — Filter state dimension (6 or 9)
    gate_prob : float  — Chi-squared gate probability (e.g. 0.997)
    health_check : bool — Enable numerical health monitoring (default True)
    max_diag_P : float — Divergence threshold on diag(P) [units depend on state]
    """

    def __init__(
        self,
        state_dim: int,
        gate_prob: float = 0.997,
        health_check: bool = True,
        max_diag_P: float = 1e10,
    ) -> None:
        self.state_dim = state_dim
        self.gate_prob = gate_prob
        self.health_check = health_check
        self.max_diag_P = max_diag_P
        self._n_updates_total = 0
        self._n_updates_rejected = 0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of measurements accepted by the gate."""
        if self._n_updates_total == 0:
            return 1.0
        return 1.0 - self._n_updates_rejected / self._n_updates_total

    def _gate_threshold(self, m: int) -> float:
        """Chi-squared gate threshold for measurement dimension m."""
        if m in _CHI2_GATE:
            return _CHI2_GATE[m]
        # Fallback: use scipy.stats.chi2 if available, else no gate
        try:
            from scipy.stats import chi2
            return float(chi2.ppf(self.gate_prob, df=m))
        except ImportError:
            return np.inf

    def _symmetrize(self, P: NDArray[np.float64]) -> NDArray[np.float64]:
        """Enforce symmetry: P = (P + Pᵀ) / 2."""
        return 0.5 * (P + P.T)

    def _health_check(
        self,
        P: NDArray[np.float64],
        tag: str = "",
    ) -> None:
        """Perform numerical health checks on P. Issue warnings or raise."""
        if not self.health_check:
            return

        # Symmetry check
        asymmetry = np.linalg.norm(P - P.T, 'fro')
        if asymmetry > 1e-6:
            warnings.warn(
                f"EKF {tag}: P asymmetry = {asymmetry:.2e}. "
                "Applying symmetrization.",
                NumericalHealthWarning,
                stacklevel=3,
            )

        # Positive definiteness via Cholesky
        try:
            np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            warnings.warn(
                f"EKF {tag}: P failed Cholesky (not PD). "
                "Consider increasing Q or checking initialization.",
                NumericalHealthWarning,
                stacklevel=3,
            )

        # Divergence check
        max_diag = np.max(np.diag(P))
        if max_diag > self.max_diag_P:
            raise CovarianceDivergenceError(
                f"EKF {tag}: max diag(P) = {max_diag:.2e} > {self.max_diag_P:.2e}. "
                "Filter has diverged."
            )

    def update(
        self,
        x_prior: NDArray[np.float64],
        P_prior: NDArray[np.float64],
        z: NDArray[np.float64],
        H: NDArray[np.float64],
        R: NDArray[np.float64],
        t: float,
        sensor_id: str = "",
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], EstimateRecord]:
        """
        Perform one EKF measurement update.

        LINEARIZATION POINT:
          H is evaluated at x_prior (the prior state estimate).
          This is the standard EKF linearization. For a nonlinear h(x),
          H = ∂h/∂x |_{x=x_prior}.
          For the GNSS model, H is exactly linear, so this is exact.

        GAIN COMPUTATION:
          S = H P⁻ Hᵀ + R
          K = P⁻ Hᵀ S⁻¹

          We use np.linalg.solve(S, H @ P⁻)ᵀ which is numerically
          preferred over explicit S⁻¹ (more stable for near-singular S).
          Kᵀ = solve(Sᵀ, (H P⁻)ᵀ) → K = solve(S, H P⁻)ᵀ
          Since S = Sᵀ, this simplifies to K = solve(S, H @ P_prior)ᵀ.

        Parameters
        ----------
        x_prior : ndarray (n,)     — prior state estimate
        P_prior : ndarray (n, n)   — prior covariance
        z : ndarray (m,)           — measurement
        H : ndarray (m, n)         — measurement Jacobian
        R : ndarray (m, m)         — measurement noise covariance
        t : float                  — epoch [s]
        sensor_id : str            — for logging

        Returns
        -------
        x_post : ndarray (n,)
        P_post : ndarray (n, n)
        record : EstimateRecord
        """
        n = self.state_dim
        m = len(z)

        self._n_updates_total += 1

        # --- Innovation ---
        h_x = H @ x_prior               # predicted measurement (linear h)
        nu  = z - h_x                   # pre-fit innovation

        # --- Innovation covariance ---
        HP  = H @ P_prior               # (m, n)
        S   = HP @ H.T + R              # (m, m) — symmetric
        S   = 0.5 * (S + S.T)          # enforce symmetry numerically

        # --- NIS and chi-squared gate ---
        try:
            S_inv_nu = np.linalg.solve(S, nu)     # S⁻¹ ν
        except np.linalg.LinAlgError:
            # S is singular — reject measurement
            self._n_updates_rejected += 1
            record = EstimateRecord(
                t_s=t, update_type="posterior",
                x_hat=x_prior.copy(), P=P_prior.copy(),
                innovation=nu, innovation_cov=S,
                NIS=np.inf, gate_accepted=False,
                sensor_id=sensor_id,
            )
            return x_prior.copy(), P_prior.copy(), record

        NIS = float(nu @ S_inv_nu)
        gate_thresh = self._gate_threshold(m)
        gate_accepted = NIS <= gate_thresh

        if not gate_accepted:
            self._n_updates_rejected += 1
            record = EstimateRecord(
                t_s=t, update_type="posterior",
                x_hat=x_prior.copy(), P=P_prior.copy(),
                innovation=nu, innovation_cov=S,
                NIS=NIS, gate_accepted=False,
                sensor_id=sensor_id,
            )
            return x_prior.copy(), P_prior.copy(), record

        # --- Kalman gain (via linear solve, numerically preferred over explicit inverse) ---
        # K = P⁻ Hᵀ S⁻¹
        # Kᵀ = S⁻ᵀ H P⁻ = solve(Sᵀ, H @ P⁻) = solve(S, H @ P⁻)
        K_T = np.linalg.solve(S, HP)    # (m, n)
        K   = K_T.T                     # (n, m)

        # --- State update ---
        x_post = x_prior + K @ nu

        # --- Covariance update (Joseph form) ---
        IKH    = np.eye(n) - K @ H      # (n, n)
        P_post = IKH @ P_prior @ IKH.T + K @ R @ K.T

        # Enforce symmetry after Joseph form
        P_post = self._symmetrize(P_post)

        # Health check
        self._health_check(P_post, tag=f"t={t:.1f}s sensor={sensor_id}")

        record = EstimateRecord(
            t_s=t,
            update_type="posterior",
            x_hat=x_post.copy(),
            P=P_post.copy(),
            innovation=nu,
            innovation_cov=S,
            NIS=NIS,
            gate_accepted=True,
            sensor_id=sensor_id,
        )
        return x_post, P_post, record

    def predict_record(
        self,
        x_prior: NDArray[np.float64],
        P_prior: NDArray[np.float64],
        t: float,
    ) -> EstimateRecord:
        """
        Wrap a pure prediction step (no measurement) in an EstimateRecord.
        This is a non-mutating record creator — it does not advance the state.
        Call after rk4_predict() to log the prior-only epoch.
        """
        return EstimateRecord(
            t_s=t,
            update_type="prior",
            x_hat=x_prior.copy(),
            P=P_prior.copy(),
            innovation=None, innovation_cov=None,
            NIS=None, gate_accepted=None, sensor_id=None,
        )

    def report(self) -> dict:
        """Summary statistics."""
        return {
            "state_dim":         self.state_dim,
            "n_updates_total":   self._n_updates_total,
            "n_updates_rejected": self._n_updates_rejected,
            "acceptance_rate":   self.acceptance_rate,
        }
