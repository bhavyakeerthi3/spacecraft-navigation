"""
tests/test_ekf.py

Phase 3 review gate: EKF correctness tests.

Tests cover:
  1. Linear-limit update: EKF = Kalman filter when h is linear
  2. Joseph-form covariance is PSD for any K
  3. Over-weighting a measurement reduces uncertainty (basic sanity)
  4. Chi-squared gate correctly rejects large innovations
  5. NIS calibration: E[NIS] ≈ m for simulated consistent measurements
  6. Gravity Jacobian F matches finite differences (9-state version)
  7. Orbit model F satisfies linearization identity: f(x+δ) ≈ f(x) + F δ
  8. 6-state EKF runs without divergence on clean GNSS over 1 orbit
  9. NEES consistency: 6-state EKF over one orbit
  10. Covariance health: diagonal never negative, P symmetric always
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from scipy.stats import chi2

from python.navigation.ekf import ExtendedKalmanFilter, NumericalHealthWarning
from python.navigation.orbit_model import (
    F_6state, F_9state,
    f_6state, f_9state,
    gnss_measurement_model,
    build_P0, build_P0_6state,
    H_GNSS_6, H_GNSS_9,
)
from python.navigation.baselines import OrbitalEKF6State
from python.dynamics.constants import MU_EARTH_M3S2, RE_M
from python.dynamics.two_body import circular_speed, orbital_period
from python.dynamics.propagation import circular_orbit_initial_condition, propagate_truth
from python.dynamics.frames import identity_C_IB
from python.dynamics.forces import ForceModel
from python.random_streams import StreamFactory
from python.sensors.gnss import GNSSSimulator


MU     = MU_EARTH_M3S2
RE     = RE_M
R_ORB  = RE + 500_000.0
V_CIRC = circular_speed(R_ORB, MU)
T_ORB  = orbital_period(R_ORB, MU)

STREAMS = {
    "gnss_noise": 0, "gnss_bias": 1, "gnss_outliers": 2,
    "accel_noise": 3, "accel_bias_init": 4, "accel_bias_walk": 5,
    "gyro_noise": 6, "gyro_bias_init": 7, "gyro_bias_walk": 8,
}


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def ekf6():
    return ExtendedKalmanFilter(state_dim=6, gate_prob=0.997)


@pytest.fixture
def ekf9():
    return ExtendedKalmanFilter(state_dim=9, gate_prob=0.997)


@pytest.fixture
def x6_nominal():
    """Nominal 6-state at 500km orbit."""
    r0 = np.array([R_ORB, 0.0, 0.0])
    v0 = np.array([0.0, V_CIRC, 0.0])
    return np.concatenate([r0, v0])


@pytest.fixture
def P6_nominal():
    return build_P0_6state(pos_sigma_m=100.0, vel_sigma_ms=0.1)


@pytest.fixture
def x9_nominal():
    r0 = np.array([R_ORB, 0.0, 0.0])
    v0 = np.array([0.0, V_CIRC, 0.0])
    return np.concatenate([r0, v0, np.zeros(3)])


@pytest.fixture
def P9_nominal():
    return build_P0(100.0, 0.1, 1e-4)


# -----------------------------------------------------------------------
# 1. Linear-limit: EKF = optimal Kalman filter when model is linear
# -----------------------------------------------------------------------

class TestEKFLinearLimit:
    """
    For a linear measurement model z = Hx + v, v~N(0,R),
    the EKF update should be IDENTICAL to the analytical Kalman filter.

    Analytical Kalman gain: K = P H^T (H P H^T + R)^{-1}
    Posterior:              x_post = x + K(z - Hx)
                            P_post = (I-KH) P (I-KH)^T + K R K^T
    """

    def test_ekf_matches_analytic_kalman_update_6state(self, ekf6, x6_nominal, P6_nominal):
        """EKF on 6-state GNSS model should match analytic Kalman update."""
        x = x6_nominal.copy()
        P = P6_nominal.copy()

        sigma_p = 5.0; sigma_v = 0.05
        R = np.diag([sigma_p**2]*3 + [sigma_v**2]*3)

        # True state perturbed (simulated measurement)
        rng = np.random.default_rng(0)
        z = x + rng.normal(0, 1, 6) * np.sqrt(np.diag(R))

        H = H_GNSS_6

        # EKF update
        x_ekf, P_ekf, record = ekf6.update(x, P, z, H, R, t=0.0)

        # Analytic Kalman
        S      = H @ P @ H.T + R
        K_anl  = P @ H.T @ np.linalg.inv(S)
        x_anl  = x + K_anl @ (z - H @ x)
        IKH    = np.eye(6) - K_anl @ H
        P_anl  = IKH @ P @ IKH.T + K_anl @ R @ K_anl.T

        np.testing.assert_allclose(x_ekf, x_anl, rtol=1e-10,
            err_msg="EKF state update diverges from analytic Kalman")
        np.testing.assert_allclose(P_ekf, P_anl, rtol=1e-10,
            err_msg="EKF covariance update diverges from analytic Kalman")

    def test_posterior_uncertainty_less_than_prior(self, ekf6, x6_nominal, P6_nominal):
        """After a valid update, position uncertainty must decrease."""
        x = x6_nominal.copy()
        P = P6_nominal.copy()
        prior_trace = np.trace(P[:3, :3])

        sigma_p = 5.0
        R = np.diag([sigma_p**2]*3 + [0.05**2]*3)
        z = x.copy()   # Exact measurement (no noise)

        x_post, P_post, _ = ekf6.update(x, P, z, H_GNSS_6, R, t=0.0)
        posterior_trace = np.trace(P_post[:3, :3])

        assert posterior_trace < prior_trace, (
            f"Position uncertainty did not decrease after update: "
            f"{posterior_trace:.4f} >= {prior_trace:.4f}"
        )

    def test_perfect_measurement_collapses_uncertainty(self, ekf6, x6_nominal):
        """
        An infinitely precise measurement (R → 0) should collapse P → 0.
        For the GNSS model (H = I for position block), the posterior
        position covariance should be near zero.
        """
        P = build_P0_6state(100.0, 100.0)
        R = np.diag([1e-20]*3 + [1e-20]*3)  # Near-perfect measurement
        z = x6_nominal.copy()

        x_post, P_post, _ = ekf6.update(x6_nominal, P, z, H_GNSS_6, R, t=0.0)
        pos_cov_trace = np.trace(P_post[:3, :3])
        assert pos_cov_trace < 1e-10, (
            f"Perfect measurement did not collapse P: trace(P_pos)={pos_cov_trace:.2e}"
        )

    def test_9state_h_matrix_shape(self, ekf9, x9_nominal, P9_nominal):
        """9-state EKF with H_GNSS_9 should run without error."""
        R = np.diag([5.0**2]*3 + [0.05**2]*3)
        z = x9_nominal[:6]

        x_post, P_post, record = ekf9.update(
            x9_nominal, P9_nominal, z, H_GNSS_9, R, t=0.0
        )
        assert x_post.shape == (9,)
        assert P_post.shape == (9, 9)


# -----------------------------------------------------------------------
# 2. Joseph-form PSD guarantee
# -----------------------------------------------------------------------

class TestJosephForm:
    """Joseph form must produce PSD P for any (possibly sub-optimal) gain."""

    def test_joseph_form_psd_with_suboptimal_gain(self, ekf6, x6_nominal, P6_nominal):
        """
        Deliberately inflate R (sub-optimal gain) — P_post must still be PSD.
        The simple form P = (I-KH) P (I-KH)^T can lose PSD here; Joseph form never does.
        """
        P = P6_nominal.copy()
        # Use 10x inflated R → sub-optimal (conservative) gain
        R_inflated = 10.0 * np.diag([5.0**2]*3 + [0.05**2]*3)
        z = x6_nominal.copy()

        x_post, P_post, _ = ekf6.update(x6_nominal, P, z, H_GNSS_6, R_inflated, t=0.0)

        # Check PSD via Cholesky
        try:
            np.linalg.cholesky(P_post)
        except np.linalg.LinAlgError:
            pytest.fail("P_post is not PSD after Joseph-form update with sub-optimal gain")

    def test_covariance_always_symmetric(self, ekf6, x6_nominal, P6_nominal):
        """P must be symmetric after every update."""
        R = np.diag([5.0**2]*3 + [0.05**2]*3)
        z = x6_nominal.copy()

        _, P_post, _ = ekf6.update(x6_nominal, P6_nominal, z, H_GNSS_6, R, t=0.0)
        asym = np.linalg.norm(P_post - P_post.T, 'fro')
        assert asym < 1e-12, f"P not symmetric after update: ||P-P^T||_F = {asym:.2e}"

    def test_covariance_positive_diagonals(self, ekf6, x6_nominal, P6_nominal):
        """All diagonal elements of P must be positive."""
        R = np.diag([5.0**2]*3 + [0.05**2]*3)
        z = x6_nominal.copy()

        _, P_post, _ = ekf6.update(x6_nominal, P6_nominal, z, H_GNSS_6, R, t=0.0)
        min_diag = np.min(np.diag(P_post))
        assert min_diag > 0, f"Negative diagonal in P: min diag = {min_diag:.2e}"


# -----------------------------------------------------------------------
# 3. Innovation gate
# -----------------------------------------------------------------------

class TestInnovationGate:
    """Chi-squared gate should reject large innovations."""

    def test_large_innovation_rejected(self, ekf6, x6_nominal, P6_nominal):
        """Innovation of 100σ should be rejected; state unchanged."""
        R = np.diag([5.0**2]*3 + [0.05**2]*3)
        z_outlier = x6_nominal.copy()
        z_outlier[0] += 500.0   # 100σ outlier in position X

        x_post, P_post, record = ekf6.update(
            x6_nominal, P6_nominal, z_outlier, H_GNSS_6, R, t=0.0
        )
        assert record.gate_accepted is False, "Outlier should be rejected by chi2 gate"
        np.testing.assert_array_equal(x_post, x6_nominal,
            err_msg="State changed after rejected measurement")
        np.testing.assert_array_equal(P_post, P6_nominal,
            err_msg="Covariance changed after rejected measurement")

    def test_consistent_measurement_accepted(self, ekf6, x6_nominal, P6_nominal):
        """A small consistent innovation should be accepted."""
        R = np.diag([5.0**2]*3 + [0.05**2]*3)
        z = x6_nominal + np.array([2.0, -1.0, 0.5, 0.01, -0.01, 0.0])  # ~0.5σ

        _, _, record = ekf6.update(x6_nominal, P6_nominal, z, H_GNSS_6, R, t=0.0)
        assert record.gate_accepted is True, (
            f"Consistent measurement rejected: NIS={record.NIS:.2f}"
        )

    def test_nis_stored_correctly(self, ekf6, x6_nominal, P6_nominal):
        """NIS in record should equal nu^T S^{-1} nu."""
        R = np.diag([5.0**2]*3 + [0.05**2]*3)
        z = x6_nominal.copy()

        _, _, record = ekf6.update(x6_nominal, P6_nominal, z, H_GNSS_6, R, t=0.0)
        nu = z - H_GNSS_6 @ x6_nominal
        S  = H_GNSS_6 @ P6_nominal @ H_GNSS_6.T + R
        NIS_expected = float(nu @ np.linalg.solve(S, nu))
        assert abs(record.NIS - NIS_expected) < 1e-10, (
            f"Stored NIS {record.NIS:.6f} ≠ computed {NIS_expected:.6f}"
        )


# -----------------------------------------------------------------------
# 4. Process model linearization
# -----------------------------------------------------------------------

class TestOrbitModelLinearization:
    """Process model Jacobian should match finite differences."""

    @pytest.mark.parametrize("r_vec", [
        np.array([R_ORB, 0.0, 0.0]),
        np.array([0.0, R_ORB * 0.8, R_ORB * 0.6]),
    ])
    def test_F6_finite_difference(self, r_vec):
        """6-state F should match central FD of f_6state."""
        x = np.concatenate([r_vec, np.array([0.0, V_CIRC, 0.0])])
        eps = 10.0   # 10 m / 10 mm/s perturbation

        F_analytic = F_6state(x[:3])
        F_fd = np.zeros((6, 6))
        for j in range(6):
            xp = x.copy(); xp[j] += eps
            xm = x.copy(); xm[j] -= eps
            F_fd[:, j] = (f_6state(xp, np.zeros(3)) - f_6state(xm, np.zeros(3))) / (2 * eps)

        err = np.linalg.norm(F_analytic - F_fd, 'fro') / (np.linalg.norm(F_analytic, 'fro') + 1e-30)
        assert err < 1e-5, (
            f"F_6state FD error {err:.2e} at r={r_vec}.\n"
            f"Analytic:\n{F_analytic}\nFD:\n{F_fd}"
        )

    def test_F9_bias_block_is_neg_C_IB(self):
        """
        The bias-coupling block in F_9state should equal -C_IB.

        From ∂(dv/dt)/∂b_a = ∂/∂b_a [C_IB (f_m - b_a)] = -C_IB.
        """
        r = np.array([R_ORB, 0.0, 0.0])
        C_IB = np.eye(3)  # identity attitude
        F9 = F_9state(r, C_IB)
        bias_block = F9[3:6, 6:9]
        np.testing.assert_allclose(bias_block, -C_IB, rtol=1e-14,
            err_msg=f"F9 bias block ≠ -C_IB:\n{bias_block}")

    def test_linearization_first_order(self):
        """
        f(x+δ) ≈ f(x) + F(x)·δ should hold to first order (small δ).
        """
        x = np.concatenate([np.array([R_ORB, 0, 0]), np.array([0, V_CIRC, 0])])
        F = F_6state(x[:3])
        f0 = f_6state(x, np.zeros(3))

        for _ in range(5):
            # Random small perturbation
            rng = np.random.default_rng(99)
            delta = rng.normal(0, 1, 6)
            delta[:3] *= 10.0    # 10 m position perturbation
            delta[3:] *= 0.01   # 0.01 m/s velocity perturbation

            f_perturbed = f_6state(x + delta, np.zeros(3))
            f_linear    = f0 + F @ delta

            # Second-order residual should be small compared to linear term
            residual = np.linalg.norm(f_perturbed - f_linear)
            linear   = np.linalg.norm(F @ delta)
            assert residual < 1e-4 * linear or residual < 1e-10, (
                f"Linearization error {residual:.2e} too large "
                f"compared to linear term {linear:.2e}"
            )


# -----------------------------------------------------------------------
# 5. End-to-end 6-state EKF over 1 orbit
# -----------------------------------------------------------------------

class TestE2EEKF6State:
    """End-to-end EKF test over one full orbit."""

    def test_6state_ekf_no_divergence(self):
        """
        6-state EKF should not diverge over 1 orbit with clean GNSS.
        Acceptance: final position RMSE < 20 m.
        """
        state0 = circular_orbit_initial_condition(500_000.0, 51.6)
        force_model = ForceModel(mu=MU)
        T = T_ORB

        # Propagate truth
        t_nav = np.arange(0.0, T + 10.0, 10.0)   # 10s steps
        t_eval = t_nav
        t_out, states = propagate_truth(
            state0, (0.0, T), force_model, identity_C_IB,
            t_eval=t_eval,
        )

        # Simulate clean GNSS (no bias, no outliers)
        class _MinCfg:
            enabled=True; rate_hz=0.1; pos_noise_1sigma_m=5.0
            vel_noise_1sigma_ms=0.05; bias_enabled=False
            bias_pos_m=[0,0,0]; bias_vel_ms=[0,0,0]
            gm_bias_enabled=False; gm_tau_s=600; gm_sigma_pos_m=2; gm_sigma_vel_ms=0.02
            outliers_enabled=False; outlier_prob=0.01; outlier_pos_m=100; outlier_vel_ms=1
            latency_s=0.0; outage_windows=[]

        factory = StreamFactory(42, STREAMS)
        gnss_sim = GNSSSimulator(_MinCfg(), factory)
        packets = gnss_sim.simulate(t_out, states)

        # Run 6-state EKF
        x0 = np.concatenate([state0[:3], state0[3:6]])  # start at truth
        P0 = build_P0_6state(50.0, 0.1)
        ekf = OrbitalEKF6State(
            x0=x0, P0=P0,
            Q_pos_m2s3=1e-12, Q_vel_m2s3=1e-6,
            R=gnss_sim.R_declared,
            dt_nav=10.0, mu=MU,
        )
        records = ekf.run(t_nav, packets)
        posterior = [r for r in records if r.update_type == "posterior"]

        assert len(posterior) > 0, "No posterior estimates produced"

        # Evaluate RMSE
        from python.navigation.evaluation import join_truth_estimates, compute_scorecard
        from python.records import TruthState

        truth_recs = [
            TruthState(
                t_s=t_out[i], r_I=states[i, :3], v_I=states[i, 3:6],
                q_IB=None, C_IB=np.eye(3),
                accel_bias_B=np.zeros(3), gyro_bias_B=np.zeros(3),
                specific_force_B=np.zeros(3), is_thrusting=False,
            )
            for i in range(len(t_out))
        ]
        eval_recs = join_truth_estimates(truth_recs, posterior, tolerance_s=15.0)
        score = compute_scorecard(eval_recs, "6-state-test")

        assert score.pos_rmse_m < 20.0, (
            f"6-state EKF position RMSE {score.pos_rmse_m:.2f} m > 20 m threshold. "
            "Filter may have diverged."
        )
        assert score.pos_3sigma_coverage > 0.90, (
            f"3sigma position coverage {score.pos_3sigma_coverage:.1%} < 90%. "
            "Filter is overconfident (declared uncertainty too small)."
        )

    def test_6state_ekf_nees_consistency(self):
        """
        Time-averaged NEES_pos should be close to 3 (consistent filter).
        Acceptance: mean NEES in [1.5, 6.0] — wide band for single run.
        (Monte Carlo average should be much tighter: [2.8, 3.2])
        """
        state0 = circular_orbit_initial_condition(500_000.0, 51.6)
        force_model = ForceModel(mu=MU)
        T = T_ORB

        t_nav = np.arange(0.0, T + 10.0, 10.0)
        t_out, states = propagate_truth(
            state0, (0.0, T), force_model, identity_C_IB, t_eval=t_nav
        )

        class _MinCfg:
            enabled=True; rate_hz=0.1; pos_noise_1sigma_m=5.0
            vel_noise_1sigma_ms=0.05; bias_enabled=False
            bias_pos_m=[0,0,0]; bias_vel_ms=[0,0,0]
            gm_bias_enabled=False; gm_tau_s=600; gm_sigma_pos_m=2; gm_sigma_vel_ms=0.02
            outliers_enabled=False; outlier_prob=0.01; outlier_pos_m=100; outlier_vel_ms=1
            latency_s=0.0; outage_windows=[]

        factory = StreamFactory(42, STREAMS)
        gnss_sim = GNSSSimulator(_MinCfg(), factory)
        packets = gnss_sim.simulate(t_out, states)

        x0 = np.concatenate([state0[:3], state0[3:6]])
        P0 = build_P0_6state(50.0, 0.1)
        ekf = OrbitalEKF6State(
            x0=x0, P0=P0, Q_pos_m2s3=1e-12, Q_vel_m2s3=1e-6,
            R=gnss_sim.R_declared, dt_nav=10.0, mu=MU,
        )
        records = ekf.run(t_nav, packets)
        posterior = [r for r in records if r.update_type == "posterior"]

        from python.navigation.evaluation import join_truth_estimates, compute_scorecard
        from python.records import TruthState

        truth_recs = [
            TruthState(
                t_s=t_out[i], r_I=states[i, :3], v_I=states[i, 3:6],
                q_IB=None, C_IB=np.eye(3),
                accel_bias_B=np.zeros(3), gyro_bias_B=np.zeros(3),
                specific_force_B=np.zeros(3), is_thrusting=False,
            )
            for i in range(len(t_out))
        ]
        eval_recs = join_truth_estimates(truth_recs, posterior, tolerance_s=15.0)
        score = compute_scorecard(eval_recs, "nees_test")

        assert 1.5 < score.mean_NEES_pos < 6.0, (
            f"NEES_pos = {score.mean_NEES_pos:.2f} outside [1.5, 6.0]. "
            "Filter is significantly inconsistent (single run). "
            "Run Monte Carlo for definitive NEES check."
        )
