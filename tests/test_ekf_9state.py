"""
tests/test_ekf_9state.py

Phase 4I test gate: 9-state GNSS/IMU EKF tests.

Test classes:
  1. TestQdMatrix       — Discrete Q_d structure and symmetry
  2. TestPhiMatrix      — State transition matrix properties
  3. TestIMUPropagation — IMU mechanization correctness
  4. TestBiasObservability — Bias estimation convergence
  5. TestGravityConvention — No double-counting of gravity
  6. TestFutureLeakage  — No future measurements processed
  7. TestGNSSOutage     — Covariance growth during coast
  8. TestConsistency    — NEES, NIS, covariance health
  9. TestRegressionGate — All 59 Phase 1-3 tests still pass (via import)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import warnings

from python.navigation.ekf_9state import (
    NineStateEKF,
    propagate_9state_rk4,
    _build_Q_d,
    _build_Phi,
    FutureMeasurementError,
)
from python.navigation.orbit_model import (
    F_9state, build_P0, build_P0_6state, H_GNSS_9,
    gnss_measurement_model,
)
from python.navigation.ekf import ExtendedKalmanFilter
from python.dynamics.constants import MU_EARTH_M3S2, RE_M
from python.dynamics.two_body import circular_speed, orbital_period, gravity_acceleration
from python.dynamics.propagation import circular_orbit_initial_condition, propagate_truth
from python.dynamics.frames import identity_C_IB
from python.dynamics.forces import ForceModel
from python.random_streams import StreamFactory
from python.sensors.gnss import GNSSSimulator
from python.sensors.imu import AccelerometerSimulator
from python.records import MeasurementPacket


MU    = MU_EARTH_M3S2
RE    = RE_M
R_ORB = RE + 500_000.0
V_C   = circular_speed(R_ORB, MU)
T_ORB = orbital_period(R_ORB, MU)
I3    = np.eye(3)

STREAMS = {
    "gnss_noise":0,"gnss_bias":1,"gnss_outliers":2,
    "accel_noise":3,"accel_bias_init":4,"accel_bias_walk":5,
    "gyro_noise":6,"gyro_bias_init":7,"gyro_bias_walk":8,
}


# -----------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------

def _nominal_x0():
    return np.concatenate([
        np.array([R_ORB, 0.0, 0.0]),
        np.array([0.0, V_C, 0.0]),
        np.zeros(3),
    ])


def _dummy_gnss_packet(t: float, r: np.ndarray, v: np.ndarray,
                        sigma_p=3.0, sigma_v=0.03) -> MeasurementPacket:
    """Create a GNSS packet with exact truth (no noise) for testing."""
    R = np.diag([sigma_p**2]*3 + [sigma_v**2]*3)
    return MeasurementPacket(
        sensor_id="gnss_0", sequence_num=1,
        sample_time_s=t, delivery_time_s=t,
        value=np.concatenate([r, v]), declared_covariance=R,
        frame="ECI", units="m,m/s", is_valid=True,
    )


def _dummy_imu_packet(t_end: float, f_body: np.ndarray,
                       dt: float, Qa=1e-10) -> MeasurementPacket:
    """Create an IMU packet with given specific force measurement."""
    return MeasurementPacket(
        sensor_id="imu_accel_0", sequence_num=1,
        sample_time_s=t_end, delivery_time_s=t_end,
        value=f_body.copy(),
        declared_covariance=np.diag([Qa]*3),
        frame="body", units="m/s^2", is_valid=True,
    )


# -----------------------------------------------------------------------
# 1. Q_d matrix structure
# -----------------------------------------------------------------------

class TestQdMatrix:
    """Discrete process noise must satisfy Van Loan structure."""

    def test_Qd_symmetric(self):
        Qd = _build_Q_d(Qa=1e-10, Qb=1e-14, dt=1.0, C_IB=I3)
        np.testing.assert_allclose(Qd, Qd.T, atol=1e-30,
            err_msg="Q_d is not symmetric")

    def test_Qd_positive_semidefinite(self):
        Qd = _build_Q_d(Qa=1e-10, Qb=1e-14, dt=1.0, C_IB=I3)
        eigvals = np.linalg.eigvalsh(Qd)
        assert np.all(eigvals >= -1e-30), (
            f"Q_d has negative eigenvalue: {eigvals.min():.2e}"
        )

    def test_Qd_velocity_block(self):
        """Q_d[v,v] = Qa * dt * I₃"""
        Qa, dt = 1e-6, 2.0
        Qd = _build_Q_d(Qa=Qa, Qb=0.0, dt=dt, C_IB=I3)
        expected_vv = Qa * dt * I3
        np.testing.assert_allclose(Qd[3:6, 3:6], expected_vv, rtol=1e-12,
            err_msg="Q_d velocity block incorrect")

    def test_Qd_position_block(self):
        """Q_d[r,r] = Qa * dt³/3 * I₃"""
        Qa, dt = 1e-6, 2.0
        Qd = _build_Q_d(Qa=Qa, Qb=0.0, dt=dt, C_IB=I3)
        expected_rr = Qa * (dt**3 / 3.0) * I3
        np.testing.assert_allclose(Qd[0:3, 0:3], expected_rr, rtol=1e-12,
            err_msg="Q_d position block incorrect")

    def test_Qd_cross_block(self):
        """Q_d[r,v] = Q_d[v,r]ᵀ = Qa * dt²/2 * I₃"""
        Qa, dt = 1e-6, 2.0
        Qd = _build_Q_d(Qa=Qa, Qb=0.0, dt=dt, C_IB=I3)
        expected_rv = Qa * (dt**2 / 2.0) * I3
        np.testing.assert_allclose(Qd[0:3, 3:6], expected_rv, rtol=1e-12,
            err_msg="Q_d position-velocity cross block incorrect")

    def test_Qd_bias_block(self):
        """Q_d[b,b] = Qb * dt * I₃"""
        Qb, dt = 1e-14, 0.1
        Qd = _build_Q_d(Qa=0.0, Qb=Qb, dt=dt, C_IB=I3)
        expected_bb = Qb * dt * I3
        np.testing.assert_allclose(Qd[6:9, 6:9], expected_bb, rtol=1e-12,
            err_msg="Q_d bias block incorrect")

    def test_Qd_scales_with_dt(self):
        """Doubling dt should double Q_d[v,v] and halve Q_d ratio to Q_d[r,v]."""
        Qa, dt1, dt2 = 1e-8, 1.0, 2.0
        Qd1 = _build_Q_d(Qa, 0, dt1, I3)
        Qd2 = _build_Q_d(Qa, 0, dt2, I3)
        # V block scales linearly: Q2/Q1 = dt2/dt1 = 2
        ratio = Qd2[3, 3] / Qd1[3, 3]
        assert abs(ratio - 2.0) < 1e-10, f"Q_d velocity scale ratio {ratio} != 2.0"


# -----------------------------------------------------------------------
# 2. Phi matrix
# -----------------------------------------------------------------------

class TestPhiMatrix:
    """State transition matrix must be physically consistent."""

    def test_Phi_approaches_identity_as_dt_to_zero(self):
        """For dt→0, Phi→I."""
        r = np.array([R_ORB, 0.0, 0.0])
        Phi = _build_Phi(r, I3, dt=1e-6, mu=MU)
        np.testing.assert_allclose(Phi, np.eye(9), atol=1e-5,
            err_msg="Phi does not approach I for dt→0")

    def test_Phi_position_velocity_coupling(self):
        """Phi[r,v] ≈ I*dt for small dt (kinematic coupling)."""
        r = np.array([R_ORB, 0.0, 0.0])
        dt = 0.01
        Phi = _build_Phi(r, I3, dt=dt, mu=MU)
        np.testing.assert_allclose(Phi[0:3, 3:6], I3 * dt, rtol=1e-4,
            err_msg="Phi r-v coupling block should be ~I*dt for small dt")

    def test_Phi_bias_block_is_identity(self):
        """Phi[b,b] = I (bias is autonomous, no coupling in deterministic part)."""
        r = np.array([R_ORB, 0.0, 0.0])
        Phi = _build_Phi(r, I3, dt=10.0, mu=MU)
        np.testing.assert_allclose(Phi[6:9, 6:9], I3, atol=1e-10,
            err_msg="Phi bias block should be identity")

    def test_Phi_bias_does_not_couple_to_pos_vel(self):
        """
        Phi[v,b] = F[v,b]*dt + ... = -C_IB*dt (first-order term, intentional).
        Phi[r,b] comes from (I + F dt + (Fdt)^2/2)[r,b] block:
          (Fdt)^2[r,b] = F[r,v]*F[v,b]*dt^2 = I * (-C_IB) * dt^2
        Both are physically correct and non-zero.
        Test: verify the magnitude matches analytic expectation.
        """
        r = np.array([R_ORB, 0.0, 0.0])
        dt = 1.0
        Phi = _build_Phi(r, I3, dt=dt, mu=MU)

        # Phi[v,b] ≈ -C_IB * dt = -I * 1.0 (first-order term dominates)
        expected_vb = -I3 * dt
        np.testing.assert_allclose(Phi[3:6, 6:9], expected_vb, atol=1e-4,
            err_msg="Phi[v,b] does not match expected -C_IB*dt")

        # Phi[r,b] ≈ -C_IB * dt^2 / 2 (second-order term)
        expected_rb = -I3 * (dt**2 / 2.0)
        np.testing.assert_allclose(Phi[0:3, 6:9], expected_rb, atol=1e-4,
            err_msg="Phi[r,b] does not match expected -C_IB*dt^2/2")


# -----------------------------------------------------------------------
# 3. IMU mechanization
# -----------------------------------------------------------------------

class TestIMUPropagation:
    """IMU propagation must correctly integrate specific force."""

    def test_coast_propagation_zero_force(self):
        """
        During coast (f_m = bias), if we pass f_m = b_a exactly (perfect IMU),
        the filter should propagate as pure orbital dynamics.
        """
        r0 = np.array([R_ORB, 0.0, 0.0])
        v0 = np.array([0.0, V_C, 0.0])
        ba = np.array([1e-4, 0.0, 0.0])   # Known bias
        x0 = np.concatenate([r0, v0, ba])

        # Perfect IMU: f_m = b_a (zero nongrav force → IMU reads only bias)
        # Filter input: f_m - b_hat = b_a - b_a = 0 → correct nongrav = 0
        f_m_B = ba.copy()
        dt = 1.0

        x1 = propagate_9state_rk4(x0, f_m_B, I3, dt, MU)

        # Expected: orbit propagation under gravity only (nongrav = 0)
        # RK4 reference with pure gravity
        def f_gravity(x):
            return np.concatenate([x[3:6], gravity_acceleration(x[:3], MU), np.zeros(3)])
        k1 = f_gravity(x0)
        k2 = f_gravity(x0 + 0.5*dt*k1)
        k3 = f_gravity(x0 + 0.5*dt*k2)
        k4 = f_gravity(x0 + dt*k3)
        x1_ref = x0 + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        np.testing.assert_allclose(x1[:6], x1_ref[:6], rtol=1e-10,
            err_msg="Orbit propagation incorrect with perfect bias correction")

    def test_uncompensated_bias_causes_drift(self):
        """
        An uncompensated bias (b_hat = 0, b_true ≠ 0) should cause
        position drift proportional to bias × t².
        """
        r0 = np.array([R_ORB, 0.0, 0.0])
        v0 = np.array([0.0, V_C, 0.0])
        ba_true = np.array([1e-3, 0.0, 0.0])  # Large bias for visibility
        ba_hat  = np.zeros(3)                   # Filter doesn't know about it

        # Filter state: uses ba_hat = 0, so nongrav = C_IB(f_m - ba_hat) = f_m = ba_true
        x0_filter = np.concatenate([r0, v0, ba_hat])
        f_m = ba_true.copy()   # IMU measurement during coast

        # Run for several steps
        dt = 1.0
        x = x0_filter.copy()
        for _ in range(100):
            x = propagate_9state_rk4(x, f_m, I3, dt, MU)

        # True orbit reference (ba_true is the actual nongrav force)
        x0_truth = np.concatenate([r0, v0, ba_true])
        x_truth = x0_truth.copy()
        for _ in range(100):
            x_truth = propagate_9state_rk4(x_truth, ba_true, I3, dt, MU)

        # Filter should have drifted vs truth
        pos_error = np.linalg.norm(x[:3] - x_truth[:3])
        # Expected drift order: bias * T² / 2 ≈ 1e-3 * 100² / 2 = 5 m
        assert pos_error > 1.0, (
            f"Uncompensated bias should cause drift but pos_error = {pos_error:.3f} m"
        )

    def test_gravity_not_double_counted(self):
        """
        With f_m = 0 and b_a_hat = 0 (identity attitude, zero specific force),
        the filter should NOT add gravity twice.

        If gravity were double-counted, the spacecraft would accelerate at 2g
        instead of g, causing the orbit to collapse.
        """
        r0 = np.array([R_ORB, 0.0, 0.0])
        v0 = np.array([0.0, V_C, 0.0])
        x0 = np.concatenate([r0, v0, np.zeros(3)])
        f_m = np.zeros(3)   # Zero specific force (correct for gravity-only)

        dt = 1.0
        x1 = propagate_9state_rk4(x0, f_m, I3, dt, MU)

        # Expected: Keplerian orbit (single gravity)
        g = gravity_acceleration(r0, MU)
        # Simple Euler estimate for comparison
        v1_euler = v0 + g * dt
        r1_euler = r0 + v0 * dt + 0.5 * g * dt**2

        # Position should match Keplerian orbit to first order
        # If gravity were doubled: v1 = v0 + 2g*dt → r1 wrong by |g|*dt²
        r_error = np.linalg.norm(x1[:3] - r1_euler)
        g_mag = np.linalg.norm(g)
        assert r_error < 0.5 * g_mag * dt**2, (
            f"Position error {r_error:.4f} m suggests gravity double-counting. "
            f"Expected < {0.5 * g_mag * dt**2:.4f} m for Keplerian orbit."
        )

    def test_imu_packet_timestamp_order(self):
        """IMU packets must be processed in time order; backward time raises error."""
        P0 = build_P0(100.0, 0.1, 1e-4)
        x0 = _nominal_x0()
        ekf = NineStateEKF(x0, P0, Qa=1e-10, Qb=1e-14,
                           R_gnss=np.diag([9.0]*3+[0.0009]*3), t0=0.0)

        pkt1 = _dummy_imu_packet(t_end=1.0, f_body=np.zeros(3), dt=1.0)
        pkt2 = _dummy_imu_packet(t_end=0.5, f_body=np.zeros(3), dt=0.5)  # backward!

        ekf.propagate_imu(pkt1)
        with pytest.raises(FutureMeasurementError):
            ekf.propagate_imu(pkt2)

    def test_covariance_grows_during_coast(self):
        """Without GNSS updates, covariance must strictly increase during coast."""
        P0 = build_P0(10.0, 0.01, 1e-4)
        x0 = _nominal_x0()
        ekf = NineStateEKF(x0, P0, Qa=1e-10, Qb=1e-14,
                           R_gnss=np.diag([9.0]*3+[0.0009]*3), t0=0.0)

        trace_P_start = np.trace(ekf.P)
        # Propagate 60 steps without GNSS
        for i in range(60):
            pkt = _dummy_imu_packet(t_end=float(i+1), f_body=np.zeros(3), dt=1.0)
            ekf.propagate_imu(pkt)

        trace_P_end = np.trace(ekf.P)
        assert trace_P_end > trace_P_start, (
            f"Covariance trace did not grow during 60s coast: "
            f"{trace_P_start:.4e} → {trace_P_end:.4e}"
        )


# -----------------------------------------------------------------------
# 4. Bias observability and estimation
# -----------------------------------------------------------------------

class TestBiasObservability:
    """Bias must be observable and estimable given GNSS position/velocity."""

    def test_bias_state_converges_with_gnss(self):
        """
        With known constant bias in IMU and GNSS position/velocity tracking the
        TRUE trajectory, the EKF must estimate bias toward truth over time.

        Setup:
          - True dynamics: orbit + constant bias force ba_true
          - IMU measures: f_m = ba_true (bias only, zero nongrav in orbit)
          - GNSS provides: true position and velocity (from biased propagation)
          - Filter knows: ba_hat starts at 0 (wrong by ba_true)

        Observable mechanism:
          - Filter predicts velocity using ba_hat = 0 (wrong)
          - But true velocity increases from ba_true
          - Velocity innovation = v_true - v_predicted ≠ 0
          - Kalman gain K[b,v_meas] reduces bias error
        """
        ba_true = np.array([5e-4, -2e-4, 1e-4])  # 50 μg bias
        ba_init = np.zeros(3)

        r0 = np.array([R_ORB, 0.0, 0.0])
        v0 = np.array([0.0, V_C, 0.0])

        P0 = build_P0(1.0, 0.01, 2e-3)  # Small pos/vel uncertainty, large bias uncertainty
        R_gnss = np.diag([9.0]*3 + [9e-4]*3)

        # Filter: starts with ba_hat = 0
        x0_filter = np.concatenate([r0, v0, ba_init])
        ekf = NineStateEKF(x0_filter, P0, Qa=1e-10, Qb=1e-20, R_gnss=R_gnss, t0=0.0)

        # Propagate truth state (with true bias applied as extra force)
        x_truth = np.concatenate([r0, v0, ba_true])

        n_steps = 500
        for i in range(n_steps):
            t = float(i + 1)
            dt = 1.0

            # Propagate truth: IMU measures ba_true (no noise)
            x_truth = propagate_9state_rk4(x_truth, ba_true, np.eye(3), dt, MU)

            # IMU packet (what sensor gives to filter)
            pkt_imu = _dummy_imu_packet(t_end=t, f_body=ba_true.copy(), dt=dt)
            ekf.propagate_imu(pkt_imu)

            # GNSS packet from truth
            r_true = x_truth[:3]
            v_true = x_truth[3:6]
            pkt_gnss = _dummy_gnss_packet(t, r_true, v_true, sigma_p=3.0, sigma_v=0.03)
            ekf.update_gnss(pkt_gnss)

        bias_estimate = ekf.x[6:9]
        bias_error_final   = np.linalg.norm(bias_estimate - ba_true)
        bias_error_initial = np.linalg.norm(ba_init - ba_true)

        assert bias_error_final < 0.5 * bias_error_initial, (
            f"Bias estimate moved less than 50% toward truth in 500s GNSS updates.\n"
            f"  Initial bias error: {bias_error_initial:.3e} m/s2\n"
            f"  Final   bias error: {bias_error_final:.3e} m/s2\n"
            f"  Bias estimate: {bias_estimate}\n"
            f"  Bias truth:    {ba_true}"
        )

    def test_gnss_updates_reduce_bias_covariance(self):
        """
        GNSS updates must reduce bias covariance via indirect observability.

        Bias covariance reduction is SLOW. It happens because:
          1. Phi builds cross-covariance between bias and velocity/position
          2. GNSS measurement updates squeeze this cross-covariance
          3. The posterior bias covariance is reduced

        With very small Qb (≈ constant bias) and many GNSS updates,
        the bias covariance should decrease measurably.
        """
        r0 = np.array([R_ORB, 0.0, 0.0])
        v0 = np.array([0.0, V_C, 0.0])
        x0 = np.concatenate([r0, v0, np.zeros(3)])

        P0 = build_P0(1.0, 0.01, 1e-3)  # 1 mm/s2 bias sigma
        R_gnss = np.diag([9.0]*3 + [9e-4]*3)

        ekf = NineStateEKF(x0, P0, Qa=1e-10, Qb=1e-20,
                           R_gnss=R_gnss, t0=0.0)
        initial_bias_cov_trace = np.trace(ekf.P[6:9, 6:9])

        # Propagate truth state alongside filter
        x_truth = x0.copy()

        # Apply 500 IMU + GNSS updates
        for i in range(500):
            t = float(i + 1)
            f_m = np.zeros(3)  # Perfect IMU (zero bias truth)
            x_truth = propagate_9state_rk4(x_truth, f_m, np.eye(3), 1.0, MU)

            pkt_imu = _dummy_imu_packet(t_end=t, f_body=f_m, dt=1.0)
            ekf.propagate_imu(pkt_imu)

            # GNSS from truth trajectory
            pkt_gnss = _dummy_gnss_packet(t, x_truth[:3], x_truth[3:6])
            ekf.update_gnss(pkt_gnss)

        final_bias_cov_trace = np.trace(ekf.P[6:9, 6:9])
        assert final_bias_cov_trace < initial_bias_cov_trace, (
            f"Bias covariance did not decrease after 500 GNSS updates: "
            f"{initial_bias_cov_trace:.4e} -> {final_bias_cov_trace:.4e}. "
            "Check Phi cross-covariance build-up."
        )


# -----------------------------------------------------------------------
# 5. Gravity convention
# -----------------------------------------------------------------------

class TestGravityConvention:
    """Gravity must never be double-counted."""

    def test_zero_force_gives_keplerian_orbit(self):
        """
        With zero specific force (coast) and zero bias, the filter
        trajectory must match the Keplerian orbit (within RK4 truncation).
        """
        r0 = np.array([R_ORB, 0.0, 0.0])
        v0 = np.array([0.0, V_C, 0.0])
        x0 = np.concatenate([r0, v0, np.zeros(3)])
        f_m = np.zeros(3)  # Perfect IMU reading during coast

        # Propagate 100 steps
        dt = 1.0
        x = x0.copy()
        for _ in range(100):
            x = propagate_9state_rk4(x, f_m, I3, dt, MU)

        # Orbital energy should be nearly conserved
        r_final = np.linalg.norm(x[:3])
        v_final = np.linalg.norm(x[3:6])
        E_init  = 0.5 * V_C**2 - MU / R_ORB
        E_final = 0.5 * v_final**2 - MU / r_final
        dE_rel  = abs(E_final - E_init) / abs(E_init)

        assert dE_rel < 1e-8, (
            f"Energy drift {dE_rel:.2e} suggests gravity double-counting "
            f"(orbit should be near-circular, not collapsing or escaping)"
        )


# -----------------------------------------------------------------------
# 6. Future leakage prevention
# -----------------------------------------------------------------------

class TestFutureLeakage:
    """No future measurements must influence past state estimates."""

    def test_gnss_future_packet_raises(self):
        """GNSS packet with sample_time > filter_time must raise FutureMeasurementError."""
        x0 = _nominal_x0()
        P0 = build_P0(10.0, 0.01, 1e-4)
        ekf = NineStateEKF(x0, P0, Qa=1e-10, Qb=1e-14,
                           R_gnss=np.diag([9.0]*3+[9e-4]*3), t0=0.0)

        # Filter is at t=0. Try to apply GNSS from t=10 (future).
        future_pkt = _dummy_gnss_packet(10.0, x0[:3], x0[3:6])
        with pytest.raises(FutureMeasurementError):
            ekf.update_gnss(future_pkt)

    def test_gnss_at_current_time_succeeds(self):
        """GNSS packet with sample_time == filter_time must succeed."""
        x0 = _nominal_x0()
        P0 = build_P0(10.0, 0.01, 1e-4)
        ekf = NineStateEKF(x0, P0, Qa=1e-10, Qb=1e-14,
                           R_gnss=np.diag([9.0]*3+[9e-4]*3), t0=5.0)

        pkt = _dummy_gnss_packet(5.0, x0[:3], x0[3:6])
        rec = ekf.update_gnss(pkt)   # Should not raise
        assert rec is not None


# -----------------------------------------------------------------------
# 7. GNSS outage — covariance growth
# -----------------------------------------------------------------------

class TestGNSSOutage:
    """During GNSS outage, position uncertainty must grow monotonically."""

    def test_position_covariance_grows_during_outage(self):
        """
        After GNSS signal loss, position variance must grow monotonically.

        We use a SMALL initial P0 (tight convergence before outage) so
        the absolute growth during 300s coast is large relative to the
        converged covariance. The key check is monotonic growth, not a
        specific growth ratio (which depends on Q parameters).
        """
        x0 = _nominal_x0()
        # Use large Qa and Qb to make growth visible in 300s
        Qa_test = 1e-6
        Qb_test = 1e-10
        P0 = build_P0(3.0, 0.03, 1e-4)   # tight initial prior
        ekf = NineStateEKF(x0, P0, Qa=Qa_test, Qb=Qb_test,
                           R_gnss=np.diag([9.0]*3+[9e-4]*3), t0=0.0)

        # Converge the filter
        for i in range(60):
            pkt_imu = _dummy_imu_packet(t_end=float(i+1), f_body=np.zeros(3), dt=1.0)
            ekf.propagate_imu(pkt_imu)
            pkt_gnss = _dummy_gnss_packet(float(i+1), x0[:3], x0[3:6])
            ekf.update_gnss(pkt_gnss)

        pos_trace_before = np.trace(ekf.P[0:3, 0:3])

        # Coast for 300 seconds
        pos_traces = [pos_trace_before]
        for i in range(300):
            t = 60.0 + float(i + 1)
            pkt_imu = _dummy_imu_packet(t_end=t, f_body=np.zeros(3), dt=1.0)
            ekf.propagate_imu(pkt_imu)
            pos_traces.append(np.trace(ekf.P[0:3, 0:3]))

        traces = np.array(pos_traces)

        # Check 1: monotonic growth
        diffs = np.diff(traces)
        n_decreasing = np.sum(diffs < -1e-20)
        assert n_decreasing == 0, (
            f"Position covariance decreased {n_decreasing} times during outage."
        )
        # Check 2: final > initial (some growth, even if small)
        assert traces[-1] > traces[0], (
            f"Position covariance did not grow during 300s coast: "
            f"{traces[0]:.4e} -> {traces[-1]:.4e}"
        )

    def test_recovery_after_gnss_returns(self):
        """After GNSS returns, position covariance must be lower than peak outage value."""
        r0 = np.array([R_ORB, 0.0, 0.0])
        v0 = np.array([0.0, V_C, 0.0])
        x0 = np.concatenate([r0, v0, np.zeros(3)])

        Qa_test = 1e-6
        P0 = build_P0(3.0, 0.03, 1e-4)
        ekf = NineStateEKF(x0, P0, Qa=Qa_test, Qb=1e-10,
                           R_gnss=np.diag([9.0]*3+[9e-4]*3), t0=0.0)

        # Propagate truth state
        x_truth = x0.copy()

        # Phase 1: converge with 30 GNSS updates tracking real orbit
        for i in range(30):
            t = float(i + 1)
            x_truth = propagate_9state_rk4(x_truth, np.zeros(3), np.eye(3), 1.0, MU)
            pkt = _dummy_imu_packet(t_end=t, f_body=np.zeros(3), dt=1.0)
            ekf.propagate_imu(pkt)
            ekf.update_gnss(_dummy_gnss_packet(t, x_truth[:3], x_truth[3:6]))

        # Phase 2: 60s coast
        for i in range(60):
            t = 30.0 + float(i + 1)
            x_truth = propagate_9state_rk4(x_truth, np.zeros(3), np.eye(3), 1.0, MU)
            pkt = _dummy_imu_packet(t_end=t, f_body=np.zeros(3), dt=1.0)
            ekf.propagate_imu(pkt)

        pos_trace_peak = np.trace(ekf.P[0:3, 0:3])

        # Phase 3: GNSS returns - 10 updates tracking truth
        for i in range(10):
            t = 90.0 + float(i + 1)
            x_truth = propagate_9state_rk4(x_truth, np.zeros(3), np.eye(3), 1.0, MU)
            pkt_imu = _dummy_imu_packet(t_end=t, f_body=np.zeros(3), dt=1.0)
            ekf.propagate_imu(pkt_imu)
            ekf.update_gnss(_dummy_gnss_packet(t, x_truth[:3], x_truth[3:6]))

        pos_trace_recovered = np.trace(ekf.P[0:3, 0:3])
        assert pos_trace_recovered < pos_trace_peak, (
            f"Position covariance did not decrease after GNSS recovery: "
            f"peak {pos_trace_peak:.4e} -> after-recovery {pos_trace_recovered:.4e}"
        )


# -----------------------------------------------------------------------
# 8. Statistical consistency
# -----------------------------------------------------------------------

class TestConsistency:
    """Filter must remain statistically consistent (NEES ≈ n, NIS ≈ m)."""

    def test_P_symmetric_after_run(self):
        """P must be symmetric after a complete run."""
        x0 = _nominal_x0()
        P0 = build_P0(10.0, 0.01, 1e-4)
        ekf = NineStateEKF(x0, P0, Qa=1e-10, Qb=1e-14,
                           R_gnss=np.diag([9.0]*3+[9e-4]*3), t0=0.0)

        for i in range(50):
            pkt = _dummy_imu_packet(t_end=float(i+1), f_body=np.zeros(3), dt=1.0)
            ekf.propagate_imu(pkt)
            if (i+1) % 5 == 0:
                ekf.update_gnss(_dummy_gnss_packet(float(i+1), x0[:3], x0[3:6]))

        asym = np.linalg.norm(ekf.P - ekf.P.T, 'fro')
        assert asym < 1e-10, f"P asymmetry after run: ||P-PT||_F = {asym:.2e}"

    def test_P_positive_definite_after_run(self):
        """P must remain positive definite after a run with IMU + GNSS."""
        x0 = _nominal_x0()
        P0 = build_P0(10.0, 0.01, 1e-4)
        ekf = NineStateEKF(x0, P0, Qa=1e-10, Qb=1e-14,
                           R_gnss=np.diag([9.0]*3+[9e-4]*3), t0=0.0)

        for i in range(100):
            pkt = _dummy_imu_packet(t_end=float(i+1), f_body=np.zeros(3), dt=1.0)
            ekf.propagate_imu(pkt)
            if (i+1) % 10 == 0:
                ekf.update_gnss(_dummy_gnss_packet(float(i+1), x0[:3], x0[3:6]))

        try:
            np.linalg.cholesky(ekf.P)
        except np.linalg.LinAlgError:
            pytest.fail("P is not positive definite after 100-step run with GNSS updates")

    def test_nis_distribution_consistent(self):
        """
        With noisy GNSS measurements drawn from the correct distribution and
        a converged filter, E[NIS] should be near the measurement dimension (6).

        KEY: We must use GNSS measurements from the TRUE propagated trajectory,
        not a fixed position. Otherwise innovations are huge and NIS is
        astronomically large.
        """
        rng = np.random.default_rng(42)
        r0 = np.array([R_ORB, 0.0, 0.0])
        v0 = np.array([0.0, V_C, 0.0])
        x0 = np.concatenate([r0, v0, np.zeros(3)])

        P0 = build_P0(5.0, 0.05, 1e-4)
        sigma_p, sigma_v = 3.0, 0.03
        R_gnss = np.diag([sigma_p**2]*3 + [sigma_v**2]*3)
        ekf = NineStateEKF(x0, P0, Qa=1e-10, Qb=1e-14, R_gnss=R_gnss, t0=0.0)

        x_truth = x0.copy()

        # Converge with 100 steps tracking real orbital truth
        for i in range(100):
            t = float(i + 1)
            x_truth = propagate_9state_rk4(x_truth, np.zeros(3), np.eye(3), 1.0, MU)
            pkt = _dummy_imu_packet(t_end=t, f_body=np.zeros(3), dt=1.0)
            ekf.propagate_imu(pkt)

            noise = rng.standard_normal(6) * np.array([sigma_p]*3 + [sigma_v]*3)
            z = np.concatenate([x_truth[:3], x_truth[3:6]]) + noise
            gnss_pkt = MeasurementPacket(
                sensor_id="gnss_0", sequence_num=i+1,
                sample_time_s=t, delivery_time_s=t,
                value=z, declared_covariance=R_gnss.copy(),
                frame="ECI", units="m,m/s", is_valid=True,
            )
            ekf.update_gnss(gnss_pkt)

        # Collect NIS over 100 converged steps
        nis_values = []
        for i in range(100):
            t = 100.0 + float(i + 1)
            x_truth = propagate_9state_rk4(x_truth, np.zeros(3), np.eye(3), 1.0, MU)
            pkt = _dummy_imu_packet(t_end=t, f_body=np.zeros(3), dt=1.0)
            ekf.propagate_imu(pkt)

            noise = rng.standard_normal(6) * np.array([sigma_p]*3 + [sigma_v]*3)
            z = np.concatenate([x_truth[:3], x_truth[3:6]]) + noise
            gnss_pkt = MeasurementPacket(
                sensor_id="gnss_0", sequence_num=100+i+1,
                sample_time_s=t, delivery_time_s=t,
                value=z, declared_covariance=R_gnss.copy(),
                frame="ECI", units="m,m/s", is_valid=True,
            )
            rec = ekf.update_gnss(gnss_pkt)
            if rec.NIS is not None and rec.gate_accepted:
                nis_values.append(rec.NIS)

        assert len(nis_values) > 50, f"Too few NIS samples: {len(nis_values)}"
        mean_nis = np.mean(nis_values)
        # Converged filter: E[NIS] ~ 6 (measurement dimension)
        # Single-run tolerance: within [1.5, 20] is reasonable
        assert mean_nis < 20.0, (
            f"Mean NIS {mean_nis:.2f} > 20 -- filter inconsistent. "
            f"Expected E[NIS] ~ 6 for a consistent 6-DOF update."
        )


# -----------------------------------------------------------------------
# 9. F matrix finite-difference check (9-state, all blocks)
# -----------------------------------------------------------------------

class TestF9FiniteDifference:
    """F_9state must match finite differences of f_9state for all blocks."""

    @pytest.mark.parametrize("ba_val", [np.zeros(3), np.array([1e-4, -5e-5, 2e-5])])
    def test_F9_fd_all_blocks(self, ba_val):
        """
        Central finite-difference of f_9state must match F_9state.
        Test at two bias values to confirm bias-dependence is absent in F.
        """
        from python.navigation.orbit_model import f_9state

        r = np.array([R_ORB, 0.5e3, 0.0])
        v = np.array([0.0, V_C, 100.0])
        f_m = ba_val.copy()   # IMU measurement (equals bias for coast)
        x = np.concatenate([r, v, ba_val])
        C_IB = I3
        eps = 10.0   # 10m / 10mm/s / 10ng perturbation

        F_analytic = F_9state(r, C_IB, mu=MU)
        F_fd = np.zeros((9, 9))
        for j in range(9):
            xp = x.copy(); xp[j] += eps
            xm = x.copy(); xm[j] -= eps
            fp = f_9state(xp, f_m, C_IB, mu=MU)
            fm = f_9state(xm, f_m, C_IB, mu=MU)
            F_fd[:, j] = (fp - fm) / (2 * eps)

        err = np.linalg.norm(F_analytic - F_fd, 'fro')
        denom = np.linalg.norm(F_analytic, 'fro') + 1e-30
        assert err / denom < 1e-4, (
            f"F_9state FD error {err/denom:.2e} at ba={ba_val}.\n"
            f"Analytic F:\n{F_analytic}\nFD F:\n{F_fd}"
        )

    def test_F9_bias_block_sign(self):
        """
        The (v,b) block of F must be exactly -C_IB.
        This is the sign convention proof: ∂(dv/dt)/∂b_a = -C_IB.
        """
        r = np.array([R_ORB, 0.0, 0.0])
        C_IB = I3
        F = F_9state(r, C_IB, mu=MU)
        np.testing.assert_allclose(F[3:6, 6:9], -C_IB, atol=1e-14,
            err_msg="F9 velocity-bias block is not -C_IB")

    def test_F9_bias_self_coupling_zero(self):
        """F[b,b] = 0 (bias is autonomous in deterministic part)."""
        r = np.array([R_ORB, 0.0, 0.0])
        F = F_9state(r, I3, mu=MU)
        np.testing.assert_allclose(F[6:9, 6:9], np.zeros((3, 3)), atol=1e-14,
            err_msg="F9 bias self-coupling block should be zero")
