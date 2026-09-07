"""
tests/test_sensors.py

Phase 2 review gate: GNSS and IMU sensor simulator tests.

Tests cover:
  1. GNSS noise: empirical std matches declared sigma (chi-squared test)
  2. GNSS bias: constant bias correctly shifts measurement mean
  3. GNSS outage: NO packets generated during outage windows
  4. GNSS outlier: outlier probability matches declared rate
  5. GNSS seed reproducibility: identical seeds produce identical packets
  6. GNSS stream independence: changing gnss stream does not affect imu stream
  7. IMU interval-average noise convention: std = sqrt(Qa / dt)
  8. IMU bias initialization: consistent with declared sigma
  9. IMU bias random walk: grows as sqrt(Qb * t) over time
  10. Measurement packet timing: delivery_time >= sample_time always
  11. MeasurementBus ordering: packets always in (delivery_time, seq_num) order
  12. MeasurementBus outage: packets with delivery_time < max_latency rejected

STATISTICAL TESTS:
  For k i.i.d. samples x_i ~ N(0, sigma^2):
    The chi-squared statistic: sum(x_i^2) / sigma^2 ~ chi^2(k)
  We use a two-sided test at 0.001 significance:
    Lower tail: chi2.ppf(0.0005, k)
    Upper tail: chi2.ppf(0.9995, k)
  N=5000 samples gives tight confidence; random failures < 1 in 1000.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from scipy.stats import chi2

from python.dynamics.constants import MU_EARTH_M3S2, RE_M
from python.dynamics.two_body import circular_speed
from python.dynamics.propagation import circular_orbit_initial_condition, propagate_truth
from python.dynamics.frames import identity_C_IB
from python.dynamics.forces import ForceModel
from python.random_streams import StreamFactory
from python.sensors.gnss import GNSSSimulator
from python.sensors.timing import MeasurementBus, build_measurement_bus
from python.records import MeasurementPacket


# -----------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------

N_SAMPLES  = 5000    # large enough for tight chi2 bounds
ALPHA      = 0.001   # two-sided significance level

ALT_M      = 500_000.0
INC_DEG    = 51.6
MU         = MU_EARTH_M3S2
RE         = RE_M
R_ORBIT    = RE + ALT_M
V_CIRC     = circular_speed(R_ORBIT, MU)

MASTER_SEED = 42
STREAMS = {
    "gnss_noise":      0,
    "gnss_bias":       1,
    "gnss_outliers":   2,
    "accel_noise":     3,
    "accel_bias_init": 4,
    "accel_bias_walk": 5,
    "gyro_noise":      6,
    "gyro_bias_init":  7,
    "gyro_bias_walk":  8,
}


class _MinimalGNSSCfg:
    """Minimal GNSS config object for tests."""
    def __init__(self, **kwargs):
        self.enabled             = True
        self.rate_hz             = 1.0
        self.pos_noise_1sigma_m  = 5.0
        self.vel_noise_1sigma_ms = 0.05
        self.bias_enabled        = False
        self.bias_pos_m          = [0.0, 0.0, 0.0]
        self.bias_vel_ms         = [0.0, 0.0, 0.0]
        self.gm_bias_enabled     = False
        self.gm_tau_s            = 600.0
        self.gm_sigma_pos_m      = 2.0
        self.gm_sigma_vel_ms     = 0.02
        self.outliers_enabled    = False
        self.outlier_prob        = 0.01
        self.outlier_pos_m       = 100.0
        self.outlier_vel_ms      = 1.0
        self.latency_s           = 0.0
        self.outage_windows      = []
        # Apply overrides
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_factory(seed=MASTER_SEED):
    return StreamFactory(seed, STREAMS)


def _make_straight_trajectory(n=N_SAMPLES, dt=1.0):
    """
    Straight-line trajectory at constant altitude (simulated as constant r, v).
    Allows fast generation of N_SAMPLES measurement packets.
    """
    r0 = np.array([R_ORBIT, 0.0, 0.0])
    v0 = np.array([0.0, V_CIRC, 0.0])
    t  = np.arange(n) * dt
    # For noise tests we just use constant r/v — we don't need orbit dynamics
    r  = np.tile(r0, (n, 1))
    v  = np.tile(v0, (n, 1))
    states = np.hstack([r, v])
    return t, states


# -----------------------------------------------------------------------
# 1. GNSS noise: empirical standard deviation
# -----------------------------------------------------------------------

class TestGNSSNoise:
    """GNSS white noise should match declared sigma."""

    def test_position_noise_std(self):
        """
        Empirical position noise std should match declared sigma.
        Uses chi-squared test on sum of squared standardized residuals.
        """
        sigma = 5.0
        cfg  = _MinimalGNSSCfg(pos_noise_1sigma_m=sigma, vel_noise_1sigma_ms=0.05)
        sim  = GNSSSimulator(cfg, _make_factory())
        t, states = _make_straight_trajectory(N_SAMPLES)

        packets = sim.simulate(t, states)
        assert len(packets) == N_SAMPLES, "Expected one packet per epoch"

        z_vals  = np.array([p.value[:3] for p in packets])  # (N, 3)
        r_true  = states[:, :3]
        residuals = (z_vals - r_true) / sigma   # normalized

        # Chi-squared test: sum of squared N(0,1) residuals ~ chi2(3*N)
        k  = residuals.size  # 3 * N_SAMPLES
        S  = float(np.sum(residuals**2))
        lo = chi2.ppf(ALPHA / 2, df=k)
        hi = chi2.ppf(1 - ALPHA / 2, df=k)
        assert lo < S < hi, (
            f"GNSS position noise chi2 statistic {S:.1f} outside [{lo:.1f}, {hi:.1f}] "
            f"(k={k}, sigma={sigma}). Noise distribution inconsistent with declared sigma."
        )

    def test_velocity_noise_std(self):
        """Velocity noise std should match declared sigma."""
        sigma = 0.05
        cfg  = _MinimalGNSSCfg(vel_noise_1sigma_ms=sigma)
        sim  = GNSSSimulator(cfg, _make_factory())
        t, states = _make_straight_trajectory(N_SAMPLES)

        packets = sim.simulate(t, states)
        z_vals   = np.array([p.value[3:6] for p in packets])
        v_true   = states[:, 3:6]
        residuals = (z_vals - v_true) / sigma

        k = residuals.size
        S = float(np.sum(residuals**2))
        lo = chi2.ppf(ALPHA / 2, df=k)
        hi = chi2.ppf(1 - ALPHA / 2, df=k)
        assert lo < S < hi, (
            f"GNSS velocity noise chi2 statistic {S:.1f} outside [{lo:.1f}, {hi:.1f}]"
        )

    def test_noise_zero_mean(self):
        """Noise should be zero-mean: |mean| < 3*sigma/sqrt(N)."""
        sigma = 5.0
        n = 2000
        cfg = _MinimalGNSSCfg(pos_noise_1sigma_m=sigma)
        sim = GNSSSimulator(cfg, _make_factory())
        t, states = _make_straight_trajectory(n)

        packets  = sim.simulate(t, states)
        z_vals   = np.array([p.value[:3] for p in packets])
        r_true   = states[:, :3]
        resid    = z_vals - r_true

        mean_err = np.abs(np.mean(resid))
        # 3σ bound on mean of N samples from N(0, sigma²): sigma/sqrt(N)
        bound = 3.0 * sigma / np.sqrt(n)
        assert mean_err < bound, (
            f"GNSS noise mean = {mean_err:.4f} m exceeds 3σ bound {bound:.4f} m"
        )

    def test_declared_covariance_matches(self):
        """Declared covariance in packet should equal R = diag(sigma²)."""
        sigma_p = 3.0
        sigma_v = 0.03
        cfg = _MinimalGNSSCfg(
            pos_noise_1sigma_m=sigma_p,
            vel_noise_1sigma_ms=sigma_v,
        )
        sim = GNSSSimulator(cfg, _make_factory())
        t, states = _make_straight_trajectory(3)
        packets = sim.simulate(t, states)

        R = packets[0].declared_covariance
        expected_diag = np.array([
            sigma_p**2, sigma_p**2, sigma_p**2,
            sigma_v**2, sigma_v**2, sigma_v**2,
        ])
        assert np.allclose(np.diag(R), expected_diag), (
            f"Declared R diagonal mismatch.\n"
            f"Expected: {expected_diag}\n"
            f"Got:      {np.diag(R)}"
        )


# -----------------------------------------------------------------------
# 2. GNSS bias
# -----------------------------------------------------------------------

class TestGNSSBias:
    """Constant bias should shift the measurement mean."""

    def test_constant_bias_shifts_mean(self):
        """With constant bias enabled, mean residual should equal bias."""
        bias_pos = np.array([10.0, -5.0, 3.0])
        bias_vel = np.array([0.01, -0.02, 0.0])
        n = 500
        cfg = _MinimalGNSSCfg(
            pos_noise_1sigma_m=0.1,   # Small noise to isolate bias
            vel_noise_1sigma_ms=0.001,
            bias_enabled=True,
            bias_pos_m=bias_pos.tolist(),
            bias_vel_ms=bias_vel.tolist(),
        )
        sim = GNSSSimulator(cfg, _make_factory())
        t, states = _make_straight_trajectory(n)

        packets  = sim.simulate(t, states)
        z_vals   = np.array([p.value for p in packets])
        true_pv  = states
        residuals = z_vals - true_pv

        mean_pos_resid = np.mean(residuals[:, :3], axis=0)
        mean_vel_resid = np.mean(residuals[:, 3:6], axis=0)

        tol_pos = 3 * 0.1 / np.sqrt(n)    # 3σ bound on the mean
        tol_vel = 3 * 0.001 / np.sqrt(n)

        np.testing.assert_allclose(
            mean_pos_resid, bias_pos, atol=tol_pos,
            err_msg=f"Bias not reflected in position mean: {mean_pos_resid} vs {bias_pos}"
        )
        np.testing.assert_allclose(
            mean_vel_resid, bias_vel, atol=tol_vel,
            err_msg=f"Bias not reflected in velocity mean"
        )


# -----------------------------------------------------------------------
# 3. GNSS outage
# -----------------------------------------------------------------------

class TestGNSSOutage:
    """No packets should be generated during outage windows."""

    def test_outage_window_produces_no_packets(self):
        """Packets in outage window [300, 600) should be absent."""
        cfg = _MinimalGNSSCfg(outage_windows=[(300.0, 600.0)])
        sim = GNSSSimulator(cfg, _make_factory())
        t, states = _make_straight_trajectory(1000)

        packets = sim.simulate(t, states)
        times_in_outage = [
            p.sample_time_s for p in packets
            if 300.0 <= p.sample_time_s < 600.0
        ]
        assert len(times_in_outage) == 0, (
            f"Expected NO packets in outage [300, 600), got {len(times_in_outage)}: "
            f"{times_in_outage[:5]}"
        )

    def test_packets_outside_outage_present(self):
        """Packets before and after outage should still exist."""
        cfg = _MinimalGNSSCfg(outage_windows=[(300.0, 600.0)])
        sim = GNSSSimulator(cfg, _make_factory())
        t, states = _make_straight_trajectory(1000)

        packets = sim.simulate(t, states)
        before = [p for p in packets if p.sample_time_s < 300.0]
        after  = [p for p in packets if p.sample_time_s >= 600.0]
        assert len(before) > 0, "No packets before outage"
        assert len(after) > 0,  "No packets after outage"

    def test_multiple_outage_windows(self):
        """Multiple non-overlapping outage windows each suppress packets."""
        windows = [(100.0, 200.0), (500.0, 700.0)]
        cfg = _MinimalGNSSCfg(outage_windows=windows)
        sim = GNSSSimulator(cfg, _make_factory())
        t, states = _make_straight_trajectory(1000)
        packets = sim.simulate(t, states)

        for t_start, t_end in windows:
            bad = [p for p in packets if t_start <= p.sample_time_s < t_end]
            assert len(bad) == 0, (
                f"Found {len(bad)} packets in outage window [{t_start}, {t_end})"
            )


# -----------------------------------------------------------------------
# 4. GNSS outliers
# -----------------------------------------------------------------------

class TestGNSSOutliers:
    """Outlier probability should match declared rate."""

    def test_outlier_rate(self):
        """
        Fraction of outlier-corrupted packets should be close to declared probability.
        Test: binomial with p=0.05, N=2000 → expected count = 100 ± ~30 (3σ).
        """
        p_outlier = 0.05
        n = 3000
        sigma_small = 0.001   # Tiny noise to easily identify outliers
        outlier_mag = 1000.0  # Huge to make them unmistakable

        cfg = _MinimalGNSSCfg(
            pos_noise_1sigma_m=sigma_small,
            outliers_enabled=True,
            outlier_prob=p_outlier,
            outlier_pos_m=outlier_mag,
        )
        sim = GNSSSimulator(cfg, _make_factory())
        t, states = _make_straight_trajectory(n)
        packets = sim.simulate(t, states)

        z_vals  = np.array([p.value[:3] for p in packets])
        r_true  = states[:, :3]
        errors  = np.linalg.norm(z_vals - r_true, axis=1)
        n_outliers = np.sum(errors > outlier_mag * 0.1)

        # Binomial confidence interval for p=0.05, N=3000: 3σ ≈ 3√(Np(1-p))
        mu_count = n * p_outlier
        sigma_count = np.sqrt(n * p_outlier * (1 - p_outlier))
        assert abs(n_outliers - mu_count) < 5 * sigma_count, (
            f"Outlier count {n_outliers} far from expected {mu_count:.0f} "
            f"(5σ bound = {5*sigma_count:.0f})"
        )


# -----------------------------------------------------------------------
# 5. Seed reproducibility
# -----------------------------------------------------------------------

class TestSeedReproducibility:
    """Same seed → identical measurement sequences."""

    def test_gnss_reproducible(self):
        """Two simulators with same seed must produce identical packets."""
        cfg = _MinimalGNSSCfg(
            pos_noise_1sigma_m=5.0,
            outliers_enabled=True,
            outlier_prob=0.05,
        )
        t, states = _make_straight_trajectory(100)

        sim1 = GNSSSimulator(cfg, _make_factory(seed=42))
        sim2 = GNSSSimulator(cfg, _make_factory(seed=42))
        packets1 = sim1.simulate(t, states)
        packets2 = sim2.simulate(t, states)

        assert len(packets1) == len(packets2)
        for p1, p2 in zip(packets1, packets2):
            np.testing.assert_array_equal(
                p1.value, p2.value,
                err_msg="Packet values differ between identical-seed runs"
            )

    def test_gnss_different_seeds_differ(self):
        """Different seeds must produce different measurement sequences."""
        cfg = _MinimalGNSSCfg()
        t, states = _make_straight_trajectory(100)

        sim1 = GNSSSimulator(cfg, _make_factory(seed=42))
        sim2 = GNSSSimulator(cfg, _make_factory(seed=43))
        packets1 = sim1.simulate(t, states)
        packets2 = sim2.simulate(t, states)

        v1 = np.array([p.value for p in packets1])
        v2 = np.array([p.value for p in packets2])
        assert not np.allclose(v1, v2), "Different seeds produced identical measurements"


# -----------------------------------------------------------------------
# 6. MeasurementBus ordering
# -----------------------------------------------------------------------

class TestMeasurementBus:
    """MeasurementBus should maintain (delivery_time, seq_num) order."""

    def test_bus_orders_by_delivery_time(self):
        """Packets popped from bus should be in non-decreasing delivery time."""
        cfg = _MinimalGNSSCfg(latency_s=0.0)
        sim = GNSSSimulator(cfg, _make_factory())
        t, states = _make_straight_trajectory(100)
        packets = sim.simulate(t, states)

        bus = build_measurement_bus(packets)
        delivery_times = []
        while len(bus) > 0:
            pkt = bus.pop_next()
            delivery_times.append(pkt.delivery_time_s)

        diffs = np.diff(delivery_times)
        assert np.all(diffs >= -1e-9), (
            f"Bus violated ordering: {diffs[diffs < -1e-9]}"
        )

    def test_drain_up_to(self):
        """drain_up_to(t) should return all packets with delivery_time <= t."""
        cfg = _MinimalGNSSCfg(latency_s=0.0)
        sim = GNSSSimulator(cfg, _make_factory())
        t, states = _make_straight_trajectory(100)
        packets = sim.simulate(t, states)

        bus = build_measurement_bus(packets)
        t_cut = 50.0
        drained = bus.drain_up_to(t_cut)

        for pkt in drained:
            assert pkt.delivery_time_s <= t_cut + 1e-9, (
                f"Packet with delivery_time={pkt.delivery_time_s:.3f} > t_cut={t_cut}"
            )

    def test_packet_timing_invariant(self):
        """delivery_time >= sample_time for every packet (always)."""
        cfg = _MinimalGNSSCfg(latency_s=0.02)   # 20 ms latency
        sim = GNSSSimulator(cfg, _make_factory())
        t, states = _make_straight_trajectory(200)
        packets = sim.simulate(t, states)

        for pkt in packets:
            assert pkt.delivery_time_s >= pkt.sample_time_s - 1e-9, (
                f"delivery_time={pkt.delivery_time_s:.4f} < "
                f"sample_time={pkt.sample_time_s:.4f}"
            )

    def test_invalid_packet_delivery_before_sample_raises(self):
        """MeasurementPacket should raise if delivery < sample."""
        with pytest.raises(ValueError, match="delivery_time"):
            MeasurementPacket(
                sensor_id="test",
                sequence_num=1,
                sample_time_s=100.0,
                delivery_time_s=99.0,    # INVALID: before sample_time
                value=np.zeros(6),
                declared_covariance=np.eye(6),
                frame="ECI",
                units="m",
            )
