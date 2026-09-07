"""
python/sensors/imu.py

IMU (accelerometer + gyroscope) measurement simulator.

PHYSICS — CRITICAL:
  An accelerometer on a spacecraft in free-fall orbit measures ZERO
  in ideal conditions, because both the sensor and spacecraft are in
  the same gravitational free-fall. Only non-gravitational forces
  (thrust, drag, SRP) produce a non-zero accelerometer reading.

  The accelerometer output is called "specific force":
      f_measured = A_a * f_true_B + b_a + n_a
  where:
      f_true_B  = actual non-gravitational specific force in body frame B
                = ZERO during coasting (no thrust, no drag in this model)
      A_a       = (I + S + M) calibration matrix
      S         = diagonal scale-factor error matrix
      M         = cross-axis misalignment matrix (zero diagonal)
      b_a       = accelerometer bias vector in B
      n_a       = white noise (interval-average model)

NOISE DISCRETIZATION CONTRACT (must match propagation.py Qd):
  IMU delivers interval-average rate samples at rate f_imu [Hz].
  Interval duration Δt = 1/f_imu.

  White noise:
      n_{a,k} ~ N(0, Qa/Δt)    where Qa = noise spectral density [m²/s³]
      (Qa/Δt converts spectral density to interval-average variance)

  Bias random walk:
      b_{a,k+1} = b_{a,k} + ξ_{b,k},  ξ_{b,k} ~ N(0, Qb·Δt)
      Applied at END of interval k.
      (This is the same convention used in propagation.py Qd derivation.)

CONVENTION:
  - Bias is drawn once from N(0, bias_init_sigma²·I) at initialization.
  - Bias changes each step by the random walk.
  - Scale and misalignment errors are constant per run.
  - All errors are independently seeded and independently switchable.

WHAT THE NAVIGATION FILTER RECEIVES:
  - f_measured (the corrupted specific force in B)
  - Sample time and delivery time (for timing module)
  - No true bias, no calibration matrix, no fault labels
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional

from ..records import MeasurementPacket


def _build_calibration_matrix(
    scale_factor: NDArray[np.float64],   # 3-element, dimensionless
    misalignment_rad: NDArray[np.float64],  # [xy, xz, yz] off-diagonal elements
    enabled_scale: bool,
    enabled_misalign: bool,
) -> NDArray[np.float64]:
    """
    Build the 3×3 calibration matrix A = I + S + M.

    S is diagonal with scale factors, M is upper-triangular with zero diagonal.

    Parameters
    ----------
    scale_factor : (3,) dimensionless scale errors (e.g. 500e-6 = 500 ppm)
    misalignment_rad : [s_xy, s_xz, s_yz] off-diagonal misalignment [rad]

    Returns
    -------
    A : (3, 3) calibration matrix
    """
    A = np.eye(3, dtype=np.float64)
    if enabled_scale:
        A += np.diag(scale_factor)
    if enabled_misalign:
        # xy is M[0,1], xz is M[0,2], yz is M[1,2]
        A[0, 1] += misalignment_rad[0]
        A[0, 2] += misalignment_rad[1]
        A[1, 2] += misalignment_rad[2]
    return A


class AccelerometerSimulator:
    """
    Accelerometer simulator for one spacecraft IMU.

    Produces interval-average specific force measurements in the body frame.
    """

    SENSOR_ID = "imu_accel_0"

    def __init__(self, accel_cfg, stream_factory) -> None:
        self._cfg = accel_cfg
        self._rng_noise = stream_factory.get("accel_noise")
        self._rng_bias_init = stream_factory.get("accel_bias_init")
        self._rng_bias_walk = stream_factory.get("accel_bias_walk")

        # Build calibration matrix
        self._A = _build_calibration_matrix(
            accel_cfg.scale_factor,
            accel_cfg.misalignment_rad,
            accel_cfg.scale_enabled,
            accel_cfg.misalign_enabled,
        )

        # Initialize bias: draw from N(0, σ_init² I)
        if accel_cfg.bias_enabled:
            sigma_init = accel_cfg.bias_init_1sigma_ms2
            self._bias = self._rng_bias_init.normal(0.0, sigma_init, 3)
        else:
            self._bias = np.zeros(3, dtype=np.float64)

        self._sequence_counter = 0

    @property
    def current_bias(self) -> NDArray[np.float64]:
        """True current bias (Simulation boundary only — NOT given to filter)."""
        return self._bias.copy()

    def generate_measurement(
        self,
        t_start: float,
        dt: float,
        specific_force_B: NDArray[np.float64],
    ) -> MeasurementPacket:
        """
        Generate one accelerometer interval-average measurement.

        Applies calibration, noise, bias in the correct order.
        Advances bias random walk at end of interval.

        Parameters
        ----------
        t_start : float
            Start of the IMU interval [s].
        dt : float
            Interval duration [s] = 1/rate_hz.
        specific_force_B : ndarray (3,)
            True specific force in body frame B [m/s²].
            = 0 during free-fall coast; = thrust/kg during burn.

        Returns
        -------
        packet : MeasurementPacket
            Contains f_measured in body frame B.
        """
        cfg = self._cfg

        # Step 1: Apply calibration matrix to true signal
        # A_a * f_true_B  (scale + misalignment)
        f_calibrated = self._A @ specific_force_B

        # Step 2: Add bias (using bias at START of interval)
        f_biased = f_calibrated + self._bias

        # Step 3: Add white noise (interval-average convention)
        # n_k ~ N(0, Qa/Δt)
        if cfg.noise_enabled:
            sigma_noise = np.sqrt(cfg.noise_density_m2s3 / dt)
            noise = self._rng_noise.normal(0.0, sigma_noise, 3)
            f_measured = f_biased + noise
        else:
            f_measured = f_biased.copy()

        # Step 4: Apply latency
        sample_t   = t_start + dt  # measurement refers to end of interval
        delivery_t = sample_t + cfg.latency_s

        # Step 5: Advance bias random walk (END of interval)
        if cfg.bias_enabled and cfg.walk_enabled:
            sigma_walk = np.sqrt(cfg.bias_walk_density_m2s5 * dt)
            self._bias += self._rng_bias_walk.normal(0.0, sigma_walk, 3)

        self._sequence_counter += 1

        return MeasurementPacket(
            sensor_id=self.SENSOR_ID,
            sequence_num=self._sequence_counter,
            sample_time_s=sample_t,
            delivery_time_s=delivery_t,
            value=f_measured,
            declared_covariance=np.diag(
                [cfg.noise_density_m2s3 / dt] * 3
            ),
            frame="body",
            units="[m/s^2, m/s^2, m/s^2]",
            is_valid=True,
        )

    def simulate(
        self,
        t_truth: NDArray[np.float64],
        states_truth: NDArray[np.float64],
        specific_force_truth: NDArray[np.float64],
        dt: float,
    ) -> list[MeasurementPacket]:
        """
        Generate all accelerometer packets for a truth trajectory.

        Parameters
        ----------
        t_truth : ndarray (N,)
        states_truth : ndarray (N, 6)
        specific_force_truth : ndarray (N, 3)
            True specific force in body B at each truth epoch [m/s²].
        dt : float
            IMU interval [s].

        Returns
        -------
        packets : list of MeasurementPacket
        """
        t_start = t_truth[0]
        t_end   = t_truth[-1]

        n_intervals = int(np.floor((t_end - t_start) / dt))
        packets = []

        for i in range(n_intervals):
            ts = t_start + i * dt
            te = ts + dt

            # Average specific force over interval (linear interpolation approx.)
            f_s = np.array([
                np.interp(ts, t_truth, specific_force_truth[:, j])
                for j in range(3)
            ])
            f_e = np.array([
                np.interp(te, t_truth, specific_force_truth[:, j])
                for j in range(3)
            ])
            f_avg = 0.5 * (f_s + f_e)

            pkt = self.generate_measurement(ts, dt, f_avg)
            packets.append(pkt)

        return packets


class GyroSimulator:
    """
    Gyroscope simulator for one spacecraft IMU.

    Models interval-average angular rate measurements in body frame B.
    Follows the same noise/bias/walk convention as AccelerometerSimulator.
    """

    SENSOR_ID = "imu_gyro_0"

    def __init__(self, gyro_cfg, stream_factory) -> None:
        self._cfg = gyro_cfg
        self._rng_noise     = stream_factory.get("gyro_noise")
        self._rng_bias_init = stream_factory.get("gyro_bias_init")
        self._rng_bias_walk = stream_factory.get("gyro_bias_walk")

        self._A = _build_calibration_matrix(
            gyro_cfg.scale_factor,
            gyro_cfg.misalignment_rad,
            gyro_cfg.scale_enabled,
            gyro_cfg.misalign_enabled,
        )

        if gyro_cfg.bias_enabled:
            self._bias = self._rng_bias_init.normal(
                0.0, gyro_cfg.bias_init_1sigma_rads, 3
            )
        else:
            self._bias = np.zeros(3, dtype=np.float64)

        self._sequence_counter = 0

    @property
    def current_bias(self) -> NDArray[np.float64]:
        return self._bias.copy()

    def generate_measurement(
        self,
        t_start: float,
        dt: float,
        omega_true_B: NDArray[np.float64],
    ) -> MeasurementPacket:
        """
        Generate one gyro interval-average measurement.

        omega_true_B : true angular velocity of body in I, expressed in B [rad/s].
        For identity attitude (Phase 1-7 assumption): omega = 0.
        """
        cfg = self._cfg

        omega_cal = self._A @ omega_true_B
        omega_biased = omega_cal + self._bias

        if cfg.noise_enabled:
            sigma_noise = np.sqrt(cfg.noise_density_rad2s / dt)
            omega_measured = omega_biased + self._rng_noise.normal(0.0, sigma_noise, 3)
        else:
            omega_measured = omega_biased.copy()

        sample_t   = t_start + dt
        delivery_t = sample_t + cfg.latency_s

        if cfg.bias_enabled and cfg.walk_enabled:
            sigma_walk = np.sqrt(cfg.bias_walk_density_rad2s3 * dt)
            self._bias += self._rng_bias_walk.normal(0.0, sigma_walk, 3)

        self._sequence_counter += 1

        return MeasurementPacket(
            sensor_id=self.SENSOR_ID,
            sequence_num=self._sequence_counter,
            sample_time_s=sample_t,
            delivery_time_s=delivery_t,
            value=omega_measured,
            declared_covariance=np.diag([cfg.noise_density_rad2s / dt] * 3),
            frame="body",
            units="[rad/s, rad/s, rad/s]",
            is_valid=True,
        )
