"""
python/navigation/baselines.py

Baseline estimators for comparison against the EKF.

BASELINES:
  B0 — Propagation only (no measurements, initial condition = truth)
       Tests: Does the orbit model drift?

  B1 — Raw GNSS (dead-simple: use GNSS position/velocity directly)
       No filtering, no propagation.
       Tests: What does unfiltered GNSS noise look like?

  B2 — GNSS-only 6-state EKF (no IMU bias state)
       Tests: How much does filtering help vs raw GNSS?

These baselines are compared against the full 9-state EKF in Phase 5
to demonstrate the value of each design choice: filtering, IMU fusion,
and bias estimation.

COMPARISON TABLE (populated after experiments):
  ┌─────────────────────┬────────────────┬──────────────────┬─────────────┐
  │ Estimator           │ Pos RMSE (m)   │ Vel RMSE (m/s)  │ Bias known? │
  ├─────────────────────┼────────────────┼──────────────────┼─────────────┤
  │ B0 Propagate-only   │ TBD (Phase 3)  │ TBD             │ –           │
  │ B1 Raw GNSS         │ TBD (Phase 2)  │ TBD             │ No          │
  │ B2 6-state EKF      │ TBD (Phase 3)  │ TBD             │ No          │
  │ Full 9-state EKF    │ TBD (Phase 4)  │ TBD             │ Yes         │
  └─────────────────────┴────────────────┴──────────────────┴─────────────┘
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional

from ..records import MeasurementPacket, EstimateRecord, TruthState
from ..dynamics.propagation import rk4_predict
from ..dynamics.frames import identity_C_IB
from ..dynamics.constants import MU_EARTH_M3S2
from .orbit_model import (
    gnss_measurement_model,
    build_P0_6state,
    F_6state,
)
from .ekf import ExtendedKalmanFilter


# -----------------------------------------------------------------------
# B1: Raw GNSS baseline
# -----------------------------------------------------------------------

class RawGNSSBaseline:
    """
    B1: Use GNSS measurements directly as position/velocity estimates.

    No filtering, no propagation, no covariance propagation.
    At epochs without a GNSS measurement (outage), the last-known
    position is held (zero-order hold — explicitly NOT a propagation,
    to isolate the value of the dynamics model).

    The declared uncertainty at each epoch is the GNSS R matrix.
    During outage, uncertainty is set to infinity (NaN covariance).

    Parameters
    ----------
    R_gnss : ndarray (6, 6) — GNSS declared measurement covariance
    """

    def __init__(self, R_gnss: NDArray[np.float64]) -> None:
        self._R = R_gnss
        self._last_z: Optional[NDArray[np.float64]] = None
        self._last_t: float = 0.0
        self._records: list[EstimateRecord] = []

    def process_packet(self, packet: MeasurementPacket) -> EstimateRecord:
        """Accept a GNSS packet and store as estimate."""
        if packet.sensor_id != "gnss_0" or not packet.is_valid:
            return None

        self._last_z = packet.value.copy()
        self._last_t = packet.sample_time_s

        record = EstimateRecord(
            t_s=packet.sample_time_s,
            update_type="posterior",
            x_hat=packet.value.copy(),  # [r; v] directly
            P=packet.declared_covariance.copy(),
        )
        self._records.append(record)
        return record

    def run(
        self,
        gnss_packets: list[MeasurementPacket],
    ) -> list[EstimateRecord]:
        """Process all GNSS packets and return estimate history."""
        self._records = []
        for pkt in gnss_packets:
            rec = self.process_packet(pkt)
            if rec is not None:
                self._records.append(rec)
        return self._records


# -----------------------------------------------------------------------
# B2: 6-state orbital EKF (GNSS only, no IMU bias)
# -----------------------------------------------------------------------

class OrbitalEKF6State:
    """
    B2: 6-state GNSS-only EKF.

    State: x = [r(3), v(3)]
    Process model: Keplerian two-body gravity (no thrust, no bias)
    Measurement model: GNSS position + velocity (H = I₆)

    This is the simplest meaningful orbital navigation filter.
    It demonstrates:
      1. Covariance propagation via Φ
      2. Gravity-gradient dynamics improvement over raw GNSS
      3. Baseline for the 9-state GNSS/IMU EKF

    Parameters
    ----------
    x0 : ndarray (6,)   — initial state [r, v]
    P0 : ndarray (6, 6) — initial covariance
    Q_pos_m2s3 : float  — process noise spectral density, position [m²/s³]
                          Added on velocity states; models unmodelled accel
    Q_vel_m2s3 : float  — process noise spectral density, velocity
    R : ndarray (6, 6)  — GNSS measurement noise covariance
    dt_nav : float      — navigation step [s]
    mu : float          — gravitational parameter
    gate_prob : float   — chi-squared gate probability
    """

    def __init__(
        self,
        x0: NDArray[np.float64],
        P0: NDArray[np.float64],
        Q_pos_m2s3: float,
        Q_vel_m2s3: float,
        R: NDArray[np.float64],
        dt_nav: float,
        mu: float = MU_EARTH_M3S2,
        gate_prob: float = 0.997,
    ) -> None:
        self.x = x0.copy()
        self.P = P0.copy()
        self._Q_pos = Q_pos_m2s3
        self._Q_vel = Q_vel_m2s3
        self._R = R.copy()
        self._dt = dt_nav
        self._mu = mu
        self._ekf = ExtendedKalmanFilter(state_dim=6, gate_prob=gate_prob)
        self._records: list[EstimateRecord] = []
        self._t = 0.0

    def _discrete_Q6(self, dt: float) -> NDArray[np.float64]:
        """
        Discrete process noise Q_d for the 6-state filter.

        Simplified: white noise on velocity states only (unmodelled acceleration).
        A full derivation would use the STM to compute the continuous-to-discrete
        conversion; this approximation is valid for small dt and weak perturbations.
        """
        q_p = self._Q_pos * dt
        q_v = self._Q_vel * dt
        Qd = np.diag(np.array([q_p, q_p, q_p, q_v, q_v, q_v]))
        return Qd

    def _propagate(self, dt: float) -> None:
        """Propagate state and covariance by dt using 4th-order Runge-Kutta."""
        # RK4 for state
        def f(x):
            r = x[:3]; v = x[3:]
            from ..dynamics.two_body import gravity_acceleration
            g = gravity_acceleration(r, mu=self._mu)
            return np.concatenate([v, g])

        k1 = f(self.x)
        k2 = f(self.x + 0.5*dt*k1)
        k3 = f(self.x + 0.5*dt*k2)
        k4 = f(self.x + dt*k3)
        self.x = self.x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Linearized covariance propagation: P = Φ P Φᵀ + Qd
        # Φ ≈ I + F dt + (F dt)²/2  (truncated Van Loan — valid for small dt)
        F = F_6state(self.x[:3], mu=self._mu)
        Fdt = F * dt
        Phi = np.eye(6) + Fdt + 0.5 * Fdt @ Fdt

        Qd = self._discrete_Q6(dt)
        self.P = Phi @ self.P @ Phi.T + Qd

    def _update_gnss(
        self,
        t: float,
        z: NDArray[np.float64],
        R: NDArray[np.float64],
    ) -> EstimateRecord:
        """Apply GNSS update."""
        from .orbit_model import H_GNSS_6
        H = H_GNSS_6
        x_post, P_post, record = self._ekf.update(
            self.x, self.P, z, H, R, t=t, sensor_id="gnss_0"
        )
        self.x = x_post
        self.P = P_post
        return record

    def run(
        self,
        t_nav: NDArray[np.float64],
        gnss_packets: list[MeasurementPacket],
    ) -> list[EstimateRecord]:
        """
        Run the 6-state EKF over the navigation time array.

        At each epoch:
          1. Propagate to current time
          2. Process any GNSS packets with delivery_time ≤ t
          3. Log estimate

        Parameters
        ----------
        t_nav : ndarray (M,)           — navigation time array [s]
        gnss_packets : list            — all GNSS packets (sorted by delivery_time)

        Returns
        -------
        records : list of EstimateRecord
        """
        self._records = []
        self._t = t_nav[0]

        # Build a time-indexed lookup for GNSS packets
        pkt_idx = 0
        n_pkts  = len(gnss_packets)

        for i, t in enumerate(t_nav):
            # Propagate from previous epoch
            if i > 0:
                dt_actual = t - t_nav[i-1]
                self._propagate(dt_actual)
            self._t = t

            # Log prior
            prior_record = self._ekf.predict_record(self.x, self.P, t)
            self._records.append(prior_record)

            # Process all GNSS packets with delivery_time ≤ t
            while pkt_idx < n_pkts and \
                  gnss_packets[pkt_idx].delivery_time_s <= t + 1e-9:
                pkt = gnss_packets[pkt_idx]
                pkt_idx += 1
                if pkt.is_valid:
                    post = self._update_gnss(t, pkt.value, pkt.declared_covariance)
                    self._records.append(post)

        return self._records
