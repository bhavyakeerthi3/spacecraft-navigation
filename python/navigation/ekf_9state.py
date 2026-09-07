"""
python/navigation/ekf_9state.py

9-state GNSS/IMU Extended Kalman Filter.

STATE:  x = [r_I(3), v_I(3), b_a_B(3)]
              ECI pos  ECI vel  accel bias in body B

ARCHITECTURE:
  This filter implements the loosely-coupled GNSS/INS architecture:
    - IMU provides the propagation control input (not a measurement)
    - GNSS provides the measurement update
    - Gravity is computed from the orbital dynamics model

ASSUMPTION (Phase 4):
  Attitude C_IB is known and provided externally.
  C_IB = I₃ (identity) unless a prescribed attitude schedule is given.
  This assumption is explicitly documented at every usage site.

IMU MECHANIZATION TIMING CONVENTION:
  An IMU interval [t_start, t_end] covers dt = t_end - t_start seconds.
  The specific force f_m is the interval-average (same convention as imu.py).
  The filter propagates from t_start → t_end using f_m as a constant input.

PROCESS NOISE Q_d (derived in phase4_derivation.md, §4A.7):

  For isotropic noise (Qa, Qb scalars), Q_d is:

    Q_d[r,r] = Qa · dt³/3 · I₃
    Q_d[r,v] = Q_d[v,r] = Qa · dt²/2 · I₃
    Q_d[v,v] = Qa · dt · I₃
    Q_d[b,b] = Qb · dt · I₃

  All other blocks are zero.

  WHY Qa/dt³/3 MATTERS:
    If this is set to zero, position covariance never grows during pure
    IMU propagation (it only grows through velocity covariance coupling
    via the STM). For dt=0.1s and Qa=1e-10, the term is ~3e-14 m²
    (negligible), but it is non-zero and physically required.

NO GRAVITY DOUBLE-COUNTING:
  The state equation is:
    dv/dt = g(r) + C_IB @ (f_m - b_a_hat)

  g(r) comes from the dynamics model.
  C_IB @ f_m is the IMU contribution.
  This is NOT g(r) + IMU_total. The IMU measures only the nongravitational part.

NO FUTURE LEAKAGE:
  Every packet is checked: filter_time >= packet.sample_time_s
  Packets with sample_time > current filter time are rejected with a
  FutureMeasurementError.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..dynamics.two_body import gravity_acceleration, gravity_jacobian
from ..dynamics.constants import MU_EARTH_M3S2
from ..records import MeasurementPacket, EstimateRecord
from .ekf import (
    ExtendedKalmanFilter,
    NumericalHealthWarning,
    CovarianceDivergenceError,
)
from .orbit_model import H_GNSS_9, F_9state


# -----------------------------------------------------------------------
# Constants and helper
# -----------------------------------------------------------------------

class FutureMeasurementError(RuntimeError):
    """Raised when a packet has sample_time > current filter time."""
    pass


def _build_Q_d(Qa: float, Qb: float, dt: float, C_IB: NDArray) -> NDArray:
    """
    Discrete process noise matrix Q_d (9×9).

    Derived by Van Loan method applied to the 9-state system.
    See phase4_derivation.md §4A.7.

    For isotropic noise (Qa, Qb scalars), Q_d is attitude-independent:
    C_IB Qa I C_IBᵀ = Qa I (rotation cancels).

    Parameters
    ----------
    Qa : float — accelerometer white noise PSD [m²/s³] per axis
    Qb : float — bias random walk PSD [m²/s⁵] per axis
    dt : float — propagation interval [s]
    C_IB : (3,3) — body-to-inertial rotation (unused for isotropic Qa)

    Returns
    -------
    Qd : (9,9)
    """
    Qd = np.zeros((9, 9))
    # Position-velocity coupling (kinematic)
    Qd[0:3, 0:3] = Qa * (dt**3 / 3.0) * np.eye(3)
    Qd[0:3, 3:6] = Qa * (dt**2 / 2.0) * np.eye(3)
    Qd[3:6, 0:3] = Qa * (dt**2 / 2.0) * np.eye(3)
    Qd[3:6, 3:6] = Qa * dt * np.eye(3)
    # Bias random walk
    Qd[6:9, 6:9] = Qb * dt * np.eye(3)
    return Qd


def _build_Phi(r: NDArray, C_IB: NDArray, dt: float, mu: float) -> NDArray:
    """
    Discrete state-transition matrix Phi (9×9), truncated at second order.

    Phi ≈ I + F·dt + (F·dt)²/2

    Second-order is required to correctly propagate the position
    uncertainty through the velocity-gravity coupling.

    Parameters
    ----------
    r : (3,) ECI position [m]
    C_IB : (3,3) body-to-inertial rotation
    dt : float propagation interval [s]
    mu : float gravitational parameter

    Returns
    -------
    Phi : (9,9)
    """
    F = F_9state(r, C_IB, mu=mu)
    Fdt = F * dt
    Phi = np.eye(9) + Fdt + 0.5 * (Fdt @ Fdt)
    return Phi


# -----------------------------------------------------------------------
# 9-state state propagation (one IMU step)
# -----------------------------------------------------------------------

def propagate_9state_rk4(
    x: NDArray[np.float64],
    f_m_B: NDArray[np.float64],
    C_IB: NDArray[np.float64],
    dt: float,
    mu: float = MU_EARTH_M3S2,
) -> NDArray[np.float64]:
    """
    Propagate 9-state x one IMU step via 4th-order Runge-Kutta.

    The IMU measurement f_m_B is held constant over the interval dt
    (interval-average convention, matches imu.py).

    Gravity is NOT added from f_m_B. It is computed from the dynamics model.
    This prevents double-counting gravity during free-fall coast.

    Parameters
    ----------
    x : (9,) current state [r, v, b_a]
    f_m_B : (3,) IMU specific force measurement in body frame [m/s²]
    C_IB : (3,3) body→inertial rotation (assumed constant over dt)
    dt : float propagation interval [s]
    mu : float

    Returns
    -------
    x_new : (9,)
    """
    def f_ode(xk):
        r  = xk[0:3]
        v  = xk[3:6]
        ba = xk[6:9]
        g  = gravity_acceleration(r, mu=mu)
        # Non-gravitational acceleration in inertial frame:
        #   C_IB (f_m_B - b_a_hat)   = specific force corrected for bias
        # + gravity from model        = gravitational acceleration
        dv = g + C_IB @ (f_m_B - ba)
        db = np.zeros(3)   # Bias deterministic rate = 0
        return np.concatenate([v, dv, db])

    k1 = f_ode(x)
    k2 = f_ode(x + 0.5 * dt * k1)
    k3 = f_ode(x + 0.5 * dt * k2)
    k4 = f_ode(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# -----------------------------------------------------------------------
# Main 9-state EKF class
# -----------------------------------------------------------------------

@dataclass
class FilterState:
    """Snapshot of filter state at one epoch. Used for history buffering."""
    t_s: float
    x: NDArray[np.float64]
    P: NDArray[np.float64]


class NineStateEKF:
    """
    9-state GNSS/IMU Extended Kalman Filter.

    Implements loosely-coupled GNSS/INS navigation:
      1. IMU packets drive the propagation step (predict)
      2. GNSS packets trigger the measurement update step

    State: x = [r_I(3), v_I(3), b_a_B(3)]

    Parameters
    ----------
    x0 : (9,) initial state [r, v, b_a]
    P0 : (9,9) initial covariance
    Qa : float accelerometer noise PSD [m²/s³]
    Qb : float bias random walk PSD [m²/s⁵]
    R_gnss : (6,6) GNSS measurement noise covariance
    mu : float gravitational parameter
    gate_prob : float chi-squared gate probability for GNSS updates
    t0 : float initial epoch [s]
    history_size : int number of past states to retain (for OOS — Phase 6)
    """

    def __init__(
        self,
        x0: NDArray[np.float64],
        P0: NDArray[np.float64],
        Qa: float,
        Qb: float,
        R_gnss: NDArray[np.float64],
        mu: float = MU_EARTH_M3S2,
        gate_prob: float = 0.997,
        t0: float = 0.0,
        history_size: int = 20,
    ) -> None:
        assert x0.shape == (9,), f"x0 must be (9,), got {x0.shape}"
        assert P0.shape == (9, 9), f"P0 must be (9,9), got {P0.shape}"

        self.x    = x0.copy()
        self.P    = P0.copy()
        self._Qa  = Qa
        self._Qb  = Qb
        self._R   = R_gnss.copy()
        self._mu  = mu
        self._t   = t0

        self._ekf = ExtendedKalmanFilter(
            state_dim=9, gate_prob=gate_prob, health_check=True
        )

        self._history_size = history_size
        self._history: list[FilterState] = [FilterState(t0, x0.copy(), P0.copy())]

        # Records for post-processing
        self._records: list[EstimateRecord] = []
        self._n_imu_steps = 0
        self._n_gnss_updates = 0
        self._n_gnss_rejected = 0

    @property
    def t(self) -> float:
        return self._t

    @property
    def records(self) -> list[EstimateRecord]:
        return self._records

    def _C_IB(self, t: float = 0.0) -> NDArray[np.float64]:
        """
        Return body-to-inertial rotation matrix at time t.

        ASSUMPTION (Phase 4): Attitude is identity. This is documented
        explicitly so it is obvious when Phase 6 needs to change this.
        """
        return np.eye(3, dtype=np.float64)

    def _push_history(self) -> None:
        """Add current state to history buffer (circular)."""
        self._history.append(FilterState(self._t, self.x.copy(), self.P.copy()))
        if len(self._history) > self._history_size:
            self._history.pop(0)

    def propagate_imu(
        self,
        packet: MeasurementPacket,
    ) -> EstimateRecord:
        """
        Propagate filter state using one IMU packet.

        The IMU packet provides the specific force measurement f_m_B.
        The filter uses this as the input to the mechanization equations.
        Gravity is provided by the orbital dynamics model — NOT the IMU.

        No GNSS update is performed here.

        Parameters
        ----------
        packet : MeasurementPacket
            sensor_id must be 'imu_accel_0'.
            sample_time_s is the END of the IMU interval.

        Returns
        -------
        record : EstimateRecord (update_type='predict')
        """
        if packet.sensor_id != "imu_accel_0":
            raise ValueError(
                f"propagate_imu received packet from '{packet.sensor_id}', "
                "expected 'imu_accel_0'"
            )

        t_end = packet.sample_time_s
        dt = t_end - self._t

        if dt < -1e-9:
            raise FutureMeasurementError(
                f"IMU packet sample_time {t_end:.4f} < filter time {self._t:.4f}. "
                "IMU packets must be monotonically increasing."
            )
        if dt < 1e-9:
            # Zero-dt step: skip propagation, return current state
            return self._ekf.predict_record(self.x, self.P, self._t)

        f_m_B  = packet.value.copy()         # (3,) specific force in body
        C_IB   = self._C_IB(self._t)

        # --- State propagation (RK4) ---
        self.x = propagate_9state_rk4(self.x, f_m_B, C_IB, dt, self._mu)

        # --- Covariance propagation: P = Phi P Phiᵀ + Qd ---
        Phi = _build_Phi(self.x[:3], C_IB, dt, self._mu)
        Qd  = _build_Q_d(self._Qa, self._Qb, dt, C_IB)
        self.P = Phi @ self.P @ Phi.T + Qd
        self.P = 0.5 * (self.P + self.P.T)   # enforce symmetry

        self._t = t_end
        self._n_imu_steps += 1
        self._push_history()

        record = EstimateRecord(
            t_s=self._t,
            update_type="predict",
            x_hat=self.x.copy(),
            P=self.P.copy(),
        )
        self._records.append(record)
        return record

    def update_gnss(
        self,
        packet: MeasurementPacket,
    ) -> EstimateRecord:
        """
        Apply a GNSS measurement update.

        The GNSS packet is processed AT the current filter time.
        If the packet's sample_time differs from filter time, a warning
        is issued (indicates a missed IMU step before the GNSS update).

        The GNSS packet gives [r_meas; v_meas] in ECI.
        H = [I₃  0₃  0₃; 0₃  I₃  0₃]  →  does NOT observe bias directly.

        Parameters
        ----------
        packet : MeasurementPacket
            sensor_id must be 'gnss_0', is_valid must be True.

        Returns
        -------
        record : EstimateRecord (update_type='posterior')
        """
        if not packet.is_valid:
            # Outage packet — skip silently
            return self._ekf.predict_record(self.x, self.P, self._t)

        if packet.sensor_id != "gnss_0":
            raise ValueError(
                f"update_gnss received packet from '{packet.sensor_id}', "
                "expected 'gnss_0'"
            )

        if packet.sample_time_s > self._t + 1e-6:
            raise FutureMeasurementError(
                f"GNSS sample_time {packet.sample_time_s:.4f} > "
                f"filter time {self._t:.4f}. "
                "Cannot apply a GNSS measurement from the future. "
                "Ensure IMU has been propagated to GNSS epoch first."
            )

        t_offset = abs(packet.sample_time_s - self._t)
        if t_offset > 0.1:
            warnings.warn(
                f"GNSS sample_time ({packet.sample_time_s:.3f}) differs from "
                f"filter time ({self._t:.3f}) by {t_offset:.3f}s. "
                "Ensure IMU was propagated to GNSS epoch before calling update_gnss.",
                stacklevel=2,
            )

        z = packet.value[:6]   # [r_meas(3); v_meas(3)]
        R = packet.declared_covariance
        H = H_GNSS_9

        x_post, P_post, record = self._ekf.update(
            self.x, self.P, z, H, R,
            t=self._t, sensor_id="gnss_0"
        )

        self.x = x_post
        self.P = P_post
        self._n_gnss_updates += 1
        if not record.gate_accepted:
            self._n_gnss_rejected += 1

        self._push_history()
        self._records.append(record)
        return record

    def coast_propagate(
        self,
        t_target: float,
        dt_step: float = 1.0,
        use_zero_force: bool = True,
    ) -> list[EstimateRecord]:
        """
        Pure-inertial coast propagation with NO IMU packets.

        Used during GNSS outage simulation when IMU packets may not be
        available (e.g., for the outage analysis comparing different
        propagation strategies).

        The specific force is set to zero (coast assumption) unless
        use_zero_force=False and actual IMU packets are provided separately.

        Gravity is still computed from the dynamics model.

        Parameters
        ----------
        t_target : float target epoch [s]
        dt_step : float propagation substep [s]
        use_zero_force : bool if True, f_m_B = 0 (coast, no thrust)

        Returns
        -------
        records : list of EstimateRecord at each substep
        """
        records = []
        t_remaining = t_target - self._t

        if t_remaining < 1e-9:
            return records

        n_steps = max(1, int(np.round(t_remaining / dt_step)))
        dt_actual = t_remaining / n_steps

        f_m_B = np.zeros(3)  # Coast: no non-gravitational force
        C_IB  = self._C_IB(self._t)

        for _ in range(n_steps):
            dummy_pkt = MeasurementPacket(
                sensor_id="imu_accel_0",
                sequence_num=-1,
                sample_time_s=self._t + dt_actual,
                delivery_time_s=self._t + dt_actual,
                value=f_m_B.copy(),
                declared_covariance=np.eye(3) * self._Qa,
                frame="body",
                units="[m/s^2]",
                is_valid=True,
            )
            rec = self.propagate_imu(dummy_pkt)
            records.append(rec)

        return records

    def run(
        self,
        t_nav: NDArray[np.float64],
        imu_packets: list[MeasurementPacket],
        gnss_packets: list[MeasurementPacket],
    ) -> list[EstimateRecord]:
        """
        Run the 9-state EKF over a complete scenario.

        PROCESSING ORDER (strict):
          For each navigation epoch t:
            1. Process all IMU packets with sample_time <= t (in order)
            2. Process GNSS packet at t (if any)
            3. Log posterior estimate

        NO FUTURE MEASUREMENT LEAKAGE:
          IMU packets with sample_time > current_filter_time are never
          processed until the filter reaches that epoch.

        Parameters
        ----------
        t_nav : (M,) navigation epoch array [s]
        imu_packets : list of IMU accelerometer packets (sorted by sample_time)
        gnss_packets : list of GNSS packets (sorted by delivery_time)

        Returns
        -------
        records : list of EstimateRecord
        """
        self._records = []

        # Build GNSS lookup by sample_time (rounded to nearest ms)
        gnss_by_t: dict[int, MeasurementPacket] = {}
        for pkt in gnss_packets:
            key = int(round(pkt.sample_time_s * 1000))
            gnss_by_t[key] = pkt

        # Sort IMU by sample_time (must be monotonically increasing)
        imu_sorted = sorted(imu_packets, key=lambda p: p.sample_time_s)
        imu_idx = 0
        n_imu = len(imu_sorted)

        for t in t_nav:
            # --- Process all IMU packets up to this epoch ---
            while imu_idx < n_imu and \
                  imu_sorted[imu_idx].sample_time_s <= t + 1e-9:
                self.propagate_imu(imu_sorted[imu_idx])
                imu_idx += 1

            # --- If no IMU, propagate with zero force to current epoch ---
            if abs(self._t - t) > 1e-6:
                self.coast_propagate(t, dt_step=min(1.0, t - self._t))

            # --- GNSS update ---
            key = int(round(t * 1000))
            if key in gnss_by_t:
                pkt = gnss_by_t[key]
                if pkt.delivery_time_s <= t + 1e-9:
                    self.update_gnss(pkt)

        return self._records

    def report(self) -> dict:
        """Summary statistics."""
        return {
            "state_dim":         9,
            "n_imu_steps":       self._n_imu_steps,
            "n_gnss_updates":    self._n_gnss_updates,
            "n_gnss_rejected":   self._n_gnss_rejected,
            "gnss_acceptance":   (
                1.0 - self._n_gnss_rejected / max(1, self._n_gnss_updates)
            ),
            "ekf_report":        self._ekf.report(),
        }
