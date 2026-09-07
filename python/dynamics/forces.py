"""
python/dynamics/forces.py

Force model interface for the spacecraft truth simulation.

DESIGN PRINCIPLE:
  Forces are strictly classified into two categories:

  1. GRAVITATIONAL forces: modelled analytically in the propagator.
     They are NOT sensed by an accelerometer (free-fall principle).
     Examples: two-body gravity, J2, lunar/solar gravity.

  2. NON-GRAVITATIONAL (specific) forces: appear in accelerometer output.
     Examples: thrust, aerodynamic drag, solar radiation pressure.

  This separation is the most critical physics decision in the project.
  Getting it wrong (e.g., feeding gravity into the IMU model) will
  produce a filter that double-counts gravity and diverges.

EXTENSIBILITY:
  To add a new force:
    - Gravitational perturbation (e.g., J2):
        Implement in a new module (e.g., dynamics/perturbations.py),
        add its acceleration to the ODE right-hand side,
        and add its Jacobian contribution to F in orbit_model.py.
        Do NOT add it to the IMU signal.
    - Non-gravitational force (e.g., drag):
        Add a ScheduledForce or ForceModel subclass below.
        Route it into both the ODE (via nongravitational_force) AND
        the IMU accelerometer model (as the truth-level specific force).

UNITS: All forces in [m/s²] in the ECI body frame unless noted.

INTERVIEW NOTE:
  "Why is J2 a gravitational perturbation that doesn't appear in the IMU?"
  Because J2 is the oblateness correction to Earth's gravity field. The
  spacecraft is still in free fall under this modified field. Just as
  two-body gravity doesn't appear in the IMU, neither does J2.
  Drag, however, is a contact force and DOES appear in the IMU.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from numpy.typing import NDArray

from .two_body import gravity_acceleration
from .constants import MU_EARTH_M3S2


# -----------------------------------------------------------------------
# Force schedule entry
# -----------------------------------------------------------------------

@dataclass
class ThrustSegment:
    """
    A single thrust segment: constant force in body frame over [t_start, t_end].

    Parameters
    ----------
    t_start_s : float
        Start time of thrust [s].
    t_end_s : float
        End time of thrust [s].
    force_body_ms2 : array-like, shape (3,)
        Commanded specific force in body frame B [m/s²].
    execution_sigma_ms2 : float
        1-sigma execution error [m/s²] per axis. Drawn once per segment.
        Zero means perfectly executed command.
    actual_force_body_ms2 : ndarray or None
        Actual (truth-level) force after adding execution error.
        Set by the simulator; never accessed by the navigation filter.
    """
    t_start_s: float
    t_end_s: float
    force_body_ms2: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3)
    )
    execution_sigma_ms2: float = 0.0
    actual_force_body_ms2: Optional[NDArray[np.float64]] = None

    def __post_init__(self) -> None:
        self.force_body_ms2 = np.asarray(
            self.force_body_ms2, dtype=np.float64
        )


# -----------------------------------------------------------------------
# Force model
# -----------------------------------------------------------------------

class ForceModel:
    """
    Spacecraft force model combining gravitational and non-gravitational forces.

    The ODE right-hand side for truth integration is:

        dr/dt = v
        dv/dt = g(r) + C_IB @ f_nongrav_B(t)

    where:
        g(r)         = two-body (+ perturbations) gravitational acceleration in I
        C_IB         = rotation matrix body → inertial (from attitude source)
        f_nongrav_B  = specific force in body frame (thrust, drag, etc.)

    The navigation filter uses the same decomposition:
        - g(r) is computed from the filter's estimated position.
        - f_nongrav is inferred from the IMU accelerometer measurement.

    Parameters
    ----------
    mu : float
        Gravitational parameter [m³/s²].
    thrust_segments : list of ThrustSegment
        Scheduled non-gravitational force segments.
    """

    def __init__(
        self,
        mu: float = MU_EARTH_M3S2,
        thrust_segments: Optional[list[ThrustSegment]] = None,
    ) -> None:
        self.mu = mu
        self.thrust_segments: list[ThrustSegment] = thrust_segments or []

    def gravitational_accel_I(
        self,
        r_I: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Gravitational acceleration in the ECI frame [m/s²].

        This is the ONLY place where gravity is computed.
        It feeds the ODE propagator but NOT the IMU model.

        Returns
        -------
        g_I : ndarray, shape (3,)
        """
        return gravity_acceleration(r_I, mu=self.mu)

    def nongravitational_force_I(
        self,
        t: float,
        C_IB: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Non-gravitational specific force in ECI frame [m/s²].

        This is the quantity that an ideal accelerometer would measure
        (in the body frame). It is rotated into I for the ODE.

        For the truth simulator: returns the ACTUAL force (including
        execution errors). For the filter: this is inferred from IMU.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        C_IB : ndarray, shape (3, 3)
            Rotation matrix from body to inertial frame.

        Returns
        -------
        f_I : ndarray, shape (3,)
            Specific force in I frame [m/s²].
        """
        f_B = self._active_force_B(t, use_actual=True)
        return C_IB @ f_B

    def commanded_force_B(self, t: float) -> NDArray[np.float64]:
        """
        Commanded (ideal) force in body frame — accessible to filter/guidance.
        This does NOT include execution errors.
        """
        return self._active_force_B(t, use_actual=False)

    def _active_force_B(
        self, t: float, use_actual: bool = True
    ) -> NDArray[np.float64]:
        """Return the force vector active at time t in body frame."""
        for seg in self.thrust_segments:
            if seg.t_start_s <= t < seg.t_end_s:
                if use_actual and seg.actual_force_body_ms2 is not None:
                    return seg.actual_force_body_ms2
                return seg.force_body_ms2
        return np.zeros(3, dtype=np.float64)

    def ode_rhs(
        self,
        t: float,
        state: NDArray[np.float64],
        C_IB: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Full ODE right-hand side for truth integration.

            ẋ = f(x, t)

        where x = [r(3), v(3)] and:

            dr/dt = v
            dv/dt = g(r) + C_IB @ f_nongrav_B(t)

        Parameters
        ----------
        t : float
            Current time [s].
        state : ndarray, shape (6,)
            State vector [r(3), v(3)] in ECI [m, m/s].
        C_IB : ndarray, shape (3, 3)
            Body-to-inertial rotation matrix.

        Returns
        -------
        dstate_dt : ndarray, shape (6,)
        """
        r = state[:3]
        v = state[3:6]

        g_I   = self.gravitational_accel_I(r)
        f_ng_I = self.nongravitational_force_I(t, C_IB)

        dstate_dt = np.empty(6, dtype=np.float64)
        dstate_dt[:3] = v
        dstate_dt[3:] = g_I + f_ng_I
        return dstate_dt

    def is_thrusting(self, t: float) -> bool:
        """True if any thrust segment is active at time t."""
        for seg in self.thrust_segments:
            if seg.t_start_s <= t < seg.t_end_s:
                return True
        return False

    def discontinuity_times(self) -> list[float]:
        """
        Return all thrust start/end times — used to split the
        integration interval at force discontinuities.

        The truth integrator must NOT step across these boundaries;
        doing so would smear the force discontinuity and violate
        conservation checking within each smooth segment.
        """
        times = set()
        for seg in self.thrust_segments:
            times.add(seg.t_start_s)
            times.add(seg.t_end_s)
        return sorted(times)
