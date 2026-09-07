"""
python/records.py

Typed data records for the spacecraft navigation simulation.

DESIGN:
  Every piece of information that flows between simulation components
  is wrapped in one of these typed dataclasses. This enforces the
  strict boundary between Simulation, Navigation, and Evaluation.

  KEY RULE:
    - TruthState contains true position, velocity, attitude, and biases.
      Navigation code must NEVER receive TruthState directly.
    - MeasurementPacket contains only what a real sensor would deliver:
      a value, a declared covariance, and timing metadata.
    - EstimateRecord contains the filter output — what navigation knows.
    - EvaluationRecord joins truth and estimate at matched epochs
      and is only used by the evaluation module.

  TIMING CONTRACT (critical):
    Every measurement has two times:
      - sample_time_s : the epoch the measurement REFERS TO (physics)
      - delivery_time_s : when the measurement became available to the filter

    For zero latency: delivery_time_s == sample_time_s.
    For delayed measurements: delivery_time_s > sample_time_s.

    The filter must use sample_time_s to place the measurement correctly
    in the state-transition history. It must use delivery_time_s to decide
    when to process it. This distinction is essential for out-of-sequence
    measurement handling.

INTERVIEW NOTE:
  "What is the difference between sample time and delivery time?"
  An accelerometer samples at t=1.00 s. The measurement is digitized
  and transmitted to the navigation computer, arriving at t=1.02 s.
  The navigation state at t=1.02 s has already been propagated past t=1.00.
  A correct implementation must re-insert the measurement at t=1.00 s
  by restoring a stored state snapshot and replaying subsequent events.
  Applying the measurement naively at t=1.02 s introduces a positioning
  error proportional to spacecraft velocity × latency ≈ 7.6 km/s × 0.02 s ≈ 150 m.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from numpy.typing import NDArray


# -----------------------------------------------------------------------
# Truth records (Simulation boundary — never passed to Navigation)
# -----------------------------------------------------------------------

@dataclass
class TruthState:
    """
    Complete true spacecraft state at one epoch.

    This record is accessible to:
      - The sensor simulator (to compute measurement values)
      - The evaluation module (to score filter accuracy)

    It is NEVER accessible to the navigation filter.

    Units: SI throughout (m, m/s, m/s², rad, rad/s, dimensionless)
    """
    t_s: float                          # Simulation time [s]
    r_I: NDArray[np.float64]            # ECI position [m], shape (3,)
    v_I: NDArray[np.float64]            # ECI velocity [m/s], shape (3,)
    q_IB: Optional[NDArray[np.float64]] # Quaternion body→ECI, [q0,q1,q2,q3], shape (4,)
                                         # None until attitude module added
    C_IB: NDArray[np.float64]           # Rotation matrix body→ECI, shape (3,3)
    accel_bias_B: NDArray[np.float64]   # True accelerometer bias in B [m/s²], shape (3,)
    gyro_bias_B: NDArray[np.float64]    # True gyro bias in B [rad/s], shape (3,)
    specific_force_B: NDArray[np.float64]  # True specific force in B [m/s²], shape (3,)
                                           # = actual nongravitational accel
    is_thrusting: bool                  # Whether a burn is active


@dataclass
class SensorTruthLabel:
    """
    Fault labels for a measurement (Simulation boundary only).
    Navigation never receives this — it is used only by evaluation
    to classify accepted/rejected measurements.
    """
    sample_time_s: float
    sensor_id: str
    is_outlier: bool
    outlier_magnitude_m: float = 0.0   # 0 if not outlier


# -----------------------------------------------------------------------
# Measurement records (Simulation → Navigation boundary)
# -----------------------------------------------------------------------

@dataclass
class MeasurementPacket:
    """
    A single measurement packet delivered to the navigation filter.

    The filter receives ONLY the fields defined here. It does not
    know whether the measurement contains an outlier, the true bias,
    or any other simulation-internal quantity.

    Fields
    ------
    sensor_id : str
        Unique sensor identifier, e.g. 'gnss_0', 'imu_accel_0'.
    sequence_num : int
        Monotonically increasing counter per sensor. Used to break
        timestamp ties deterministically.
    sample_time_s : float
        Epoch this measurement refers to [s]. Use for filter insertion.
    delivery_time_s : float
        Epoch when measurement becomes available to filter [s].
        sample_time_s <= delivery_time_s always.
    value : NDArray
        Measurement vector. Shape and units depend on sensor_id:
          - GNSS: [rx, ry, rz, vx, vy, vz] [m, m/s], shape (6,)
          - IMU accel: [fx, fy, fz] in B [m/s²], shape (3,)
          - IMU gyro:  [ωx, ωy, ωz] in B [rad/s], shape (3,)
          - Star tracker: quaternion [q0,q1,q2,q3], shape (4,)
    declared_covariance : NDArray
        Navigation-declared noise covariance. Square matrix matching
        value dimension. This is what the filter uses — it does NOT
        have access to the true noise level.
    frame : str
        Coordinate frame of the measurement value (e.g. 'ECI', 'body').
    units : str
        Physical units string for documentation.
    is_valid : bool
        False if sensor is in outage. Filter must ignore invalid packets.
    """
    sensor_id: str
    sequence_num: int
    sample_time_s: float
    delivery_time_s: float
    value: NDArray[np.float64]
    declared_covariance: NDArray[np.float64]
    frame: str
    units: str
    is_valid: bool = True

    def __post_init__(self) -> None:
        if self.delivery_time_s < self.sample_time_s - 1e-9:
            raise ValueError(
                f"Packet {self.sensor_id}[{self.sequence_num}]: "
                f"delivery_time ({self.delivery_time_s:.3f}s) < "
                f"sample_time ({self.sample_time_s:.3f}s). "
                "Measurements cannot be delivered before they are sampled."
            )


# -----------------------------------------------------------------------
# Navigation filter output (Navigation → Evaluation boundary)
# -----------------------------------------------------------------------

@dataclass
class EstimateRecord:
    """
    Navigation filter state estimate at one epoch.

    This is what the filter outputs. Evaluation uses it alongside
    TruthState to compute errors. The filter never sees TruthState.

    Fields
    ------
    t_s : float
        Estimate epoch [s]. This must be precisely defined:
        is this BEFORE or AFTER the measurement update at this epoch?
        Tag 'prior' (pre-update) or 'posterior' (post-update).
    update_type : str
        'prior'     — state after predict step, before update
        'posterior' — state after measurement update
        'predict'   — state at a non-measurement epoch (filter propagation)
    x_hat : NDArray (9,)
        Estimated state [rx, ry, rz, vx, vy, vz, bax, bay, baz] [m, m/s, m/s²]
    P : NDArray (9, 9)
        Estimated state covariance [m², (m/s)², (m/s²)², cross-terms]
    innovation : Optional[NDArray]
        Pre-fit innovation ν = z − h(x⁻) at this update [same units as z].
        None for predict-only epochs.
    innovation_cov : Optional[NDArray]
        Innovation covariance S = H P⁻ Hᵀ + R. None for predict epochs.
    NIS : Optional[float]
        Normalized Innovation Squared = νᵀ S⁻¹ ν. None for predict epochs.
    gate_accepted : Optional[bool]
        Whether the measurement passed the χ² gate. None for predict epochs.
    sensor_id : Optional[str]
        Which sensor triggered this update. None for predict epochs.
    """
    t_s: float
    update_type: str           # 'prior', 'posterior', or 'predict'
    x_hat: NDArray[np.float64]
    P: NDArray[np.float64]
    innovation: Optional[NDArray[np.float64]] = None
    innovation_cov: Optional[NDArray[np.float64]] = None
    NIS: Optional[float] = None
    gate_accepted: Optional[bool] = None
    sensor_id: Optional[str] = None

    def pos_hat(self) -> NDArray[np.float64]:
        return self.x_hat[:3]

    def vel_hat(self) -> NDArray[np.float64]:
        return self.x_hat[3:6]

    def bias_hat(self) -> NDArray[np.float64]:
        return self.x_hat[6:9]

    def pos_std(self) -> NDArray[np.float64]:
        """Marginal 1-sigma position [m], shape (3,)."""
        return np.sqrt(np.diag(self.P[:3, :3]))

    def vel_std(self) -> NDArray[np.float64]:
        """Marginal 1-sigma velocity [m/s], shape (3,)."""
        return np.sqrt(np.diag(self.P[3:6, 3:6]))

    def bias_std(self) -> NDArray[np.float64]:
        """Marginal 1-sigma accelerometer bias [m/s²], shape (3,)."""
        return np.sqrt(np.diag(self.P[6:9, 6:9]))


# -----------------------------------------------------------------------
# Evaluation record (joined truth + estimate — Evaluation boundary only)
# -----------------------------------------------------------------------

@dataclass
class EvaluationRecord:
    """
    Joined truth and estimate at a matched epoch.

    Created by the evaluation module; never by navigation.
    Contains all quantities needed for RMSE, NEES, NIS scoring.

    Note: NEES is computed here at matched epochs. Do NOT compute NEES
    using P⁻¹ from a posterior epoch to evaluate a prior-epoch error,
    or vice versa. The update_type must match.
    """
    t_s: float
    truth: TruthState
    estimate: EstimateRecord

    # Computed error fields (filled by evaluation module)
    pos_error_I: Optional[NDArray[np.float64]] = None   # r_hat - r_true [m]
    vel_error_I: Optional[NDArray[np.float64]] = None   # v_hat - v_true [m/s]
    bias_error_B: Optional[NDArray[np.float64]] = None  # b_hat - b_true [m/s²]
    NEES: Optional[float] = None                         # full 9-state NEES
    NEES_pos: Optional[float] = None                     # 3-state positional NEES
    pos_error_RTN: Optional[NDArray[np.float64]] = None  # error in RTN coords [m]
    vel_error_RTN: Optional[NDArray[np.float64]] = None  # error in RTN coords [m/s]
