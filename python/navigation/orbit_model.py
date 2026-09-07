"""
python/navigation/orbit_model.py

Navigation filter state definition, process model, and measurement model.

STATE DEFINITION:
  9-state filter:  x = [r(3), v(3), b_a(3)]   in ECI / body-frame bias
  6-state filter:  x = [r(3), v(3)]            GNSS-only, no IMU bias

PROCESS MODEL (9-state):
  dx/dt = f(x, f_m, t) =
    [v;
     g(r) + C_IB @ (f_m - b_a);    # f_m = IMU measurement, b_a = bias estimate
     0]                              # bias deterministic part = 0

MEASUREMENT MODEL (GNSS position + velocity):
  h(x) = [r; v]  →  H = [I₃  0₃  0₃]  (position rows)
                         [0₃  I₃  0₃]  (velocity rows)

JACOBIAN F (9×9):
  F = [[0₃    I₃    0₃   ]
       [Gr    0₃   -C_IB  ]
       [0₃    0₃    0₃   ]]

  where Gr = μ(3rrᵀ/|r|⁵ - I/|r|³) is the gravity gradient tensor.

DESIGN NOTE:
  The navigation filter uses measured IMU force f_m ONCE — via the
  process model ODE. It is NOT added again as a correction in the
  measurement update. That would double-count the force.

EXTENSIBILITY:
  To add the 6→9 state extension: the 9-state model already includes
  the bias block. To go from 9 → 15 states (add gyro bias + attitude),
  implement a separate MEKF (attitude_mekf.py). Do NOT add attitude
  states here; this file stays as the orbital filter model.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..dynamics.two_body import gravity_acceleration, gravity_jacobian
from ..dynamics.constants import MU_EARTH_M3S2


# -----------------------------------------------------------------------
# Measurement model (shared between 6- and 9-state filters)
# -----------------------------------------------------------------------

# H matrix for GNSS position + velocity measurement on 6-state filter
H_GNSS_6 = np.hstack([np.eye(6), np.zeros((6, 0))])  # shape (6, 6)

# H matrix for GNSS position + velocity measurement on 9-state filter
H_GNSS_9 = np.hstack([np.eye(6), np.zeros((6, 3))])   # shape (6, 9)

# H matrix for GNSS position-only on 9-state filter
H_POS_9 = np.hstack([np.eye(3), np.zeros((3, 6))])    # shape (3, 9)


def gnss_measurement_model(
    x: NDArray[np.float64],
    state_dim: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    GNSS position+velocity measurement model.

    Parameters
    ----------
    x : ndarray (n,)  — filter state (6 or 9 dim)
    state_dim : int    — 6 or 9

    Returns
    -------
    h_x : ndarray (6,) — predicted measurement [r; v]
    H   : ndarray (6, n) — measurement Jacobian
    """
    h_x = x[:6].copy()
    H   = H_GNSS_6 if state_dim == 6 else H_GNSS_9
    return h_x, H


# -----------------------------------------------------------------------
# 6-state process model
# -----------------------------------------------------------------------

def f_6state(
    x6: NDArray[np.float64],
    accel_I: NDArray[np.float64],
    mu: float = MU_EARTH_M3S2,
) -> NDArray[np.float64]:
    """
    6-state ODE RHS: dx/dt = [v; g(r) + accel_I].

    For the pure orbital-dynamics (GNSS-only) filter, accel_I = 0.
    For the model-aided filter with commanded thrust, accel_I = C_IB @ f_commanded.

    Parameters
    ----------
    x6 : ndarray (6,)
    accel_I : ndarray (3,) — additional acceleration in I [m/s²]
    mu : float

    Returns
    -------
    dx_dt : ndarray (6,)
    """
    r = x6[:3]; v = x6[3:6]
    g = gravity_acceleration(r, mu=mu)
    dx = np.empty(6)
    dx[:3] = v
    dx[3:] = g + accel_I
    return dx


def F_6state(
    r: NDArray[np.float64],
    mu: float = MU_EARTH_M3S2,
) -> NDArray[np.float64]:
    """
    6×6 Jacobian ∂f/∂x for the 6-state filter.

    F = [[0₃  I₃]
         [Gr  0₃]]
    """
    Gr = gravity_jacobian(r, mu=mu)
    F  = np.zeros((6, 6))
    F[0:3, 3:6] = np.eye(3)
    F[3:6, 0:3] = Gr
    return F


# -----------------------------------------------------------------------
# 9-state process model
# -----------------------------------------------------------------------

def f_9state(
    x9: NDArray[np.float64],
    f_measured_B: NDArray[np.float64],
    C_IB: NDArray[np.float64],
    mu: float = MU_EARTH_M3S2,
) -> NDArray[np.float64]:
    """
    9-state ODE RHS for the IMU-aided filter.

    dx/dt = [v;
             g(r) + C_IB @ (f_measured_B - b_a);
             0]

    IMPORTANT: The filter uses the IMU measurement to correct for the
    actual non-gravitational force. The bias is estimated and subtracted.
    Gravity comes from the model, NOT from the IMU.

    Parameters
    ----------
    x9 : ndarray (9,)  — [r, v, b_a]
    f_measured_B : ndarray (3,)  — IMU measurement in body frame [m/s²]
    C_IB : ndarray (3, 3)        — body→inertial rotation
    mu : float

    Returns
    -------
    dx_dt : ndarray (9,)
    """
    r  = x9[0:3]
    v  = x9[3:6]
    ba = x9[6:9]

    g_I   = gravity_acceleration(r, mu=mu)
    f_I   = C_IB @ (f_measured_B - ba)  # corrected specific force in I

    dx    = np.empty(9)
    dx[0:3] = v
    dx[3:6] = g_I + f_I
    dx[6:9] = np.zeros(3)   # bias deterministic drift = 0
    return dx


def F_9state(
    r: NDArray[np.float64],
    C_IB: NDArray[np.float64],
    mu: float = MU_EARTH_M3S2,
) -> NDArray[np.float64]:
    """
    9×9 Jacobian ∂f/∂x for the 9-state IMU-aided filter.

    F = [[0₃    I₃    0₃  ]
         [Gr    0₃   -C_IB]
         [0₃    0₃    0₃  ]]

    Note the sign: ∂(dv/dt)/∂b_a = ∂/∂b_a [C_IB(f_m - b_a)] = -C_IB.
    """
    Gr = gravity_jacobian(r, mu=mu)
    F  = np.zeros((9, 9))
    F[0:3, 3:6] = np.eye(3)
    F[3:6, 0:3] = Gr
    F[3:6, 6:9] = -C_IB
    return F


def initial_state_9(
    r0_hat: NDArray[np.float64],
    v0_hat: NDArray[np.float64],
    bias_sigma_ms2: float,
    rng,   # np.random.Generator for initial error
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Initialize the 9-state EKF prior.

    DESIGN:
      The initial state is drawn from a Gaussian prior N(x_true_0, P_0).
      Equivalently: start with estimated bias = 0 and draw true bias from
      N(0, bias_sigma²·I). Both interpretations give the same distribution.

      IMPORTANT: The navigation filter receives (x_hat_0, P_0). It does NOT
      receive x_true_0 or the true bias. The prior covariance P_0 declares
      our uncertainty about the initial state.

    Parameters
    ----------
    r0_hat : ndarray (3,)  — initial position estimate (from GNSS or config)
    v0_hat : ndarray (3,)  — initial velocity estimate
    bias_sigma_ms2 : float — 1-sigma initial bias uncertainty [m/s²]
    rng : numpy Generator  — for initial position/velocity error (unused here;
                             caller applies error to r0_hat, v0_hat already)

    Returns
    -------
    x0 : ndarray (9,)     — initial state estimate [r_hat, v_hat, 0_bias]
    P0 : ndarray (9, 9)   — initial covariance
    """
    x0 = np.concatenate([r0_hat, v0_hat, np.zeros(3)])
    return x0


def build_P0(
    pos_sigma_m: float,
    vel_sigma_ms: float,
    bias_sigma_ms2: float,
) -> NDArray[np.float64]:
    """
    Build the 9×9 initial covariance matrix P₀.

    P₀ = diag(σ_pos² I₃, σ_vel² I₃, σ_bias² I₃)

    All cross-correlations are zero initially (independent priors).
    """
    diag_vals = np.concatenate([
        np.full(3, pos_sigma_m**2),
        np.full(3, vel_sigma_ms**2),
        np.full(3, bias_sigma_ms2**2),
    ])
    return np.diag(diag_vals)


def build_P0_6state(
    pos_sigma_m: float,
    vel_sigma_ms: float,
) -> NDArray[np.float64]:
    """6×6 initial covariance for the GNSS-only filter."""
    diag_vals = np.concatenate([
        np.full(3, pos_sigma_m**2),
        np.full(3, vel_sigma_ms**2),
    ])
    return np.diag(diag_vals)
