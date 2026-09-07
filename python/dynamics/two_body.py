"""
python/dynamics/two_body.py

Two-body gravitational acceleration and its Jacobian (gravity gradient tensor).

MATHEMATICAL FOUNDATION:
  For a spacecraft at position r in an Earth-Centred Inertial (ECI) frame,
  the gravitational acceleration due to a point-mass Earth is:

      g(r) = -μ r / |r|³

  where μ = GM_Earth is Earth's gravitational parameter.

  The Jacobian ∂g/∂r is called the gravity gradient tensor:

      ∂g/∂r = μ (3 r rᵀ / |r|⁵ - I / |r|³)

  This appears in the EKF's process-model Jacobian F:

      F = [[0  I  0 ]
           [Gr 0  -C]
           [0  0  0 ]]

  where Gr = ∂g/∂r is computed here.

CRITICAL PHYSICS (INTERVIEW NOTE):
  g(r) is the GRAVITATIONAL acceleration. It is NOT what an accelerometer
  measures. In free-fall orbit a perfect accelerometer reads exactly zero
  because the spacecraft and the sensor are both in free fall together.
  Only non-gravitational forces (thrust, drag, SRP) appear in accelerometer
  output. This is why we call the accelerometer signal "specific force."

  Do NOT feed g(r) into the accelerometer model. It is used only in the
  propagation equations (truth and filter).

COMMON IMPLEMENTATION MISTAKES:
  1. Computing |r|² inside the force function instead of |r|³ → wrong units.
  2. Using position in kilometres without converting → wrong by 10⁹ factor.
  3. Getting the sign wrong in the Jacobian (forgetting the minus sign
     in the spherical field term, or the plus in the gradient term).
  4. Computing ∂g/∂r by numerically differentiating after normalization
     instead of deriving the analytic form → potential cancellation errors.

VERIFICATION:
  For a circular orbit: |g| = μ/|r|² = ω² |r| where ω = √(μ/|r|³).
  The Jacobian should reproduce this via finite differences to < 1e-8
  relative error (tested in test_dynamics.py).

EXTENSIBILITY:
  To add J2, third-body gravity, SRP, or drag:
  - Do NOT modify this file.
  - Instead, implement new force functions in forces.py and combine them
    through the ForceModel interface defined there.
  - Never route gravitational perturbation terms into the accelerometer signal.
"""

import numpy as np
from numpy.typing import NDArray

from .constants import MU_EARTH_M3S2


def gravity_acceleration(
    r: NDArray[np.float64],
    mu: float = MU_EARTH_M3S2,
) -> NDArray[np.float64]:
    """
    Compute two-body gravitational acceleration in ECI frame.

    Parameters
    ----------
    r : ndarray, shape (3,)
        Position vector in ECI [m].
    mu : float
        Gravitational parameter [m³/s²]. Defaults to Earth.

    Returns
    -------
    g : ndarray, shape (3,)
        Gravitational acceleration [m/s²].
        g = -mu * r / |r|³

    Raises
    ------
    ValueError
        If |r| is unrealistically small (below 1 km) — guards against
        numerical singularity at the origin.
    """
    r = np.asarray(r, dtype=np.float64)
    r_norm = np.linalg.norm(r)

    if r_norm < 1.0e3:
        raise ValueError(
            f"Position magnitude {r_norm:.3e} m is below 1 km. "
            "Two-body gravity is undefined at the origin. "
            "Check units (expected metres, not km)."
        )

    return -mu * r / (r_norm ** 3)


def gravity_jacobian(
    r: NDArray[np.float64],
    mu: float = MU_EARTH_M3S2,
) -> NDArray[np.float64]:
    """
    Compute the gravity gradient tensor ∂g/∂r.

    This is the Jacobian of gravitational acceleration with respect to
    position. It forms the upper-left block of the EKF process-model
    Jacobian F.

    Parameters
    ----------
    r : ndarray, shape (3,)
        Position vector in ECI [m].
    mu : float
        Gravitational parameter [m³/s²].

    Returns
    -------
    Gr : ndarray, shape (3, 3)
        Gravity gradient matrix [s⁻²].

        Gr = μ (3 r rᵀ / |r|⁵ - I / |r|³)

    Mathematical derivation:
        g = -μ r |r|⁻³
        ∂g_i/∂r_j = -μ [δ_ij |r|⁻³ + r_i ∂(|r|⁻³)/∂r_j]
                   = -μ [δ_ij |r|⁻³ - 3 r_i r_j |r|⁻⁵]
                   =  μ [3 r_i r_j / |r|⁵ - δ_ij / |r|³]

    In matrix form:
        Gr = μ (3 rrᵀ / |r|⁵ - I / |r|³)

    UNITS: [m/s² / m] = [s⁻²] — same as frequency squared.

    INTERVIEW NOTE:
        This matrix has one negative eigenvalue and two smaller negative
        eigenvalues. The positive eigenvalue corresponds to the radial
        direction (gravity increases as altitude decreases → destabilizing).
        The negative eigenvalues correspond to the transverse directions.
        This is the physical basis of tidal forces and gravity gradient
        stabilization.
    """
    r = np.asarray(r, dtype=np.float64)
    r_norm = np.linalg.norm(r)

    if r_norm < 1.0e3:
        raise ValueError(
            f"Position magnitude {r_norm:.3e} m is below 1 km."
        )

    r_norm3 = r_norm ** 3
    r_norm5 = r_norm ** 5

    outer = np.outer(r, r)          # r rᵀ,  shape (3, 3)
    I3    = np.eye(3, dtype=np.float64)

    Gr = mu * (3.0 * outer / r_norm5 - I3 / r_norm3)
    return Gr


def orbital_period(a: float, mu: float = MU_EARTH_M3S2) -> float:
    """
    Kepler's third law: period of a circular/elliptical orbit.

    Parameters
    ----------
    a : float
        Semi-major axis [m].
    mu : float
        Gravitational parameter [m³/s²].

    Returns
    -------
    T : float
        Orbital period [s].
    """
    return 2.0 * np.pi * np.sqrt(a ** 3 / mu)


def circular_speed(r_mag: float, mu: float = MU_EARTH_M3S2) -> float:
    """
    Speed of a circular orbit at given radius.

    Parameters
    ----------
    r_mag : float
        Orbital radius [m].
    mu : float
        Gravitational parameter [m³/s²].

    Returns
    -------
    v_c : float
        Circular orbital speed [m/s].
    """
    return np.sqrt(mu / r_mag)
