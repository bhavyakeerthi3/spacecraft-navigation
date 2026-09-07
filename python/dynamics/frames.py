"""
python/dynamics/frames.py

Reference frame utilities for the spacecraft navigation simulation.

FRAMES USED:
    I  — Earth-Centred Inertial (ECI), idealized, fixed axes.
         In reality this is GCRF; we explicitly ignore Earth orientation
         parameters, polar motion, and precession/nutation. Sufficient
         for simulation scope.

    B  — Spacecraft body frame, right-handed.
         C_IB maps vectors from B into I: v_I = C_IB @ v_B.

    R/T/N — Radial/Along-Track/Cross-Track, a local orbit frame.
         Defined at the truth spacecraft position.
         NOT a rotating state-space frame; used only for error evaluation
         and observability reporting.

IMPORTANT DISTINCTION:
    We resolve error vectors into R/T/N axes for plotting, but the
    filter state and covariance remain in the inertial frame I.
    Covariance in R/T/N is C_RI @ P_rr @ C_RI^T, where P_rr is the
    3×3 position block. This is a coordinate change, not a state
    transformation.

CONVENTION:
    R — Radial outward (r̂ = r/|r|)
    N — Orbit normal (r × v direction, ĥ = (r×v)/|r×v|)
    T — Along-track, completes right-handed frame (N × R)

NOTE ON "ALONG-TRACK":
    Along-track T = N × R is NOT the velocity direction unless the orbit
    is exactly circular and in the defined frame. For a circular orbit
    these coincide, but for general orbits they differ. Use T = N × R.

INTERVIEW NOTE:
    Rotating the inertial velocity vector into R/T/N axes gives
    v_RTN = C_RI @ v_I.
    The time derivative of position in R/T/N is NOT the same as this
    rotated velocity — the frame itself rotates at orbital rate ω.
    The correct kinematic relationship is:
        ρ̇_RTN = C_RI @ ṙ_I − ω_RTN × ρ_RTN
    This distinction matters critically in the relative navigation module.
"""

import numpy as np
from numpy.typing import NDArray


def identity_C_IB(t: float) -> NDArray[np.float64]:
    """
    Identity attitude: body frame = inertial frame.

    This is the Phase 1–7 default assumption. No rotation is applied.
    The spacecraft is treated as inertially fixed (not LVLH-pointing).

    Parameters
    ----------
    t : float
        Time [s] — not used (returns constant identity).

    Returns
    -------
    C_IB : ndarray, shape (3, 3)
    """
    return np.eye(3, dtype=np.float64)


def rtn_from_state(
    r_I: NDArray[np.float64],
    v_I: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the R/T/N unit vectors from the ECI position and velocity.

    Parameters
    ----------
    r_I : ndarray, shape (3,)
        Position in ECI [m].
    v_I : ndarray, shape (3,)
        Velocity in ECI [m/s].

    Returns
    -------
    R_hat : ndarray, shape (3,)  — radial outward unit vector
    T_hat : ndarray, shape (3,)  — along-track unit vector (N × R)
    N_hat : ndarray, shape (3,)  — orbit-normal unit vector (r × v)

    Raises
    ------
    ValueError
        If angular momentum vector is near zero (degenerate orbit).
    """
    r_norm = np.linalg.norm(r_I)
    R_hat  = r_I / r_norm

    h      = np.cross(r_I, v_I)
    h_norm = np.linalg.norm(h)
    if h_norm < 1e-3:
        raise ValueError(
            "Angular momentum vector near zero — degenerate orbit. "
            "Cannot define R/T/N frame."
        )
    N_hat = h / h_norm
    T_hat = np.cross(N_hat, R_hat)

    return R_hat, T_hat, N_hat


def C_rtn_from_eci(
    r_I: NDArray[np.float64],
    v_I: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Rotation matrix from ECI to RTN frame.

    C_RI such that v_RTN = C_RI @ v_I.

    Rows of C_RI are the RTN unit vectors expressed in ECI.

    Parameters
    ----------
    r_I : ndarray (3,)
    v_I : ndarray (3,)

    Returns
    -------
    C_RI : ndarray (3, 3)
    """
    R_hat, T_hat, N_hat = rtn_from_state(r_I, v_I)
    C_RI = np.stack([R_hat, T_hat, N_hat], axis=0)  # shape (3, 3)
    return C_RI


def eci_to_rtn(
    vec_I: NDArray[np.float64],
    r_I: NDArray[np.float64],
    v_I: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Project a vector from ECI into RTN components.

    Parameters
    ----------
    vec_I : ndarray (3,)
        Vector in ECI frame.
    r_I : ndarray (3,)
        Reference position in ECI.
    v_I : ndarray (3,)
        Reference velocity in ECI.

    Returns
    -------
    vec_RTN : ndarray (3,)
        [radial, along-track, cross-track] components.
    """
    C_RI = C_rtn_from_eci(r_I, v_I)
    return C_RI @ vec_I


def orbital_rate(
    r_I: NDArray[np.float64],
    v_I: NDArray[np.float64],
) -> float:
    """
    Instantaneous orbital angular rate |ω| = |r × v| / |r|² [rad/s].

    For a circular orbit this equals √(μ/|r|³).
    Used in relative navigation for the Coriolis term.
    """
    h_vec  = np.cross(r_I, v_I)
    h_norm = np.linalg.norm(h_vec)
    r_norm = np.linalg.norm(r_I)
    return h_norm / (r_norm ** 2)


def rodrigues_rotate(
    v: NDArray[np.float64],
    axis: NDArray[np.float64],
    angle_rad: float,
) -> NDArray[np.float64]:
    """
    Rotate vector v by angle_rad about unit axis using Rodrigues' formula.

        v_rot = v cos(θ) + (axis × v) sin(θ) + axis (axis·v)(1 - cos(θ))

    Parameters
    ----------
    v : ndarray (3,)
    axis : ndarray (3,)  — should be unit vector
    angle_rad : float

    Returns
    -------
    v_rot : ndarray (3,)
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)
