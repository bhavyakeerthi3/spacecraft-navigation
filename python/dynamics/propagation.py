"""
python/dynamics/propagation.py

Numerical integrators for truth simulation and navigation prediction.

TWO INDEPENDENT INTEGRATORS — WHY?
  1. TRUTH integrator (DOP853, adaptive step):
     High-accuracy reference. Uses scipy's DOP853 (Dormand-Prince 8th order)
     with tight tolerances (rtol=atol=1e-12). This is an independent benchmark
     for verifying that the navigation filter is not hiding a dynamics bug.

  2. NAVIGATION integrator (RK4, fixed step, dt=0.1 s):
     Used in the EKF prediction step. Fixed-step RK4 is simple to replicate
     in C++ and MATLAB. The step size is chosen to be small relative to
     orbital dynamics time scales (period ~5700 s → dt/T < 2e-4).

WHY NOT USE THE SAME INTEGRATOR FOR BOTH?
  If truth and filter share the same integrator, a systematic integration
  error would be invisible (both are equally wrong). Using independent
  methods allows integration error to appear as a distinct residual
  signature in the filter innovations.

STEP-SIZE CONVERGENCE:
  Phase 1 requirement: halve dt_nav and verify that position error
  relative to DOP853 truth decreases by ~16x (4th-order method).

FORCE DISCONTINUITIES:
  The truth integrator splits the integration interval at thrust
  start/end times to prevent smoothing across step functions.
  The navigation RK4 also respects these boundaries by clamping steps.

NAVIGATION RK4 + VARIATIONAL EQUATIONS:
  To propagate the state transition matrix Φ and input sensitivity B,
  we augment the state with:
      Φ ∈ ℝ^{n×n}  (n = filter state dimension)
      B ∈ ℝ^{n×3}  (3 = accelerometer dimension)

  The augmented ODE is:
      dΦ/dt = F(x,t) Φ,    Φ(0) = I
      dB/dt = F(x,t) B + G_a(t),  B(0) = 0

  where G_a = [0; C_IB; 0] maps accelerometer errors into state space.

  This propagation is used in ekf.py to compute the discrete process
  noise covariance Q_d.

UNITS: All positions in [m], velocities in [m/s], time in [s].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from typing import Optional, Callable

from .forces import ForceModel
from .two_body import gravity_acceleration, gravity_jacobian
from .constants import MU_EARTH_M3S2


# -----------------------------------------------------------------------
# Initial condition helper
# -----------------------------------------------------------------------

def circular_orbit_initial_condition(
    altitude_m: float,
    inclination_deg: float,
    raan_deg: float = 0.0,
    arg_lat_deg: float = 0.0,
    mu: float = MU_EARTH_M3S2,
    Re: float = 6_378_137.0,
) -> NDArray[np.float64]:
    """
    Compute the ECI state vector [r(3), v(3)] for a circular orbit.

    Parameters
    ----------
    altitude_m : float
        Altitude above Re [m].
    inclination_deg : float
        Orbital inclination [deg].
    raan_deg : float
        Right Ascension of Ascending Node [deg]. Default 0.
    arg_lat_deg : float
        Argument of latitude (= argument of perigee + true anomaly) [deg].
        Default 0 places the spacecraft at the ascending node.
    mu : float
        Gravitational parameter [m³/s²].
    Re : float
        Reference radius [m] (used to define altitude).

    Returns
    -------
    state0 : ndarray, shape (6,)
        [rx, ry, rz, vx, vy, vz] in ECI [m, m/s].

    Mathematics:
        r_mag = Re + h
        v_c   = sqrt(mu / r_mag)

        At arg_lat = 0 (ascending node):
            r₀ = r_mag [cos(Ω), sin(Ω), 0]ᵀ
            v₀ = v_c   [-sin(Ω)cos(i), cos(Ω)cos(i), sin(i)]ᵀ

        General arg_lat u:
            The orbit-normal axis k = [-sin(Ω)sin(i), cos(Ω)sin(i), cos(i)]ᵀ
            Position is r₀ rotated by u about k.
            Velocity is perpendicular in the orbit plane.

    INTERVIEW NOTE:
        RAAN = 0 and arg_lat = 0 is not "arbitrary" — it places the
        spacecraft at the equatorial crossing on the Greenwich meridian at
        t=0. For a pure simulation with an idealized ECI frame and no
        Earth rotation model, this choice has no physical consequence but
        must be declared so results are reproducible.
    """
    inc = np.deg2rad(inclination_deg)
    raan = np.deg2rad(raan_deg)
    u    = np.deg2rad(arg_lat_deg)

    r_mag = Re + altitude_m
    v_c   = np.sqrt(mu / r_mag)

    # Perifocal → ECI rotation
    # Orbit normal unit vector (in direction of angular momentum)
    k = np.array([
        -np.sin(raan) * np.sin(inc),
         np.cos(raan) * np.sin(inc),
         np.cos(inc),
    ])

    # Ascending node direction in ECI
    n_hat = np.array([np.cos(raan), np.sin(raan), 0.0])

    # Position at arg_lat = 0
    r_node = r_mag * n_hat

    # Rotate r_node by u about k using Rodrigues' formula
    r_0 = (
        r_node * np.cos(u)
        + np.cross(k, r_node) * np.sin(u)
        + k * np.dot(k, r_node) * (1.0 - np.cos(u))
    )

    # Velocity is perpendicular to r and to orbit normal
    r_hat = r_0 / np.linalg.norm(r_0)
    v_0   = v_c * np.cross(k, r_hat)

    state0 = np.concatenate([r_0, v_0])
    return state0


# -----------------------------------------------------------------------
# Truth integrator (DOP853)
# -----------------------------------------------------------------------

def propagate_truth(
    state0: NDArray[np.float64],
    t_span: tuple[float, float],
    force_model: ForceModel,
    C_IB_func: Callable[[float], NDArray[np.float64]],
    rtol: float = 1e-12,
    atol: float = 1e-12,
    t_eval: Optional[NDArray[np.float64]] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Propagate the truth orbit using scipy's DOP853 adaptive integrator.

    The integration interval is automatically split at force discontinuities
    to prevent numerical smearing of step forces.

    Parameters
    ----------
    state0 : ndarray, shape (6,)
        Initial state [r(3), v(3)] in ECI.
    t_span : tuple (t_start, t_end)
        Integration time interval [s].
    force_model : ForceModel
        Force model providing ODE RHS.
    C_IB_func : callable
        Function C_IB_func(t) → (3,3) rotation matrix body → inertial.
        For Phase 1–7: returns identity (no rotation).
    rtol, atol : float
        DOP853 tolerances.
    t_eval : ndarray or None
        Times at which to return state. If None, solver chooses.

    Returns
    -------
    t_out : ndarray, shape (N,)
        Times [s].
    states : ndarray, shape (N, 6)
        States [r(3), v(3)] at each time.
    """
    t_start, t_end = t_span

    # Split integration at force discontinuities
    disc_times = [
        t for t in force_model.discontinuity_times()
        if t_start < t < t_end
    ]
    boundaries = [t_start] + sorted(disc_times) + [t_end]

    all_t: list[NDArray] = []
    all_y: list[NDArray] = []

    state = state0.copy()

    for i in range(len(boundaries) - 1):
        t0 = boundaries[i]
        t1 = boundaries[i + 1]

        if t_eval is not None:
            t_eval_seg = t_eval[(t_eval >= t0) & (t_eval <= t1)]
        else:
            t_eval_seg = None

        def rhs(t: float, y: NDArray) -> NDArray:
            return force_model.ode_rhs(t, y, C_IB_func(t))

        sol = solve_ivp(
            rhs,
            (t0, t1),
            state,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            t_eval=t_eval_seg,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(
                f"DOP853 integration failed on [{t0}, {t1}]: {sol.message}"
            )

        # Collect output, avoiding duplicate boundary points
        t_seg = sol.t
        y_seg = sol.y.T   # shape (N, 6)

        if len(all_t) > 0 and len(t_seg) > 0 and t_seg[0] == all_t[-1][-1]:
            t_seg = t_seg[1:]
            y_seg = y_seg[1:]

        all_t.append(t_seg)
        all_y.append(y_seg)
        state = sol.y[:, -1]  # carry-over for next segment

    t_out   = np.concatenate(all_t)
    states  = np.vstack(all_y)
    return t_out, states


# -----------------------------------------------------------------------
# Navigation integrator: fixed-step RK4 with variational equations
# -----------------------------------------------------------------------

def _nav_ode_rhs_9state(
    t: float,
    x: NDArray[np.float64],
    C_IB: NDArray[np.float64],
    mu: float,
    accel_measurement: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    ODE RHS for the 9-state navigation filter propagation.

    State x = [r(3), v(3), b_a(3)] in ECI.

    Equations:
        dr/dt = v
        dv/dt = g(r) + C_IB @ (accel_measurement - b_a)
        db/dt = 0   (bias modelled as random walk; deterministic part is zero)

    Parameters
    ----------
    accel_measurement : ndarray (3,)
        IMU accelerometer measurement (specific force in body frame) [m/s²].
        In the filter prediction: this is the held IMU interval average.

    Returns
    -------
    dx_dt : ndarray (9,)
    """
    r  = x[0:3]
    v  = x[3:6]
    ba = x[6:9]

    g_I  = gravity_acceleration(r, mu=mu)
    f_B  = accel_measurement - ba      # corrected specific force in B
    f_I  = C_IB @ f_B                  # rotated to I

    dx_dt = np.empty(9, dtype=np.float64)
    dx_dt[0:3] = v
    dx_dt[3:6] = g_I + f_I
    dx_dt[6:9] = np.zeros(3)           # deterministic bias drift = 0
    return dx_dt


def _nav_jacobian_9state(
    r: NDArray[np.float64],
    C_IB: NDArray[np.float64],
    mu: float,
) -> NDArray[np.float64]:
    """
    9×9 Jacobian F = ∂f/∂x for the navigation filter ODE.

    F = [[0₃   I₃   0₃  ]
         [Gr   0₃  -C_IB]
         [0₃   0₃   0₃  ]]

    where Gr = ∂g/∂r = μ(3rrᵀ/|r|⁵ - I/|r|³).

    DERIVATION:
        ∂(dr/dt)/∂r = 0,  ∂(dr/dt)/∂v = I,  ∂(dr/dt)/∂b = 0
        ∂(dv/dt)/∂r = Gr, ∂(dv/dt)/∂v = 0,  ∂(dv/dt)/∂b = -C_IB
        ∂(db/dt)/∂r = 0,  ∂(db/dt)/∂v = 0,  ∂(db/dt)/∂b = 0

    Returns
    -------
    F : ndarray, shape (9, 9)
    """
    Gr = gravity_jacobian(r, mu=mu)
    F  = np.zeros((9, 9), dtype=np.float64)
    F[0:3, 3:6] = np.eye(3)
    F[3:6, 0:3] = Gr
    F[3:6, 6:9] = -C_IB
    return F


def rk4_predict(
    x: NDArray[np.float64],
    P: NDArray[np.float64],
    dt: float,
    C_IB: NDArray[np.float64],
    accel_measurement: NDArray[np.float64],
    Qa_m2s3: float,
    Qb_m2s5: float,
    mu: float = MU_EARTH_M3S2,
) -> tuple[NDArray[np.float64], NDArray[np.float64],
           NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    One fixed-step RK4 prediction step for the 9-state EKF.

    Integrates the state, state-transition matrix Φ, and input sensitivity
    matrix B simultaneously.

    State transition ODE (Lyapunov/variational equations):
        dΦ/dt = F(x,t) Φ,    Φ(0) = I₉
        dB/dt = F(x,t) B + G_a,  B(0) = 0

    where G_a = [0; C_IB; 0] is the input-error sensitivity.

    Discrete process noise (held interval-average noise convention):
        Q_d = B Qa/dt Bᵀ + E_b Qb·dt E_bᵀ

    Units:
        Qa_m2s3 [m²/s³] — accelerometer noise spectral density
        Qb_m2s5 [m²/s⁵] — bias random-walk spectral density

    Returns
    -------
    x_pred : ndarray (9,)    — propagated state
    P_pred : ndarray (9, 9)  — propagated covariance
    Phi    : ndarray (9, 9)  — state transition matrix
    Qd     : ndarray (9, 9)  — discrete process noise
    B_mat  : ndarray (9, 3)  — input sensitivity matrix
    """
    n = 9

    # ----- Augmented state for variational equations -----
    # Pack [x(9), Phi(81), B(27)] → total 117 elements
    aug0 = np.zeros(n + n * n + n * 3, dtype=np.float64)
    aug0[:n] = x
    aug0[n : n + n * n] = np.eye(n).ravel()  # Phi = I

    def aug_rhs(t: float, aug: NDArray) -> NDArray:
        x_   = aug[:n]
        Phi_ = aug[n : n + n * n].reshape(n, n)
        B_   = aug[n + n * n :].reshape(n, 3)

        r_   = x_[0:3]
        F_   = _nav_jacobian_9state(r_, C_IB, mu)
        G_a  = np.zeros((n, 3), dtype=np.float64)
        G_a[3:6, :] = C_IB

        dx_dt   = _nav_ode_rhs_9state(t, x_, C_IB, mu, accel_measurement)
        dPhi_dt = F_ @ Phi_
        dB_dt   = F_ @ B_ + G_a

        daug_dt = np.empty_like(aug)
        daug_dt[:n] = dx_dt
        daug_dt[n : n + n * n] = dPhi_dt.ravel()
        daug_dt[n + n * n :] = dB_dt.ravel()
        return daug_dt

    # ----- RK4 -----
    k1 = aug_rhs(0.0,      aug0)
    k2 = aug_rhs(dt / 2.0, aug0 + dt / 2.0 * k1)
    k3 = aug_rhs(dt / 2.0, aug0 + dt / 2.0 * k2)
    k4 = aug_rhs(dt,       aug0 + dt * k3)

    aug1 = aug0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    x_pred = aug1[:n]
    Phi    = aug1[n : n + n * n].reshape(n, n)
    B_mat  = aug1[n + n * n :].reshape(n, 3)

    # ----- Discrete process noise Q_d -----
    # Held interval-average accelerometer noise contribution:
    #   Q_d,accel = B (Qa/dt) Bᵀ
    # Bias random-walk contribution (end-of-interval jump):
    #   Q_d,bias  = E_b Qb·dt E_bᵀ
    #
    # CRITICAL: Do NOT add the same noise twice. Q_d,accel and Q_d,bias
    # are derived from separate, independent stochastic processes.
    E_b = np.zeros((n, 3), dtype=np.float64)
    E_b[6:9, :] = np.eye(3)

    Qd  = B_mat @ (Qa_m2s3 / dt * np.eye(3)) @ B_mat.T
    Qd += E_b @ (Qb_m2s5 * dt * np.eye(3)) @ E_b.T

    # ----- Covariance propagation -----
    P_pred = Phi @ P @ Phi.T + Qd

    # Enforce symmetry to remove floating-point asymmetry
    P_pred = 0.5 * (P_pred + P_pred.T)

    return x_pred, P_pred, Phi, Qd, B_mat


# -----------------------------------------------------------------------
# Six-state ODE RHS (for Phase 3: GNSS-only EKF, no IMU bias state)
# -----------------------------------------------------------------------

def rk4_predict_6state(
    x6: NDArray[np.float64],
    P6: NDArray[np.float64],
    dt: float,
    C_IB: NDArray[np.float64],
    accel_I: NDArray[np.float64],
    Q_model_m2s3: float,
    mu: float = MU_EARTH_M3S2,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    One fixed-step RK4 prediction for the 6-state orbital EKF (no bias).

    State x6 = [r(3), v(3)].
    accel_I is the externally applied (commanded or measured) acceleration
    in ECI [m/s²]. For pure GNSS coasting: pass zeros.

    Q_model_m2s3 is a declared model uncertainty acceleration PSD [m²/s³].
    It represents unmodelled force uncertainty (NOT accelerometer noise —
    there is no accelerometer in this baseline filter).

    Returns
    -------
    x6_pred : ndarray (6,)
    P6_pred : ndarray (6, 6)
    Phi6    : ndarray (6, 6)
    """
    n = 6

    def rhs6(t: float, y: NDArray) -> NDArray:
        r_ = y[0:3]
        v_ = y[3:6]
        g_ = gravity_acceleration(r_, mu=mu)
        dy = np.empty(6)
        dy[0:3] = v_
        dy[3:6] = g_ + accel_I
        return dy

    def F6(r_: NDArray) -> NDArray:
        Gr_ = gravity_jacobian(r_, mu=mu)
        F   = np.zeros((6, 6))
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 0:3] = Gr_
        return F

    # Pack [x6, Phi6(36)]
    aug0 = np.zeros(n + n * n)
    aug0[:n] = x6
    aug0[n:] = np.eye(n).ravel()

    def aug_rhs(t: float, aug: NDArray) -> NDArray:
        x_   = aug[:n]
        Phi_ = aug[n:].reshape(n, n)
        r_   = x_[0:3]
        F_   = F6(r_)
        daug = np.empty_like(aug)
        daug[:n] = rhs6(t, x_)
        daug[n:] = (F_ @ Phi_).ravel()
        return daug

    k1 = aug_rhs(0.0,        aug0)
    k2 = aug_rhs(dt / 2.0,   aug0 + dt / 2.0 * k1)
    k3 = aug_rhs(dt / 2.0,   aug0 + dt / 2.0 * k2)
    k4 = aug_rhs(dt,          aug0 + dt * k3)

    aug1 = aug0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    x6_pred = aug1[:n]
    Phi6    = aug1[n:].reshape(n, n)

    # Model-error process noise: Q_d,6 ~ Lv Qc dt² / 2 (simplified Van Loan)
    # For the GNSS+dynamics baseline, use a simple velocity-noise model:
    #   process noise maps as [0; dt I] @ Q_model/dt @ [0; dt I]ᵀ → Δv block
    Lv = np.zeros((n, 3))
    Lv[3:6, :] = np.eye(3)
    Qd6 = Lv @ (Q_model_m2s3 / dt * np.eye(3)) @ Lv.T

    P6_pred = Phi6 @ P6 @ Phi6.T + Qd6
    P6_pred = 0.5 * (P6_pred + P6_pred.T)

    return x6_pred, P6_pred, Phi6
