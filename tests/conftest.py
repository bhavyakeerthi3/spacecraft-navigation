# tests/conftest.py
"""
Shared pytest fixtures available to all test modules.
"""
import numpy as np
import pytest

from python.dynamics.constants import MU_EARTH_M3S2, RE_M
from python.dynamics.two_body import circular_speed, orbital_period
from python.dynamics.propagation import circular_orbit_initial_condition

@pytest.fixture(scope="session")
def orbit_constants():
    """Shared orbital constants for all tests."""
    alt = 500_000.0
    Re  = RE_M
    mu  = MU_EARTH_M3S2
    r   = Re + alt
    return {
        "alt_m":    alt,
        "Re_m":     Re,
        "mu_m3s2":  mu,
        "r_orbit_m": r,
        "v_circ_ms": circular_speed(r, mu),
        "period_s":  orbital_period(r, mu),
    }
