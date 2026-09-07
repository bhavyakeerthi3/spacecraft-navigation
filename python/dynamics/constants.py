"""
python/dynamics/constants.py

Physical constants used throughout the simulation.

DESIGN DECISION:
  All constants are defined here and imported everywhere else.
  Never hard-code numerical values in dynamics or filter code.
  This makes it trivial to switch between Earth, Moon, Mars scenarios
  or to update constants without hunting through the codebase.

PROVENANCE:
  mu_m3s2  : IAU 2012 system; see also IERS Conventions 2010, Table 1.1
  Re_m     : WGS84 semi-major axis — used only for altitude computation
              and initial conditions; not used inside the filter.
  J2       : EGM2008 second zonal harmonic coefficient — provided for
              future higher-fidelity extension (Phase 10+).
              Not used in the current two-body propagator.

UNITS:
  All quantities in SI: metres, seconds, kilograms, radians.
  Conversion factors provided for convenience at config/plot boundaries.

INTERVIEW NOTE:
  An interviewer will ask: "Why is GM a single constant and not G*M_Earth
  separately?" Answer: GM is measured with much higher precision than G or
  M individually because satellite tracking directly constrains the
  combined product through Kepler's third law.
"""

import math

# -----------------------------------------------------------------------
# Gravitational parameter  [m³ / s²]
# -----------------------------------------------------------------------
MU_EARTH_M3S2: float = 3.986004418e14

# -----------------------------------------------------------------------
# Reference radii  [m]
# -----------------------------------------------------------------------
RE_M: float = 6_378_137.0          # WGS84 equatorial semi-major axis
RP_M: float = 6_356_752.3142       # WGS84 polar radius (not used yet)

# -----------------------------------------------------------------------
# J2 zonal harmonic (dimensionless) — for future J2 extension
# -----------------------------------------------------------------------
J2_EARTH: float = 1.08262668e-3

# -----------------------------------------------------------------------
# Rotation rate of Earth  [rad/s]
# -----------------------------------------------------------------------
OMEGA_EARTH_RADS: float = 7.2921150e-5

# -----------------------------------------------------------------------
# Speed of light  [m/s]
# -----------------------------------------------------------------------
C_LIGHT_MS: float = 299_792_458.0

# -----------------------------------------------------------------------
# Derived convenience values
# -----------------------------------------------------------------------
TWO_PI: float = 2.0 * math.pi

# -----------------------------------------------------------------------
# Unit conversion factors (applied only at config/plot boundaries)
# -----------------------------------------------------------------------
DEG2RAD: float = math.pi / 180.0
RAD2DEG: float = 180.0 / math.pi
PPM2SI:  float = 1.0e-6        # parts-per-million → dimensionless
MRAD2RAD: float = 1.0e-3       # milli-radians → radians
ARCSEC2RAD: float = math.pi / (180.0 * 3600.0)
KM2M: float = 1.0e3
M2KM: float = 1.0e-3
