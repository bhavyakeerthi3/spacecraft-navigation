"""
python/config.py

Configuration loading, validation, and resolved-manifest generation.

DESIGN:
  All physical quantities live in configs/*.yaml.
  This module loads a config, validates it, and returns a strongly-typed
  dataclass. Any unit conversion (deg→rad, ppm→dimensionless) happens here.
  Downstream code never does unit conversions — it always works in SI.

REPRODUCIBILITY:
  The resolved config (with computed derived quantities) is saved as part of
  every run manifest. That manifest, together with the software version and
  random seeds, is sufficient to reproduce any result.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

import yaml
import numpy as np

from .dynamics.constants import MU_EARTH_M3S2, RE_M, DEG2RAD


# -----------------------------------------------------------------------
# Config dataclass
# -----------------------------------------------------------------------

@dataclass
class GNSSConfig:
    enabled: bool
    rate_hz: float
    pos_noise_1sigma_m: float
    vel_noise_1sigma_ms: float
    bias_enabled: bool
    bias_pos_m: np.ndarray
    bias_vel_ms: np.ndarray
    gm_bias_enabled: bool
    gm_tau_s: float
    gm_sigma_pos_m: float
    gm_sigma_vel_ms: float
    outliers_enabled: bool
    outlier_prob: float
    outlier_pos_m: float
    outlier_vel_ms: float
    latency_s: float
    outage_windows: list[tuple[float, float]]


@dataclass
class AccelConfig:
    noise_density_m2s3: float
    bias_init_1sigma_ms2: float
    bias_walk_density_m2s5: float
    scale_factor: np.ndarray    # dimensionless, 3 axes
    misalignment_rad: np.ndarray  # radians, [xy, xz, yz]
    noise_enabled: bool
    bias_enabled: bool
    walk_enabled: bool
    scale_enabled: bool
    misalign_enabled: bool
    latency_s: float


@dataclass
class GyroConfig:
    noise_density_rad2s: float
    bias_init_1sigma_rads: float
    bias_walk_density_rad2s3: float
    scale_factor: np.ndarray
    misalignment_rad: np.ndarray
    noise_enabled: bool
    bias_enabled: bool
    walk_enabled: bool
    scale_enabled: bool
    misalign_enabled: bool
    latency_s: float


@dataclass
class IMUConfig:
    enabled: bool
    rate_hz: float
    accel: AccelConfig
    gyro: GyroConfig


@dataclass
class FilterConfig:
    state_dim: int
    init_pos_1sigma_m: float
    init_vel_1sigma_ms: float
    init_bias_1sigma_ms2: float
    Qa_m2s3: float
    Qb_m2s5: float
    gnss_R_pos_1sigma_m: float
    gnss_R_vel_1sigma_ms: float
    gate_probability: float
    attitude_source: str


@dataclass
class SimConfig:
    """Fully resolved, SI-unit simulation configuration."""
    scenario: str
    schema_version: str

    # Derived orbital parameters
    mu_m3s2: float
    Re_m: float
    altitude_m: float
    inclination_rad: float
    raan_rad: float
    arg_lat_rad: float
    r_orbit_m: float        # Re + altitude
    v_circ_ms: float        # sqrt(mu/r)
    period_s: float         # 2π sqrt(r³/mu)

    # Simulation time
    t_start_s: float
    t_end_s: float
    dt_nav_s: float

    # Truth integrator
    truth_rtol: float
    truth_atol: float

    # Sensor configs
    gnss: GNSSConfig
    imu: IMUConfig

    # Filter config
    filter: FilterConfig

    # Random seeds (master + stream offsets)
    master_seed: int
    streams: dict[str, int]

    # Output
    results_dir: Path
    figures_dir: Path
    tables_dir: Path
    manifests_dir: Path
    figure_dpi: int

    # Hash of the resolved config for reproducibility
    config_hash: str = ""

    def to_manifest_dict(self) -> dict:
        """Serialize to a JSON-compatible dict for manifest writing."""
        d = {
            "scenario":        self.scenario,
            "schema_version":  self.schema_version,
            "config_hash":     self.config_hash,
            "orbital": {
                "altitude_m":      self.altitude_m,
                "inclination_deg": np.degrees(self.inclination_rad),
                "r_orbit_m":       self.r_orbit_m,
                "v_circ_ms":       self.v_circ_ms,
                "period_s":        self.period_s,
            },
            "time": {
                "t_start_s":  self.t_start_s,
                "t_end_s":    self.t_end_s,
                "dt_nav_s":   self.dt_nav_s,
            },
            "random": {
                "master_seed": self.master_seed,
                "streams":     self.streams,
            },
        }
        return d


# -----------------------------------------------------------------------
# Loader
# -----------------------------------------------------------------------

def _ppm_list_to_array(values: list[float]) -> np.ndarray:
    """Convert ppm list to dimensionless array."""
    return np.array(values, dtype=np.float64) * 1e-6


def _mrad_list_to_array(values: list[float]) -> np.ndarray:
    """Convert mrad list to radians array."""
    return np.array(values, dtype=np.float64) * 1e-3


def load_config(config_path: str | Path) -> SimConfig:
    """
    Load and validate a YAML configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    cfg : SimConfig
        Fully resolved, validated configuration in SI units.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If required fields are missing or values are physically unreasonable.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # --- Constants ---
    mu     = raw["constants"]["mu_m3s2"]
    Re     = raw["constants"]["Re_m"]
    alt    = raw["orbit"]["altitude_m"]
    inc_d  = raw["orbit"]["inclination_deg"]
    raan_d = raw["orbit"]["raan_deg"]
    al_d   = raw["orbit"]["arg_lat_deg"]

    r_orbit = Re + alt
    v_circ  = np.sqrt(mu / r_orbit)
    period  = 2.0 * np.pi * np.sqrt(r_orbit**3 / mu)

    # --- Validate physical ranges ---
    if not (200e3 <= alt <= 2000e3):
        raise ValueError(f"Altitude {alt/1e3:.0f} km outside expected LEO range [200, 2000] km")
    if not (0 <= inc_d <= 180):
        raise ValueError(f"Inclination {inc_d}° outside [0, 180]")

    # --- GNSS ---
    gc = raw["gnss"]
    gnss = GNSSConfig(
        enabled=gc["enabled"],
        rate_hz=gc["rate_hz"],
        pos_noise_1sigma_m=gc["pos_noise_1sigma_m"],
        vel_noise_1sigma_ms=gc["vel_noise_1sigma_ms"],
        bias_enabled=gc["bias_enabled"],
        bias_pos_m=np.array(gc["bias_pos_m"], dtype=np.float64),
        bias_vel_ms=np.array(gc["bias_vel_ms"], dtype=np.float64),
        gm_bias_enabled=gc["gm_bias_enabled"],
        gm_tau_s=gc["gm_tau_s"],
        gm_sigma_pos_m=gc["gm_sigma_pos_m"],
        gm_sigma_vel_ms=gc["gm_sigma_vel_ms"],
        outliers_enabled=gc["outliers_enabled"],
        outlier_prob=gc["outlier_prob"],
        outlier_pos_m=gc["outlier_pos_m"],
        outlier_vel_ms=gc["outlier_vel_ms"],
        latency_s=gc["latency_s"],
        outage_windows=[tuple(w) for w in gc.get("outage_windows", [])],
    )

    # --- IMU ---
    ic = raw["imu"]
    ac = ic["accel"]
    accel = AccelConfig(
        noise_density_m2s3=ac["noise_density_m2s3"],
        bias_init_1sigma_ms2=ac["bias_init_1sigma_ms2"],
        bias_walk_density_m2s5=ac["bias_walk_density_m2s5"],
        scale_factor=_ppm_list_to_array(ac["scale_factor_ppm"]),
        misalignment_rad=_mrad_list_to_array(ac["misalignment_mrad"]),
        noise_enabled=ac["noise_enabled"],
        bias_enabled=ac["bias_enabled"],
        walk_enabled=ac["walk_enabled"],
        scale_enabled=ac["scale_enabled"],
        misalign_enabled=ac["misalign_enabled"],
        latency_s=ac["latency_s"],
    )

    gy = ic["gyro"]
    gyro = GyroConfig(
        noise_density_rad2s=gy["noise_density_rad2s"],
        bias_init_1sigma_rads=gy["bias_init_1sigma_rads"],
        bias_walk_density_rad2s3=gy["bias_walk_density_rad2s3"],
        scale_factor=_ppm_list_to_array(gy["scale_factor_ppm"]),
        misalignment_rad=_mrad_list_to_array(gy["misalignment_mrad"]),
        noise_enabled=gy["noise_enabled"],
        bias_enabled=gy["bias_enabled"],
        walk_enabled=gy["walk_enabled"],
        scale_enabled=gy["scale_enabled"],
        misalign_enabled=gy["misalign_enabled"],
        latency_s=gy["latency_s"],
    )

    imu = IMUConfig(
        enabled=ic["enabled"],
        rate_hz=ic["rate_hz"],
        accel=accel,
        gyro=gyro,
    )

    # --- Filter ---
    fc = raw["filter"]
    filt = FilterConfig(
        state_dim=fc["state_dim"],
        init_pos_1sigma_m=fc["init"]["pos_1sigma_m"],
        init_vel_1sigma_ms=fc["init"]["vel_1sigma_ms"],
        init_bias_1sigma_ms2=fc["init"]["bias_1sigma_ms2"],
        Qa_m2s3=fc["process_noise"]["Qa_m2s3"],
        Qb_m2s5=fc["process_noise"]["Qb_m2s5"],
        gnss_R_pos_1sigma_m=fc["gnss_R"]["pos_1sigma_m"],
        gnss_R_vel_1sigma_ms=fc["gnss_R"]["vel_1sigma_ms"],
        gate_probability=fc["gate_probability"],
        attitude_source=fc["attitude"]["source"],
    )

    # --- Random ---
    rng_cfg = raw["random"]

    # --- Output paths ---
    out = raw.get("output", {})
    proj_root = config_path.parent.parent
    results_dir   = proj_root / out.get("results_dir",  "results")
    figures_dir   = proj_root / out.get("figures_dir",  "results/figures")
    tables_dir    = proj_root / out.get("tables_dir",   "results/tables")
    manifests_dir = proj_root / out.get("manifests_dir","results/manifests")

    cfg = SimConfig(
        scenario=raw["meta"]["scenario"],
        schema_version=raw["meta"]["schema_version"],
        mu_m3s2=mu,
        Re_m=Re,
        altitude_m=alt,
        inclination_rad=np.deg2rad(inc_d),
        raan_rad=np.deg2rad(raan_d),
        arg_lat_rad=np.deg2rad(al_d),
        r_orbit_m=r_orbit,
        v_circ_ms=v_circ,
        period_s=period,
        t_start_s=raw["time"]["t_start_s"],
        t_end_s=raw["time"]["t_end_s"],
        dt_nav_s=raw["time"]["dt_nav_s"],
        truth_rtol=raw["truth_integrator"]["rtol"],
        truth_atol=raw["truth_integrator"]["atol"],
        gnss=gnss,
        imu=imu,
        filter=filt,
        master_seed=rng_cfg["master_seed"],
        streams=rng_cfg["streams"],
        results_dir=results_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        manifests_dir=manifests_dir,
        figure_dpi=out.get("figure_dpi", 300),
    )

    # Compute reproducibility hash over the resolved config dict
    cfg.config_hash = _hash_config(cfg)
    return cfg


def _hash_config(cfg: SimConfig) -> str:
    """SHA-256 hash of the serialized config manifest dict."""
    d = cfg.to_manifest_dict()
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


# -----------------------------------------------------------------------
# Quick test / entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    config_path = (
        Path(__file__).parent.parent / "configs" / "nominal.yaml"
    )
    try:
        cfg = load_config(config_path)
        print(f"Config loaded: {cfg.scenario} (hash={cfg.config_hash})")
        print(f"  Altitude:   {cfg.altitude_m/1e3:.0f} km")
        print(f"  Period:     {cfg.period_s:.1f} s ({cfg.period_s/60:.2f} min)")
        print(f"  v_circ:     {cfg.v_circ_ms:.2f} m/s")
        print(f"  State dim:  {cfg.filter.state_dim}")
        print(f"  GNSS rate:  {cfg.gnss.rate_hz} Hz")
        print(f"  IMU rate:   {cfg.imu.rate_hz} Hz")
        print("  Config hash:", cfg.config_hash)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
