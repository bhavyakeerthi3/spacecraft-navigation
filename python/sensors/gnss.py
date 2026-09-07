"""
python/sensors/gnss.py

GNSS solution-level measurement simulator.

WHAT THIS MODELS:
  A spacecraft GNSS receiver that outputs a position and velocity solution
  at a fixed rate. This is a LOOSELY COUPLED model — the filter receives
  the solved position/velocity, NOT raw pseudoranges or carrier phases.

WHAT THIS DOES NOT MODEL (explicitly declared):
  - Satellite geometry (DOP, PDOP, elevation masks)
  - Pseudorange / carrier-phase observables
  - Receiver clock bias and drift
  - Multipath or ionospheric/tropospheric delays
  - Ambiguity resolution or cycle slips
  - RF tracking loop dynamics or acquisition
  - ECEF ↔ ECI transformation (we work in a single idealized ECI frame)
  These omissions are consistent with the scope declared in the blueprint.
  A tightly coupled architecture would model the pseudorange level.

ERROR MODEL (per blueprint §5.1):
  z_k = [r(t_k); v(t_k)] + b_k^G + ε_k + o_k

  where:
    ε_k ~ N(0, R_k)              White Gaussian noise
    b_k^G                        Constant or Gauss-Markov bias
    o_k                          Outlier (mixture component, Bernoulli p)
    R_k = diag(σ_pos² I, σ_vel² I)   Navigation-declared covariance

NAVIGATION RECEIVES:
  - z_k (measurement value, possibly corrupted by bias/outlier)
  - R_k (declared noise covariance — does NOT include bias or outlier)
  - sample_time_s, delivery_time_s, is_valid
  - No fault labels, no truth, no bias values

INTERVIEW NOTE:
  "Why is increasing R insufficient to handle temporally correlated bias?"
  Because R models independent, white measurement noise. A bias is persistent
  across epochs and creates correlated innovations. Inflating R changes the
  Kalman gain but does not capture the cross-epoch correlation structure.
  The correct fix is to augment the state with a bias state (not done in the
  nominal 9-state filter, which is intentionally mismatched for stress tests).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional

from ..records import MeasurementPacket


class GNSSSimulator:
    """
    GNSS solution-level measurement simulator for a spacecraft in ECI.

    Parameters are set via the GNSSConfig dataclass from config.py.
    All physical parameters are in SI units (metres, m/s, seconds).

    Usage
    -----
    sim = GNSSSimulator(cfg.gnss, cfg.streams, dt_nav_s=cfg.dt_nav_s)
    packets = sim.simulate(truth_states, t_array)
    """

    SENSOR_ID = "gnss_0"

    def __init__(
        self,
        gnss_cfg,           # GNSSConfig from config.py
        stream_factory,     # StreamFactory from random_streams.py
    ) -> None:
        self._cfg = gnss_cfg
        self._rng_noise   = stream_factory.get("gnss_noise")
        self._rng_bias    = stream_factory.get("gnss_bias")
        self._rng_outlier = stream_factory.get("gnss_outliers")

        # Build declared measurement covariance R (6×6, position/velocity)
        sigma_p = gnss_cfg.pos_noise_1sigma_m
        sigma_v = gnss_cfg.vel_noise_1sigma_ms
        self._R_declared = np.diag([
            sigma_p**2, sigma_p**2, sigma_p**2,
            sigma_v**2, sigma_v**2, sigma_v**2,
        ])

        # Initialize Gauss-Markov bias state
        if gnss_cfg.gm_bias_enabled:
            # Draw initial bias from stationary distribution N(0, Σ_b)
            sp = gnss_cfg.gm_sigma_pos_m
            sv = gnss_cfg.gm_sigma_vel_ms
            self._gm_bias = np.concatenate([
                self._rng_bias.normal(0, sp, 3),
                self._rng_bias.normal(0, sv, 3),
            ])
        else:
            self._gm_bias = np.zeros(6)

        self._sequence_counter = 0

    @property
    def R_declared(self) -> NDArray[np.float64]:
        """Navigation-declared 6×6 measurement covariance."""
        return self._R_declared.copy()

    def _in_outage(self, t: float) -> bool:
        """True if time t falls within any configured outage window."""
        for (t_start, t_end) in self._cfg.outage_windows:
            if t_start <= t < t_end:
                return True
        return False

    def _update_gm_bias(self, dt: float) -> None:
        """
        Advance the Gauss-Markov bias one step.

        b_{k+1} = α b_k + η_k
        α = exp(-Δt / τ)
        η_k ~ N(0, (1 - α²) Σ_b)

        This produces a stationary first-order Gauss-Markov process
        with steady-state variance Σ_b.
        """
        tau  = self._cfg.gm_tau_s
        alpha = np.exp(-dt / tau)

        sigma_p = self._cfg.gm_sigma_pos_m
        sigma_v = self._cfg.gm_sigma_vel_ms
        sigma_eta = np.sqrt(1.0 - alpha**2) * np.array([
            sigma_p, sigma_p, sigma_p,
            sigma_v, sigma_v, sigma_v,
        ])

        eta = self._rng_bias.normal(0.0, 1.0, 6) * sigma_eta
        self._gm_bias = alpha * self._gm_bias + eta

    def generate_packet(
        self,
        t: float,
        r_true_I: NDArray[np.float64],
        v_true_I: NDArray[np.float64],
        prev_t: Optional[float] = None,
    ) -> Optional[MeasurementPacket]:
        """
        Generate one GNSS measurement packet at time t.

        Returns None if the receiver is in an outage window.

        Parameters
        ----------
        t : float
            Sample time [s].
        r_true_I : ndarray (3,)
            True ECI position [m].
        v_true_I : ndarray (3,)
            True ECI velocity [m/s].
        prev_t : float or None
            Previous packet time, for Gauss-Markov bias update.

        Returns
        -------
        packet : MeasurementPacket or None
        """
        # --- Outage check ---
        if self._in_outage(t):
            return None   # No packet during outage; do NOT send zeros

        # --- Gauss-Markov bias update ---
        if self._cfg.gm_bias_enabled and prev_t is not None:
            dt_gm = t - prev_t
            if dt_gm > 0:
                self._update_gm_bias(dt_gm)

        # --- White noise ---
        sigma_p = self._cfg.pos_noise_1sigma_m
        sigma_v = self._cfg.vel_noise_1sigma_ms
        noise_pos = self._rng_noise.normal(0.0, sigma_p, 3)
        noise_vel = self._rng_noise.normal(0.0, sigma_v, 3)
        noise = np.concatenate([noise_pos, noise_vel])

        # --- Constant bias ---
        const_bias = np.zeros(6)
        if self._cfg.bias_enabled:
            const_bias = np.concatenate([
                np.asarray(self._cfg.bias_pos_m, dtype=np.float64),
                np.asarray(self._cfg.bias_vel_ms, dtype=np.float64),
            ])

        # --- Gauss-Markov bias ---
        gm_bias = self._gm_bias if self._cfg.gm_bias_enabled else np.zeros(6)

        # --- Outlier ---
        outlier = np.zeros(6)
        if self._cfg.outliers_enabled:
            draw = self._rng_outlier.uniform()
            if draw < self._cfg.outlier_prob:
                # Broad error, independent per axis
                outlier[:3] = self._rng_outlier.normal(
                    0, self._cfg.outlier_pos_m, 3
                )
                outlier[3:] = self._rng_outlier.normal(
                    0, self._cfg.outlier_vel_ms, 3
                )

        # --- Compose measurement ---
        true_pv = np.concatenate([r_true_I, v_true_I])
        z = true_pv + const_bias + gm_bias + noise + outlier

        # --- Delivery time (latency) ---
        delivery_t = t + self._cfg.latency_s

        self._sequence_counter += 1

        return MeasurementPacket(
            sensor_id=self.SENSOR_ID,
            sequence_num=self._sequence_counter,
            sample_time_s=t,
            delivery_time_s=delivery_t,
            value=z,
            declared_covariance=self._R_declared.copy(),
            frame="ECI",
            units="[m, m, m, m/s, m/s, m/s]",
            is_valid=True,
        )

    def simulate(
        self,
        t_truth: NDArray[np.float64],
        states_truth: NDArray[np.float64],
    ) -> list[MeasurementPacket]:
        """
        Generate all GNSS packets for a truth trajectory.

        Measurement epochs are chosen at the configured rate, aligned
        to integer multiples of the measurement interval starting at t=0.

        Parameters
        ----------
        t_truth : ndarray (N,)
            Truth time array [s].
        states_truth : ndarray (N, 6)
            Truth states [r(3), v(3)] at each truth time.

        Returns
        -------
        packets : list of MeasurementPacket
            One packet per GNSS epoch (None entries excluded).
        """
        dt_gnss = 1.0 / self._cfg.rate_hz
        t_start = t_truth[0]
        t_end   = t_truth[-1]

        # GNSS epochs: integer multiples of dt_gnss within [t_start, t_end]
        n_meas = int(np.floor((t_end - t_start) / dt_gnss)) + 1
        t_gnss = t_start + np.arange(n_meas) * dt_gnss
        t_gnss = t_gnss[t_gnss <= t_end + 1e-9]

        packets: list[MeasurementPacket] = []
        prev_t: Optional[float] = None

        for tk in t_gnss:
            # Interpolate truth state at measurement epoch
            r_k = np.interp(tk, t_truth, states_truth[:, 0]), \
                  np.interp(tk, t_truth, states_truth[:, 1]), \
                  np.interp(tk, t_truth, states_truth[:, 2])
            v_k = np.interp(tk, t_truth, states_truth[:, 3]), \
                  np.interp(tk, t_truth, states_truth[:, 4]), \
                  np.interp(tk, t_truth, states_truth[:, 5])
            r_k = np.array(r_k)
            v_k = np.array(v_k)

            pkt = self.generate_packet(tk, r_k, v_k, prev_t)
            if pkt is not None:
                packets.append(pkt)
                prev_t = tk
            else:
                # In outage — do NOT update prev_t for GM bias
                # (bias is frozen during outage, resumes on recovery)
                pass

        return packets
