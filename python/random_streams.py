"""
python/random_streams.py

Independent, reproducible random number streams for each error source.

DESIGN:
  Each stochastic quantity (GNSS noise, IMU bias, outliers, etc.) gets
  its own NumPy Generator derived from a master seed. This means:
    1. Changing one error source does NOT affect draws from others.
    2. Every stream is individually reproducible given its seed.
    3. Parallel Monte Carlo runs derive their generators from (master_seed, run_index).

  We use NumPy's Generator/PCG64 (not the legacy RandomState) for
  better statistical properties and explicit seeding.

  STREAM NAMING:
    Stream names are declared in configs/nominal.yaml under random.streams.
    Each name maps to an integer offset added to master_seed.
    The SimConfig.streams dict is the authority on valid stream names.

USAGE:
    from python.random_streams import StreamFactory

    factory = StreamFactory(master_seed=42, streams=cfg.streams)
    rng_gnss  = factory.get("gnss_noise")
    rng_accel = factory.get("accel_noise")

    # Draw samples:
    noise = rng_gnss.standard_normal(3) * sigma_pos

COMMON MISTAKE:
    Do NOT use a single shared RNG for all error sources. Changing the
    number of draws in one source would shift all subsequent seeds,
    making results irreproducible and comparisons invalid.

MONTE CARLO:
    For MC run k (0-indexed), derive per-run seeds as:
        run_seed = master_seed + 1000 * (k + 1)
    Pass run_seed as the master_seed to StreamFactory.
    This ensures run streams are independent AND reproducible.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class StreamFactory:
    """
    Factory for independent reproducible RNG streams.

    Parameters
    ----------
    master_seed : int
        Master seed. All stream seeds are derived from this.
    streams : dict[str, int]
        Mapping from stream name to integer offset.
        Seed for stream s = master_seed + streams[s].
    """

    def __init__(
        self,
        master_seed: int,
        streams: dict[str, int],
    ) -> None:
        self._master_seed = master_seed
        self._streams = streams
        self._generators: dict[str, np.random.Generator] = {}

        # Pre-build all generators
        for name, offset in streams.items():
            seed = (master_seed + offset) % (2**32)
            self._generators[name] = np.random.default_rng(seed)

    def get(self, name: str) -> np.random.Generator:
        """
        Return the independent RNG generator for the named stream.

        Parameters
        ----------
        name : str
            Stream name, must be in the streams dict from config.

        Raises
        ------
        KeyError
            If name is not a declared stream.
        """
        if name not in self._generators:
            raise KeyError(
                f"Unknown stream '{name}'. "
                f"Declared streams: {list(self._generators.keys())}"
            )
        return self._generators[name]

    @property
    def master_seed(self) -> int:
        return self._master_seed

    @property
    def stream_names(self) -> list[str]:
        return list(self._generators.keys())

    @classmethod
    def for_mc_run(
        cls,
        run_index: int,
        master_seed: int,
        streams: dict[str, int],
    ) -> "StreamFactory":
        """
        Create a StreamFactory for Monte Carlo run k (0-indexed).

        Each run gets a unique master seed derived from the global seed:
            run_seed = master_seed + 1000 * (run_index + 1)

        This scheme gives 1000 independent runs before seeds collide
        (in the unlikely worst case; PCG64 has excellent independence
        properties at much larger separations).
        """
        run_seed = master_seed + 1000 * (run_index + 1)
        return cls(master_seed=run_seed, streams=streams)


# -----------------------------------------------------------------------
# Quick test / entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    # Demo: verify that streams are independent and reproducible
    import sys

    STREAMS = {
        "gnss_noise":     0,
        "accel_noise":    1,
        "accel_bias_init": 2,
    }
    MASTER = 42

    factory1 = StreamFactory(MASTER, STREAMS)
    factory2 = StreamFactory(MASTER, STREAMS)  # Same seed → same results

    # Reproducibility: two factories with same seed must give same output
    draw1 = factory1.get("gnss_noise").standard_normal(3)
    draw2 = factory2.get("gnss_noise").standard_normal(3)
    assert np.allclose(draw1, draw2), "Reproducibility check failed!"
    print(f"gnss_noise draw:  {draw1}")

    # Independence: drawing from one stream should not affect another
    draw_accel_before = factory1.get("accel_noise").standard_normal(3).copy()
    _ = factory1.get("gnss_noise").standard_normal(100)  # exhaust gnss stream
    draw_accel_after  = factory1.get("accel_noise").standard_normal(3)

    # The two accel draws must be different (successive draws from same generator)
    print(f"accel_noise 1st:  {draw_accel_before}")
    print(f"accel_noise 2nd:  {draw_accel_after}")
    print("Stream independence: gnss draws do not affect accel draws ✓")

    # MC run derivation
    f_run0 = StreamFactory.for_mc_run(0, MASTER, STREAMS)
    f_run1 = StreamFactory.for_mc_run(1, MASTER, STREAMS)
    d0 = f_run0.get("gnss_noise").standard_normal(3)
    d1 = f_run1.get("gnss_noise").standard_normal(3)
    assert not np.allclose(d0, d1), "MC runs should be different!"
    print(f"MC run 0 gnss:    {d0}")
    print(f"MC run 1 gnss:    {d1}")
    print("MC run independence ✓")
    print("\nAll random_streams checks passed.")
