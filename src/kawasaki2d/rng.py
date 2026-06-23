"""Random-number generation and seeding utilities.

Reproducibility requirement (task card, FAIR-Reusability gate): every run must
be regenerable from a recorded seed, and the RNG library + version is captured
in the manifest. We standardise on NumPy's ``Generator`` backed by the PCG64
bit generator, seeded through ``SeedSequence`` so that a single integer seed
deterministically yields the full stream and independent sub-streams.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BIT_GENERATOR = "PCG64"


def make_rng(seed: int) -> np.random.Generator:
    """Return a deterministic ``Generator`` for an integer ``seed``.

    The same ``seed`` always yields the same stream on the same NumPy version,
    which is what the bitwise-reproducibility gate checks.
    """
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed))))


def spawn_rngs(seed: int, n: int) -> list[np.random.Generator]:
    """Return ``n`` independent generators derived from one base ``seed``.

    Used to give each ensemble realisation a statistically independent but
    fully reproducible stream. ``SeedSequence.spawn`` guarantees the child
    sequences are (with overwhelming probability) non-overlapping.
    """
    ss = np.random.SeedSequence(int(seed))
    return [np.random.Generator(np.random.PCG64(child)) for child in ss.spawn(int(n))]


@dataclass(frozen=True)
class RngFingerprint:
    """Identifies the RNG implementation for the provenance manifest."""

    library: str
    library_version: str
    bit_generator: str

    def as_dict(self) -> dict[str, str]:
        return {
            "library": self.library,
            "library_version": self.library_version,
            "bit_generator": self.bit_generator,
        }


def rng_fingerprint() -> RngFingerprint:
    """Capture the RNG identity for the manifest."""
    return RngFingerprint(
        library="numpy",
        library_version=np.__version__,
        bit_generator=BIT_GENERATOR,
    )
