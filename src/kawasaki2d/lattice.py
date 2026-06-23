"""Lattice construction and global quantities for the 2D Ising model.

Spins are stored as an ``(N, N)`` array of ``int8`` with values in ``{+1, -1}``.
Boundary conditions are periodic (a torus). ``J = k_B = 1``.

This module holds the *clear, slow, reference* implementations of the energy
and of the single-exchange energy change. The performance-critical inner loop
lives in :mod:`kawasaki2d.dynamics`; the reference functions here are what the
validation tests compare against (``tests/test_energy.py``).
"""

from __future__ import annotations

import numpy as np

SPIN_DTYPE = np.int8


def init_lattice(n: int, magnetisation: int = 0, *, rng: np.random.Generator) -> np.ndarray:
    """Return a random ``(n, n)`` spin configuration with exact magnetisation.

    Exactly ``n_up = (n*n + magnetisation) // 2`` sites are set to ``+1`` and the
    rest to ``-1``, with the up-spins placed uniformly at random. This fixes the
    conserved sector ``M = magnetisation`` exactly.

    Raises ``ValueError`` if ``magnetisation`` is not reachable on an ``n × n``
    lattice (wrong parity, or out of range).
    """
    if n <= 0:
        raise ValueError(f"lattice size must be positive, got {n}")
    nsites = n * n
    if abs(magnetisation) > nsites or (nsites + magnetisation) % 2 != 0:
        raise ValueError(
            f"magnetisation {magnetisation} is not reachable on a {n}x{n} lattice "
            f"({nsites} sites)"
        )
    n_up = (nsites + magnetisation) // 2
    flat = np.full(nsites, -1, dtype=SPIN_DTYPE)
    flat[:n_up] = 1
    rng.shuffle(flat)
    return flat.reshape(n, n)


def magnetisation(lattice: np.ndarray) -> int:
    """Total magnetisation ``M = Σ s_i`` (exact integer)."""
    return int(lattice.sum())


def total_energy(lattice: np.ndarray) -> int:
    """Total energy ``H = -J Σ_⟨ij⟩ s_i s_j`` counting each bond once.

    Each bond is counted exactly once by summing the products with the
    *forward* (down and right) neighbours under periodic wrap. Returns an exact
    integer (energies are integers when ``J = 1``).
    """
    s = lattice.astype(np.int64)
    right = np.roll(s, -1, axis=1)
    down = np.roll(s, -1, axis=0)
    return int(-(s * right).sum() - (s * down).sum())


def neighbour_sum(lattice: np.ndarray) -> np.ndarray:
    """Per-site sum of the four nearest-neighbour spins (periodic)."""
    s = lattice.astype(np.int64)
    return (
        np.roll(s, 1, axis=0)
        + np.roll(s, -1, axis=0)
        + np.roll(s, 1, axis=1)
        + np.roll(s, -1, axis=1)
    )


# Neighbour direction offsets (row, col): up, down, left, right.
NEIGHBOUR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def delta_E_for_swap(lattice: np.ndarray, r0: int, c0: int, direction: int) -> int:
    """Reference ``ΔE`` for exchanging site ``(r0, c0)`` with a neighbour.

    ``direction`` indexes :data:`NEIGHBOUR_OFFSETS`. Implements the closed-form
    ``ΔE = J (s_i - s_j)(a - b)`` from ``docs/physics.md``, where ``a``, ``b``
    are the neighbour-sums of the two sites excluding each other. Returns ``0``
    when the two spins are equal (the swap is a no-op).

    This is the slow, transparent reference; the test suite checks it against a
    full recomputation of :func:`total_energy` before and after the swap.
    """
    n = lattice.shape[0]
    dr, dc = NEIGHBOUR_OFFSETS[direction]
    r1, c1 = (r0 + dr) % n, (c0 + dc) % n
    si = int(lattice[r0, c0])
    sj = int(lattice[r1, c1])
    if si == sj:
        return 0
    # Neighbour sums including the partner, then remove the partner's spin.
    nsum_i = (
        int(lattice[(r0 - 1) % n, c0])
        + int(lattice[(r0 + 1) % n, c0])
        + int(lattice[r0, (c0 - 1) % n])
        + int(lattice[r0, (c0 + 1) % n])
    )
    nsum_j = (
        int(lattice[(r1 - 1) % n, c1])
        + int(lattice[(r1 + 1) % n, c1])
        + int(lattice[r1, (c1 - 1) % n])
        + int(lattice[r1, (c1 + 1) % n])
    )
    a = nsum_i - sj  # neighbours of i excluding j
    b = nsum_j - si  # neighbours of j excluding i
    return (si - sj) * (a - b)
