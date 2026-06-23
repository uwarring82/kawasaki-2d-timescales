"""Validation gate: energy and local ΔE correctness.

Checks (task card, validation gate):
* local ΔE against brute-force total-energy differences on small lattices;
* total_energy against an independent naive double-loop computation.
"""

import numpy as np
import pytest

from kawasaki2d import lattice
from kawasaki2d.rng import make_rng


def naive_total_energy(s: np.ndarray) -> int:
    """Independent O(N²) reference: sum -s_i s_j over forward NN bonds (periodic)."""
    n = s.shape[0]
    e = 0
    for r in range(n):
        for c in range(n):
            e -= int(s[r, c]) * int(s[(r + 1) % n, c])  # down bond
            e -= int(s[r, c]) * int(s[r, (c + 1) % n])  # right bond
    return e


@pytest.mark.parametrize("n", [4, 6, 8, 16])
def test_total_energy_matches_naive(n):
    rng = make_rng(100 + n)
    for _ in range(20):
        s = lattice.init_lattice(n, 0, rng=rng)
        assert lattice.total_energy(s) == naive_total_energy(s)


@pytest.mark.parametrize("n", [4, 6, 8, 16])
def test_delta_E_matches_brute_force(n):
    """ΔE(swap) must equal total_energy(after) − total_energy(before) exactly."""
    rng = make_rng(7 * n)
    s = lattice.init_lattice(n, 0, rng=rng)
    for _ in range(3000):
        r, c = int(rng.integers(n)), int(rng.integers(n))
        d = int(rng.integers(4))
        dE_local = lattice.delta_E_for_swap(s, r, c, d)
        dr, dc = lattice.NEIGHBOUR_OFFSETS[d]
        r1, c1 = (r + dr) % n, (c + dc) % n
        e0 = lattice.total_energy(s)
        s[r, c], s[r1, c1] = s[r1, c1], s[r, c]
        e1 = lattice.total_energy(s)
        s[r, c], s[r1, c1] = s[r1, c1], s[r, c]  # restore
        assert dE_local == (e1 - e0)


def test_equal_spin_swap_is_zero():
    rng = make_rng(1)
    s = lattice.init_lattice(8, 0, rng=rng)
    # find an equal-spin neighbour pair and confirm ΔE == 0
    n = 8
    found = False
    for r in range(n):
        for c in range(n):
            for d in range(4):
                dr, dc = lattice.NEIGHBOUR_OFFSETS[d]
                if s[r, c] == s[(r + dr) % n, (c + dc) % n]:
                    assert lattice.delta_E_for_swap(s, r, c, d) == 0
                    found = True
    assert found


def test_energy_bounds():
    """Ground state (fully aligned) has E = -2N²; energy is integer-valued."""
    n = 8
    s = np.ones((n, n), dtype=np.int8)
    assert lattice.total_energy(s) == -2 * n * n
