"""Validation gate: exact conservation of M under the dynamics.

Magnetisation must be invariant after every block of proposed moves, for both
the local (kinetic) and non-local (preparation) kernels, at any temperature and
acceptance rate.
"""

import numpy as np
import pytest

from kawasaki2d import dynamics, lattice
from kawasaki2d.rng import make_rng


@pytest.mark.parametrize("n,M", [(8, 0), (8, 4), (16, 0), (16, -8)])
@pytest.mark.parametrize("T", [0.5, 1.5, 5.0])
def test_local_kawasaki_conserves_M(n, M, T):
    rng = make_rng(hash((n, M)) % 2**31)
    s = lattice.init_lattice(n, M, rng=rng)
    assert lattice.magnetisation(s) == M
    for _ in range(20):  # repeated blocks; M must never drift
        dynamics.run_kawasaki(s, T, 5, rng)
        assert lattice.magnetisation(s) == M


@pytest.mark.parametrize("n,M", [(8, 0), (8, 6), (12, -4)])
def test_nonlocal_exchange_conserves_M(n, M):
    rng = make_rng(999 + n + M)
    s = lattice.init_lattice(n, M, rng=rng)
    for _ in range(10):
        dynamics.run_nonlocal_exchange(s, 2.5, 2, rng)
        assert lattice.magnetisation(s) == M


def test_spins_stay_in_pm1():
    rng = make_rng(3)
    s = lattice.init_lattice(16, 0, rng=rng)
    dynamics.run_kawasaki(s, 1.0, 50, rng)
    assert set(np.unique(s)).issubset({-1, 1})
