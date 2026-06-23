"""Validation gate: bitwise reproducibility under fixed seeds.

Same seed ⇒ identical trajectory; different seed ⇒ (almost surely) different.
Because the random stream is drawn in NumPy and passed into the inner loop, the
Numba and pure-Python kernels also agree bitwise — so this gate holds regardless
of whether Numba is installed.
"""

import numpy as np

from kawasaki2d import dynamics, lattice
from kawasaki2d.rng import make_rng, spawn_rngs


def _run(seed, n=16, sweeps=40, T=1.2):
    rng = make_rng(seed)
    s = lattice.init_lattice(n, 0, rng=rng)
    dynamics.run_kawasaki(s, T, sweeps, rng)
    return s


def test_same_seed_identical():
    assert np.array_equal(_run(123), _run(123))


def test_different_seed_differs():
    assert not np.array_equal(_run(123), _run(124))


def test_init_is_deterministic():
    a = lattice.init_lattice(32, 0, rng=make_rng(5))
    b = lattice.init_lattice(32, 0, rng=make_rng(5))
    assert np.array_equal(a, b)


def test_spawned_streams_are_independent_and_reproducible():
    rngs_a = spawn_rngs(2024, 4)
    rngs_b = spawn_rngs(2024, 4)
    draws_a = [r.random(10) for r in rngs_a]
    draws_b = [r.random(10) for r in rngs_b]
    # reproducible across spawn calls
    for da, db in zip(draws_a, draws_b):
        assert np.array_equal(da, db)
    # distinct sub-streams differ from each other
    assert not np.array_equal(draws_a[0], draws_a[1])
