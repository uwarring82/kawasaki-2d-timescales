"""Tests for equilibration diagnostics (gate criterion (b) + prep-kernel check)."""

import numpy as np
import pytest

from kawasaki2d import equilibration as eq
from kawasaki2d.rng import spawn_rngs

pytest.importorskip("scipy")


def test_histogram_overlap_bounds():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 1000)
    b = rng.normal(0, 1, 1000)
    far = rng.normal(20, 1, 1000)
    assert eq.histogram_overlap(a, b) > 0.8
    assert eq.histogram_overlap(a, far) < 0.05


def test_two_independent_runs_indistinguishable():
    """Gate criterion (b): two independent equilibration runs agree."""
    ra, rb = spawn_rngs(31, 2)
    comp = eq.compare_equilibration(
        16, 3.0, 0, rng_a=ra, rng_b=rb,
        kernel_a="nonlocal", kernel_b="nonlocal", n_sweeps=3000,
    )
    assert comp.indistinguishable, comp.detail
    assert comp.overlap > 0.7


def test_prep_kernel_matches_local_baseline():
    """Preparation-kernel validation: non-local prep == local-Kawasaki equilibrium."""
    ra, rb = spawn_rngs(42, 2)
    comp = eq.compare_equilibration(
        16, 3.0, 0, rng_a=ra, rng_b=rb,
        kernel_a="nonlocal", kernel_b="local", n_sweeps=3000,
    )
    assert comp.indistinguishable, comp.detail
    # the two kernels must agree on the mean equilibrium energy
    assert abs(comp.mean_a - comp.mean_b) < 5 * np.hypot(comp.sem_a, comp.sem_b)


def test_energy_trace_saturates():
    (rng,) = spawn_rngs(5, 1)
    sweeps, e = eq.energy_trace(16, 10.0, 0, rng, kernel="nonlocal", n_sweeps=1500, sample_every=25)
    assert sweeps[0] == 0 and sweeps[-1] == 1500
    assert len(sweeps) == len(e)
    # late-time plateau: the last third has small spread relative to the full drop
    drop = abs(e[0] - e[-1]) + 1e-9
    late_spread = np.std(e[len(e) * 2 // 3:])
    assert late_spread < 0.5 * drop
