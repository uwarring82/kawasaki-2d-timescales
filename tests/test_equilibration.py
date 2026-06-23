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


def test_estimate_equilibrium_energy_temperature_ordering():
    """E_inf(T) per spin: strongly negative at low T, near zero at high T."""
    (ra,) = spawn_rngs(13, 1)
    (rb,) = spawn_rngs(14, 1)
    lowT = eq.estimate_equilibrium_energy(8, 1.0, 0, ra, sweeps_burn=1500, sample_every=30, n_samples=25)
    hiT = eq.estimate_equilibrium_energy(8, 8.0, 0, rb, sweeps_burn=1500, sample_every=30, n_samples=25)
    assert lowT.mean < -1.4          # nearly phase-separated
    assert hiT.mean > lowT.mean      # disordered sits higher
    assert hiT.mean < 0.0            # still below 0 at finite T
    assert lowT.sd >= 0


def test_calibrate_prep_budget_passes_at_high_T():
    """The independent-chains gate converges at an easy (high-T) point."""
    res = eq.calibrate_prep_budget(
        8, 6.0, 0, candidates=[100, 200, 400], ref_burn=1500, base_seed=3,
        kernel="nonlocal", n_test=16, n_ref=20, safety_tau_mult=10,
    )
    assert res["gate_passed"]
    assert res["gated_budget"] in (100, 200, 400)
    assert res["tau_E_sweeps"] >= 1.0


def test_autocorrelation_time_white_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20000)
    tau = eq.integrated_autocorrelation_time(x)
    assert 0.8 < tau < 1.5            # white noise -> tau ~ 1


def test_autocorrelation_time_ar1():
    """AR(1) with coefficient phi has tau_int = (1+phi)/(1-phi)."""
    rng = np.random.default_rng(1)
    phi = 0.8
    n = 60000
    x = np.empty(n)
    x[0] = 0.0
    noise = rng.normal(size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]
    tau = eq.integrated_autocorrelation_time(x)
    expected = (1 + phi) / (1 - phi)   # = 9
    assert abs(tau - expected) / expected < 0.25


def test_energy_trace_saturates():
    (rng,) = spawn_rngs(5, 1)
    sweeps, e = eq.energy_trace(16, 10.0, 0, rng, kernel="nonlocal", n_sweeps=1500, sample_every=25)
    assert sweeps[0] == 0 and sweeps[-1] == 1500
    assert len(sweeps) == len(e)
    # late-time plateau: the last third has small spread relative to the full drop
    drop = abs(e[0] - e[-1]) + 1e-9
    late_spread = np.std(e[len(e) * 2 // 3:])
    assert late_spread < 0.5 * drop
