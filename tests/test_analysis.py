"""Tests for the pre-registered analysis machinery."""

import numpy as np
import pytest

from kawasaki2d import analysis
from kawasaki2d.rng import make_rng

pytest.importorskip("scipy")


def test_ensemble_stats_mean_and_ci():
    rng = make_rng(0)
    samples = rng.normal(5.0, 2.0, size=(500, 3))
    st = analysis.ensemble_stats(samples)
    assert np.allclose(st.mean, 5.0, atol=0.3)
    assert st.n == 500
    assert np.all(st.ci_low < st.mean) and np.all(st.mean < st.ci_high)


def test_benjamini_hochberg_basic():
    p = np.array([0.001, 0.002, 0.6, 0.7, 0.8])
    res = analysis.benjamini_hochberg(p, alpha=0.05)
    assert res.rejected[0] and res.rejected[1]
    assert not res.rejected[2:].any()


def test_benjamini_hochberg_none_significant():
    p = np.array([0.4, 0.5, 0.9])
    res = analysis.benjamini_hochberg(p, alpha=0.05)
    assert not res.rejected.any()


def test_difference_bootstrap_detects_separation():
    rng = make_rng(1)
    times = np.array([10, 100])
    hot = rng.normal(10.0, 1.0, size=(200, 2))   # hot clearly larger
    cold = rng.normal(7.0, 1.0, size=(200, 2))
    dci = analysis.difference_bootstrap(hot, cold, times, rng, n_boot=1000)
    assert np.all(dci.excludes_zero)
    assert np.all(dci.sign == 1)


def test_crossing_test_finds_sign_change():
    # construct ensembles where hot<cold early, hot>cold late
    rng = make_rng(2)
    times = np.array([1, 2, 3, 4])
    hot = np.vstack([rng.normal(m, 0.2, size=300) for m in (5, 6, 8, 9)]).T
    cold = np.vstack([rng.normal(m, 0.2, size=300) for m in (7, 7, 7, 7)]).T
    dci = analysis.difference_bootstrap(hot, cold, times, rng, n_boot=1000)
    verdict = analysis.crossing_test(dci)
    assert verdict.crossed
    assert verdict.sign_before == -1 and verdict.sign_after == 1


def test_fit_offset_growth_recovers_parameters():
    rng = make_rng(3)
    t = np.linspace(1, 1000, 200)
    R0_true, lam_true = 2.5, 0.8
    L = analysis.growth_model(t, R0_true, lam_true) + rng.normal(0, 0.02, size=t.size)
    fit = analysis.fit_offset_growth(t, L)
    assert abs(fit.R0 - R0_true) < 0.2
    assert abs(fit.lam - lam_true) < 0.1


def test_effective_exponent_of_cube_root():
    t = np.geomspace(1, 1e4, 100)
    L = np.cbrt(t)  # pure t^(1/3)
    tm, alpha = analysis.effective_exponent(t, L)
    assert np.allclose(np.median(alpha), 1.0 / 3.0, atol=0.02)


def test_required_ensemble_size_scales_with_inverse_signal_squared():
    n_small_signal = analysis.required_ensemble_size(1.0, 1.0, signal=0.5)
    n_large_signal = analysis.required_ensemble_size(1.0, 1.0, signal=1.0)
    # smaller signal needs more samples (∝ 1/δ²)
    assert n_small_signal > n_large_signal
    assert abs(n_small_signal - 4 * n_large_signal) <= 2
