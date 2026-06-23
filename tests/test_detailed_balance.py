"""Validation gate: ergodicity + detailed balance + correct Metropolis acceptance.

On a 4×4 lattice the fixed-M=0 sector (C(16,8) = 12870 configurations) is small
enough to enumerate exactly. We compare the energy distribution sampled by a long
local-Kawasaki Monte Carlo run against the exact canonical distribution
``P(E) ∝ g(E) e^{-E/T}``. Agreement requires (a) the proposal to connect the whole
sector (ergodicity), (b) detailed balance, and (c) the correct acceptance rule —
the three things this gate certifies together. The non-local preparation kernel
is checked to sample the *same* equilibrium.
"""

import itertools
import math

import numpy as np
import pytest

from kawasaki2d import dynamics, lattice
from kawasaki2d.rng import make_rng


def _exact_energy_distribution(n: int, n_up: int, T: float):
    """Exact canonical energy distribution over the fixed-M sector."""
    base = np.full(n * n, -1, dtype=np.int8)
    degeneracy: dict[int, int] = {}
    for up in itertools.combinations(range(n * n), n_up):
        flat = base.copy()
        flat[list(up)] = 1
        E = lattice.total_energy(flat.reshape(n, n))
        degeneracy[E] = degeneracy.get(E, 0) + 1
    levels = np.array(sorted(degeneracy))
    g = np.array([degeneracy[E] for E in levels], dtype=float)
    w = g * np.exp(-levels / T)
    w /= w.sum()
    return levels, w


def _mc_energy_distribution(kernel, levels, T, *, burn, gap, n_samp, seed):
    rng = make_rng(seed)
    s = lattice.init_lattice(4, 0, rng=rng)
    kernel(s, T, burn, rng)
    counts = {int(E): 0 for E in levels}
    energies = np.empty(n_samp)
    for i in range(n_samp):
        kernel(s, T, gap, rng)
        E = lattice.total_energy(s)
        counts[E] += 1
        energies[i] = E
    probs = np.array([counts[int(E)] / n_samp for E in levels])
    return probs, energies


def test_local_kawasaki_samples_canonical_distribution():
    T = 2.5
    levels, p_exact = _exact_energy_distribution(4, 8, T)
    p_mc, energies = _mc_energy_distribution(
        dynamics.run_kawasaki, levels, T, burn=300, gap=12, n_samp=3000, seed=2024
    )
    # (a) every energy level's probability matches within tolerance
    assert np.max(np.abs(p_mc - p_exact)) < 0.02
    # (b) mean energy matches within a few standard errors
    e_exact = float((levels * p_exact).sum())
    sem = energies.std(ddof=1) / math.sqrt(len(energies))
    assert abs(energies.mean() - e_exact) < 5 * sem


def test_nonlocal_prep_samples_same_equilibrium():
    """The non-local preparation sampler must reproduce the same equilibrium."""
    T = 2.5
    levels, p_exact = _exact_energy_distribution(4, 8, T)
    p_mc, energies = _mc_energy_distribution(
        dynamics.run_nonlocal_exchange, levels, T, burn=200, gap=6, n_samp=2500, seed=77
    )
    assert np.max(np.abs(p_mc - p_exact)) < 0.025
    e_exact = float((levels * p_exact).sum())
    sem = energies.std(ddof=1) / math.sqrt(len(energies))
    assert abs(energies.mean() - e_exact) < 5 * sem


def test_low_T_drives_energy_down():
    """Sanity: a deep quench lowers the energy toward the phase-separated state."""
    rng = make_rng(11)
    s = lattice.init_lattice(16, 0, rng=rng)
    e0 = lattice.total_energy(s)
    dynamics.run_kawasaki(s, 0.6 * 2.269185, 300, rng)
    assert lattice.total_energy(s) < e0
