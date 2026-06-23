"""Tests for observables: correlation, structure factor, lengths, clusters."""

import numpy as np
import pytest

from kawasaki2d import dynamics, lattice, observables as obs
from kawasaki2d.rng import make_rng


def test_correlation_C0_is_one_at_M0():
    rng = make_rng(1)
    s = lattice.init_lattice(32, 0, rng=rng)
    r, c = obs.correlation_function(s)
    assert abs(c[0] - 1.0) < 1e-9


def test_broken_bond_density_consistent_with_energy():
    """E = -2N² + 2·N_unsat ⇒ broken_bond_density = (E + 2N²) / (2·2N²)."""
    rng = make_rng(2)
    s = lattice.init_lattice(16, 0, rng=rng)
    dynamics.run_kawasaki(s, 1.0, 30, rng)
    n2 = s.size
    E = lattice.total_energy(s)
    n_unsat = (E + 2 * n2) / 2
    expected = n_unsat / (2 * n2)
    assert abs(obs.broken_bond_density(s) - expected) < 1e-12


def test_aligned_lattice_has_zero_broken_bonds():
    s = np.ones((8, 8), dtype=np.int8)
    assert obs.broken_bond_density(s) == 0.0


def test_cluster_areas_sum_to_site_count():
    rng = make_rng(3)
    s = lattice.init_lattice(32, 0, rng=rng)
    dynamics.run_kawasaki(s, 1.36, 100, rng)
    up = obs.cluster_areas(s, 1)
    down = obs.cluster_areas(s, -1)
    assert up.sum() == int((s == 1).sum())
    assert down.sum() == int((s == -1).sum())


def test_periodic_cluster_wraps():
    """A stripe touching both boundaries is one cluster under periodic BCs."""
    s = -np.ones((8, 8), dtype=np.int8)
    s[:, 0] = 1
    s[:, 7] = 1  # left and right columns connect through the periodic wrap
    areas = obs.cluster_areas(s, 1)
    assert len(areas) == 1
    assert areas[0] == 16


def test_coarsening_increases_length_scales():
    rng = make_rng(4)
    s = lattice.init_lattice(64, 0, rng=rng)
    Tf = 0.6 * 2.269185
    dynamics.run_kawasaki(s, Tf, 30, rng)
    L_S_early = obs.length_from_structure(s)
    k_early = obs.characteristic_k(s)
    dynamics.run_kawasaki(s, Tf, 600, rng)
    L_S_late = obs.length_from_structure(s)
    k_late = obs.characteristic_k(s)
    assert L_S_late > L_S_early       # domains grow
    assert k_late < k_early           # characteristic wavevector shrinks


def test_cluster_number_density_decreases_with_coarsening():
    rng = make_rng(5)
    s = lattice.init_lattice(64, 0, rng=rng)
    Tf = 0.6 * 2.269185
    dynamics.run_kawasaki(s, Tf, 30, rng)
    early = obs.cluster_number_density(s)
    dynamics.run_kawasaki(s, Tf, 600, rng)
    late = obs.cluster_number_density(s)
    assert late < early
