"""Tests for the exact-diagonalisation spectral tier."""

import numpy as np
import pytest

pytest.importorskip("scipy")
import scipy.sparse as sp  # noqa: E402

from kawasaki2d import spectral, lattice  # noqa: E402


def test_enumerate_sector_4x4_M0():
    sec = spectral.enumerate_sector(4, 0)
    from math import comb
    assert len(sec.states) == comb(16, 8) == 12870
    # energies match the lattice reference for a few states
    for r in (0, 100, 5000, 12869):
        mask = int(sec.states[r])
        flat = np.array([(mask >> p) & 1 for p in range(16)], dtype=np.int8) * 2 - 1
        assert sec.energies[r] == lattice.total_energy(flat.reshape(4, 4))


def _sector_3x3():
    return spectral.enumerate_sector(3, 1)  # 126 states, non-degenerate periodic


def test_transition_matrix_is_stochastic_and_reversible():
    sec = _sector_3x3()
    T = 1.5
    P = spectral.build_transition_matrix(sec, T)
    rs = np.asarray(P.sum(1)).ravel()
    assert np.allclose(rs, 1.0)                       # rows sum to 1
    pi = spectral.boltzmann(sec.energies, T)
    # stationary: pi P = pi
    assert np.max(np.abs(pi @ P - pi)) < 1e-12
    # detailed balance: pi_i P_ij = pi_j P_ji
    Pd = P.toarray()
    lhs = pi[:, None] * Pd
    assert np.max(np.abs(lhs - lhs.T)) < 1e-12


def test_spectrum_top_eigenvalue_is_one():
    sec = _sector_3x3()
    pi = spectral.boltzmann(sec.energies, 1.5)
    spec = spectral.spectrum(spectral.build_transition_matrix(sec, 1.5), pi, k=10)
    assert abs(spec.eigenvalues[0] - 1.0) < 1e-9
    assert np.all(np.diff(spec.eigenvalues) <= 1e-12)   # descending
    assert spec.eigenvalues[1] < 1.0                    # a genuine gap


def test_slowest_excited_mode_has_nonzero_overlap():
    sec = _sector_3x3()
    pi = spectral.boltzmann(sec.energies, 1.5)
    spec = spectral.spectrum(spectral.build_transition_matrix(sec, 1.5), pi, k=30)
    sm = spectral.slowest_excited_mode(spec, sec.energies, T_probe=3.0)
    assert sm.index >= 1 and 0 < sm.eigenvalue < 1
    a = spectral.mpemba_coefficient(sm.v_slow, sec.energies, [3.0])
    assert abs(a[0]) > 1e-6
