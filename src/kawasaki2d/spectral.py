"""Small-N exact-diagonalisation spectral tier (the spectral Mpemba probe).

For a lattice small enough to enumerate the fixed-``M`` sector (default ``4×4``,
``M=0`` ⇒ C(16,8)=12870 states), we build the *exact* local-Kawasaki Metropolis
transition matrix at the bath temperature ``T_f`` and diagonalise it. This is the
only tier at which a genuine **spectral Mpemba** verdict is awardable (task card
diagnosability note): the relaxation is governed by the overlap of the initial
distribution with the slowest relaxation mode (Lu–Raz / Klich et al.).

Conventions
-----------
The transition matrix ``P`` reproduces the simulation's kinetic kernel exactly:
each step proposes one of ``4N²`` (site, direction) pairs uniformly; an
opposite-spin proposal is a nearest-neighbour swap accepted with Metropolis
probability ``min(1, e^{-ΔE/T_f})``; equal-spin or rejected proposals are
self-loops. ``P`` is reversible w.r.t. the canonical distribution
``π(T_f) ∝ e^{-E/T_f}`` restricted to the sector.

The slowest mode is the right eigenvector ``v_2`` of ``P`` (second-largest
eigenvalue ``λ_2 < 1``). The Mpemba coefficient of a preparation at ``T_i`` is
the projection of its Boltzmann distribution onto ``v_2``:
``a_2(T_i) = Σ_σ π_{T_i}(σ) · v_2(σ)`` — the amplitude with which the initial
state excites the slow mode. A vanishing ``a_2(T_i)`` is a *strong* spectral
Mpemba point; ``|a_2(T_hot)| < |a_2(T_cold)|`` is a weak spectral Mpemba
inversion (the hotter preparation relaxes faster asymptotically).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

try:
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


def _bond_list(n: int) -> list[tuple[int, int]]:
    """Unique nearest-neighbour bonds (i, j) as flat site indices (periodic)."""
    bonds = []
    for r in range(n):
        for c in range(n):
            i = r * n + c
            bonds.append((i, ((r + 1) % n) * n + c))   # down
            bonds.append((i, r * n + (c + 1) % n))      # right
    return bonds


def _neighbour_dirs(n: int) -> np.ndarray:
    """For each site, the 4 neighbour site indices (up, down, left, right)."""
    nb = np.empty((n * n, 4), dtype=np.int64)
    for r in range(n):
        for c in range(n):
            i = r * n + c
            nb[i, 0] = ((r - 1) % n) * n + c
            nb[i, 1] = ((r + 1) % n) * n + c
            nb[i, 2] = r * n + (c - 1) % n
            nb[i, 3] = r * n + (c + 1) % n
    return nb


@dataclass
class SpectralSector:
    """Enumerated fixed-M sector: states (bitmasks), energies, index map."""

    n: int
    magnetisation: int
    states: np.ndarray          # (S,) uint32 bitmasks (bit p set ⇒ site p is +1)
    energies: np.ndarray        # (S,) int energies
    index: dict                 # bitmask -> row index


def enumerate_sector(n: int, magnetisation: int = 0) -> SpectralSector:
    """Enumerate every configuration in the fixed-``M`` sector of an ``n×n`` lattice."""
    nsites = n * n
    n_up = (nsites + magnetisation) // 2
    if (nsites + magnetisation) % 2 != 0 or not (0 <= n_up <= nsites):
        raise ValueError(f"magnetisation {magnetisation} unreachable on {n}x{n}")
    bonds = _bond_list(n)
    states = []
    energies = []
    index = {}
    for combo in combinations(range(nsites), n_up):
        mask = 0
        for p in combo:
            mask |= (1 << p)
        # energy = -sum s_i s_j ; s_i s_j = +1 if bits equal else -1
        diff = 0
        for i, j in bonds:
            bi = (mask >> i) & 1
            bj = (mask >> j) & 1
            diff += (bi ^ bj)            # 1 if spins differ (broken bond)
        E = 2 * diff - len(bonds)        # = -(#equal - #diff)
        index[mask] = len(states)
        states.append(mask)
        energies.append(E)
    return SpectralSector(n=n, magnetisation=magnetisation,
                          states=np.array(states, dtype=np.uint32),
                          energies=np.array(energies, dtype=np.int64), index=index)


def boltzmann(energies: np.ndarray, T: float) -> np.ndarray:
    """Canonical distribution ``π ∝ e^{-E/T}`` over the sector (normalised)."""
    w = np.exp(-(energies - energies.min()) / T)
    return w / w.sum()


def build_transition_matrix(sector: SpectralSector, T: float):
    """Exact local-Kawasaki Metropolis transition matrix ``P`` at temperature ``T``.

    Returns a CSR matrix. Each of the ``4N²`` (site, dir) proposals has weight
    ``1/(4N²)``; an opposite-spin proposal swaps and is accepted by Metropolis.
    Rows sum to 1 (self-loops absorb equal-spin / rejected proposals).
    """
    if not _HAVE_SCIPY:
        raise RuntimeError("spectral tier requires scipy")
    n = sector.n
    nsites = n * n
    nb = _neighbour_dirs(n)
    norm = 1.0 / (4 * nsites)
    S = sector.states.shape[0]
    E = sector.energies
    idx = sector.index

    rows, cols, vals = [], [], []
    offdiag_rowsum = np.zeros(S)
    for r in range(S):
        mask = int(sector.states[r])
        Er = E[r]
        for i in range(nsites):
            bi = (mask >> i) & 1
            for d in range(4):
                j = int(nb[i, d])
                bj = (mask >> j) & 1
                if bi == bj:
                    continue  # equal spins: no-op (self-loop), skip
                # swap bits i and j
                m2 = mask ^ ((1 << i) | (1 << j))
                c = idx[m2]
                dE = E[c] - Er
                acc = 1.0 if dE <= 0 else float(np.exp(-dE / T))
                p = norm * acc
                rows.append(r); cols.append(c); vals.append(p)
                offdiag_rowsum[r] += p
    P = sp.coo_matrix((vals, (rows, cols)), shape=(S, S)).tocsr()
    P = P + sp.diags(1.0 - offdiag_rowsum)
    return P


@dataclass
class Spectrum:
    """Top-``k`` reversible spectrum of ``P`` (right eigenvectors of ``P``)."""

    eigenvalues: np.ndarray     # (k,) descending; eigenvalues[0] = 1 (stationary)
    right_vectors: np.ndarray   # (S, k) right eigenvectors v_k = Π^{-1/2} φ_k


def spectrum(P, pi: np.ndarray, k: int = 30) -> Spectrum:
    """Top-``k`` eigenpairs of the reversible ``P`` via the symmetric form.

    ``P`` is symmetrised to ``S = Π^{1/2} P Π^{-1/2}`` (entries bounded by detailed
    balance); ``eigsh`` returns the algebraically-largest eigenvalues. Right
    eigenvectors of ``P`` are ``v_k = Π^{-1/2} φ_k``.
    """
    sqrt_pi = np.sqrt(pi)
    Ssym = (sp.diags(sqrt_pi) @ P @ sp.diags(1.0 / sqrt_pi)).tocsr()
    Ssym = 0.5 * (Ssym + Ssym.T)  # clean round-off asymmetry
    vals, vecs = eigsh(Ssym, k=min(k, P.shape[0] - 1), which="LA")
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    V = (vecs[:, order]) / sqrt_pi[:, None]
    return Spectrum(eigenvalues=vals, right_vectors=V)


@dataclass
class SlowMode:
    index: int                  # column in the spectrum (0 = stationary)
    eigenvalue: float
    v_slow: np.ndarray          # right eigenvector of the slowest *excited* mode
    relaxation_time_steps: float
    eigenvalues: np.ndarray     # the full top-k spectrum (for reporting)


def slowest_excited_mode(spec: Spectrum, energies: np.ndarray, *,
                         T_probe: float, overlap_tol: float = 1e-6) -> SlowMode:
    """Slowest mode with non-zero overlap with a symmetric Boltzmann initial state.

    Antisymmetric (k≠0 / spin-flip-odd) modes are orthogonal to every
    lattice-symmetric initial distribution and so are never excited; the
    Mpemba-relevant slow mode is the slowest one a Boltzmann state actually
    projects onto. ``T_probe`` selects the probe distribution (any single
    temperature suffices to detect symmetric modes).
    """
    pi_probe = boltzmann(energies, T_probe)
    overlaps = np.abs(pi_probe @ spec.right_vectors)  # |a_k| for each mode
    for k in range(1, spec.eigenvalues.shape[0]):     # skip stationary (k=0)
        if overlaps[k] > overlap_tol:
            lam = float(spec.eigenvalues[k])
            tau = -1.0 / np.log(lam) if 0 < lam < 1 else float("inf")
            return SlowMode(index=k, eigenvalue=lam, v_slow=spec.right_vectors[:, k],
                            relaxation_time_steps=tau, eigenvalues=spec.eigenvalues)
    raise RuntimeError("no excited mode found within the computed spectrum; increase k")


def mpemba_coefficient(v_slow: np.ndarray, energies: np.ndarray, T_i) -> np.ndarray:
    """``a_2(T_i) = Σ_σ π_{T_i}(σ) v_2(σ)`` for one or many initial temperatures."""
    T_i = np.atleast_1d(np.asarray(T_i, float))
    out = np.empty(T_i.shape[0])
    for k, T in enumerate(T_i):
        out[k] = float(np.dot(boltzmann(energies, T), v_slow))
    return out
