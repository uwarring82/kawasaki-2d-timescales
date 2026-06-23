"""Equilibration diagnostics for the initial-state convergence gate.

The equilibration gate (task card) requires, for each ``T_i``, evidence that the
prepared initial state has actually reached canonical equilibrium *before* the
quench — either (a) the equal-time observables saturate, or (b) two independent
equilibration runs give statistically indistinguishable energy distributions.
This module provides reusable tools for both, plus a preparation-kernel
validation (non-local prep vs local-Kawasaki baseline must agree on equal-time
observables).

These diagnostics matter most near ``T_c`` (critical slowing down): a too-short
preparation leaves residual memory of the random start. They are used by the
Milestone-2 driver and will gate the Milestone-4 ``T_i`` sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dynamics
from .lattice import init_lattice, total_energy

try:
    from scipy.stats import ks_2samp

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

_KERNELS = {"local": dynamics.run_kawasaki, "nonlocal": dynamics.run_nonlocal_exchange}


def energy_trace(
    n: int,
    T: float,
    magnetisation: int,
    rng: np.random.Generator,
    *,
    kernel: str = "nonlocal",
    n_sweeps: int = 2000,
    sample_every: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-spin energy vs sweep during equilibration (for a saturation plot).

    Returns ``(sweeps, energy_per_spin)``.
    """
    run = _KERNELS[kernel]
    lattice = init_lattice(n, magnetisation, rng=rng)
    sweeps = [0]
    energies = [total_energy(lattice) / lattice.size]
    done = 0
    while done < n_sweeps:
        block = min(sample_every, n_sweeps - done)
        run(lattice, T, block, rng)
        done += block
        sweeps.append(done)
        energies.append(total_energy(lattice) / lattice.size)
    return np.asarray(sweeps), np.asarray(energies)


def equilibrium_energy_samples(
    n: int,
    T: float,
    magnetisation: int,
    rng: np.random.Generator,
    *,
    kernel: str = "nonlocal",
    n_sweeps: int = 3000,
    sample_every: int = 20,
    burn_fraction: float = 0.5,
) -> np.ndarray:
    """Decorrelated per-spin energy samples from the equilibrium region.

    Runs ``n_sweeps`` of ``kernel`` and samples the energy every
    ``sample_every`` sweeps, discarding the first ``burn_fraction`` of samples
    as burn-in.
    """
    run = _KERNELS[kernel]
    lattice = init_lattice(n, magnetisation, rng=rng)
    n_samples = max(1, n_sweeps // sample_every)
    samples = np.empty(n_samples)
    for i in range(n_samples):
        run(lattice, T, sample_every, rng)
        samples[i] = total_energy(lattice) / lattice.size
    burn = int(burn_fraction * n_samples)
    return samples[burn:]


def histogram_overlap(a: np.ndarray, b: np.ndarray, *, bins: int | None = None) -> float:
    """Overlap coefficient of two samples' histograms (1 = identical, 0 = disjoint).

    Bins are shared across both samples; returns ``Σ min(p_a, p_b)`` over bins,
    where ``p`` are normalised (sum-to-one) histogram heights. Note: the overlap
    coefficient is biased downward at small sample sizes (finite-binning noise),
    so it is reported as a *descriptive* statistic; the formal same-distribution
    decision in :func:`compare_equilibration` rests on the two-sample KS test and
    mean agreement. ``bins`` defaults to ``max(10, √n)``.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if bins is None:
        bins = max(10, int(np.sqrt(min(len(a), len(b)))))
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi == lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges, density=False)
    pb, _ = np.histogram(b, bins=edges, density=False)
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    return float(np.minimum(pa, pb).sum())


@dataclass
class EquilibrationComparison:
    """Result of comparing two equilibration runs' equilibrium energy samples."""

    mean_a: float
    mean_b: float
    sem_a: float
    sem_b: float
    overlap: float
    ks_stat: float
    ks_pvalue: float
    kernel_a: str
    kernel_b: str
    indistinguishable: bool
    detail: str


def compare_equilibration(
    n: int,
    T: float,
    magnetisation: int,
    *,
    rng_a: np.random.Generator,
    rng_b: np.random.Generator,
    kernel_a: str = "nonlocal",
    kernel_b: str = "nonlocal",
    n_sweeps: int = 4000,
    sample_every: int = 10,
    burn_fraction: float = 0.5,
    ks_alpha: float = 0.05,
    mean_sigma: float = 4.0,
) -> EquilibrationComparison:
    """Compare two independent equilibration runs (gate criterion (b)).

    With ``kernel_a == kernel_b`` this is the two-independent-runs check. With
    ``kernel_a="nonlocal"``, ``kernel_b="local"`` it is the preparation-kernel
    validation (fast sampler vs local-Kawasaki baseline must agree on the
    equal-time energy distribution).

    The same-distribution decision is: the two-sample KS test is non-significant
    (``p >= ks_alpha``) **and** the means agree to within ``mean_sigma`` combined
    standard errors. The histogram overlap is reported descriptively.
    """
    sa = equilibrium_energy_samples(
        n, T, magnetisation, rng_a, kernel=kernel_a,
        n_sweeps=n_sweeps, sample_every=sample_every, burn_fraction=burn_fraction,
    )
    sb = equilibrium_energy_samples(
        n, T, magnetisation, rng_b, kernel=kernel_b,
        n_sweeps=n_sweeps, sample_every=sample_every, burn_fraction=burn_fraction,
    )
    overlap = histogram_overlap(sa, sb)
    if _HAVE_SCIPY:
        ks = ks_2samp(sa, sb)
        ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
    else:  # pragma: no cover
        ks_stat, ks_p = float("nan"), float("nan")
    sem_a = float(sa.std(ddof=1) / np.sqrt(len(sa)))
    sem_b = float(sb.std(ddof=1) / np.sqrt(len(sb)))
    combined_sem = float(np.hypot(sem_a, sem_b))
    mean_gap = abs(float(sa.mean()) - float(sb.mean()))
    mean_agree = combined_sem == 0 or mean_gap <= mean_sigma * combined_sem
    ks_pass = (not np.isfinite(ks_p)) or ks_p >= ks_alpha
    indistinguishable = bool(ks_pass and mean_agree)
    return EquilibrationComparison(
        mean_a=float(sa.mean()),
        mean_b=float(sb.mean()),
        sem_a=sem_a,
        sem_b=sem_b,
        overlap=overlap,
        ks_stat=ks_stat,
        ks_pvalue=ks_p,
        kernel_a=kernel_a,
        kernel_b=kernel_b,
        indistinguishable=indistinguishable,
        detail=(
            f"overlap={overlap:.3f}; KS p={ks_p:.3f} (>= {ks_alpha} ⇒ same); "
            f"mean gap={mean_gap:.4f} vs {mean_sigma}·SEM={mean_sigma * combined_sem:.4f}"
        ),
    )
