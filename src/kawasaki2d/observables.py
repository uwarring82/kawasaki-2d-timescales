"""Observables for coarsening: energy, correlations, structure factor, length
scales, interface density, and cluster-area statistics.

All functions take an ``(N, N)`` int8 spin array (periodic BCs). Length scales
are returned in lattice units. Definitions follow ``docs/physics.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lattice import total_energy

try:
    from scipy.ndimage import label as _ndi_label

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# Energy / magnetisation                                                       #
# --------------------------------------------------------------------------- #


def energy(lattice: np.ndarray) -> int:
    """Total energy (see :func:`kawasaki2d.lattice.total_energy`)."""
    return total_energy(lattice)


def energy_per_spin(lattice: np.ndarray) -> float:
    return total_energy(lattice) / lattice.size


def broken_bond_density(lattice: np.ndarray) -> float:
    """Fraction of nearest-neighbour bonds that are unsatisfied (anti-aligned).

    Note: this is an affine function of the energy (``E = -J(N_sat - N_unsat)``)
    and is therefore *not* energy-independent. Use :func:`cluster_number_density`
    for the energy-independent morphology check required by the Mpemba gate.
    """
    s = lattice.astype(np.int64)
    nbonds = 2 * lattice.size  # 2N^2 bonds on the torus
    aligned = (s * np.roll(s, -1, axis=1) == 1).sum() + (s * np.roll(s, -1, axis=0) == 1).sum()
    return float(nbonds - aligned) / nbonds


# --------------------------------------------------------------------------- #
# Radial averaging helper                                                      #
# --------------------------------------------------------------------------- #


def _radial_average(field_centered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially average a 2D field whose origin is at the array centre.

    Returns ``(radii, values)`` where ``radii`` are integer pixel distances
    ``0, 1, 2, ...`` and ``values`` the mean of the field over each annulus.
    """
    n = field_centered.shape[0]
    cy = cx = n // 2
    y, x = np.indices(field_centered.shape)
    r = np.hypot(y - cy, x - cx)
    r_int = r.astype(np.int64)
    rmax = n // 2
    radii = np.arange(0, rmax + 1)
    sums = np.bincount(r_int.ravel(), weights=field_centered.ravel(), minlength=rmax + 1)
    counts = np.bincount(r_int.ravel(), minlength=rmax + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)
    return radii[: rmax + 1], means[: rmax + 1]


# --------------------------------------------------------------------------- #
# Correlation function and structure factor                                    #
# --------------------------------------------------------------------------- #


def correlation_function(lattice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged connected correlation ``C(r) = ⟨s_i s_{i+r}⟩ − ⟨s⟩²``.

    Computed via the Wiener–Khinchin theorem (FFT autocorrelation), then
    radially averaged. ``C(0) = 1 − ⟨s⟩²`` (``= 1`` at ``M = 0``).
    """
    s = lattice.astype(np.float64)
    mean = s.mean()
    f = np.fft.fft2(s - mean)
    autocorr = np.fft.ifft2(np.abs(f) ** 2).real / s.size
    autocorr = np.fft.fftshift(autocorr)
    return _radial_average(autocorr)


def structure_factor(lattice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged structure factor ``S(k) = |FFT(s − ⟨s⟩)|² / N²``.

    Returns ``(k, S)`` with physical wavevector ``k = 2π m / N`` (lattice spacing
    = 1). The ``k = 0`` component (the conserved magnetisation) is excluded from
    the returned arrays' first entry being meaningful — callers that take moments
    must drop ``k = 0`` (see :func:`characteristic_k`).
    """
    s = lattice.astype(np.float64)
    mean = s.mean()
    f = np.fft.fft2(s - mean)
    sk = np.abs(f) ** 2 / s.size
    sk = np.fft.fftshift(sk)
    radii, sk_radial = _radial_average(sk)
    n = lattice.shape[0]
    k = 2.0 * np.pi * radii / n
    return k, sk_radial


def characteristic_k(lattice: np.ndarray) -> float:
    """First moment of the structure factor, ``⟨k⟩ = Σ k S(k) / Σ S(k)`` (k>0)."""
    k, sk = structure_factor(lattice)
    sel = k > 0
    ks, sks = k[sel], sk[sel]
    good = np.isfinite(sks)
    denom = sks[good].sum()
    if denom <= 0:
        return float("nan")
    return float((ks[good] * sks[good]).sum() / denom)


# --------------------------------------------------------------------------- #
# Length-scale estimators (L_C, L_S, L_E)                                       #
# --------------------------------------------------------------------------- #


def length_from_correlation(lattice: np.ndarray, threshold: float = 1.0 / np.e) -> float:
    """``L_C``: smallest ``r`` where ``C(r)`` first falls to ``threshold·C(0)``.

    Linear interpolation between the bracketing radii. Returns ``nan`` if no
    crossing is found within the radial range.
    """
    r, c = correlation_function(lattice)
    if c[0] <= 0:
        return float("nan")
    target = threshold * c[0]
    cn = c / c[0]
    for i in range(1, len(cn)):
        if np.isfinite(cn[i]) and cn[i] <= threshold:
            # interpolate between i-1 and i
            c0, c1 = cn[i - 1], cn[i]
            if c1 == c0:
                return float(r[i])
            frac = (c0 - threshold) / (c0 - c1)
            return float(r[i - 1] + frac * (r[i] - r[i - 1]))
    return float("nan")


def length_from_correlation_zero(lattice: np.ndarray) -> float:
    """Alternative ``L_C``: first zero crossing of ``C(r)`` (interpolated)."""
    r, c = correlation_function(lattice)
    for i in range(1, len(c)):
        if np.isfinite(c[i]) and c[i] <= 0:
            c0, c1 = c[i - 1], c[i]
            if c1 == c0:
                return float(r[i])
            frac = c0 / (c0 - c1)
            return float(r[i - 1] + frac * (r[i] - r[i - 1]))
    return float("nan")


def length_from_structure(lattice: np.ndarray) -> float:
    """``L_S = 2π / ⟨k⟩`` from the structure-factor first moment."""
    kbar = characteristic_k(lattice)
    if not np.isfinite(kbar) or kbar <= 0:
        return float("nan")
    return float(2.0 * np.pi / kbar)


def length_from_energy(lattice: np.ndarray, e_inf: float) -> float:
    """``L_E ∝ 1/(E − E_∞)`` (late-stage interfacial relation).

    Returns the *proportionality-free* quantity ``1/(e − e_inf)`` in per-spin
    energy units; the amplitude is non-universal and is fixed elsewhere. ``e``
    and ``e_inf`` are per-spin energies. Returns ``nan`` when the excess energy
    is non-positive (outside the interfacial regime / within noise of ``E_∞``).
    """
    e = energy_per_spin(lattice)
    excess = e - e_inf
    if excess <= 0:
        return float("nan")
    return float(1.0 / excess)


# --------------------------------------------------------------------------- #
# Cluster analysis (periodic connected components)                             #
# --------------------------------------------------------------------------- #


def _periodic_label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 4-connected clusters of ``True`` cells with periodic wrap.

    Uses :func:`scipy.ndimage.label` then merges labels that touch across the
    periodic boundary with a union–find pass.
    """
    if not _HAVE_SCIPY:
        raise RuntimeError("cluster analysis requires scipy")
    lbl, num = _ndi_label(mask)  # 4-connectivity (default cross structure)
    if num <= 1:
        return lbl, num

    parent = list(range(num + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    n = mask.shape[0]
    # wrap rows (top<->bottom) and columns (left<->right)
    for c in range(n):
        if mask[0, c] and mask[n - 1, c]:
            union(lbl[0, c], lbl[n - 1, c])
    for r in range(n):
        if mask[r, 0] and mask[r, n - 1]:
            union(lbl[r, 0], lbl[r, n - 1])

    roots = np.array([find(i) for i in range(num + 1)])
    # relabel to contiguous ids
    uniq = {root: i for i, root in enumerate(sorted(set(roots[1:])), start=1)}
    remap = np.zeros(num + 1, dtype=np.int64)
    for i in range(1, num + 1):
        remap[i] = uniq[roots[i]]
    merged = remap[lbl]
    return merged, len(uniq)


def cluster_areas(lattice: np.ndarray, spin_value: int = 1) -> np.ndarray:
    """Areas (site counts) of periodic 4-connected clusters of ``spin_value``.

    Returns a sorted (descending) array of cluster sizes — the empirical
    ``P(A, t)`` sample for the requested spin species.
    """
    mask = lattice == spin_value
    if mask.sum() == 0:
        return np.zeros(0, dtype=np.int64)
    merged, num = _periodic_label(mask)
    counts = np.bincount(merged.ravel())[1:]  # drop background label 0
    counts = counts[counts > 0]
    return np.sort(counts)[::-1]


def cluster_number_density(lattice: np.ndarray) -> float:
    """Energy-independent morphology measure: clusters per site.

    Counts periodic 4-connected same-spin clusters of *both* species and
    divides by ``N²``. Unlike :func:`broken_bond_density` this is a topological
    (connectivity) quantity, not an affine function of the energy, so it can
    distinguish a route inversion (different domain-size distribution) from a
    genuine slow-mode effect (Mpemba morphology gate).
    """
    up = cluster_areas(lattice, 1)
    down = cluster_areas(lattice, -1)
    return float(len(up) + len(down)) / lattice.size


# --------------------------------------------------------------------------- #
# Bundled snapshot of all scalar observables                                   #
# --------------------------------------------------------------------------- #


@dataclass
class ObservableSnapshot:
    """Scalar observables at a single time, for one configuration."""

    energy: int
    energy_per_spin: float
    magnetisation: int
    broken_bond_density: float
    L_C: float
    L_S: float
    characteristic_k: float
    cluster_number_density: float


def snapshot(lattice: np.ndarray, *, with_clusters: bool = True) -> ObservableSnapshot:
    """Compute the bundled scalar observables for one configuration.

    ``with_clusters=False`` skips the (slower) connected-component analysis.
    """
    e = total_energy(lattice)
    return ObservableSnapshot(
        energy=e,
        energy_per_spin=e / lattice.size,
        magnetisation=int(lattice.sum()),
        broken_bond_density=broken_bond_density(lattice),
        L_C=length_from_correlation(lattice),
        L_S=length_from_structure(lattice),
        characteristic_k=characteristic_k(lattice),
        cluster_number_density=cluster_number_density(lattice) if with_clusters else float("nan"),
    )
