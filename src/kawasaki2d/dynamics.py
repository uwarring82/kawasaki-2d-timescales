"""Kinetic kernels: local Kawasaki exchange and (preparation) non-local exchange.

The **post-quench kinetic kernel** is local nearest-neighbour Kawasaki exchange
with Metropolis acceptance (:func:`run_kawasaki`). Magnetisation is conserved
exactly because every accepted move swaps two spins.

Reproducibility design
----------------------
The random stream is drawn in NumPy (PCG64) and *passed into* the inner loop,
rather than drawn inside it. Consequences:

* The PCG64 stream recorded in the manifest is the one actually consumed.
* The optional Numba-JIT inner loop and the pure-Python fallback consume the
  identical stream and therefore produce **bitwise-identical** trajectories.
  Correctness never depends on Numba being installed; only speed does.

Each sweep consumes, in order: ``N²`` site indices, ``N²`` direction indices,
``N²`` acceptance uniforms. One sweep = ``N²`` *attempted* updates regardless of
acceptance (time-normalisation gate).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

# Set KAWASAKI2D_NO_NUMBA=1 to force the pure-Python inner loop (for
# benchmarking, or to prove the two paths agree bitwise). Correctness is
# identical either way; only speed changes.
_DISABLE_NUMBA = os.environ.get("KAWASAKI2D_NO_NUMBA", "") not in ("", "0", "false", "False")

try:  # optional acceleration; correctness is identical without it
    if _DISABLE_NUMBA:
        raise ImportError("numba disabled via KAWASAKI2D_NO_NUMBA")
    from numba import njit

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - exercised only when numba is absent
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        """No-op stand-in for :func:`numba.njit` when Numba is unavailable."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(func):
            return func

        return _decorator


def numba_available() -> bool:
    """True when the Numba-accelerated inner loop is active."""
    return _HAVE_NUMBA


def _kawasaki_inner_impl(spins, n, sites, dirs, accept, T):
    """One sweep of local Kawasaki Metropolis. Mutates ``spins`` in place.

    Returns ``(n_swaps, n_active)``: accepted swaps and proposals whose two
    spins differed (``ΔE`` actually evaluated). Equal-spin proposals are no-ops
    that still count as attempts.
    """
    n_attempts = sites.shape[0]
    n_swaps = 0
    n_active = 0
    for k in range(n_attempts):
        i = sites[k]
        r0 = i // n
        c0 = i % n
        d = dirs[k]
        if d == 0:  # up
            r1 = r0 - 1 if r0 > 0 else n - 1
            c1 = c0
        elif d == 1:  # down
            r1 = r0 + 1 if r0 < n - 1 else 0
            c1 = c0
        elif d == 2:  # left
            r1 = r0
            c1 = c0 - 1 if c0 > 0 else n - 1
        else:  # right
            r1 = r0
            c1 = c0 + 1 if c0 < n - 1 else 0

        si = int(spins[r0, c0])
        sj = int(spins[r1, c1])
        if si == sj:
            continue  # no-op exchange; still an attempt
        n_active += 1

        up0 = r0 - 1 if r0 > 0 else n - 1
        dn0 = r0 + 1 if r0 < n - 1 else 0
        lf0 = c0 - 1 if c0 > 0 else n - 1
        rt0 = c0 + 1 if c0 < n - 1 else 0
        nsum_i = (
            int(spins[up0, c0])
            + int(spins[dn0, c0])
            + int(spins[r0, lf0])
            + int(spins[r0, rt0])
        )
        up1 = r1 - 1 if r1 > 0 else n - 1
        dn1 = r1 + 1 if r1 < n - 1 else 0
        lf1 = c1 - 1 if c1 > 0 else n - 1
        rt1 = c1 + 1 if c1 < n - 1 else 0
        nsum_j = (
            int(spins[up1, c1])
            + int(spins[dn1, c1])
            + int(spins[r1, lf1])
            + int(spins[r1, rt1])
        )
        a = nsum_i - sj  # neighbours of i excluding j
        b = nsum_j - si  # neighbours of j excluding i
        dE = (si - sj) * (a - b)

        if dE <= 0 or accept[k] < math.exp(-dE / T):
            spins[r0, c0] = sj
            spins[r1, c1] = si
            n_swaps += 1
    return n_swaps, n_active


_kawasaki_inner = njit(cache=True)(_kawasaki_inner_impl)


@dataclass
class SweepStats:
    """Diagnostics accumulated over a block of sweeps."""

    sweeps: int
    attempts: int
    swaps: int
    active: int

    @property
    def acceptance_rate(self) -> float:
        """Accepted swaps per attempted update (includes equal-spin no-ops)."""
        return self.swaps / self.attempts if self.attempts else 0.0

    @property
    def active_acceptance_rate(self) -> float:
        """Accepted swaps per *active* (unequal-spin) proposal."""
        return self.swaps / self.active if self.active else 0.0


def run_kawasaki(
    spins: np.ndarray,
    T: float,
    n_sweeps: int,
    rng: np.random.Generator,
) -> SweepStats:
    """Run ``n_sweeps`` of local Kawasaki Metropolis dynamics, in place.

    This is the post-quench kinetic kernel. ``spins`` is modified in place and
    its magnetisation is preserved exactly. Returns sweep diagnostics including
    the acceptance rate.
    """
    if spins.dtype != np.int8:
        raise TypeError("spins must be an int8 array (use lattice.init_lattice)")
    n = spins.shape[0]
    n_attempts = n * n
    total_swaps = 0
    total_active = 0
    for _ in range(int(n_sweeps)):
        sites = rng.integers(0, n_attempts, size=n_attempts, dtype=np.int64)
        dirs = rng.integers(0, 4, size=n_attempts, dtype=np.int64)
        accept = rng.random(n_attempts)
        swaps, active = _kawasaki_inner(spins, n, sites, dirs, accept, float(T))
        total_swaps += int(swaps)
        total_active += int(active)
    return SweepStats(
        sweeps=int(n_sweeps),
        attempts=int(n_sweeps) * n_attempts,
        swaps=total_swaps,
        active=total_active,
    )


def _nonlocal_inner_impl(flat, n, plus, minus, ip_arr, im_arr, accept, T):
    """One pass of non-local opposite-spin exchange. Mutates ``flat`` in place.

    ``flat`` is the lattice flattened to 1D (a view of the 2D spins). ``plus``
    and ``minus`` hold the flat indices of the ``+1`` and ``-1`` sites; they are
    updated in O(1) on each accepted swap (a ``+`` site that flips simply trades
    places between the two index lists). ``ip_arr`` / ``im_arr`` are random
    indices into ``plus`` / ``minus`` (drawn in NumPy for reproducibility).

    Returns ``(n_swaps, n_active)``. Every proposal here is "active" (the two
    chosen sites carry opposite spins by construction).
    """
    n_plus = plus.shape[0]
    n_minus = minus.shape[0]
    n_attempts = ip_arr.shape[0]
    n_swaps = 0
    n_active = 0
    for k in range(n_attempts):
        if n_plus == 0 or n_minus == 0:
            break
        ip = ip_arr[k]
        im = im_arr[k]
        p = plus[ip]
        m = minus[im]
        n_active += 1

        pr = p // n
        pc = p % n
        mr = m // n
        mc = m % n
        sp = int(flat[p])  # +1
        sm = int(flat[m])  # -1

        up_p = ((pr - 1) % n) * n + pc
        dn_p = ((pr + 1) % n) * n + pc
        lf_p = pr * n + (pc - 1) % n
        rt_p = pr * n + (pc + 1) % n
        nsum_p = int(flat[up_p]) + int(flat[dn_p]) + int(flat[lf_p]) + int(flat[rt_p])

        up_m = ((mr - 1) % n) * n + mc
        dn_m = ((mr + 1) % n) * n + mc
        lf_m = mr * n + (mc - 1) % n
        rt_m = mr * n + (mc + 1) % n
        nsum_m = int(flat[up_m]) + int(flat[dn_m]) + int(flat[lf_m]) + int(flat[rt_m])

        # If the two sites are nearest neighbours the shared bond is unchanged
        # (stays anti-aligned); remove it from both neighbour sums so it is not
        # spuriously counted in ΔE.
        adjacent = (pc == mc and (pr - mr) % n == 1) or (pc == mc and (mr - pr) % n == 1) \
            or (pr == mr and (pc - mc) % n == 1) or (pr == mr and (mc - pc) % n == 1)
        if adjacent:
            nsum_p -= sm
            nsum_m -= sp

        # ΔE = -J[ s'_p nsum_p + s'_m nsum_m - s_p nsum_p - s_m nsum_m ],
        # with s'_p = sm, s'_m = sp.
        dE = -((sm - sp) * nsum_p + (sp - sm) * nsum_m)

        if dE <= 0 or accept[k] < math.exp(-dE / T):
            flat[p] = sm
            flat[m] = sp
            plus[ip] = m
            minus[im] = p
            n_swaps += 1
    return n_swaps, n_active


_nonlocal_inner = njit(cache=True)(_nonlocal_inner_impl)


def run_nonlocal_exchange(
    spins: np.ndarray,
    T: float,
    n_sweeps: int,
    rng: np.random.Generator,
) -> SweepStats:
    """Non-local opposite-spin exchange (a *preparation* sampler only).

    Picks a random ``+`` site and a random ``-`` site and proposes swapping
    them with Metropolis acceptance. Conserves ``M`` and samples the same
    fixed-``M`` canonical equilibrium as local Kawasaki, but mixes faster — used
    only to equilibrate initial states, never as the post-quench kinetics. One
    "sweep" is ``N²`` attempted opposite-spin exchanges.

    O(1) per accepted move: the ``+``/``-`` site-index lists are updated in
    place rather than recomputed. Like the local kernel, the random stream is
    drawn in NumPy and passed in, so the Numba and pure-Python paths agree
    bitwise.
    """
    if spins.dtype != np.int8:
        raise TypeError("spins must be an int8 array (use lattice.init_lattice)")
    n = spins.shape[0]
    n_attempts = n * n
    flat = spins.reshape(-1)  # view; mutations propagate to spins
    plus = np.flatnonzero(flat == 1).astype(np.int64)
    minus = np.flatnonzero(flat == -1).astype(np.int64)
    n_plus, n_minus = plus.shape[0], minus.shape[0]
    total_swaps = 0
    total_active = 0
    if n_plus == 0 or n_minus == 0:
        return SweepStats(int(n_sweeps), int(n_sweeps) * n_attempts, 0, 0)
    for _ in range(int(n_sweeps)):
        ip_arr = rng.integers(0, n_plus, size=n_attempts, dtype=np.int64)
        im_arr = rng.integers(0, n_minus, size=n_attempts, dtype=np.int64)
        accept = rng.random(n_attempts)
        swaps, active = _nonlocal_inner(flat, n, plus, minus, ip_arr, im_arr, accept, float(T))
        total_swaps += int(swaps)
        total_active += int(active)
    return SweepStats(
        sweeps=int(n_sweeps),
        attempts=int(n_sweeps) * n_attempts,
        swaps=total_swaps,
        active=total_active,
    )
