"""Quench protocol: equilibrate at ``T_i`` → quench to ``T_f`` → track coarsening.

The preparation (equilibration) and the post-quench *kinetic* kernel are kept
strictly separate, per the equilibration gate:

* **Kinetic kernel** (post-quench, what we study): always local Kawasaki
  Metropolis (:func:`kawasaki2d.dynamics.run_kawasaki`).
* **Preparation kernel** (pre-quench): ``"local"`` Kawasaki (baseline) or
  ``"nonlocal"`` opposite-spin exchange (faster sampler, same fixed-``M``
  equilibrium). Recorded in the result and the manifest; never conflated with
  the kinetic kernel.

``L_E`` is intentionally *not* stored per step: it depends on an independently
estimated ``E_∞(N, T_f, M)`` and is derived in :mod:`kawasaki2d.analysis` from
the recorded energy column, so the ``E_∞`` choice and its horizon dependence are
explicit rather than baked into the trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import dynamics, observables
from .lattice import init_lattice, magnetisation, total_energy

_PREP_KERNELS = {
    "local": dynamics.run_kawasaki,
    "nonlocal": dynamics.run_nonlocal_exchange,
}


@dataclass
class Preparation:
    """An equilibrated initial state plus its preparation provenance."""

    lattice: np.ndarray
    T_i: float
    kernel: str
    n_sweeps: int
    energy: int
    magnetisation: int
    acceptance_rate: float
    energy_trace: np.ndarray = field(default_factory=lambda: np.zeros(0))


def prepare_initial_state(
    n: int,
    T_i: float,
    magnetisation_target: int,
    rng: np.random.Generator,
    *,
    kernel: str = "nonlocal",
    n_sweeps: int = 2000,
    trace_every: int | None = None,
) -> Preparation:
    """Equilibrate a fresh random ``M``-fixed state at ``T_i``.

    ``trace_every`` (if set) records the per-spin energy every that-many sweeps,
    so the equilibration gate can check saturation. ``kernel`` selects the
    preparation sampler (``"local"`` or ``"nonlocal"``).
    """
    if kernel not in _PREP_KERNELS:
        raise ValueError(f"unknown preparation kernel {kernel!r}; choose from {list(_PREP_KERNELS)}")
    run = _PREP_KERNELS[kernel]
    lattice = init_lattice(n, magnetisation_target, rng=rng)

    trace: list[float] = []
    if trace_every and trace_every > 0:
        done = 0
        while done < n_sweeps:
            block = min(trace_every, n_sweeps - done)
            run(lattice, T_i, block, rng)
            done += block
            trace.append(total_energy(lattice) / lattice.size)
        # final acceptance is not accumulated across blocks here; recompute a
        # short diagnostic block for a representative acceptance rate
        stats = run(lattice, T_i, max(1, trace_every // 4), rng)
        acc = stats.acceptance_rate
    else:
        stats = run(lattice, T_i, n_sweeps, rng)
        acc = stats.acceptance_rate

    return Preparation(
        lattice=lattice,
        T_i=float(T_i),
        kernel=kernel,
        n_sweeps=int(n_sweeps),
        energy=total_energy(lattice),
        magnetisation=magnetisation(lattice),
        acceptance_rate=acc,
        energy_trace=np.asarray(trace),
    )


def log_schedule(t_max: int, n_points: int, t_min: int = 1) -> np.ndarray:
    """Log-spaced integer sweep checkpoints in ``[t_min, t_max]``, plus ``0``.

    Deduplicated and sorted. ``0`` (the quench instant) is always included.
    """
    if t_max < t_min:
        raise ValueError("t_max must be >= t_min")
    pts = np.unique(
        np.round(np.geomspace(t_min, t_max, num=int(n_points))).astype(np.int64)
    )
    return np.unique(np.concatenate(([0], pts)))


@dataclass
class CoarseningTrajectory:
    """A single quench trajectory: scalar rows plus optional ``S(k, t)`` curves."""

    sweeps: np.ndarray            # (T,) checkpoint sweeps
    rows: list[dict]              # per-checkpoint scalar observables (TRAJECTORY_COLUMNS)
    sk_k: np.ndarray              # (K,) radial wavevectors (same for all t), or empty
    sk: np.ndarray                # (T, K) structure factor S(k, t), or empty


def coarsening_trajectory(
    lattice: np.ndarray,
    T_f: float,
    schedule: np.ndarray,
    rng: np.random.Generator,
    *,
    with_clusters: bool = True,
    record_sk: bool = False,
) -> CoarseningTrajectory:
    """Quench ``lattice`` to ``T_f`` and record observables on ``schedule``.

    ``lattice`` (already equilibrated) is evolved in place by the local Kawasaki
    kinetic kernel. ``schedule`` is a sorted array of sweep checkpoints (include
    0 to record the pre-quench state). When ``record_sk`` is set, the radially
    averaged structure factor ``S(k, t)`` is captured at every checkpoint.

    Acceptance rate is reported per interval (diagnostic only; never a clock for
    cross-preparation comparison).
    """
    schedule = np.unique(np.asarray(schedule, dtype=np.int64))
    rows: list[dict] = []
    sk_rows: list[np.ndarray] = []
    sk_k = np.zeros(0)
    prev = 0
    for t in schedule:
        if t > prev:
            stats = dynamics.run_kawasaki(lattice, T_f, int(t - prev), rng)
            acc = stats.acceptance_rate
        else:
            acc = float("nan")  # the t=0 checkpoint: no interval yet
        prev = int(t)
        snap = observables.snapshot(lattice, with_clusters=with_clusters)
        rows.append(
            {
                "sweep": int(t),
                "energy": snap.energy,
                "energy_per_spin": snap.energy_per_spin,
                "magnetisation": snap.magnetisation,
                "broken_bond_density": snap.broken_bond_density,
                "L_C": snap.L_C,
                "L_S": snap.L_S,
                "characteristic_k": snap.characteristic_k,
                "cluster_number_density": snap.cluster_number_density,
                "acceptance_rate": acc,
            }
        )
        if record_sk:
            k, sk = observables.structure_factor(lattice)
            sk_k = k
            sk_rows.append(sk)
    return CoarseningTrajectory(
        sweeps=schedule,
        rows=rows,
        sk_k=sk_k,
        sk=np.asarray(sk_rows) if sk_rows else np.zeros((0, 0)),
    )


def track_quench(
    lattice: np.ndarray,
    T_f: float,
    schedule: np.ndarray,
    rng: np.random.Generator,
    *,
    with_clusters: bool = True,
) -> list[dict]:
    """Scalar-only quench trajectory (rows keyed by ``io.TRAJECTORY_COLUMNS``).

    Thin wrapper over :func:`coarsening_trajectory` kept for the CLI and tests.
    """
    return coarsening_trajectory(
        lattice, T_f, schedule, rng, with_clusters=with_clusters, record_sk=False
    ).rows
