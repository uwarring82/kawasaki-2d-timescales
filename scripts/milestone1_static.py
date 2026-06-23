#!/usr/bin/env python
"""Milestone 1 — static phase separation.

Equilibrate M=0 lattices at temperatures spanning T_c and record snapshots plus
equilibrium observables (energy, broken-bond density, cluster statistics,
correlation length). Demonstrates conserved-order-parameter phase separation:
below T_c the system phase-separates into a few large domains; above T_c it
stays disordered with small correlations.

Produces an append-only ``results/<run_id>/`` directory with snapshots, an
observables CSV, figures, and a provenance manifest.

Usage:
    python scripts/milestone1_static.py [configs/milestone1_static.yaml]
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make the script runnable without an editable install.
sys.path.insert(0, str(REPO_ROOT / "src"))

from kawasaki2d import io, observables as obs, protocols, provenance, T_C  # noqa: E402
from kawasaki2d.rng import spawn_rngs  # noqa: E402


def main(config_path: str) -> Path:
    config = io.load_config(config_path)
    run_id = config["run_id"]
    n = int(config["model"]["N"])
    M = int(config["model"].get("magnetisation", 0))
    temps = [float(t) for t in config["temperatures"]]
    prep = config["preparation"]
    kernel = prep.get("kernel", "nonlocal")
    n_sweeps = int(prep["n_sweeps"])
    base_seed = int(config["ensemble"]["seed"])
    with_clusters = bool(config.get("observables", {}).get("with_clusters", True))

    run_dir = io.new_run_directory(REPO_ROOT / "results", run_id)
    (run_dir / "snapshots").mkdir()
    io.dump_config(config, run_dir / "config.yaml")

    manifest = provenance.Manifest.build(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        config=config,
        seeds={"base": base_seed},
        repo_root=REPO_ROOT,
        notes="Milestone 1: static phase separation",
    )

    rngs = spawn_rngs(base_seed, len(temps))
    rows = []
    snapshots = {}
    for T, rng in zip(temps, rngs):
        p = protocols.prepare_initial_state(
            n, T, M, rng, kernel=kernel, n_sweeps=n_sweeps
        )
        snap = obs.snapshot(p.lattice, with_clusters=with_clusters)
        areas = obs.cluster_areas(p.lattice, 1)
        largest_frac = float(areas[0]) / (n * n / 2) if areas.size else 0.0
        rows.append(
            {
                "temperature": T,
                "T_over_Tc": T / T_C,
                "energy_per_spin": snap.energy_per_spin,
                "broken_bond_density": snap.broken_bond_density,
                "L_C": snap.L_C,
                "cluster_number_density": snap.cluster_number_density,
                "n_up_clusters": int(areas.size),
                "largest_cluster_frac": largest_frac,
            }
        )
        snap_path = run_dir / "snapshots" / f"T_{T:.3f}.npy"
        np.save(snap_path, p.lattice)
        manifest.add_output(f"snapshot_T{T:.3f}", snap_path)
        snapshots[T] = p.lattice.copy()
        print(
            f"T={T:5.3f} (T/Tc={T/T_C:4.2f})  e/spin={snap.energy_per_spin:+.3f}  "
            f"broken={snap.broken_bond_density:.3f}  L_C={snap.L_C:5.2f}  "
            f"largest_domain_frac={largest_frac:.3f}"
        )

    # observables CSV
    obs_cols = (
        "temperature", "T_over_Tc", "energy_per_spin", "broken_bond_density",
        "L_C", "cluster_number_density", "n_up_clusters", "largest_cluster_frac",
    )
    obs_csv = run_dir / "static_observables.csv"
    io.write_trajectory_csv(obs_csv, rows, columns=obs_cols)
    manifest.add_output("static_observables", obs_csv)

    # ---- figures ----
    _plot_snapshots(snapshots, run_dir / "snapshots_grid.png")
    _plot_observables(rows, run_dir / "static_observables.png")
    manifest.add_output("snapshots_grid", run_dir / "snapshots_grid.png")
    manifest.add_output("static_observables_fig", run_dir / "static_observables.png")

    manifest.write(run_dir / "manifest.json")
    print(f"\nwrote {run_dir}")
    return run_dir


def _plot_snapshots(snapshots: dict, path: Path) -> None:
    temps = sorted(snapshots)
    fig, axes = plt.subplots(1, len(temps), figsize=(3 * len(temps), 3.4))
    if len(temps) == 1:
        axes = [axes]
    for ax, T in zip(axes, temps):
        ax.imshow(snapshots[T], cmap="binary", interpolation="nearest")
        ax.set_title(f"T={T:.3g}  (T/T_c={T / T_C:.2f})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Equilibrium configurations, M=0 (Kawasaki-conserved)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_observables(rows: list[dict], path: Path) -> None:
    T = np.array([r["temperature"] for r in rows])
    e = np.array([r["energy_per_spin"] for r in rows])
    nb = np.array([r["broken_bond_density"] for r in rows])
    ncd = np.array([r["cluster_number_density"] for r in rows])
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a in ax:
        a.axvline(T_C, color="r", ls="--", lw=1, label="$T_c$")
    ax[0].plot(T, e, "o-"); ax[0].set_xlabel("T"); ax[0].set_ylabel("energy / spin")
    ax[1].plot(T, nb, "o-"); ax[1].set_xlabel("T"); ax[1].set_ylabel("broken-bond density")
    ax[2].plot(T, ncd, "o-"); ax[2].set_xlabel("T"); ax[2].set_ylabel("cluster number density")
    ax[0].legend()
    fig.suptitle("Static equilibrium observables vs temperature (N fixed, M=0)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "configs" / "milestone1_static.yaml")
    main(cfg)
