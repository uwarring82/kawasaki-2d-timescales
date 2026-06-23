"""Command-line runner: ``kawasaki-run CONFIG``.

Drives a single-quench (Milestone-2-style) coarsening run from a YAML config,
producing an append-only ``results/<run_id>/`` directory with a per-realisation
and ensemble trajectory plus a provenance manifest. Wall-clock timestamps are
taken here, at the application boundary, never inside the library.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import io, protocols, provenance
from .rng import spawn_rngs


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_single_quench(config: dict, results_root: str | Path = "results",
                      repo_root: Path | None = None) -> Path:
    """Execute a single-quench ensemble run and write an append-only result dir."""
    run_id = config["run_id"]
    model = config["model"]
    prep_cfg = config["preparation"]
    quench_cfg = config["quench"]
    ens_cfg = config.get("ensemble", {})
    obs_cfg = config.get("observables", {})

    n = int(model["N"])
    M = int(model.get("magnetisation", 0))
    T_i = float(prep_cfg["T_i"])
    prep_kernel = prep_cfg.get("kernel", "nonlocal")
    prep_sweeps = int(prep_cfg["n_sweeps"])
    T_f = float(quench_cfg["T_f"])
    schedule = protocols.log_schedule(
        t_max=int(quench_cfg["schedule"]["t_max"]),
        n_points=int(quench_cfg["schedule"]["n_points"]),
    )
    n_real = int(ens_cfg.get("n_realisations", 1))
    base_seed = int(ens_cfg.get("seed", 0))
    with_clusters = bool(obs_cfg.get("with_clusters", True))

    run_dir = io.new_run_directory(results_root, run_id)
    io.dump_config(config, run_dir / "config.yaml")

    # Independent, reproducible sub-streams per realisation.
    rngs = spawn_rngs(base_seed, n_real)
    manifest = provenance.Manifest.build(
        run_id=run_id,
        timestamp=_utc_now_iso(),
        config=config,
        seeds={"base": base_seed, "n_realisations": n_real},
        repo_root=repo_root,
        notes="single-quench ensemble run",
    )

    # Accumulate per-realisation length observables for the ensemble file.
    per_real_L_S = []
    for k, rng in enumerate(rngs):
        prep = protocols.prepare_initial_state(
            n, T_i, M, rng, kernel=prep_kernel, n_sweeps=prep_sweeps
        )
        rows = protocols.track_quench(
            prep.lattice, T_f, schedule, rng, with_clusters=with_clusters
        )
        csv_path = run_dir / f"trajectory_real{k:03d}.csv"
        io.write_trajectory_csv(csv_path, rows)
        manifest.add_output(f"trajectory_real{k:03d}", csv_path)
        per_real_L_S.append([r["L_S"] for r in rows])

    # Ensemble mean L_S vs sweep (a quick-look summary; full stats live in analysis).
    arr = np.asarray(per_real_L_S, dtype=float)
    ens_rows = [
        {"sweep": int(schedule[j]), "L_S": float(np.nanmean(arr[:, j]))}
        for j in range(len(schedule))
    ]
    ens_path = run_dir / "ensemble_LS.csv"
    io.write_trajectory_csv(ens_path, ens_rows, columns=("sweep", "L_S"))
    manifest.add_output("ensemble_LS", ens_path)

    manifest.write(run_dir / "manifest.json")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Kawasaki-Ising quench from a YAML config.")
    parser.add_argument("config", help="path to a YAML run config")
    parser.add_argument("--results-root", default="results", help="root output directory")
    args = parser.parse_args(argv)

    config = io.load_config(args.config)
    run_dir = run_single_quench(config, results_root=args.results_root)
    print(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
