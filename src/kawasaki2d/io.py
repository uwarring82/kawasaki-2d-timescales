"""I/O: YAML configs, append-only run directories, and the output schema.

Interoperability (FAIR): configs are YAML; numerical trajectories are CSV with a
documented header (see ``docs/output_schema.md``); units are ``J = k_B = 1`` and
time is in *sweeps*. Results are **append-only**: a run directory is never
overwritten; corrections are new dated runs.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

# Column names of the standard coarsening-trajectory CSV. Documented in
# docs/output_schema.md; consumed by analysis.py.
TRAJECTORY_COLUMNS = (
    "sweep",              # attempted-update time: integer number of sweeps since quench
    "energy",             # total energy (J=1)
    "energy_per_spin",    # energy / N^2
    "magnetisation",      # M = sum s_i (must be constant within a run)
    "broken_bond_density",
    "L_C",                # length from correlation (1/e crossing)
    "L_S",                # length from structure-factor first moment
    "characteristic_k",   # <k>
    "cluster_number_density",
    "acceptance_rate",    # accepted swaps / attempted updates over the preceding interval
)


def load_config(path: str | Path) -> dict:
    """Load a YAML config into a plain dict."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    return cfg


def dump_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Write a config dict to YAML (sorted keys for stable diffs)."""
    with open(path, "w") as fh:
        yaml.safe_dump(dict(config), fh, sort_keys=True, default_flow_style=False)


def new_run_directory(results_root: str | Path, run_id: str) -> Path:
    """Create ``results_root/run_id`` for a new run.

    Append-only contract: raises ``FileExistsError`` if the directory already
    exists, so a prior result can never be silently overwritten.
    """
    root = Path(results_root)
    run_dir = root / run_id
    if run_dir.exists():
        raise FileExistsError(
            f"run directory {run_dir} already exists; results are append-only "
            f"(use a new run_id for a correction)"
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_trajectory_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] = TRAJECTORY_COLUMNS,
) -> None:
    """Write coarsening-trajectory rows to CSV with a fixed header.

    Each row is a mapping with (at least) the keys in ``columns``; missing keys
    are written empty. A units header comment line precedes the column header.
    """
    path = Path(path)
    with open(path, "w", newline="") as fh:
        fh.write("# units: J=k_B=1; time in sweeps (one sweep = N^2 attempted updates)\n")
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def read_trajectory_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Read a trajectory CSV (skipping the units comment) into column arrays."""
    path = Path(path)
    with open(path) as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    reader = csv.DictReader(lines)
    cols: dict[str, list] = {name: [] for name in reader.fieldnames or []}
    for row in reader:
        for name, val in row.items():
            cols[name].append(float(val) if val not in ("", None) else np.nan)
    return {name: np.asarray(vals) for name, vals in cols.items()}
