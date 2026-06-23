# Output schema

All numerical outputs use open, non-proprietary formats. Units are `J = k_B = 1`;
time is in **sweeps** (one sweep = `N²` attempted nearest-neighbour bond updates).

## Run directory

Each run produces an append-only directory `results/<run_id>/` containing:

| File | Format | Contents |
|---|---|---|
| `manifest.json` | JSON | provenance: config, config hash, seeds, git state, environment, output checksums |
| `config.yaml` | YAML | the exact config used (echoed for self-containment) |
| `trajectory*.csv` | CSV | per-checkpoint observables (schema below); one file per realisation or an ensemble file |
| `snapshots/*.npy` | NumPy | optional saved spin configurations (`int8`, `(N,N)`) |
| `*.png` / `*.svg` | image | figures regenerated from the data |

No file enters a run directory without a `manifest.json`. Directories are never
overwritten; a correction is a new dated `run_id`.

## Trajectory CSV columns

The first line is a `#`-prefixed units comment. Columns (see
`kawasaki2d.io.TRAJECTORY_COLUMNS`):

| Column | Units | Meaning |
|---|---|---|
| `sweep` | sweeps | attempted-update time since the quench (integer; 0 = quench instant) |
| `energy` | J | total energy `H = -J Σ s_i s_j` |
| `energy_per_spin` | J | `energy / N²` |
| `magnetisation` | spins | `M = Σ s_i` (constant within a run; recorded as a check) |
| `broken_bond_density` | — | fraction of NN bonds anti-aligned (affine in energy; *not* energy-independent) |
| `L_C` | lattice units | length from `C(r)` (1/e crossing) |
| `L_S` | lattice units | length from structure-factor first moment, `2π/⟨k⟩` |
| `characteristic_k` | 1/lattice | `⟨k⟩` |
| `cluster_number_density` | 1/site | periodic same-spin clusters per site (energy-independent morphology measure) |
| `acceptance_rate` | — | accepted swaps / attempted updates over the preceding interval (diagnostic only) |

`L_E` is **not** a stored column: it depends on an independently estimated
`E_∞(N, T_f, M)` and is derived in `analysis.py` from the `energy` column, so the
`E_∞` choice and its `N`/horizon dependence remain explicit.

## Manifest fields (`manifest.json`)

`run_id`, `timestamp` (ISO-8601), `config`, `config_hash` (SHA-256 of canonical
JSON), `seeds`, `git` (`commit`, `dirty`, `branch`), `environment`
(`python_version`, `platform`, `kawasaki2d_version`, `numba_active`, `rng`,
`packages`), `outputs` (label → SHA-256), `schema_version`, `notes`.
