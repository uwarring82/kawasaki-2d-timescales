# kawasaki2d

A small, reproducible **2D Kawasaki–Ising** demonstrator of conserved-order-parameter
phase separation, built to study quench-dependent relaxation and coarsening
timescales — and to map the *operational boundary* between finite-system,
mode-resolved anomalous relaxation and thermodynamic-limit coarsening (the
"spectral vs. coarsening-route" question for the Mpemba effect under conserved
dynamics).

The governing specification is the task card
[`TASK-kawasaki-mpemba-boundary-v4.md`](TASK-kawasaki-mpemba-boundary-v4.md).
This README documents what is implemented and how to reproduce it.

## Model

- Square lattice, periodic boundary conditions.
- Hamiltonian `H = -J Σ_⟨ij⟩ s_i s_j`, spins `s ∈ {+1, -1}`, `J = k_B = 1`.
- Critical temperature `T_c = 2 / ln(1 + √2) ≈ 2.269185`.
- **Post-quench kinetic kernel:** nearest-neighbour **Kawasaki exchange** with
  Metropolis acceptance. Magnetisation `M = Σ s_i` is conserved exactly.
- **Time unit:** one *sweep* `= N²` *attempted* nearest-neighbour bond updates,
  independent of acceptance (per the time-normalisation gate). Accepted-move
  counts are diagnostics only and are never used for cross-preparation
  comparison.

## Design decisions (committed)

| Decision | Choice | Rationale |
|---|---|---|
| Local move | uniform random site + uniform random of 4 NN directions | gives a symmetric, detailed-balance-preserving proposal over the fixed-`M` sector |
| Equal-spin proposal | counts as an attempt, leaves config unchanged (`ΔE = 0`) | keeps the attempted-update clock honest (time independent of acceptance) |
| `ΔE` | `J (s_i − s_j)(a − b)`, `a,b` = neighbour-sums of `i,j` excluding each other | derived in `docs/physics.md`; verified against brute-force total energy in tests |
| RNG | NumPy `Generator(PCG64)`, seed recorded per run | bitwise reproducible; library + version captured in the manifest |
| Acceleration | optional Numba JIT of the inner loop, pure-Python fallback | correctness never depends on Numba being installed |

## Repository layout

```
src/kawasaki2d/
  lattice.py      # lattice construction, energy, magnetisation, ΔE
  dynamics.py     # Kawasaki NN-exchange Metropolis kernel + sweeps
  observables.py  # E, M, C(r,t), S(k,t), L_C/L_S/L_E, k*, interface density, P(A,t)
  protocols.py    # equilibrate → quench → track
  analysis.py     # bootstrap, scaling collapse, crossing tests (pre-registered)
  io.py           # YAML config, append-only results, output schema
  provenance.py   # per-run manifest: config hash, git commit, seed, env, checksums
  rng.py          # seeding utilities
  cli.py          # `kawasaki-run` entry point
configs/          # YAML run configs (pre-registered analysis plan lives here)
scripts/          # milestone driver scripts
tests/            # validation-gate tests (must pass before any physics claim)
results/          # append-only run outputs, each with a manifest
logbook/          # dated, append-only human logbook
docs/             # physics notes, output schema
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,plot,analysis]"        # add ",accel" for Numba JIT
pytest                                         # validation gate
```

## Gate status

The task card defines gates that **must pass before any physics claim**. Current state:

- [x] Exact conservation of `M` after every proposed move (test)
- [x] Local `ΔE` verified against brute-force total-energy differences (test)
- [x] Correct Metropolis acceptance / fixed-`T` equilibrium behaviour (test)
- [x] Ergodicity + detailed balance of the exchange proposal on a small lattice (test)
- [x] Bitwise reproducibility under fixed seeds (test)
- [x] Ensemble averaging over independent realisations (Milestone 2; finite-size `N=32/64/128` comparison in Milestone 3)

**Milestones:** M1 (static phase separation) and M2 (single-quench benchmark)
are complete — see `results/` and `logbook/`. M2 confirms diffusive coarsening:
`L = R₀ + (λt)^{1/3}` fits both `L_C` and `L_S` to R² > 0.996, with the
offset-corrected exponent recovering 1/3. The equilibration diagnostic also
flags (as designed) that the near-`T_c` `T_i=2.4` leg needs more preparation
sweeps before Milestone 4.

FAIR / provenance / equilibration / Mpemba-claim / reporting gates: see the task
card. Provenance manifests are emitted by `provenance.py`; the human logbook is
in `logbook/`.

## Reproducibility & integrity

Every reported number, figure, and verdict is regenerable from a recorded
`(config, code commit, seed, environment)` tuple. Results are **append-only**;
corrections are new dated entries, never silent overwrites. All runs —
including negative, underdetermined, and discarded — are recorded and counted.
The analysis plan (fitting windows, estimator definitions, verdict thresholds,
the declared `(T_i, T_f, N)` grid and the primary pair) is committed to version
control **before** the crossing search (Milestone 5) is run.

## Licence

Code: MIT (`LICENSE`). Data and figures: CC-BY-4.0 (`LICENSE-DATA`).
