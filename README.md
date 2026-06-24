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

## Findings in plain language

**The question.** Cool a two-component system (think a binary alloy) below its
ordering temperature and it *phase-separates*: the two species clump into
domains that grow over time — "coarsening". We asked a sharp version of the
**Mpemba question**: can a system that *started hotter* end up ordering *faster*
than one that started colder? Concretely, does a near-infinite-temperature start
(`T_i=10`, tiny initial domains) ever overtake a just-above-critical start
(`T_i=2.4`, which begins with larger domains)?

**What we built.** A small, fully reproducible simulator of this system
(conserved-density *Kawasaki* dynamics), in which every number is traceable to
the exact code, configuration, and random seed that produced it. It was
validated hard before any physics claim: it conserves the right quantity exactly,
its energetics match a brute-force calculation, and on a tiny lattice it
reproduces the *exact* textbook equilibrium distribution.

**What we found — simulation at realistic sizes.** Domains grow by the expected
diffusive law (size ∝ time^(1/3)) across every size tested. For the headline
comparison at 128×128, the colder start leads the entire time and is never
overtaken. We then checked whether that lead is merely a *head start* (the colder
system simply begins with bigger domains) hiding a hidden speed advantage for the
hotter one. After mathematically subtracting the head start, the hotter system is
*still* not faster — equal or slower by every measure, with proper statistical
controls (bootstrap confidence intervals, false-discovery-rate correction, and a
verified ability to detect an effect had one existed). **No Mpemba inversion.**

**What we found — exact theory at tiny size.** On a 4×4 lattice we solved the
dynamics *exactly* by diagonalising the full 12,870-state transition matrix (no
simulation noise). The spectral theory of the Mpemba effect says a faster
relaxer overlaps *less* with the slowest-decaying mode. We found the hotter start
consistently overlaps *more* — the opposite of Mpemba — across bath temperatures.
The exact calculation reproduces the simulator's measured relaxation time to
within 1%, and it *predicts* the no-inversion result later seen at large size.

**Bottom line.** Two independent methods — exact diagonalisation on a tiny
lattice and a statistically controlled simulation at a realistic size — **agree:
there is no Mpemba-like inversion in this system at the conditions studied.** A
hotter start does not order faster here. This is a clean, well-controlled
*negative* result, which is as scientifically valuable as a positive one would
have been. (Scope: one operational point — `M=0`, `T_f=0.6 T_c`, the designated
primary `T_i` pair; the wider grid is set up but deferred.)

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
  lattice.py        # lattice construction, energy, magnetisation, ΔE
  dynamics.py       # Kawasaki NN-exchange kinetic kernel + non-local prep sampler
  observables.py    # E, M, C(r,t), S(k,t), L_C/L_S/L_E, k*, interface density, P(A,t)
  protocols.py      # equilibrate → quench → track (CoarseningTrajectory, S(k,t))
  equilibration.py  # autocorrelation time + independent-chains equilibration gate, E_inf
  analysis.py       # offset fit, difference + offset-corrected bootstrap, FDR, exponents
  spectral.py       # exact-diagonalisation spectral tier (transition matrix, slow mode)
  io.py             # YAML config, append-only results, output schema
  provenance.py     # per-run manifest: config hash, git commit, seed, env, checksums
  rng.py            # seeding utilities
  cli.py            # `kawasaki-run` entry point
configs/            # YAML run configs + the pre-registration (preregistration_m5.yaml)
scripts/            # milestone drivers: milestone1_static … milestone5_crossing, milestone_spectral
tests/              # validation-gate + analysis tests (70; must pass before any claim)
results/            # append-only run outputs, each with a manifest
logbook/            # dated, append-only human logbook (+ M5 protocol amendment)
docs/               # physics notes, output schema
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

## Milestones & verdicts (technical summary)

**Milestones:** M1 (static phase separation), M2 (single-quench benchmark), and
M3 (coarsening-law assessment across `N=32/64/128`) are complete — see `results/`
and `logbook/`. M2/M3 confirm diffusive coarsening: `L = R₀ + (λt)^{1/3}` fits
`L_C`, `L_S`, `L_E` to R² > 0.99 at every size, and `(L_S−R₀)/N` vs `t/N³`
collapses across sizes once the offset is controlled. The free exponent is
stated as illustrative (≈ 1/3, not precision-pinned at these sizes, per the task
card). M4 (initial-temperature sweep) adds an independent-chains equilibration
gate (all five `T_i` legs validated; the earlier `T_i=2.4` flag was a
correlated-sampling artifact) and the common-protocol sweep: initial correlation
length rises sharply toward `T_c`, and the `L_S(t)` curves are strictly ordered
with no raw crossing.

**Milestone 5 (pre-registered crossing search, primary pair `T_i`=10 vs 2.4,
`T_f`=0.6 `T_c`, N=128) — verdict: _no supported inversion_.** No raw overtaking;
the offset-corrected difference `(L−R₀)_hot − (L−R₀)_cold` is null for `L_S`
(equal coarsening rate) and negative for `L_C`/`L_E` (cold stays ahead) — it
favours the hot leg in no estimator, with BH-FDR control and resolving power
confirmed (not variance-limited). A clean, controlled deflationary result.

**Spectral tier (4×4 exact diagonalisation) — _no spectral Mpemba_.** The exact
local-Kawasaki transition matrix on the 12870-state M=0 sector gives a
slow-mode overlap `a₂(T_i)` that is monotone with no zero-crossing across the
`T_f` scan: the hotter prep always has *more* slow-mode overlap (relaxes slower,
anti-Mpemba). The spectral gap is validated against the simulated autocorrelation
(τ_int 955 vs predicted 951, ratio 1.00). Both tiers agree — the spectral picture
*predicts* the N=128 no-inversion. The full `(T_i,T_f,N)` grid is deferred (see
the M5 logbook + amendment).

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

## Acknowledgements

This project was inspired by **"Simulating and understanding phase change"**, a
3Blue1Brown guest video by **Vilas Winstein** (Spectral Collective) — deriving
the Boltzmann distribution, defining temperature, and simulating a liquid/vapour
(lattice-gas) phase change with Monte-Carlo dynamics:
https://www.youtube.com/watch?v=itRV2jEtV8Q

The video's lattice model, Boltzmann sampling, and phase-change picture are the
foundation this project builds on. The key technical difference: the video uses
**Glauber** (non-conserved, single-spin-flip) dynamics, whereas this project
studies the **Kawasaki** (conserved-order-parameter, exchange) variant and its
coarsening / Mpemba behaviour — and a companion Spectral Collective video
analyses the mean-field phase change. Home page: https://www.3blue1brown.com

## Citation

If you use this software, please cite it (see `CITATION.cff`). Release `v1.0.0`
is archived to Zenodo:

> Kawasaki-2D contributors (2026). *kawasaki2d: 2D Kawasaki–Ising coarsening and
> the operational boundary of the Mpemba effect* (v1.0.0). Zenodo.
> https://doi.org/10.5281/zenodo.XXXXXXX

(The DOI placeholder is replaced with the minted Zenodo DOI on release — see
`RELEASE.md`.)

## Licence

Code: MIT (`LICENSE`). Data and figures: CC-BY-4.0 (`LICENSE-DATA`).

