# Task Card — Kawasaki–Ising coarsening and the operational limits of the Mpemba effect

**Repository:** `kawasaki-2d-timescales`
**ID:** _(assign)_ · **Owner:** _(assign)_ · **Version:** 0.4 (freeze candidate) · **Priority:** _(assign)_

### Changelog
- **0.3 → 0.4 (freeze candidate):** qualified the stationary-state claim with sector ergodicity and a detailed-balance-preserving exchange proposal; corrected the BKL statement (a continuous-time kinetic model, not an automatic continuum limit of Metropolis — equivalence must be demonstrated); softened the spectral-tier size claim to be benchmark-gated; specified false-discovery-rate control with a pre-designated primary pair; adjusted two framing phrases to avoid prejudging the late-time scaling and to avoid implying a single universal Mpemba mechanism.
- **0.2 → 0.3:** added a FAIR / provenance / scientific-integrity commitment and an enforceable gate; specified logbook management (human layer vs automated manifest); extended the definition of done with archival, metadata, and traceability requirements.
- **0.1 → 0.2:** corrected the framing to the finite-N-vs-thermodynamic-limit crossover; replaced the binary verdict with a four-outcome scheme; added an equilibration gate and preparation-kernel specification; corrected the time-normalisation gate; relaxed the estimator gate to two-of-three; split Milestone 3; added a `T_f` scan, morphology observables, and the small-lattice diagnosability note.

## Objective

**Primary.** Build a small, reproducible 2D Kawasaki–Ising demonstrator of conserved-order-parameter phase separation, and use it to study quench-dependent relaxation and coarsening timescales.

**Subordinate (operational).** Test the crossover between finite-system, mode-resolved anomalous relaxation and thermodynamic-limit coarsening, asking whether distinct initial equilibrium correlation structures produce an inversion of post-quench coarsening times under the same conserved dynamics — and whether any such inversion constitutes a spectral Mpemba effect or only an initial-condition-dependent coarsening-route inversion.

## Framing and rationale

- On the relaxation side, a widely used mechanistic account is established for finite-state reversible Markov descriptions: anomalous relaxation is governed by the overlap of the initial state with the slowest relaxation mode, the strong version corresponding to a vanishing overlap coefficient (eigenmode / Mpemba-index picture). It is one account among several proposed mechanisms, not a unique reduction of all Mpemba-like phenomena.
- **This picture is not violated by conserved dynamics at finite N.** For every finite lattice, `T_f > 0`, and fixed magnetisation sector, a detailed-balance-preserving nearest-neighbour exchange that is ergodic over the sector gives a reversible chain with a unique canonical stationary state restricted to that sector and a discrete, real relaxation spectrum. A slow-mode-overlap analysis therefore remains formally available, and the equilibrium is reachable in finite time. (The reference implementation must use such a proposal; restricted or "efficient" exchange variants must be checked for ergodicity over the sector and for detailed balance, as some fail to sample the fixed-`M` Boltzmann distribution correctly.)
- What changes is **operational**: the spectral gap closes with system size, and on any realistic budget the accessible post-quench dynamics is the coarsening transient — better described by a growing length scale and ageing/scaling forms (`L(t) ~ t^(1/3)`) than by a few isolated, accessible exponential modes. In the scaling regime both preparations are expected to share the same late-time scaling form, up to non-universal amplitudes, offsets, and finite-size corrections — which is itself one of the things this project tests rather than assumes.
- The contribution is therefore **boundary-mapping**: locating where finite-system spectral relaxation gives way to thermodynamic-limit coarsening. A clean, statistically defensible negative/deflationary result is a valid and complete outcome.
- **Directional hypothesis (to be tested, not assumed).** At `M = 0`, the hottest preparation (`T_i ≈ 10`, near-infinite T) is maximally disordered with small initial correlations; the coldest above-`T_c` preparation (`T_i = 2.4`) carries the largest near-critical `ξ` and so starts "ahead" in domain scale. An inversion requires the hotter preparation to overtake — plausibly because near-critical morphology is slow to reorganise under conserved dynamics. The analysis must test this directional claim explicitly rather than fish for any crossing.

## Diagnosability note (read before scoping verdicts)

A **spectral** diagnosis requires the actual relaxation spectrum and the initial-state overlap with the slowest mode. Exact diagonalisation is straightforward only for very small sectors (e.g. `4×4`); sparse Krylov access to a few slow modes may extend the spectral tier to modest lattices, subject to memory and convergence benchmarks — and reliable eigenvector *overlaps* across several initial distributions are more demanding to converge than the gap alone. The production coarsening sizes (`64² = 4096`, `128² = 16384` sites) are far beyond reach. Consequently:
- The **spectral Mpemba** verdict is a small-lattice consistency probe, not a production deliverable.
- Production-size runs can support only **coarsening-route inversion**, **no inversion**, or **underdetermined**.
- Small-`N` exact diagonalisation has a defined role: test whether the spectral picture *predicts* the qualitative route behaviour seen at larger `N`.

## Working practices — FAIR, provenance, and integrity

The project is run to FAIR principles and to a standard of scientific honesty in which the analysis is protected from its own author's hopes. Three commitments underpin everything below:

- **Reproducibility is the baseline, not the achievement.** Every reported number, figure, and verdict must be regenerable from a recorded `(config, code commit, seed, environment)` tuple. FAIR-Reusability is the formal version of the reproducibility gates already in this card.
- **All runs are reported.** Negative, underdetermined, and discarded runs are recorded and counted, never filtered. The four-outcome verdict scheme is a recording duty, not only a classification.
- **The analysis plan is fixed before the crossing is examined.** The gates in this card constitute a pre-registered analysis plan. Because the central risk in a Mpemba search is talking oneself into a visual crossing, fitting windows, estimator definitions, verdict thresholds, and the declared `(T_i, T_f, N)` grid — including the primary pair — are committed to version control *before* Milestone 5 is run. Deviations are recorded with justification and labelled exploratory.

## Scope (in)

- Square lattice, periodic BCs, `H = -J Σ⟨ij⟩ s_i s_j`, `J = k_B = 1`; `T_c = 2/ln(1+√2) ≈ 2.269`.
- Nearest-neighbour Kawasaki exchange (post-quench kinetic kernel) with Metropolis acceptance; exact conservation of `M`.
- Quench protocol: equilibrate at `T_i > T_c` → quench at `t = 0` to common `T_f < T_c` → track coarsening.
- Default grid: `N ∈ {32, 64, 128}` (plus `N = 4` for the spectral tier, larger small lattices only if Krylov benchmarks permit), `M = 0`, `T_i ∈ {2.4, 2.8, 3.5, 5.0, 10.0}`, `T_f` scan `T_f/T_c ∈ {0.5, 0.6, 0.75}` (default `≈ 0.6`).
- Observables: `E(t)`, `E(t) − E_∞`, snapshots, `C(r,t)`, `L(t)`, `S(k,t)`, `k*(t)`, acceptance rate; cluster-area distribution `P(A,t)`; an energy-independent interface-density measure; ensemble mean / variance / CIs.

## Non-goals (out of scope)

- Full fluid hydrodynamics / momentum transport (pure Kawasaki is the diffusive conserved sector only).
- Using a finite-volume equilibrium-distance measure as the *sole* diagnostic during the pre-saturation coarsening regime. The fixed-`N` equilibrium energy `E_∞(N, T_f, M)` is a legitimate reference if independently estimated or sampled long enough — but its dependence on `N`, `T_f`, and simulation horizon must be reported, and it must not be the only diagnostic.
- Claiming a new universal scaling law or exponent for the Mpemba crossing.
- Promising a spectral Mpemba demonstration at coarsening sizes (see diagnosability note).
- Single-trajectory or snapshot-level "evidence" of an effect.

## Deliverables

- [ ] Repository skeleton (`src/kawasaki2d/{lattice,dynamics,observables,protocols,analysis,io}.py`, `configs/`, `scripts/`, `tests/`, `notebooks/`, `results/`, `logbook/`).
- [ ] **Milestone 1 — static phase separation:** snapshots + basic observables above and below `T_c`.
- [ ] **Milestone 2 — single-quench benchmark:** random `M = 0` → `T_f`, with `E(t)`, `L(t)`, evolving `S(k,t)`.
- [ ] **Milestone 3 — coarsening-law assessment:** see split criteria below.
- [ ] **Milestone 4 — initial-temperature sweep:** ensemble-averaged post-quench trajectories for the `T_i` grid at common `T_f`, on validated equilibrated initial states.
- [ ] **Milestone 5 — boundary test:** crossing search with the full gates below; explicit four-outcome verdict, with uncertainty bands, finite-size checks, and (where feasible) the small-`N` spectral consistency probe.

## Acceptance criteria

### Validation gate (must pass before any physics claim)
- [ ] Exact conservation of `M` after every proposed move (test).
- [ ] Local `ΔE` verified against brute-force total-energy differences on small lattices (test).
- [ ] Correct Metropolis acceptance and fixed-`T` equilibrium behaviour (test).
- [ ] Ergodicity/detailed-balance check on the exchange proposal: confirm the implemented nearest-neighbour exchange connects the fixed-`M` sector and reproduces the canonical distribution on a small lattice (test).
- [ ] Bitwise reproducibility under fixed seeds (test).
- [ ] Ensemble averaging over many independent realisations; finite-size comparison across `N = 32, 64, 128`.

### FAIR & integrity gate

**FAIR**
- [ ] **Findable:** version-controlled with semantic versioning; each release tagged and archived to a persistent identifier (Zenodo DOI); machine-readable metadata (`CITATION.cff` / `codemeta.json`) with keywords and description.
- [ ] **Accessible:** open repository plus archived snapshot under a stated open licence (code: OSI-approved; data and figures: CC-BY or equivalent); retrievable over standard protocols; metadata persists independently of any large data artefacts.
- [ ] **Interoperable:** open, non-proprietary formats throughout (YAML configs; CSV/Parquet or HDF5/NetCDF for numerical data; PNG/SVG figures; Markdown docs); a documented output schema; units stated (`J = k_B = 1`) and carried in data headers or sidecars.
- [ ] **Reusable:** pinned environment (`pyproject.toml` + lockfile, or container/conda spec, including the RNG library and version); README sufficient to reproduce; every artefact carries its provenance (below).

**Provenance and reproducibility**
- [ ] Each run emits a machine-generated manifest: config hash, git commit, seed(s), environment fingerprint, timestamp, and output checksums. No result enters `results/` without one.
- [ ] Results are append-only; corrections are new dated entries, never silent overwrites of prior outputs.
- [ ] Bug-impact protocol: when a defect is found, the manifest and logbook identify which prior results it invalidates; affected results are re-run or struck and labelled — never quietly patched.

**Logbook (human layer, complementary to the automated manifest)**
- [ ] A chronological, version-controlled logbook (`logbook/` dated entries, or `LOGBOOK.md`), append-only by convention. Each entry is dated and attributed and, where applicable, linked to the run ID / config hash / commit it concerns.
- [ ] Entries record intent, observations, decisions, surprises, dead-ends, parameter changes, and discovered bugs with their effect on earlier results — the judgement and context the automated manifest cannot capture.
- [ ] The writeup distinguishes pre-registered (confirmatory) analyses from exploratory/post-hoc ones, and every figure and number names the `(config, commit, seed)` that produced it.

### Equilibration gate (initial-state convergence before the quench)
- [ ] For each `T_i`, define and justify `t_eq`. Verify either (a) `C(r, t_eq)` saturates / is consistent with the known correlation form at that `T_i`, or (b) two independent equilibration runs give statistically indistinguishable energy distributions and `C(r)`.
- [ ] Treat `T_i = 2.4` (near `T_c`) explicitly: confirm critical slowing down has not left residual memory of the random start.
- [ ] **Preparation kernel:** local Kawasaki is the baseline preparation. A faster sampler may be used *only* if it respects the fixed-`M` sector and reproduces the same restricted canonical equilibrium — non-local (global) opposite-spin exchange is the natural choice; cluster algorithms (Wolff/Swendsen–Wang) do **not** conserve `M` and are excluded. Validate that prep and baseline give indistinguishable equal-time observables. **Record the preparation kernel; never conflate it with the post-quench kinetic kernel (which must be local Kawasaki).**

### Mpemba-claim gate (specific to the subordinate objective)
- [ ] **Operational reference state** defined explicitly (excess interfacial energy, a fixed late-time configuration, or a scaling-collapsed observable). If `E_∞(N,T_f,M)` is used, report its `N`/horizon dependence.
- [ ] **Scaling-collapse with offset control:** check whether the offset `R₀` in `R(t) ≈ R₀ + (λt)^(1/3)` is itself preparation-dependent. Attempt collapse after subtracting an independently extracted `R₀(T_i)`. A crossing that disappears under `R₀`-correction is a route/offset artefact, not an asymptotic inversion.
- [ ] **Time-normalisation:** physical time is defined by *attempted* nearest-neighbour bond updates — one sweep = `N²` attempts, independent of acceptance. Accepted-move count is a diagnostic only and is illegitimate for cross-preparation comparison (it applies a preparation- and temperature-dependent clock). A continuous-time (BKL) implementation accumulates time from explicitly specified transition rates — advancing by the current configuration's total escape rate, not by counting accepted events — and defines a continuous-time kinetic model. It is not automatically identical to discrete attempted-update Metropolis dynamics; equivalence of observables and time calibration must be demonstrated at overlapping parameters before results are pooled.
- [ ] **Length-estimator agreement (two-of-three):** at least two of `L_C`, `L_S`, `L_E` show consistent ordering and compatible late-time scaling within stated estimator-specific systematics. `E − E_∞ ∝ 1/L` is a late-stage interfacial relation (sensitive to interfacial width, roughness, pinning), not a general identity, so disagreement is a diagnostic outcome to analyse — not automatic code failure.
- [ ] **Difference-bootstrap:** report a crossing only if the bootstrap CI on `L_hot(t) − L_cold(t)` changes sign with the CI excluding zero on both sides of the crossing.
- [ ] **Seed budget:** run a pilot to estimate the per-realisation variance of `L` at the target `t`; size the ensemble so the standard error on the difference is a stated fraction of the expected signal (fraction set by the desired power to detect the sign change, not hardcoded). Expect O(10²–10³); if variance-limited, the verdict is **underdetermined**, not "no effect".
- [ ] **Multiple comparisons:** apply false-discovery-rate control (Benjamini–Hochberg) across the declared `(T_i, T_f, N)` grid, with the per-pair crossing test reduced to a well-defined significance quantity for the procedure to act on, and the primary `(T_i^hot, T_i^cold)` pair — at the default `T_f` and largest `N` — designated before production runs.
- [ ] **Morphology check:** use `P(A,t)` and the energy-independent interface density to test whether a "hotter relaxes faster" signal is driven by different initial domain-size distributions (a route inversion) rather than slow-mode suppression.

### Reporting gate — four outcomes
- [ ] **Spectral Mpemba effect** — a robust inversion tied to reduced overlap with the slowest relevant mode (or an equivalent finite-state diagnostic). Awardable only at the small-`N` spectral tier.
- [ ] **Coarsening-route inversion** — a robust crossing in a coarsening observable, explained by initial morphology, amplitude, lag, or ageing rather than demonstrated slow-mode suppression.
- [ ] **No supported inversion** — no crossing survives uncertainty, estimator, size, offset, and time-origin checks.
- [ ] **Underdetermined** — statistics insufficient to separate the above; reported honestly as a power limitation.

## Milestone 3 criteria (split)
- **Demonstrator:** show a pre-saturation window compatible with `L(t) ~ t^(1/3)`, including local effective exponents and sensitivity to the lower fitting cutoff.
- **Robustness:** compare `N = 32, 64, 128`, show the candidate window lies sufficiently below the equilibrium domain scale, and collapse against a finite-size scaling variable only if data quality warrants. (Conserved-dynamics studies report exponents near 1/3 across sizes ~16–128, with finite-size effects entering near the equilibrium domain scale.)

## Performance notes
- `N = 64/128` is adequate for the demonstrator and the crossing search but thin for a precision exponent — use the robustness criterion, or state the exponent as illustrative.
- At `T_f ≈ 0.6 T_c` acceptance is low and interface-dominated; the acceptance-rate diagnostic is the right early signal.
- **BKL / rejection-free is not a transparent acceleration.** It defines a continuous-time kinetic model that must be validated against attempted-update Metropolis at overlapping parameters (observables and time calibration), and introduced only after the baseline kernel passes all physics and reproducibility tests.
- Inner loop in a JIT/compiled kernel (e.g. Numba); conserved exchange does not vectorise as cleanly as Glauber checkerboard.

## References (prior art — consult before claiming novelty)
- Review (primary overview) — *Speedups in nonequilibrium thermal relaxation: Mpemba and related effects*, arXiv:2502.01758 (2025). Frames Mpemba-type effects across Markovian, kinetic, and phase-transition settings; read the phase-transition section.
- Lu & Raz, *PNAS* (2017) — Markovian Mpemba effect; slow-mode framework.
- Klich, Raz, Hirschberg & Kessler, *PRX* (2019) — strong Mpemba effect and Mpemba index.
- Pemartín, Mompó, Lasanta, Martín-Mayor & Salas, *Phys. Rev. E* **104**, 044114 (2021) — Glauber Ising quench; slow domain growth aiding fast evolution (closest prior art; non-conserved; a "coarsening-route" precedent).
- Das & Vadakkayil, *Phys. Chem. Chem. Phys.* (2021) — hotter paramagnet ordering faster.
- Ghosh, Pathak, Chatterjee & Das, arXiv:2407.06954 (2024) — metastability vs critical-fluctuation routes after quench; scaling.
- Tartaglia, Cugliandolo & Picco, arXiv:1805.05775 (2018) — 2D Ising, local and nonlocal Kawasaki coarsening + percolation; observable/`L(t)` conventions and coexisting geometric lengths.
- Majumder & Das, arXiv:1101.4524 (2011) — diffusive coarsening, finite-size scaling, weak size effects, `R₀ + (λt)^(1/3)`.
- Quantum cross-link — *The quantum Mpemba effects*, Nature Reviews Physics (2025).

## Definition of done
Repository runs end-to-end from config; all validation, FAIR/integrity, and equilibration gates pass in CI; Milestones 1–5 produced; Milestone 5 carries an explicit four-outcome verdict with uncertainty bands, finite-size checks, offset control, and the small-`N` spectral probe where feasible; the repository is archived to a persistent identifier with machine-readable metadata and an open licence; the logbook is complete, and every reported figure and number is traceable to its generating `(config, commit, seed)`.
