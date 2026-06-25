# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [1.0.1] — 2026-06-25

Incorporates four independent adversarial reviews of v1.0.0 (`reviews/2026-06-24-*.md`;
two via the primed v1 brief, two via the open/unprimed v2 brief) plus a
release-readiness review. All converge: **no blocker; the scoped
`no_supported_inversion` headline stands.** This release adds the grid search (M6),
a software-robustness fix, a provenance-completeness fix, and wording
clarifications — not a change to the science. The committed `N=128` verdict is
unaffected by every fix below.

### Fixed
- **M6 sweep manifest provenance** (release review): the grid *sweep* driver had
  registered only `grid_meta.json`, omitting `sweeps_N*.npy` and the 90 per-cell
  `{E,LC,LS}.npy` arrays. `scripts/milestone6_grid.py` now registers every output;
  the committed `results/m6_grid_sweep_v1/` manifest was completed post-hoc with
  the checksums of the (deterministic) data, preserving its generation provenance
  (data and verdict unchanged).
- **Saturation-guard `IndexError`** in `scripts/milestone5_crossing.py` (found by
  the kimi v2 review): the offset-corrected bootstrap received full-length
  estimator arrays with a truncated `times` when the saturation guard clipped the
  window (`upper < t_max`, e.g. at `N=64`). Now the arrays are sliced to the kept
  time columns, and `analysis.offset_corrected_difference_bootstrap` raises a
  clear `ValueError` on a time-dimension mismatch (regression test added). No
  effect at `N=128` (there `upper = t_max`, so the slice is a no-op).
- **`oc_late_sign` reporting** (found by the GPT-5 Codex v2 review): the M5 report
  now reports a late-window sign of `0` unless there is an FDR-significant late
  point (previously it could show a nonzero sign with no significant point —
  harmless to the verdict, which already required FDR significance, but
  misreadable).

### Added
- **Milestone 6 — grid search** (`scripts/milestone6_grid.py`, `configs/grid_m6.yaml`,
  `results/m6_grid_sweep_v1/`, `results/m6_grid_verdict_v1/`): broadens the
  primary-pair verdict across `T_i × T_f` at `N ∈ {32, 64}`. All hot>cold pairs
  tested for a directional offset-corrected inversion with per-pair saturation
  windows and BH-FDR across the grid. **Verdict: no supported inversion across the
  grid** — 0 of 48 tested pairs meet the two-of-three rule (smallest 2-of-3
  p = 0.244); the `N=64` cold-reference pairs reproduce the M5 result. (Full
  `N=128` grid still compute-deferred.)
- `analysis.offset_corrected_difference_bootstrap` gains an `exponent` parameter
  (default `1/3` = pre-registered; `None` = free per-resample linearity scan).
- `scripts/m5_offset_sensitivity.py` and `results/m5_offset_sensitivity_v1/`: a
  reproducible offset-model sensitivity artifact (fixed {0.30, 1/3, 0.36} vs
  free-exponent) for the primary pair. +2 tests (free-exponent, length-guard).

### Changed (wording, prompted by review)
- Qualified the C4 claim: the offset-corrected difference favours the hot leg in
  no estimator **under the pre-registered fixed-1/3 model** (stable across fixed
  0.30–0.36). Under a *free-exponent* stress test `L_S` flips to weakly favour the
  hot leg (`D≈+0.40`, FDR-significant) while `L_C`/`L_E` stay negative — the
  two-of-three hot-inversion rule is met under **no** offset model, so the verdict
  is robust; only the "no estimator favours hot" phrasing was model-dependent.
- Free-exponent growth estimates remain **illustrative** (not all bands centre on
  1/3); the fixed-1/3 law fit is the supported statement.
- "Spectral predicts coarsening" softened to *qualitative consistency* in both the
  technical and the plain-language README sections (two different observables; not
  a derivation).
- Citation/README: release is "to be archived" to Zenodo (DOI minting pending)
  rather than "is archived", until the DOI is minted.
- README plain-language bottom line scoped to the **tested / pre-saturation** grid
  (`N ∈ {32, 64}`, pairs with a coarsening window), noting the 12 skipped
  near-critical pairs (finite-size) and the deferred `N=128` grid (release review).

## [1.0.0] — 2026-06-24

First archival release: a complete, reproducible 2D Kawasaki–Ising coarsening
study with a two-tier (spectral + coarsening) Mpemba boundary test. All five
milestones plus the small-`N` spectral probe are delivered; every reported
number, figure, and verdict is regenerable from a recorded
`(config, commit, seed, environment)` tuple.

### Scientific result
- **No Mpemba-like inversion** in conserved 2D Kawasaki–Ising dynamics at the
  studied operational points, by two independent methods:
  - **Coarsening tier (N=128, primary pair T_i=10 vs 2.4, T_f=0.6 T_c):**
    `no_supported_inversion`. No raw overtaking; the offset-corrected difference
    `(L−R₀)_hot − (L−R₀)_cold` favours the hot leg in no estimator (null for
    `L_S`, negative for `L_C`/`L_E`), under BH-FDR control with confirmed
    resolving power.
  - **Spectral tier (4×4 exact diagonalisation, 12870 states):**
    `no_spectral_mpemba`. The slow-mode overlap `a₂(T_i)` is monotone with no
    zero-crossing across the `T_f` scan (hotter ⇒ more slow-mode overlap ⇒
    slower); the spectral gap matches the simulated autocorrelation (ratio 1.00).
  - The spectral tier **predicts** the coarsening-tier verdict.

### Added
- Core engine (`lattice`, `dynamics`, `observables`, `protocols`, `analysis`,
  `equilibration`, `spectral`, `io`, `provenance`, `rng`, `cli`).
- Local Kawasaki Metropolis kinetic kernel and non-local opposite-spin
  preparation sampler; optional Numba JIT (bitwise-identical to the pure-Python
  path).
- Length estimators `L_C`, `L_S`, `L_E`; structure factor `S(k,t)`; periodic
  cluster `P(A,t)`; energy-independent cluster-number-density morphology.
- Pre-registered analysis: difference-bootstrap, offset-corrected
  difference-bootstrap (R₀ re-fitted per resample), `R₀+(λt)^{1/3}` fit,
  data-driven exponent scan, Benjamini–Hochberg FDR, integrated autocorrelation
  time, independent-chains equilibration gate, seed-budget power.
- Exact-diagonalisation spectral tier (transition matrix, reversible spectrum,
  slowest-excited-mode selection, Mpemba coefficient).
- Milestone drivers M1–M5 + the spectral probe; configs; 70 tests.
- FAIR scaffolding: MIT (code) + CC-BY-4.0 (data/figures), `CITATION.cff`,
  `codemeta.json`, `.zenodo.json`, pinned `requirements.lock`, output-schema and
  physics docs, dated logbook, CI workflow, pre-registration + amendment.

### Provenance note
- Milestone result directories under `results/` were generated during the 0.1.0
  development line; their manifests record the exact generating commit and seed
  (the authoritative provenance). The 1.0.0 release packages them with finalised
  metadata; the code is functionally unchanged (a version-string bump only), so
  re-running at the tagged commit reproduces them bitwise.

### Deferred (documented, not omitted)
- Full `(T_i, T_f, N)` grid with across-pairs FDR, and the `T_f` scan / secondary
  `T_i` pairs at coarsening sizes (compute) — see the M5 protocol amendment.

[1.0.1]: https://github.com/uwarring82/kawasaki-2d-timescales/releases/tag/v1.0.1
[1.0.0]: https://github.com/uwarring82/kawasaki-2d-timescales/releases/tag/v1.0.0
