# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

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

[1.0.0]: https://github.com/uwarring82/kawasaki-2d-timescales/releases/tag/v1.0.0
