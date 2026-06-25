# 2026-06-25 — Milestone 6: grid crossing search

**Author:** grid session · **Software:** v1.0.0+ (unreleased)
**Runs:** `results/m6_grid_sweep_v1/` (ensembles), `results/m6_grid_verdict_v1/`
(pairwise verdict) — each with its own manifest.
**Config:** `configs/grid_m6.yaml`. Resolves the across-pairs FDR deferred in the
M5 amendment (A2), for `N ∈ {32, 64}` (the full `N=128` grid remains
compute-deferred; the `N=128` primary pair is the M5 anchor).

## Design

For each `(N, T_f)` cell, an ensemble (24 realisations) was run for every
`T_i ∈ {2.4, 2.8, 3.5, 5.0, 10.0}` at `T_f/T_c ∈ {0.5, 0.6, 0.75}`, prep budgets
reused from the N=64 equilibration gate (conservative). Every hot>cold pair is
tested for a **directional** offset-corrected inversion (hotter overtakes colder
after the head-start `R₀` is removed), fixed-1/3 model, with a **per-pair**
saturation window (both legs below `0.40 N`) and a one-sided "hot ahead" bootstrap
p-value per estimator. Per-pair significance = the 2nd-smallest of the three
one-sided p's (the two-of-three level); **BH-FDR across the whole grid** of
per-pair tests.

## Result — no supported inversion across the grid

- **48 pairs tested over 6 `(N, T_f)` cells; 12 pairs skipped** (no pre-saturation
  window — near-critical `T_i=2.4` legs at small `N` / high `T_f` already span the
  lattice; a finite-size effect, recorded not forced).
- **0 pairs meet the two-of-three hot-inversion rule even *before* FDR.** The
  smallest per-pair (2-of-3) p across the entire grid is **0.244** (≫ α=0.05);
  nothing is BH-FDR-supported. **Grid verdict: `no_supported_inversion_across_grid`.**
- The `N=64` cold-reference pairs (`T_i` vs 2.4) at `T_f/T_c = 0.5, 0.6` are
  **negative** (cold ahead) — independently **reproducing the M5 primary-pair
  finding** across the `T_f` scan.

## Honest nuance (the L_S lean, grid-wide)

The single-estimator `L_S` offset-corrected difference leans **positive** in 37 of
48 pairs (range −0.80…+1.03), and 10 pairs have ≥1 estimator with a final-point
95% CI excluding zero on the hot side — almost all `L_S`. This is the *same*
`L_S`-specific sensitivity the v1.0.0 review round surfaced (under the fixed-1/3
offset, `L_S` is the most offset-sensitive estimator and can read slightly
hot-ahead), now seen across the grid. It **never** meets the two-of-three rule:
`L_C` and `L_E` do not corroborate it in any cell. The two-of-three rule is
exactly what guards the verdict against this single-estimator artefact — and it
holds everywhere on the grid.

## Interpretation

The no-inversion conclusion is **not** an artefact of the single designated
operational point: across initial temperature, bath temperature, and two system
sizes, no hotter preparation robustly overtakes a colder one under conserved
Kawasaki coarsening. Where a coarsening window exists, the verdict is negative;
where it does not (near-critical cold leg on a small lattice), there is no
pre-saturation regime to test. This is consistent with the spectral tier's
monotone `a₂(T_i)` (no inversion for any pair) and with the M5 primary-pair
verdict.

## Scope still open

- Full `N=128` grid (beyond the primary pair) — compute-deferred.
- 24 realisations/cell is a scan-level ensemble (lower power than the 48-realisation
  primary pair); it resolves the *absence of a two-of-three signal* robustly but a
  subtle single-cell effect below this power could be missed (none indicated).

## Provenance fix (post-review, bug-impact note)

A release reviewer found that the M6 *sweep* driver registered only
`grid_meta.json` in its manifest, omitting `sweeps_N*.npy` and the 90 per-cell
`{E,LC,LS}.npy` arrays — so the central M6 data was committed and git-tracked but
not checksummed in the manifest. This is a provenance-completeness defect, not a
data defect (the arrays are correct and unchanged; the grid verdict is unaffected).

Fix: (1) `scripts/milestone6_grid.py` now registers every output; (2) the already-
committed `results/m6_grid_sweep_v1/manifest.json` was completed post-hoc with the
checksums of the existing (deterministic) data files, preserving its original
generation commit/seed/config and carrying a `notes` line recording the
completion. No result was re-run or changed.
