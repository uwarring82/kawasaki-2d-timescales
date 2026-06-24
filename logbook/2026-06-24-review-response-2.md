# 2026-06-24 — Review response, round 2 (open/unprimed reviews + bug fix)

**Author:** review-response session · **Software:** v1.0.0 → (unreleased patch)
**Reviews this round:** `reviews/2026-06-24-gpt5-codex-v2.md`,
`reviews/2026-06-24-kimi-v2.md` — two independent passes via the **open/unprimed**
brief `TASK-kawasaki-mpemba-review-v2.md` (assessment written before reading prior
material). Round 1 (primed) was `reviews/2026-06-24-codex.md`, `-kimi.md`.

## Outcome

All four reviews converge: **no blocker; the scoped `no_supported_inversion`
verdict stands** for the primary pair (`M=0`, `T_f=0.6 T_c`, `N=128`,
`T_i=10` vs `2.4`). The unprimed pair independently re-derived the round-1
free-exponent `L_S` caveat (already addressed) **and** surfaced new items the
primed reviews missed — the value of the unprimed brief.

## New finding — real software bug (minor, non-verdict-flipping)

**Saturation-guard `IndexError`** (kimi v2). In `milestone5_crossing.py`
`stage_analyse`, the offset-corrected bootstrap was called with full-length
estimator arrays but a `times` vector truncated to `sweeps <= upper`. When the
saturation guard clips the window (`upper < t_max`) the internal `mask = times >
cutoff` no longer matches the array length → `IndexError`. This fires at `N=64`
(the cold leg saturates at `L_S/N≈0.405 > 0.40`) but **not** at the committed
`N=128` (there `upper = t_max`, so it never triggered).

Fix: slice the estimator arrays to the same kept time columns as `times`
(`keep = sweeps <= upper`); and `offset_corrected_difference_bootstrap` now raises
a clear `ValueError` on a time-dimension mismatch instead of a cryptic
`IndexError`. Regression test added (`test_offset_corrected_diff_rejects_time_
length_mismatch`). Verified the `N=128` path is unchanged (`keep` all-True ⇒ slice
is a no-op; `L_S` `D_final=-0.019` reproduces the released verdict).

## Other minors fixed

- **`oc_late_sign` reporting** (GPT-5 Codex v2): the M5 report now sets the
  late-window sign to `0` unless an FDR-significant late point exists. The verdict
  logic already required FDR significance, so this is reporting clarity only.
- **Plain-language "predicts"** (GPT-5 Codex v2): the plain-language README said
  the spectral tier "predicts" the large-`N` result while the technical section
  said *qualitative consistency*; aligned both to the weaker, correct phrasing.
- **Zenodo wording** (both v2 reviews): README/citation now say `v1.0.0` is *to be
  archived* (DOI minting pending) rather than *is archived*, matching reality.

## Independent corroboration retained

The kimi v2 review ran an **independent-seed `N=128`** pipeline (seed 60606, 16
realisations) → `results/review_kimi_m5_verdict_seed2_v2/`: verdict
`no_supported_inversion` again (`L_S` null, `L_C`/`L_E` negative; wider CI at 16
realisations, two-of-three still not met). Plus a deliberate `N=64` finite-size
probe showing the saturation guard engaging. These reviewer reruns (all
manifest-bearing) are committed under `results/review_kimi_*`; the single
crashed, manifest-less `review_kimi_m5_verdict_N064_v1` directory was removed (it
violates the "no result without a manifest" rule — the crash is now covered by a
unit test instead).

## Status

72 tests pass. The committed `v1.0.0` artifacts and verdict are unchanged; this is
an unreleased patch (candidate `v1.0.1`) of a robustness fix + wording
clarifications. Remaining publication items: cut `v1.0.1`, mint the Zenodo DOI,
and (optionally) complete the deferred `(T_i, T_f, N)` grid.
