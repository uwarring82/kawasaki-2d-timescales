# 2026-06-24 — Review response (v1.0.0 review round)

**Author:** review-response session · **Software:** v1.0.0 → (unreleased patch)
**Reviews:** `reviews/2026-06-24-codex.md`, `reviews/2026-06-24-kimi.md`
(two independent adversarial passes against tag `v1.0.0`, commit `32bbcc6`).

## Outcome of the round

Both reviews independently reach the same conclusion: **no blocker**; the scoped
`no_supported_inversion` verdict (primary pair `T_i`=10 vs 2.4, `T_f`=0.6 `T_c`,
`N=128`) stands. Both reproduced the engine (70 tests, default + no-Numba), the
spectral probe, and the pre-registration commit order. Both raised the **same
major (non-verdict-flipping) caveat** — the one I had flagged in the review brief
(§4.1, offset-correction self-reference).

## The caveat, verified by the author

Under the **pre-registered fixed-1/3** offset model, the offset-corrected
difference `D = (L−R₀)_hot − (L−R₀)_cold` is null for `L_S` and negative for
`L_C`/`L_E` — no estimator favours hot. Under a **free-exponent** offset re-fit,
`L_S` flips to a significant **positive** value while `L_C`/`L_E` stay negative.

I re-derived this independently before accepting it, then made it reproducible:

| model | `L_C` D | `L_S` D | `L_E` D | two-of-three hot inversion |
|---|---:|---:|---:|:---:|
| fixed 0.30 | −0.791* | −0.013 | −1.532* | no |
| fixed 1/3  | −0.742* | −0.019 | −1.453* | no |
| fixed 0.36 | −0.710* | −0.023 | −1.401* | no |
| **free**   | −1.408* | **+0.401*** | −1.568* | **no** |

(`*` = BH-FDR-significant late difference; `results/m5_offset_sensitivity_v1/`.)

**Why the verdict survives:** the four-outcome scheme awards an inversion only on
a **two-of-three** agreement in the hot direction. Under no offset model do ≥2
estimators show a significant positive `D`. The free-exponent `L_S` flip is a
single estimator. So `no_supported_inversion` is robust; what was *not* robust was
the auxiliary wording "no estimator favours hot", which is true only for the
fixed-1/3 model.

**Why this is expected:** fitting a free exponent per leg lets each leg trade
amplitude against offset over a finite, offset-dominated window; `L_S` carries a
large offset (`R₀≈19.5`) so its residual is the most sensitive to that trade.
This is exactly the offset-correction self-reference the brief warned about — the
pre-registered fix (exponent locked at 1/3) is the principled choice; the
free-exponent run quantifies the sensitivity.

## Actions taken

1. **Code:** added an `exponent` parameter to
   `analysis.offset_corrected_difference_bootstrap` (default 1/3 = pre-registered,
   `None` = free), NaN-safe for `L_E`; `+1` test; the fixed-1/3 default reproduces
   the released verdict exactly (no regression).
2. **Artifact:** `scripts/m5_offset_sensitivity.py` →
   `results/m5_offset_sensitivity_v1/` (manifest `dirty:false`) makes the caveat a
   committed, regenerable result (table + figure).
3. **Wording (README, CHANGELOG):** qualified the C4 claim to the fixed-1/3 model
   and stated the free-exponent `L_S` behaviour explicitly; kept the free exponent
   "illustrative" (C2); softened "spectral predicts coarsening" to *qualitative
   consistency* (C5).
4. **Records:** both reviews committed under `reviews/`; the Kimi spectral rerun
   (redundant with Codex's committed reproduction) was dropped in favour of the
   offset-sensitivity artifact.

## Not changed (and why)

- The **verdict** (`no_supported_inversion`) and the released `m5_verdict_v1` are
  unchanged — the fixed-1/3 analysis is the confirmatory, pre-registered test.
- The `v1.0.0` tag/commit is **not** rewritten (it is published). These
  clarifications are an unreleased patch (candidate `v1.0.1`).

## Accepted limitations carried forward

- Single operational point; full `(T_i,T_f,N)` grid + across-pairs FDR deferred.
- `L_C`/`L_E` negative `D` is a measurement; "morphology memory" remains an
  interpretation (could be phrased as a cold-faster / anti-Mpemba tendency).
- `4×4` spectral tier is exact but tiny; sparse-Krylov larger-sector extension is
  the natural next hardening step.
