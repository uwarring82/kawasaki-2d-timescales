# 2026-06-23 — Milestone 5 protocol amendment (recorded BEFORE running)

**Author:** M5 session · **Software:** v0.1.0
**Status:** pre-analysis amendment to `configs/preregistration_m5.yaml`. Written
and committed *before* any Milestone-5 crossing analysis is examined, per the
integrity gate ("deviations recorded with justification before the crossing is
examined"). Three items; the locked pre-registration is otherwise unchanged.

## A1 — Primary verdict quantity: offset-corrected difference

The pre-registration's bootstrap quantity is the raw `L_hot(t) − L_cold(t)`; the
pre-registration *also* mandates offset control ("subtract an independently
extracted `R₀(T_i)`; a crossing that disappears under `R₀`-correction is a
route/offset artefact"). We therefore report **both** difference bootstraps and
base the **verdict** on the offset-corrected difference
`D(t) = (L_hot − R₀_hot) − (L_cold − R₀_cold)`, with `R₀` fitted independently
per leg and **re-fitted inside each bootstrap resample** so its uncertainty
propagates. Rationale: the directional hypothesis is about a genuine *rate*
overtake, not the preparation head-start; the raw difference is reported for
completeness and to apply the pre-registered crossing rule verbatim.

## A2 — FDR scope this run

The pre-registration applies Benjamini–Hochberg FDR across the declared
`(T_i, T_f, N)` grid (per-pair test → one significance quantity). Running the
full grid at N=128 is beyond this turn's compute budget. This run executes the
**designated primary pair** (`T_i`=10 vs 2.4, `T_f`=0.6 `T_c`, N=128) at full
rigour and applies BH-FDR at α=0.05 **across the time points** of the crossing
test (the multiple-`t` comparison). The full-grid (across-pairs) FDR is
**deferred**, not omitted silently; the primary pair is the designated
confirmatory test, so the headline verdict does not depend on it.

## A3 — Ensemble size from a pilot

Per the pre-registration's seed-budget provision, the ensemble size is set from a
pilot estimate of the per-realisation variance of `L` at the target time, sized
for power 0.8 at α=0.05 to resolve the expected signal. The achieved power and
SE-to-signal ratio are reported. If the affordable ensemble is variance-limited
relative to the offset-corrected signal, the verdict is **underdetermined**
(not "no effect"), exactly as the pre-registration prescribes.

No other parameter (estimator definitions, fitting window, crossing rule,
bootstrap settings, the four-outcome scheme, or the primary pair) is changed.
