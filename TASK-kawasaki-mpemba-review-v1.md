# Task Card — Critical review of the Kawasaki–Ising Mpemba boundary study

**Repository:** `kawasaki-2d-timescales`
**Artifact under review:** release `v1.0.0` (tag `v1.0.0`, commit `32bbcc6`)
**Type:** independent, adversarial scientific + software review
**Version:** review brief v1 · **Owner:** _(assign reviewer)_

---

## 0. Purpose

The project reports a **negative result**: under conserved 2D Kawasaki–Ising
dynamics, at the studied operational point, there is **no Mpemba-like inversion**
— by two independent methods (an offset-controlled coarsening crossing search at
`N=128`, and an exact-diagonalisation spectral probe at `4×4`). A negative result
is only as strong as the controls behind it. **Your job is to try to break it.**
Assume nothing; verify or refute each claim; find the analysis or code decision
that, if wrong, would flip or void the verdict.

This is not a rubber-stamp. A review that finds nothing wrong should say *what it
checked and why each check passed*. A review that finds a flaw should say how it
changes the verdict.

## 1. The claims under review (verbatim, falsifiable)

1. **C1 — Engine correctness.** The simulator exactly conserves `M`; the local
   `ΔE` matches brute-force energy differences; the Metropolis–Kawasaki kernel
   samples the canonical fixed-`M` distribution (verified against exact `4×4`
   enumeration); runs are bitwise reproducible and Numba/pure-Python agree to the
   bit.
2. **C2 — Coarsening law.** `L(t) = R₀ + (λt)^{1/3}` fits `L_C`, `L_S`, `L_E` to
   `R² > 0.99` across `N=32/64/128`; the free exponent is consistent with 1/3 but
   *illustrative* at these sizes; `(L_S−R₀)/N` vs `t/N³` collapses across sizes.
3. **C3 — Equilibration gate.** Initial states at every `T_i` are equilibrated
   (independent-chains mean-stability gate, energy **and** `L_C`); the earlier
   `T_i=2.4` flag was a correlated-sampling artefact, not under-equilibration.
4. **C4 — Coarsening verdict (primary pair `T_i`=10 vs 2.4, `T_f`=0.6 `T_c`,
   `N=128`).** `no_supported_inversion`: no raw overtaking; the offset-corrected
   difference `(L−R₀)_hot−(L−R₀)_cold` favours the hot leg in **no** estimator
   (`L_S` null; `L_C`,`L_E` negative), under BH-FDR with confirmed resolving
   power (not variance-limited).
5. **C5 — Spectral verdict (`4×4`).** `no_spectral_mpemba`: the slow-mode overlap
   `a₂(T_i)` is monotone, no zero-crossing, no `|a₂|` decrease, across the `T_f`
   scan; the hotter prep always has *more* slow-mode overlap (relaxes slower).
   The spectral gap reproduces the simulated autocorrelation (ratio ≈ 1.00), and
   the spectral picture *predicts* the `N=128` no-inversion.
6. **C6 — Integrity.** The analysis plan (`configs/preregistration_m5.yaml`) was
   committed *before* the Milestone-5 crossing was examined; deviations are
   recorded as a dated amendment *before* analysis; results are append-only and
   every number is traceable to `(config, commit, seed, environment)`.

## 2. How to reproduce (do this first)

```bash
git clone https://github.com/uwarring82/kawasaki-2d-timescales
cd kawasaki-2d-timescales && git checkout v1.0.0
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,plot]"        # add ",accel" for the Numba kernel
pytest                               # expect 70 passing (the validation gate)
```

Note: on some toolchains the editable install is honoured inconsistently; if
imports fail, run scripts with `PYTHONPATH=src python scripts/...`. Drivers:
`scripts/milestone{1..5}_*.py`, `scripts/milestone_spectral.py` (M5 staged:
`--gate`, `--sweep`, `--analyse`). Compute guide: spectral probe ~seconds; M3/M4
~minutes; the M5 `N=128` sweep ~10–15 min (96 realisations). Every run writes
`results/<run_id>/manifest.json`; the human narrative is in `logbook/`.

Reproducibility spot-check: pick any committed figure/number, open its run's
`manifest.json`, check out the recorded commit, re-run with the recorded config,
and confirm you regenerate it. Verify Numba/pure-Python agreement by running the
suite once normally and once with `KAWASAKI2D_NO_NUMBA=1`.

## 3. Review dimensions and checklist

### A. Physics correctness
- [ ] Re-derive `ΔE = J(s_i−s_j)(a−b)` for NN exchange; confirm it matches
      `docs/physics.md` and `tests/test_energy.py`.
- [ ] Confirm the time unit (1 sweep = `N²` *attempted* updates) is applied
      uniformly and that accepted-move counts are never used cross-preparation.
- [ ] Confirm the non-local preparation kernel and the local kinetic kernel are
      never conflated; the post-quench kinetics is local Kawasaki everywhere.
- [ ] Sanity-check `T_c`, ground-state energy, and the `T_i=10 ⇒ ⟨e⟩≈−2/T`
      high-temperature limit.

### B. Statistical methodology (the heart of C4)
- [ ] **Offset-correction circularity.** `R₀` is fitted with the exponent
      *fixed at 1/3* (linear fit of `L` vs `t^{1/3}`). Does subtracting this `R₀`
      bias the offset-corrected difference toward the null when the true exponent
      ≠ 1/3? Re-run the offset fit with a free exponent and with neighbouring
      fixed exponents (0.30, 0.36) and check the C4 verdict is unchanged.
- [ ] **Bootstrap validity.** `offset_corrected_difference_bootstrap` re-fits
      `R₀` inside each resample. Confirm the CIs and the two-sided p-values are
      computed correctly and that `n_boot=5000` is sufficient (CI stability).
- [ ] **FDR scope.** The run applies BH-FDR across *time points* of the primary
      pair (amendment A2), not the full grid. Judge whether that is an adequate
      multiple-comparison control for the headline claim, and whether deferring
      the across-pairs FDR weakens it.
- [ ] **Resolving power.** Verify the minimum-detectable-difference framing
      (`L_S` MDD ≈ 0.47, `L_C`/`L_E` resolved). Is "not variance-limited"
      justified, or is `n=48` too small for a credible null on `L_S`?

### C. Interpretation (the subtlest part)
- [ ] `L_S` gives `D≈0` (equal rate) but `L_C` and `L_E` give `D<0` (cold ahead
      *after* offset removal). The card reads this as "no inversion + morphology
      memory". **Challenge it:** is `D<0` better described as an *inverse* effect
      (cold genuinely faster), and does that matter for the headline? Is the
      two-of-three rule (here: two estimators agreeing on the *unhypothesised*
      sign) being applied in spirit or just in letter?
- [ ] Is calling the result "no Mpemba" fair given the hypothesis was specifically
      *hot overtakes*? (It was — `D>0` in no estimator — but state your judgement.)

### D. Spectral tier (C5)
- [ ] Reproduce the symmetry argument: the truly-slowest modes are antisymmetric
      (`k≠0`) and have machine-zero overlap with symmetric Boltzmann ICs; the
      Mpemba-relevant mode is the slowest *symmetric* one (`#8`). Is
      `slowest_excited_mode` (overlap threshold `1e-6`, probe temperature)
      robust? Does the chosen mode change with `T_probe` or `k`?
- [ ] Check the `a₂` sign/normalisation conventions and that "monotone, no
      zero-crossing" is not an artefact of the sign fix.
- [ ] Verify the gap validation: spectral `τ_exp` → predicted `τ_int = 2τ_exp` vs
      simulated. Is the factor of 2 (integrated-vs-exponential) correct, or is the
      1.00 ratio a coincidence of conventions?

### E. Equilibration & E_inf
- [ ] The equilibration criterion was changed from KS to *mean stability* after
      observing K=30 KS noise (M4 logbook). Is this a legitimate QC change (it is
      *not* the hypothesis test), or does it risk passing under-equilibrated
      states? Stress it at `T_i=2.4`.
- [ ] `E_∞(N,T_f)` is a horizon-limited estimate via the non-local kernel below
      `T_c`. Does the non-local sampler truly reach the same equilibrium in the
      budget used? `L_E` depends sensitively on `E_∞`; quantify the sensitivity.

### F. Reproducibility / FAIR / integrity (C6)
- [ ] Confirm via `git log` that `configs/preregistration_m5.yaml` predates the
      M5 machinery and results, and that the amendment predates the M5 analysis.
- [ ] Confirm every `results/*/manifest.json` records a commit and `dirty:false`,
      and that at least one result regenerates bitwise.
- [ ] Confirm append-only discipline (no silent overwrites in history).

## 4. Author-flagged weak points (start here — these are where I'd attack)

In the spirit of honest review, the strongest places to push:

1. **Offset-correction is partly self-referential** (B above): `R₀` assumes 1/3.
   The headline rests on the offset-corrected difference; if the exponent
   genuinely deviates from 1/3 at `N=128`, the `R₀` and hence `D` shift. The
   free-exponent scan is wide here ("illustrative"), so this is the softest joint.
2. **One operational point.** C4 is a single `(T_i pair, T_f, N)`. The full grid
   (other pairs, `T_f` scan at coarsening sizes, across-pairs FDR) is *deferred*,
   not done. The directional hypothesis could still hold elsewhere; the headline
   must not be over-generalised.
3. **`L_C`/`L_E` "morphology memory" is an interpretation, not a measurement.**
   The negative offset-corrected `D` in those estimators is real and FDR-robust;
   the *explanation* (initial-morphology memory vs a genuine inverse effect) is
   not independently demonstrated.
4. **`n=48` realisations** at `N=128` is modest; the `L_S` null leans on the MDD
   argument. A larger ensemble would harden (or break) the null.
5. **Spectral↔coarsening "agreement" is qualitative.** Both say "no inversion",
   but the spectral relaxation-distance and the coarsening length-crossing are
   different observables; the "predicts" claim is a consistency statement, not a
   derivation.
6. **`4×4` is tiny.** The spectral tier is exact but far from the thermodynamic
   limit; the sparse-Krylov extension to larger sectors was not attempted.

## 5. Known limitations & scope (already stated by the authors)

- Exponent reported as *illustrative*, not precision-pinned (task card; M3).
- Spectral Mpemba verdict awardable only at the small-`N` tier (diagnosability
  note); not a production deliverable.
- Full grid + `T_f` scan + secondary pairs + sparse-Krylov spectral tier:
  deferred (M5 amendment A2), documented not omitted.
- Zenodo DOI: placeholder pending minting (`RELEASE.md`).

## 6. How to record findings

For each claim C1–C6 assign one of:
- **Confirmed** — independently reproduced / verified; note what you ran.
- **Confirmed with caveats** — holds, but with a stated limitation.
- **Challenged** — a specific concern that, if correct, changes the claim;
  give the evidence and the expected effect on the verdict.
- **Refuted** — demonstrably wrong; show the counter-evidence.

Severity for any issue: **blocker** (voids a headline verdict) / **major**
(weakens or rescopes it) / **minor** (clarity, polish, robustness).

Deliver the review as a dated file `reviews/<date>-<reviewer>.md` (append-only,
like the logbook) plus, where useful, a re-run `results/<run_id>/` with its own
manifest so your checks are themselves reproducible. Open issues/PRs against
specific files; reference commits and line numbers.

## 7. Definition of done (for the review)

- [ ] Each of C1–C6 has a recorded verdict with evidence.
- [ ] Every Section-4 weak point is explicitly addressed (confirmed safe or
      escalated).
- [ ] At least one committed result is regenerated independently.
- [ ] The validation-gate suite passes on the reviewer's machine (or failures are
      reported with environment details).
- [ ] A one-paragraph overall assessment: does the **no-inversion** headline
      stand as scoped? What, if anything, must change before it is publishable?
