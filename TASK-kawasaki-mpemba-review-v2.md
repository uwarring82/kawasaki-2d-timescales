# Task Card — Independent scientific review (open brief)

**Repository:** `kawasaki-2d-timescales`
**Artifact under review:** the current `main`, and release `v1.0.0` (tag
`v1.0.0`). Review either; state which commit you assessed.
**Type:** open, unprimed, honest scientific review
**Version:** review brief v2 · **Owner:** _(assign reviewer)_

---

## 0. What we are asking for

An **honest, independent scientific assessment** of this project and its central
result. Read the work, run what you need to, and tell us what you actually
conclude. We are not asking you to confirm anything. Find errors, weak
inferences, overclaims, hidden assumptions, irreproducible steps — or, if the
work withstands scrutiny, say so *and say what you checked to earn that
conclusion*. A well-supported "this is sound as scoped" is as valuable as a
flaw; an unexamined verdict of either kind is not.

This brief deliberately **does not** point you at any suspected weak spots. We
want your unbiased reading. Decide for yourself where this work is most
vulnerable and test there.

## 1. Independence (please read first)

The repository already contains an earlier review brief
(`TASK-kawasaki-mpemba-review-v1.md`) and prior reviews under `reviews/`. We are
not hiding them. But to get an *independent* signal:

- Form and **write down your own assessment first** — your own list of what to
  check, what you found, and your verdict — **before** reading the prior brief or
  reviews.
- Then (optionally) read the prior material and add a short note on where you
  agree, disagree, or found something they missed.

If your independent pass reaches the same concerns as the prior reviews, that is
useful corroboration; if it reaches different ones, that is more useful still.

## 2. What the project asserts (so you know the target)

In plain terms, the project claims: under conserved 2D Kawasaki–Ising dynamics,
at one studied operational point (`M=0`, a bath temperature `T_f=0.6 T_c`, and a
designated initial-temperature pair `T_i=10` vs `2.4`), there is **no Mpemba-like
inversion** — a hotter-prepared system does not order faster than a colder one —
established by two independent methods: a statistically controlled coarsening
crossing search at `N=128`, and an exact-diagonalisation spectral analysis at
`4×4`. Supporting deliverables include a validated simulation engine, a
finite-size coarsening-law study, an equilibration procedure, and a
pre-registered analysis plan.

The project's own statements of its claims, methods, and caveats are in
`README.md`, the task card `TASK-kawasaki-mpemba-boundary-v4.md`, the `logbook/`,
and `configs/preregistration_m5.yaml`. Assess whether those claims are warranted
**at the scope stated** — neither under- nor over-reaching.

## 3. Reproduce before you judge

```bash
git clone https://github.com/uwarring82/kawasaki-2d-timescales
cd kawasaki-2d-timescales            # review main, or: git checkout v1.0.0
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,plot]"         # add ",accel" for the Numba kernel
pytest                               # the validation suite
```

If imports misbehave on your toolchain, run scripts with
`PYTHONPATH=src python scripts/...`. Drivers are in `scripts/`; each run writes
`results/<run_id>/manifest.json` recording `(config, commit, seed, environment)`;
the human narrative is in `logbook/`. Compute guide: the spectral analysis runs
in seconds; the finite-size and temperature studies in minutes; the largest
crossing run is ~10–15 min.

Useful independent checks (choose your own — these are just starting points, not
a checklist of suspicions): regenerate a committed figure or number from its
manifest and confirm it matches; re-run the test suite; vary a seed and see what
moves; probe any analysis choice you think a reasonable scientist might have made
differently.

## 4. Scope of the assessment

Cover whatever you judge relevant. As a non-leading frame, a complete review
usually touches:

- **Physics & modelling** — is the model, its dynamics, and its time/temperature
  bookkeeping correct and standard? Do the validations actually validate?
- **Numerics & statistics** — are the estimators, fits, uncertainty
  quantification, and significance/decision rules sound? Is the headline
  conclusion robust to reasonable alternative analysis choices and to finite
  statistics?
- **Reproducibility & provenance** — can you regenerate the reported results from
  what is recorded? Is the integrity discipline (pre-registration, append-only,
  manifests) real and intact?
- **Claims & scope** — is every stated conclusion supported by the evidence at
  the stated scope? Is anything over- or under-claimed? Are the limitations
  honest and complete?
- **Software** — is the implementation correct, clear, and consistent with what
  the text says it does?

## 5. Recording your review

Deliver an append-only file `reviews/<YYYY-MM-DD>-<reviewer>.md`. Please include:

- **Commit reviewed**, and the commands you ran (so checks are reproducible).
- A **verdict per area** you assessed: *Confirmed* / *Confirmed with caveats* /
  *Challenged* / *Refuted*, each with the evidence.
- A **severity** for any issue: *blocker* (voids a headline conclusion) / *major*
  (weakens or rescopes it) / *minor* (clarity, robustness, polish).
- If you re-ran code, drop the resulting `results/<run_id>/` (with its manifest)
  so your checks are themselves regenerable.
- A **one-paragraph overall assessment**: does the headline conclusion stand as
  scoped? What, if anything, would you require before considering it publishable?
- (Optional, after the above) your comparison to the prior brief/reviews.

## 6. Definition of done (for the review)

- [ ] An independent assessment written before consulting prior reviews.
- [ ] At least one committed result regenerated from its provenance.
- [ ] The validation suite run (pass/fail reported with environment details).
- [ ] A clear overall verdict with evidence, at the scope the project claims.

Thank you for reviewing critically and honestly. The goal is a result that is
correct and correctly scoped — whichever way the evidence points.
