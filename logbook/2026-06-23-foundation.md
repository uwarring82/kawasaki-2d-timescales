# 2026-06-23 — Foundation: engine, validation gate, FAIR scaffolding, Milestone 1

**Author:** project bootstrap session
**Software version:** v0.1.0
**Scope:** Repository skeleton, core simulation engine, validation-gate tests,
FAIR/provenance scaffolding, pre-registered Milestone-5 analysis plan, and the
Milestone-1 static-phase-separation demonstrator.

## Intent

Stand up a correct, reproducible foundation before any physics claim, per the
task card's rule that the **validation gate must pass first**. Everything in
this entry is infrastructure plus Milestone 1; no Mpemba/crossing physics has
been run.

## Decisions

- **Stack:** Python 3.13, NumPy + SciPy, optional Numba JIT for the inner loop.
  Numba is *optional*: correctness never depends on it (see surprise #1).
- **RNG:** NumPy `Generator(PCG64)` seeded via `SeedSequence`; library + version
  recorded in every manifest. Ensemble realisations use `SeedSequence.spawn`.
- **Move proposal:** uniform site × uniform neighbour direction (symmetric);
  equal-spin proposals count as attempts (`ΔE = 0`) to keep the attempted-update
  clock honest. `ΔE = J(s_i − s_j)(a − b)` derived in `docs/physics.md`.
- **Preparation vs kinetic kernel:** kept strictly separate. Kinetic = local
  Kawasaki Metropolis (always). Preparation = local or non-local opposite-spin
  exchange; both conserve M and (verified) sample the same fixed-M equilibrium.
- **Pre-registration:** `configs/preregistration_m5.yaml` freezes the declared
  grid, primary pair `(T_i^hot=10, T_i^cold=2.4)` at `T_f≈0.6 T_c`, `N=128`,
  estimator definitions, fitting windows, bootstrap/FDR settings, and the
  four-outcome verdict scheme — committed BEFORE any crossing search.

## Validation gate — PASS

52 tests green. Key certifications:
- `M` conserved exactly under both kernels at all T (test_conservation).
- Local `ΔE` matches brute-force total-energy differences over thousands of
  random swaps on `N=4,6,8,16` (test_energy).
- **Detailed balance / ergodicity / acceptance** (test_detailed_balance): on the
  `4×4`, `M=0` sector (12870 configs enumerated exactly), a long local-Kawasaki
  run reproduces the exact canonical energy distribution — max per-level
  probability deviation ≈ 0.001, mean energy within 0.09 SEM. The non-local
  preparation sampler reproduces the *same* equilibrium.
- Bitwise reproducibility under fixed seeds; spawned sub-streams independent.

## Surprises / dead-ends / bugs found

1. **Numba and pure-Python paths are bitwise identical.** Designed in
   deliberately: the random stream is drawn in NumPy and *passed into* the inner
   loop, so the manifest's PCG64 claim is exactly the stream consumed, and the
   JIT/non-JIT paths agree to the bit. Verified by hashing final lattices
   (`KAWASAKI2D_NO_NUMBA=1` toggle). Numba gives ~50× speedup (≈39M attempts/s
   vs ≈0.78M at N=128).
2. **Performance bug found and fixed (pre-commit).** The first non-local
   preparation kernel recomputed `np.argwhere` after every accepted move —
   O(N²) per swap, which hung at N=64. Rewritten to maintain `+`/`−` site-index
   arrays updated in O(1) per swap, with the same pre-generated-random-stream +
   optional-Numba pattern as the local kernel. No prior results were affected
   (none had been recorded).
3. **Editable install quirk.** On this toolchain (homebrew Python 3.13 +
   setuptools), the default `pip install -e .` wrote a bare-path `.pth` that
   `site` did not add to `sys.path`. Worked around with
   `--config-settings editable_mode=compat`; scripts also bootstrap `src/` onto
   the path so they run without an install.

## Milestone 1 — DONE

`scripts/milestone1_static.py` → `results/m1_static_v1/` (config, snapshots,
observables CSV, figures, manifest). N=64, M=0, nonlocal prep 4000 sweeps,
`T ∈ {1.2, 1.8, 2.269, 2.8, 4.0}`. Observed the expected crossover: deep below
`T_c` a single domain holds 99.8% of up-spins (`e/spin ≈ −1.89`); the
correlation length collapses from `L_C ≈ 14.5` (T=1.2) to `0.78` (T=4.0) across
`T_c`, with broken-bond density rising 0.026 → 0.367. Snapshot grid shows clean
two-slab separation → rough interface → critical fractal domains → disorder.

NOTE: the `m1_static_v1` manifest shows `git.commit = None` because it was
generated before the first commit. It will be regenerated post-commit so the
recorded result carries a real commit hash.

## Next

- Milestone 2: single-quench benchmark (E(t), L(t), S(k,t)) — CLI already wired.
- Milestone 3: coarsening-law assessment (`L(t) ~ t^{1/3}`, effective exponents,
  finite-size N=32/64/128).
- Equilibration gate: implement the two-independent-runs and saturation checks
  for each T_i; treat T_i=2.4 (critical slowing down) explicitly.
