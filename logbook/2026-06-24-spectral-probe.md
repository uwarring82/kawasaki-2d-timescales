# 2026-06-24 — Milestone 5 spectral tier: exact-diagonalisation Mpemba probe

**Author:** spectral session · **Software:** v0.1.0
**Run:** `results/m5_spectral_v1/` (own manifest).
**Tier:** 4×4, M=0 — the only tier at which a *spectral* Mpemba verdict is
awardable (task-card diagnosability note).

## Spectral verdict — no spectral Mpemba

Exact local-Kawasaki Metropolis transition matrix on the enumerated 4×4 M=0
sector (12870 states), diagonalised at the bath temperature. Across the
pre-registered `T_f` scan (0.5, 0.6, 0.75 `T_c`):

| `T_f`/`T_c` | slow mode | λ_slow | τ_exp (sw) | a₂(hot=10) | a₂(cold=2.4) | primary Mpemba? |
|---|---|---|---|---|---|---|
| 0.50 | #8 | 0.999936 | 973 | +3.42 | +2.83 | no |
| 0.60 | #8 | 0.999869 | 476 | +1.85 | +1.47 | no |
| 0.75 | #8 | 0.999657 | 182 | +0.98 | +0.66 | no |

`a₂(T_i)` is **monotonically increasing**, with **no zero-crossing** (no strong
Mpemba) and **no interval where |a₂| decreases** (no weak Mpemba for any pair).
At every `T_f`, the hotter preparation has the **larger** slow-mode overlap — it
relaxes **slower**, the anti-Mpemba direction. **Verdict: no spectral Mpemba.**

## The key subtlety — symmetry

The genuinely slowest modes (λ≈0.99997, τ≈2400 sweeps) are the **antisymmetric**
`k≠0` hydrodynamic (conserved-density) modes. Lattice-symmetric Boltzmann initial
distributions have **machine-zero** overlap with them, so they are never excited.
The Mpemba-relevant slow mode is the slowest **symmetric** mode (#8); using λ₂
naively would have given the meaningless `a₂≡0`. `slowest_excited_mode` selects
the slowest mode a symmetric state actually projects onto.

## Gap validation — spectral ↔ dynamics

At `T_f`=0.6 `T_c`, the energy observable overlaps **only** mode #8 (the slowest
symmetric mode), τ_exp = 476 sweeps. With the integrated-autocorrelation
convention τ_int ≈ 2·τ_exp this predicts τ_int = 951 sweeps; the **simulated**
energy autocorrelation (local Kawasaki, 4×4) gives τ_int = **955** sweeps —
**ratio 1.00**. The exact transition matrix reproduces the simulation's dynamics
to <1%, validating both the spectral construction and the kinetic kernel.

## Connection to the coarsening tier (the card's question)

The card asks whether the small-`N` spectral picture *predicts* the qualitative
route behaviour at production sizes. It does: the spectral analysis says the
hotter preparation has monotonically **more** slow-mode overlap (slower
relaxation) — i.e. **no inversion** — which is exactly the
`no_supported_inversion` verdict found at N=128 (`m5_verdict_v1`). Both tiers
agree, by independent methods (exact diagonalisation vs offset-controlled
difference-bootstrap on Monte-Carlo coarsening).

## Four-outcome scheme (spectral tier)

- **Spectral Mpemba effect** — **not** awarded (no weak/strong inversion, robust
  across `T_f`).
- Consistent with **no supported inversion** at coarsening sizes.

## Bottom line

Across both the spectral (exact, 4×4) and coarsening (N=128, statistically
controlled) tiers, this conserved 2D Kawasaki–Ising system shows **no Mpemba-like
inversion** at the studied operational points — a coherent, multi-method,
defensible negative result, and the spectral tier *predicts* the coarsening one.
