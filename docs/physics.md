# Physics conventions and derivations

`J = k_B = 1` throughout. Spins `s_i ∈ {+1, -1}`. Square lattice `N × N`,
periodic boundary conditions. Energy

```
H = -J Σ_⟨ij⟩ s_i s_j
```

where the sum runs over each nearest-neighbour bond once. On the `N × N` torus
there are `2N²` bonds, so `E ∈ [-2N², +2N²]` and the per-bond energy lies in
`[-1, +1]`.

## Magnetisation and the conserved sector

`M = Σ_i s_i`. Kawasaki dynamics exchanges the values on two sites, so `M` is
conserved by construction. With `n_+` up-spins and `n_- = N² − n_+` down-spins,
`M = n_+ − n_-`. The default working point is `M = 0` (`n_+ = n_- = N²/2`),
which requires `N²` even — true for all even `N`, hence the grid `N ∈ {4, 32, 64, 128}`.

## Nearest-neighbour exchange energy change

Propose exchanging the spins on neighbouring sites `i` and `j`.

- If `s_i == s_j` the exchange leaves the configuration unchanged: `ΔE = 0`. It
  still counts as one *attempted* update (time-normalisation gate).
- If `s_i != s_j`, let
  - `a` = sum of `s` over the neighbours of `i` **excluding** `j`,
  - `b` = sum of `s` over the neighbours of `j` **excluding** `i`.

  The `i–j` bond energy is unchanged (the pair stays anti-aligned after the
  swap). Only the bonds from `i` and `j` to their *other* neighbours change:

  ```
  E_other(before) = -J s_i a - J s_j b
  E_other(after)  = -J s_j a - J s_i b
  ΔE = E_after - E_before = J (s_i - s_j)(a - b).
  ```

  This local expression is verified against a brute-force recomputation of the
  total energy difference in `tests/test_energy.py`.

## Metropolis acceptance

Accept the proposed exchange with probability `min(1, exp(-ΔE / T))`. Combined
with a symmetric proposal (uniform site × uniform neighbour direction), this
gives detailed balance with respect to the canonical distribution
`π(σ) ∝ exp(-H(σ)/T)` restricted to the fixed-`M` sector.

**Ergodicity over the sector.** Any two configurations with the same `M` are
connected by a sequence of nearest-neighbour exchanges (a single misplaced spin
can be walked to any target site one NN swap at a time), so the chain is
irreducible on the sector. Irreducible + aperiodic + detailed balance ⇒ a unique
stationary distribution equal to the restricted canonical distribution. This is
checked numerically on a `4 × 4` lattice in `tests/test_detailed_balance.py` by
comparing the sampled energy histogram against exact enumeration of the sector.

## Preparation vs. kinetic kernel (do not conflate)

- **Kinetic kernel (post-quench):** *must* be local nearest-neighbour Kawasaki
  Metropolis. This is the dynamics whose timescales we study.
- **Preparation kernel (pre-quench equilibration at `T_i`):** local Kawasaki is
  the baseline. A faster sampler may be used *only* if it conserves `M` and
  samples the same restricted canonical equilibrium — non-local opposite-spin
  exchange (pick a random `+` and a random `−`, swap with Metropolis) is the
  natural choice and is also detailed-balanced over the sector. Cluster
  algorithms (Wolff / Swendsen–Wang) do **not** conserve `M` and are excluded.

The preparation kernel is always recorded in the run manifest.

## Time normalisation

Physical (Monte Carlo) time is measured in **sweeps**, one sweep `= N²`
*attempted* nearest-neighbour bond updates, independent of acceptance. A
continuous-time / rejection-free (BKL) implementation defines a *different*
continuous-time kinetic model; if introduced, it must be validated against the
attempted-update Metropolis baseline at overlapping parameters (observables and
time calibration) before any results are pooled.

## Length-scale estimators

Three independent estimators of the characteristic domain length `L(t)`:

- **`L_C`** — from the equal-time correlation function `C(r,t)`: the smallest
  `r` at which the (radially averaged) `C(r,t)` falls to a fixed fraction of
  `C(0,t)` (default: first crossing of `C = C(0)/e`, with the zero-crossing
  reported as an alternative).
- **`L_S`** — from the structure factor `S(k,t)`: `L_S = 2π / ⟨k⟩`, with the
  first moment `⟨k⟩ = Σ_k k S(k) / Σ_k S(k)` taken over the radially averaged
  `S(k)` (excluding the `k = 0` component, which carries the conserved `M`).
- **`L_E`** — from the excess energy: in the late interfacial regime
  `E(t) − E_∞ ∝ 1/L`, so `L_E ∝ 1 / (E(t) − E_∞)`. This is a late-stage
  interfacial relation, **not** a general identity; disagreement with `L_C`,
  `L_S` is a diagnostic to analyse, not an automatic code failure.

The two-of-three agreement gate uses these three estimators.
