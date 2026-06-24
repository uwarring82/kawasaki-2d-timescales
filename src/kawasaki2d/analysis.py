"""Pre-registered statistical analysis for the coarsening / Mpemba gates.

This module is the *confirmatory* analysis layer. Its definitions (bootstrap
procedure, offset model, FDR method, effective-exponent estimator) are fixed
here, in version control, before the crossing search (Milestone 5) is run. Any
deviation is recorded in the logbook and labelled exploratory.

Implements the Mpemba-claim-gate machinery:

* ensemble mean / CI over realisations (:func:`ensemble_stats`);
* the difference-bootstrap on ``L_hot(t) − L_cold(t)`` (:func:`difference_bootstrap`);
* the crossing test with sign-change-with-CI-excluding-zero (:func:`crossing_test`);
* the ``R(t) ≈ R₀ + (λ t)^{1/3}`` offset fit and ``R₀``-subtracted collapse
  (:func:`fit_offset_growth`, :func:`offset_corrected`);
* local effective growth exponents (:func:`effective_exponent`);
* Benjamini–Hochberg FDR control across the declared grid (:func:`benjamini_hochberg`);
* a seed-budget power calculation (:func:`required_ensemble_size`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.stats import norm as _norm

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# Ensemble statistics                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class EnsembleStats:
    mean: np.ndarray
    sem: np.ndarray          # standard error of the mean
    ci_low: np.ndarray
    ci_high: np.ndarray
    n: int


def ensemble_stats(samples: np.ndarray, ci: float = 0.95) -> EnsembleStats:
    """Mean, SEM and a normal-approx CI over realisations.

    ``samples`` has shape ``(n_realisations, n_times)`` (or ``(n_realisations,)``).
    Uses the sample standard deviation (ddof=1) and a Gaussian quantile.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[:, None]
    n = samples.shape[0]
    mean = np.nanmean(samples, axis=0)
    sd = np.nanstd(samples, axis=0, ddof=1) if n > 1 else np.zeros(samples.shape[1])
    sem = sd / np.sqrt(n)
    z = _z_quantile(ci)
    return EnsembleStats(mean=mean, sem=sem, ci_low=mean - z * sem, ci_high=mean + z * sem, n=n)


def _z_quantile(ci: float) -> float:
    if _HAVE_SCIPY:
        return float(_norm.ppf(0.5 + ci / 2.0))
    # fallback table for common CIs
    return {0.90: 1.6448536, 0.95: 1.959964, 0.99: 2.5758293}.get(round(ci, 2), 1.959964)


# --------------------------------------------------------------------------- #
# Difference bootstrap and crossing test                                       #
# --------------------------------------------------------------------------- #


def _bootstrap_pvalues(boots: np.ndarray) -> np.ndarray:
    """Two-sided bootstrap p-value per column: 2·min(frac≤0, frac≥0), clipped."""
    n = boots.shape[0]
    frac_le = np.mean(boots <= 0, axis=0)
    frac_ge = np.mean(boots >= 0, axis=0)
    p = 2.0 * np.minimum(frac_le, frac_ge)
    return np.clip(p, 1.0 / n, 1.0)


@dataclass
class DifferenceCI:
    times: np.ndarray
    diff_mean: np.ndarray     # mean of (hot - cold) at each time
    ci_low: np.ndarray
    ci_high: np.ndarray
    excludes_zero: np.ndarray  # bool per time: CI excludes 0
    sign: np.ndarray           # sign of diff_mean where CI excludes zero, else 0
    pvalue: np.ndarray | None = None  # two-sided bootstrap p-value per time


def difference_bootstrap(
    hot: np.ndarray,
    cold: np.ndarray,
    times: np.ndarray,
    rng: np.random.Generator,
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> DifferenceCI:
    """Bootstrap CI on ``L_hot(t) − L_cold(t)`` at each time.

    ``hot`` and ``cold`` are ``(n_realisations, n_times)`` arrays of a length
    observable for the two preparations (independent ensembles, resampled
    independently with replacement). Returns per-time difference means and
    percentile-bootstrap CIs, plus whether each CI excludes zero and the sign.
    """
    hot = np.asarray(hot, float)
    cold = np.asarray(cold, float)
    times = np.asarray(times)
    nh, nt = hot.shape
    nc = cold.shape[0]
    boots = np.empty((n_boot, nt))
    for b in range(n_boot):
        ih = rng.integers(0, nh, size=nh)
        ic = rng.integers(0, nc, size=nc)
        boots[b] = np.nanmean(hot[ih], axis=0) - np.nanmean(cold[ic], axis=0)
    alpha = (1.0 - ci) / 2.0
    ci_low = np.nanpercentile(boots, 100 * alpha, axis=0)
    ci_high = np.nanpercentile(boots, 100 * (1 - alpha), axis=0)
    diff_mean = np.nanmean(hot, axis=0) - np.nanmean(cold, axis=0)
    excludes_zero = (ci_low > 0) | (ci_high < 0)
    sign = np.where(excludes_zero, np.sign(diff_mean), 0).astype(int)
    return DifferenceCI(
        times=times,
        diff_mean=diff_mean,
        ci_low=ci_low,
        ci_high=ci_high,
        excludes_zero=excludes_zero,
        sign=sign,
        pvalue=_bootstrap_pvalues(boots),
    )


def offset_corrected_difference_bootstrap(
    hot: np.ndarray,
    cold: np.ndarray,
    times: np.ndarray,
    rng: np.random.Generator,
    *,
    cutoff: float,
    n_boot: int = 5000,
    ci: float = 0.95,
    exponent: float | None = 1.0 / 3.0,
) -> tuple["DifferenceCI", float, float]:
    """Bootstrap CI on the **offset-corrected** difference of two preparations.

    ``D(t) = (L_hot(t) − R0_hot) − (L_cold(t) − R0_cold)`` where ``R0`` is fitted
    independently per leg from ``L = R0 + (λt)^{α}`` over ``t > cutoff`` and
    **re-fitted inside every bootstrap resample**, so the offset uncertainty
    propagates into the CI. ``hot``/``cold`` are ``(n_realisations, n_times)``.
    Removing ``R0`` strips the preparation head-start, so a non-zero ``D``
    reflects a genuine coarsening-*rate* difference, not the initial offset
    (Milestone-5 verdict quantity; see the M5 protocol amendment).

    ``exponent`` fixes the growth exponent used to fit ``R0`` (default ``1/3`` —
    the **pre-registered** model). Pass ``exponent=None`` for the *free-exponent*
    sensitivity variant: each leg's exponent is chosen per resample by the
    linearity scan (:func:`preferred_growth_exponent`). The free variant is
    exploratory — the conclusion can depend on the offset model (reviewers found
    that ``L_S`` flips sign under it), so it is a robustness check, not the
    confirmatory test.

    Returns ``(DifferenceCI over t>cutoff, R0_hot, R0_cold)``.
    """
    hot = np.asarray(hot, float)
    cold = np.asarray(cold, float)
    times = np.asarray(times, float)
    mask = times > cutoff
    tt = times[mask]
    nh, nc = hot.shape[0], cold.shape[0]

    def _R0(traj_mean):
        L = traj_mean[mask]
        finite = np.isfinite(L)
        tloc, Lloc = tt[finite], L[finite]
        if tloc.size < 3:
            return float("nan")
        a = exponent if exponent is not None else preferred_growth_exponent(tloc, Lloc).best_alpha
        u = tloc ** a
        A = np.vstack([np.ones_like(u), u]).T
        return float(np.linalg.lstsq(A, Lloc, rcond=None)[0][0])

    R0h = _R0(np.nanmean(hot, axis=0))
    R0c = _R0(np.nanmean(cold, axis=0))
    d_point = (np.nanmean(hot, axis=0)[mask] - R0h) - (np.nanmean(cold, axis=0)[mask] - R0c)

    boots = np.empty((n_boot, tt.size))
    for b in range(n_boot):
        mh = np.nanmean(hot[rng.integers(0, nh, size=nh)], axis=0)
        mc = np.nanmean(cold[rng.integers(0, nc, size=nc)], axis=0)
        boots[b] = (mh[mask] - _R0(mh)) - (mc[mask] - _R0(mc))
    alpha = (1.0 - ci) / 2.0
    lo = np.nanpercentile(boots, 100 * alpha, axis=0)
    hi = np.nanpercentile(boots, 100 * (1 - alpha), axis=0)
    excludes_zero = (lo > 0) | (hi < 0)
    sign = np.where(excludes_zero, np.sign(d_point), 0).astype(int)
    return (
        DifferenceCI(times=tt, diff_mean=d_point, ci_low=lo, ci_high=hi,
                     excludes_zero=excludes_zero, sign=sign,
                     pvalue=_bootstrap_pvalues(boots)),
        float(R0h), float(R0c),
    )


@dataclass
class CrossingVerdict:
    crossed: bool
    crossing_time: float | None
    sign_before: int
    sign_after: int
    detail: str


def crossing_test(dci: DifferenceCI) -> CrossingVerdict:
    """Decide whether a *significant* sign change occurs in the difference.

    A crossing is reported only if there is a time index where the CI-excluding-
    zero sign flips from one definite sign to the opposite definite sign — i.e.
    the bootstrap CI on ``L_hot − L_cold`` excludes zero on *both* sides of the
    crossing with opposite signs (task-card difference-bootstrap gate).
    """
    sign = dci.sign
    sig_idx = np.where(sign != 0)[0]
    if sig_idx.size == 0:
        return CrossingVerdict(False, None, 0, 0, "no time point has a CI excluding zero")
    # walk significant points in time order, look for an opposite-sign pair
    for a, b in zip(sig_idx[:-1], sig_idx[1:]):
        if sign[a] != 0 and sign[b] != 0 and sign[a] == -sign[b]:
            tcross = 0.5 * (dci.times[a] + dci.times[b])
            return CrossingVerdict(
                True, float(tcross), int(sign[a]), int(sign[b]),
                f"significant sign change between t={dci.times[a]} and t={dci.times[b]}",
            )
    return CrossingVerdict(
        False, None, int(sign[sig_idx[0]]), int(sign[sig_idx[-1]]),
        "significant difference present but no opposite-sign-with-CI crossing",
    )


# --------------------------------------------------------------------------- #
# Offset growth law R(t) ≈ R0 + (λ t)^(1/3)                                     #
# --------------------------------------------------------------------------- #


def growth_model(t: np.ndarray, R0: float, lam: float) -> np.ndarray:
    """``R(t) = R0 + (λ t)^{1/3}`` (diffusive conserved-dynamics coarsening)."""
    return R0 + np.cbrt(lam * np.asarray(t, float))


@dataclass
class OffsetFit:
    R0: float
    lam: float
    R0_err: float
    lam_err: float
    r_squared: float


def fit_offset_growth(t: np.ndarray, L: np.ndarray) -> OffsetFit:
    """Fit ``L(t) ≈ R0 + (λ t)^{1/3}`` with the exponent **fixed at 1/3**.

    The pre-registration (``configs/preregistration_m5.yaml``) fixes the growth
    exponent at 1/3, so the model is *linear* in the variable ``u = t^{1/3}``:
    ``L = R0 + b·u`` with ``λ = b³``. A linear least-squares fit is robust and
    non-degenerate — unlike a free nonlinear fit of ``R0`` and ``λ`` over a short,
    offset-dominated window, which is ill-conditioned (the two parameters trade
    off). Used to test whether the offset ``R0`` is preparation-dependent: a
    crossing that disappears after subtracting an independently fitted
    ``R0(T_i)`` is a route/offset artefact, not an asymptotic inversion.
    """
    t = np.asarray(t, float)
    L = np.asarray(L, float)
    mask = np.isfinite(t) & np.isfinite(L) & (t > 0)
    t, L = t[mask], L[mask]
    if t.size < 3:
        raise ValueError("need at least 3 finite points with t>0 to fit")
    u = np.cbrt(t)
    A = np.vstack([np.ones_like(u), u]).T
    coef, residuals, *_ = np.linalg.lstsq(A, L, rcond=None)
    R0, b = float(coef[0]), float(coef[1])
    pred = A @ coef
    ss_res = float(np.sum((L - pred) ** 2))
    ss_tot = float(np.sum((L - L.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # parameter standard errors from the linear-model covariance
    dof = max(1, len(L) - 2)
    sigma2 = ss_res / dof
    cov = sigma2 * np.linalg.inv(A.T @ A)
    R0_err = float(np.sqrt(cov[0, 0]))
    b_err = float(np.sqrt(cov[1, 1]))
    lam = b**3 if b > 0 else float("nan")
    lam_err = abs(3 * b**2 * b_err) if b > 0 else float("nan")
    return OffsetFit(R0=R0, lam=lam, R0_err=R0_err, lam_err=lam_err, r_squared=r2)


def offset_corrected(L: np.ndarray, R0: float) -> np.ndarray:
    """Return ``L − R0`` (the offset-subtracted length for collapse checks)."""
    return np.asarray(L, float) - R0


@dataclass
class ExponentScan:
    """Data-driven growth exponent from a linearity scan over ``α``."""

    best_alpha: float
    best_r2: float
    alpha_lo: float          # lower edge of the within-tolerance band
    alpha_hi: float          # upper edge of the within-tolerance band
    alpha_grid: np.ndarray
    r2_grid: np.ndarray


def preferred_growth_exponent(
    t: np.ndarray,
    L: np.ndarray,
    *,
    alpha_grid: np.ndarray | None = None,
    r2_tol: float = 0.0005,
) -> ExponentScan:
    """Estimate the growth exponent by maximising the linearity of ``L`` vs ``t^α``.

    For each candidate ``α`` we fit ``L = a + b·t^α`` (a *linear* least-squares
    problem — no ill-conditioning) and record the ``R²``. For the exact
    offset-growth form ``L = R0 + (λt)^{1/3}`` the curve is perfectly linear at
    ``α = 1/3`` and curved elsewhere, so ``R²(α)`` peaks at the true exponent.
    The reported uncertainty band is the range of ``α`` whose ``R²`` is within
    ``r2_tol`` of the maximum — broad when the window is short (the honest
    "thin for a precision exponent" regime), narrow when it is long.
    """
    t = np.asarray(t, float)
    L = np.asarray(L, float)
    mask = np.isfinite(t) & np.isfinite(L) & (t > 0)
    t, L = t[mask], L[mask]
    if t.size < 4:
        raise ValueError("need at least 4 finite points with t>0")
    if alpha_grid is None:
        alpha_grid = np.linspace(0.2, 0.5, 61)
    r2 = np.empty(alpha_grid.size)
    for i, a in enumerate(alpha_grid):
        u = t**a
        A = np.vstack([np.ones_like(u), u]).T
        coef, *_ = np.linalg.lstsq(A, L, rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((L - pred) ** 2))
        ss_tot = float(np.sum((L - L.mean()) ** 2))
        r2[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    best = int(np.nanargmax(r2))
    within = np.where(r2 >= r2[best] - r2_tol)[0]
    return ExponentScan(
        best_alpha=float(alpha_grid[best]),
        best_r2=float(r2[best]),
        alpha_lo=float(alpha_grid[within.min()]),
        alpha_hi=float(alpha_grid[within.max()]),
        alpha_grid=alpha_grid,
        r2_grid=r2,
    )


# --------------------------------------------------------------------------- #
# Effective growth exponent                                                    #
# --------------------------------------------------------------------------- #


def effective_exponent(t: np.ndarray, L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local effective exponent ``d ln L / d ln t`` via centred differences.

    Returns ``(t_mid, alpha_eff)``. For ideal diffusive coarsening
    ``L ~ t^{1/3}`` so ``alpha_eff → 1/3``.
    """
    t = np.asarray(t, float)
    L = np.asarray(L, float)
    mask = np.isfinite(t) & np.isfinite(L) & (t > 0) & (L > 0)
    t, L = t[mask], L[mask]
    lnt, lnL = np.log(t), np.log(L)
    alpha = np.gradient(lnL, lnt)
    return t, alpha


# --------------------------------------------------------------------------- #
# Multiple-comparison control                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class BHResult:
    rejected: np.ndarray       # bool per hypothesis
    threshold: float           # largest p-value passing BH
    sorted_order: np.ndarray


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> BHResult:
    """Benjamini–Hochberg FDR control at level ``alpha``.

    Applied across the declared ``(T_i, T_f, N)`` grid of per-pair crossing
    tests, each reduced to a single p-value. Returns which hypotheses are
    rejected (a crossing declared) controlling the false-discovery rate.
    """
    p = np.asarray(pvalues, float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh
    if not passed.any():
        rejected = np.zeros(m, dtype=bool)
        return BHResult(rejected=rejected, threshold=0.0, sorted_order=order)
    kmax = np.max(np.where(passed)[0])
    pcut = ranked[kmax]
    rejected = p <= pcut
    return BHResult(rejected=rejected, threshold=float(pcut), sorted_order=order)


# --------------------------------------------------------------------------- #
# Seed-budget power calculation                                                #
# --------------------------------------------------------------------------- #


def required_ensemble_size(
    var_hot: float,
    var_cold: float,
    signal: float,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
) -> int:
    """Per-preparation realisation count to detect a difference ``signal``.

    Two-sample z-test sizing: ``n ≈ (z_{1−α/2} + z_{1−β})² (σ²_hot + σ²_cold) / δ²``.
    ``signal`` is the expected ``|L_hot − L_cold|`` at the target time. If the
    achieved variance leaves the difference SE too large to resolve the sign,
    the Milestone-5 verdict is **underdetermined**, not "no effect".
    """
    if signal <= 0:
        raise ValueError("signal must be positive")
    z_a = _z_quantile(1 - alpha)        # one-sided-ish; uses 1-alpha here
    z_b = _z_quantile(2 * power - 1) if power > 0.5 else 0.0
    n = (z_a + z_b) ** 2 * (var_hot + var_cold) / signal**2
    return int(np.ceil(n))
