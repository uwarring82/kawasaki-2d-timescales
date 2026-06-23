#!/usr/bin/env python
"""Milestone 3 — coarsening-law assessment across N = 32, 64, 128.

For each size (run in increasing order): estimate E_inf(N,T_f), run an ensemble
quench, derive the three length estimators L_C, L_S, L_E, and assess the
diffusive growth law L = R0 + (lambda t)^(1/3):

  * data-driven exponent via the linearity scan (maximise R^2 of L vs t^alpha);
  * offset fit (exponent fixed at 1/3) -> R0, lambda, R^2;
  * lower-cutoff sensitivity scan;
  * saturation/window guard (window must stay below ~finite_size_fraction * N);
  * two-of-three estimator agreement.

Each N emits its own append-only run directory + manifest; a summary directory
holds the cross-N comparison table and figure.

Usage:
    python scripts/milestone3_coarsening.py [configs/coarsening_m3.yaml]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kawasaki2d import T_C, analysis, equilibration as eq, io, protocols, provenance  # noqa: E402
from kawasaki2d.rng import make_rng, spawn_rngs  # noqa: E402


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fit_block(t, L, *, cutoff, alpha_grid, upper=None):
    """Offset fit + exponent scan + offset-corrected exponent over a window."""
    m = (t > cutoff) & np.isfinite(L)
    if upper is not None:
        m &= t <= upper
    if m.sum() < 4:
        return None
    tt, LL = t[m], L[m]
    fit = analysis.fit_offset_growth(tt, LL)
    scan = analysis.preferred_growth_exponent(tt, LL, alpha_grid=alpha_grid)
    tc, ac = analysis.effective_exponent(tt, LL - fit.R0)
    return {
        "R0": fit.R0, "R0_err": fit.R0_err, "lambda": fit.lam, "fit_r2": fit.r_squared,
        "preferred_alpha": scan.best_alpha, "alpha_lo": scan.alpha_lo, "alpha_hi": scan.alpha_hi,
        "scan_r2": scan.best_r2,
        "offset_corrected_exponent_median": float(np.nanmedian(ac)),
        "n_points": int(m.sum()), "window": [float(cutoff), float(upper) if upper else float(tt.max())],
    }


def run_one_size(N, cfg, run_dir, manifest, alpha_grid):
    M = int(cfg["model"]["magnetisation"])
    prep = cfg["preparation"]
    T_i, prep_kernel, prep_sweeps = float(prep["T_i"]), prep.get("kernel", "nonlocal"), int(prep["n_sweeps"])
    T_f = float(cfg["quench"]["T_f"])
    n_real = int(cfg["ensemble"]["n_realisations"])
    base_seed = int(cfg["ensemble"]["seed"])
    an = cfg["analysis"]
    cutoff = int(an["lower_cutoff_sweeps"])
    fs_frac = float(an["finite_size_fraction"])

    size_cfg = next(s for s in cfg["sizes"] if int(s["N"]) == N)
    schedule = protocols.log_schedule(int(size_cfg["t_max"]), int(size_cfg["n_points"]))
    sweeps = np.asarray(schedule, float)

    # --- E_inf(N, T_f) working estimate (own seed stream) ---
    ei = cfg["einf"]
    e_rng = make_rng(base_seed + 777 + N)
    einf = eq.estimate_equilibrium_energy(
        N, T_f, M, e_rng, kernel=ei.get("kernel", "nonlocal"),
        sweeps_burn=int(ei["sweeps_burn"]), sample_every=int(ei["sample_every"]),
        n_samples=int(ei["n_samples"]),
    )
    print(f"  E_inf(N={N}) = {einf.mean:.4f} +/- {einf.sd:.4f} /spin "
          f"(burn {einf.sweeps_burn} sweeps, {einf.n_samples} samples)")

    # --- ensemble quench ---
    rngs = spawn_rngs(base_seed + N, n_real)
    E = np.empty((n_real, len(sweeps)))
    L_C = np.empty((n_real, len(sweeps)))
    L_S = np.empty((n_real, len(sweeps)))
    for k, rng in enumerate(rngs):
        p = protocols.prepare_initial_state(N, T_i, M, rng, kernel=prep_kernel, n_sweeps=prep_sweeps)
        traj = protocols.coarsening_trajectory(p.lattice, T_f, schedule, rng,
                                               with_clusters=False, record_sk=False)
        io.write_trajectory_csv(run_dir / f"trajectory_real{k:03d}.csv", traj.rows)
        manifest.add_output(f"trajectory_real{k:03d}", run_dir / f"trajectory_real{k:03d}.csv")
        E[k] = [r["energy_per_spin"] for r in traj.rows]
        L_C[k] = [r["L_C"] for r in traj.rows]
        L_S[k] = [r["L_S"] for r in traj.rows]

    # L_E per realisation = 1/(e - e_inf), masked where excess <= 0
    excess = E - einf.mean
    with np.errstate(divide="ignore", invalid="ignore"):
        L_E = np.where(excess > 0, 1.0 / excess, np.nan)

    st_E = analysis.ensemble_stats(E)
    st_LC = analysis.ensemble_stats(L_C)
    st_LS = analysis.ensemble_stats(L_S)
    st_LE = analysis.ensemble_stats(L_E)

    # --- saturation guard: window upper bound where L_S stays below fs_frac*N ---
    below = st_LS.mean < fs_frac * N
    upper = float(sweeps[below][-1]) if below.any() else float(sweeps[-1])
    saturated = not below.all()
    L_S_max_over_N = float(np.nanmax(st_LS.mean) / N)
    print(f"  window upper={upper:.0f} sweeps (L_S<{fs_frac}*N); "
          f"max L_S/N={L_S_max_over_N:.3f}{'  [trimmed for saturation]' if saturated else ''}")

    # --- fits per estimator over the guarded window ---
    fits = {}
    for name, st in (("L_C", st_LC), ("L_S", st_LS), ("L_E", st_LE)):
        blk = _fit_block(sweeps, st.mean, cutoff=cutoff, alpha_grid=alpha_grid, upper=upper)
        fits[name] = blk
        if blk:
            print(f"  {name}: preferred alpha={blk['preferred_alpha']:.3f} "
                  f"[{blk['alpha_lo']:.3f},{blk['alpha_hi']:.3f}]  "
                  f"R0={blk['R0']:.3f} lambda={blk['lambda']:.4g} fitR2={blk['fit_r2']:.4f}")

    # --- lower-cutoff sensitivity scan (on L_S): report fixed-1/3 fit R^2 (the
    # pre-registered confirmatory quantity) and the descriptive free exponent ---
    cutoff_scan = {}
    for c in an["cutoff_scan"]:
        blk = _fit_block(sweeps, st_LS.mean, cutoff=int(c), alpha_grid=alpha_grid, upper=upper)
        if blk:
            cutoff_scan[int(c)] = {"preferred_alpha": blk["preferred_alpha"], "fit_r2": blk["fit_r2"]}
    r2s = [v["fit_r2"] for v in cutoff_scan.values()]
    # "stable" = the 1/3 offset law keeps fitting well as the cutoff moves
    cutoff_stable = (min(r2s) > 0.99) if r2s else False
    print(f"  cutoff-scan L_S [1/3-fit R^2 | free-alpha]: "
          f"{ {c: (round(v['fit_r2'],4), round(v['preferred_alpha'],3)) for c,v in cutoff_scan.items()} }")

    # Confirmatory consistency-with-1/3 (pre-registration fixes the exponent at
    # 1/3): an estimator is consistent if the offset-growth law fits to R^2>0.99.
    # The free-exponent band is reported separately as a descriptive, illustrative
    # precision statement (deliberately wide at these sizes — see the task card).
    def consistent_third(b):
        return bool(b and b["fit_r2"] is not None and b["fit_r2"] > 0.99)

    def band_includes_third(b):
        return bool(b and b["alpha_lo"] - 0.02 <= 1 / 3 <= b["alpha_hi"] + 0.02)

    consistent = [n for n, b in fits.items() if consistent_third(b)]
    two_of_three = len(consistent) >= 2
    for n in fits:
        if fits[n]:
            fits[n]["consistent_with_one_third"] = consistent_third(fits[n])
            fits[n]["band_includes_one_third"] = band_includes_third(fits[n])

    # --- ensemble CSV ---
    rows = []
    for j in range(len(sweeps)):
        rows.append({
            "sweep": int(sweeps[j]),
            "E_per_spin_mean": st_E.mean[j],
            "L_C_mean": st_LC.mean[j], "L_C_sem": st_LC.sem[j],
            "L_S_mean": st_LS.mean[j], "L_S_sem": st_LS.sem[j],
            "L_E_mean": st_LE.mean[j], "L_E_sem": st_LE.sem[j],
            "excess_energy": st_E.mean[j] - einf.mean,
        })
    cols = ("sweep", "E_per_spin_mean", "L_C_mean", "L_C_sem", "L_S_mean", "L_S_sem",
            "L_E_mean", "L_E_sem", "excess_energy")
    io.write_trajectory_csv(run_dir / "ensemble_trajectory.csv", rows, columns=cols)
    manifest.add_output("ensemble_trajectory", run_dir / "ensemble_trajectory.csv")

    summary = {
        "N": N, "T_f": T_f, "T_f_over_Tc": T_f / T_C,
        "e_inf_mean": einf.mean, "e_inf_sd": einf.sd, "e_inf_horizon_sweeps": einf.sweeps_burn,
        "window_upper": upper, "saturated_within_tmax": saturated, "L_S_max_over_N": L_S_max_over_N,
        "fits": fits, "cutoff_scan": cutoff_scan, "cutoff_stable": cutoff_stable,
        "estimators_consistent_with_one_third": consistent, "two_of_three_agreement": two_of_three,
    }
    (run_dir / "analysis.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    manifest.add_output("analysis", run_dir / "analysis.json")

    _plot_size(N, sweeps, st_LC, st_LS, st_LE, fits, cutoff, upper, alpha_grid,
               run_dir / f"coarsening_N{N:03d}.png")
    manifest.add_output(f"coarsening_N{N:03d}", run_dir / f"coarsening_N{N:03d}.png")

    return summary


def _plot_size(N, sweeps, st_LC, st_LS, st_LE, fits, cutoff, upper, alpha_grid, path):
    pos = sweeps > 0
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    a = ax[0]
    for name, st, mk in (("L_C", st_LC, "o"), ("L_S", st_LS, "s"), ("L_E", st_LE, "^")):
        a.plot(sweeps[pos], st.mean[pos], mk, ms=3, label=name)
        b = fits[name]
        if b:
            tf = sweeps[(sweeps > cutoff) & (sweeps <= upper)]
            a.plot(tf, b["R0"] + np.cbrt(b["lambda"] * tf), "k-", lw=0.8)
    a.axvline(upper, color="gray", ls=":", lw=1, label="saturation guard")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("sweeps"); a.set_ylabel("L"); a.set_title(f"N={N}: lengths + offset fits"); a.legend(fontsize=8)

    c = ax[1]
    for name, st in (("L_C", st_LC), ("L_S", st_LS), ("L_E", st_LE)):
        b = fits[name]
        if b:
            m = (sweeps > cutoff) & (sweeps <= upper) & np.isfinite(st.mean)
            scan = analysis.preferred_growth_exponent(sweeps[m], st.mean[m], alpha_grid=alpha_grid)
            c.plot(scan.alpha_grid, scan.r2_grid, "-", label=f"{name} (peak {scan.best_alpha:.3f})")
    c.axvline(1 / 3, color="r", ls="--", lw=1, label="1/3")
    c.set_xlabel(r"trial exponent $\alpha$"); c.set_ylabel(r"$R^2$ of $L$ vs $t^\alpha$")
    c.set_title("exponent by linearity"); c.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def build_summary(results, run_dir, manifest):
    Ns = [r["N"] for r in results]
    rows = []
    for r in results:
        row = {"N": r["N"], "e_inf": r["e_inf_mean"], "L_S_max_over_N": r["L_S_max_over_N"],
               "cutoff_stable": r["cutoff_stable"], "two_of_three": r["two_of_three_agreement"]}
        for est in ("L_C", "L_S", "L_E"):
            b = r["fits"].get(est)
            row[f"{est}_alpha"] = b["preferred_alpha"] if b else float("nan")
            row[f"{est}_R0"] = b["R0"] if b else float("nan")
            row[f"{est}_lambda"] = b["lambda"] if b else float("nan")
        rows.append(row)
    cols = ("N", "e_inf", "L_S_max_over_N", "cutoff_stable", "two_of_three",
            "L_C_alpha", "L_C_R0", "L_C_lambda", "L_S_alpha", "L_S_R0", "L_S_lambda",
            "L_E_alpha", "L_E_R0", "L_E_lambda")
    io.write_trajectory_csv(run_dir / "summary_table.csv", rows, columns=cols)
    manifest.add_output("summary_table", run_dir / "summary_table.csv")

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    a = ax[0, 0]
    for est, mk in (("L_C", "o"), ("L_S", "s"), ("L_E", "^")):
        al = np.array([r["fits"][est]["preferred_alpha"] if r["fits"].get(est) else np.nan for r in results])
        lo = np.array([r["fits"][est]["alpha_lo"] if r["fits"].get(est) else np.nan for r in results])
        hi = np.array([r["fits"][est]["alpha_hi"] if r["fits"].get(est) else np.nan for r in results])
        a.errorbar(Ns, al, yerr=[al - lo, hi - al], fmt=mk + "-", capsize=3, label=est)
    a.axhline(1 / 3, color="r", ls="--", lw=1, label="1/3")
    a.set_xscale("log", base=2); a.set_xticks(Ns); a.set_xticklabels(Ns)
    a.set_xlabel("N"); a.set_ylabel("preferred exponent")
    a.set_title("free exponent vs N (illustrative; band = linearity tolerance)"); a.legend(fontsize=8)

    b = ax[0, 1]
    for est, mk in (("L_C", "o"), ("L_S", "s"), ("L_E", "^")):
        R0 = [r["fits"][est]["R0"] if r["fits"].get(est) else np.nan for r in results]
        b.plot(Ns, R0, mk + "-", label=f"{est} $R_0$")
    b.set_xscale("log", base=2); b.set_xticks(Ns); b.set_xticklabels(Ns)
    b.set_xlabel("N"); b.set_ylabel("offset $R_0$"); b.set_title("offset $R_0$ vs N"); b.legend(fontsize=8)

    # Naive FSS collapse L_S/N vs t/N^3 — does NOT collapse: the offset R0/N is
    # not scaled out (separation ~ R0/N). Shown to make the point explicit.
    R0_S = {r["N"]: (r["fits"]["L_S"]["R0"] if r["fits"].get("L_S") else 0.0) for r in results}
    for title, corrected, c in (("naive collapse: $L_S/N$ vs $t/N^3$", False, ax[1, 0]),
                                ("offset-corrected: $(L_S-R_0)/N$ vs $t/N^3$", True, ax[1, 1])):
        for r in results:
            N = r["N"]
            d = io.read_trajectory_csv(run_dir.parent / f"m3_N{N:03d}_v1" / "ensemble_trajectory.csv")
            sw, ls = d["sweep"], d["L_S_mean"]
            pos = sw > 0
            y = (ls[pos] - (R0_S[N] if corrected else 0.0)) / N
            c.plot(sw[pos] / N**3, y, "-o", ms=3, label=f"N={N}")
        c.set_xscale("log"); c.set_yscale("log")
        c.set_xlabel(r"$t/N^3$"); c.set_ylabel(r"$(L_S-R_0)/N$" if corrected else r"$L_S/N$")
        c.set_title(title); c.legend(fontsize=8)

    fig.suptitle("Milestone 3 — coarsening-law assessment across N")
    fig.tight_layout(); fig.savefig(run_dir / "m3_summary.png", dpi=130); plt.close(fig)
    manifest.add_output("m3_summary_fig", run_dir / "m3_summary.png")


def main(config_path: str, summary_only: bool = False) -> None:
    cfg = io.load_config(config_path)
    an = cfg["analysis"]
    eg = an["exponent_grid"]
    alpha_grid = np.linspace(float(eg["min"]), float(eg["max"]), int(eg["n"]))
    Ns = [int(s["N"]) for s in cfg["sizes"]]

    print(f"Milestone 3: N={Ns}, T_f={cfg['quench']['T_f']} (T_f/T_c="
          f"{float(cfg['quench']['T_f'])/T_C:.2f}), {cfg['ensemble']['n_realisations']} realisations")
    results = []
    if summary_only:
        # Rebuild the cross-N summary from saved per-N analysis.json (no re-sim).
        for N in sorted(Ns):
            results.append(json.loads(
                (REPO_ROOT / "results" / f"m3_N{N:03d}_v1" / "analysis.json").read_text()))
    else:
        for N in sorted(Ns):  # increasing order, per the M3 finite-size protocol
            print(f"\n=== N = {N} ===")
            run_id = f"m3_N{N:03d}_v1"
            run_dir = io.new_run_directory(REPO_ROOT / "results", run_id)
            io.dump_config(cfg, run_dir / "config.yaml")
            manifest = provenance.Manifest.build(
                run_id=run_id, timestamp=_utc(), config=cfg,
                seeds={"base": int(cfg["ensemble"]["seed"]), "N": N},
                repo_root=REPO_ROOT, notes=f"Milestone 3 coarsening, N={N}",
            )
            summary = run_one_size(N, cfg, run_dir, manifest, alpha_grid)
            manifest.write(run_dir / "manifest.json")
            results.append(summary)

    # --- summary ---
    print("\n=== summary ===")
    sdir = io.new_run_directory(REPO_ROOT / "results", "m3_summary_v1")
    smanifest = provenance.Manifest.build(
        run_id="m3_summary_v1", timestamp=_utc(), config=cfg,
        seeds={"base": int(cfg["ensemble"]["seed"])},
        repo_root=REPO_ROOT, notes="Milestone 3 cross-N summary",
    )
    build_summary(results, sdir, smanifest)
    (sdir / "summary.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    smanifest.add_output("summary_json", sdir / "summary.json")
    smanifest.write(sdir / "manifest.json")
    def _a(b):
        return f"{b['preferred_alpha']:.3f}" if b else "  n/a"

    def _r2(b):
        return f"{b['fit_r2']:.4f}" if b else "n/a"

    for r in results:
        f = r["fits"]
        print(f"  N={r['N']:3d}: 1/3-fit R^2 "
              f"L_C={_r2(f['L_C'])} L_S={_r2(f['L_S'])} L_E={_r2(f['L_E'])} | "
              f"free-alpha L_C={_a(f['L_C'])} L_S={_a(f['L_S'])} L_E={_a(f['L_E'])} | "
              f"two-of-three(consistent w/ 1/3)={r['two_of_three_agreement']}")
    print(f"\nwrote {sdir}")


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "--summary-only"]
    summary_only = "--summary-only" in sys.argv
    cfg_path = argv[0] if argv else str(REPO_ROOT / "configs" / "coarsening_m3.yaml")
    main(cfg_path, summary_only=summary_only)
