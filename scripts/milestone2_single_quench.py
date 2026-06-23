#!/usr/bin/env python
"""Milestone 2 — single-quench benchmark, with honesty guardrails.

Random M=0 state equilibrated at ``T_i`` → quench to ``T_f < T_c`` → track
``E(t)``, ``L(t)`` (``L_C``, ``L_S``), and the evolving structure factor
``S(k,t)``.

Three phases (the guardrails):
  Phase 1 — Equilibration sanity: confirm the preparation actually reaches
            canonical equilibrium at ``T_i`` (energy-trace saturation; two
            independent runs indistinguishable; non-local prep agrees with the
            local-Kawasaki baseline). Also demonstrated at a near-T_c point.
  Phase 2 — Pilot seed sweep: 2-3 seeds first; check E(t)/L(t) are qualitatively
            consistent and not stuck in metastable configurations, before
            spending the full ensemble.
  Phase 3 — Ensemble run: per-realisation trajectories + ensemble mean/CI of
            E(t), L_C(t), L_S(t), the S(k,t) evolution, and the effective
            growth exponent. Figures + provenance manifest.

Outputs an append-only ``results/<run_id>/`` directory.

Usage:
    python scripts/milestone2_single_quench.py [configs/single_quench.yaml]
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

from kawasaki2d import (  # noqa: E402
    T_C, analysis, dynamics, equilibration as eq, io, observables as obs,
    protocols, provenance,
)
from kawasaki2d.rng import make_rng, spawn_rngs  # noqa: E402


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Phase 1 — equilibration sanity                                               #
# --------------------------------------------------------------------------- #


def phase1_equilibration(n, M, T_i, prep_kernel, prep_sweeps, base_seed, run_dir, manifest):
    print(f"\n=== Phase 1: equilibration sanity (N={n}, M={M}) ===")
    report = {}
    # Demonstrate at the working T_i and at a near-T_c point (the hard case).
    # Deterministic per-tag seed offsets (no hash(): it is per-process randomised).
    for ti, (tag, T) in enumerate((("T_i", T_i), ("near_Tc", 2.4))):
        off = 1000 * (ti + 1)
        ra, rb = spawn_rngs(base_seed + off, 2)
        two_runs = eq.compare_equilibration(
            n, T, M, rng_a=ra, rng_b=rb, kernel_a=prep_kernel, kernel_b=prep_kernel,
            n_sweeps=prep_sweeps,
        )
        rc, rd = spawn_rngs(base_seed + 5000 + off, 2)
        kernel_cmp = eq.compare_equilibration(
            n, T, M, rng_a=rc, rng_b=rd, kernel_a="nonlocal", kernel_b="local",
            n_sweeps=prep_sweeps,
        )
        (rt,) = spawn_rngs(base_seed + 9000 + off, 1)
        sweeps, etrace = eq.energy_trace(n, T, M, rt, kernel=prep_kernel,
                                         n_sweeps=prep_sweeps, sample_every=max(10, prep_sweeps // 60))
        report[tag] = {
            "T": T, "T_over_Tc": T / T_C,
            "two_runs_indistinguishable": two_runs.indistinguishable,
            "two_runs_overlap": two_runs.overlap,
            "two_runs_ks_p": two_runs.ks_pvalue,
            "prep_vs_local_indistinguishable": kernel_cmp.indistinguishable,
            "prep_vs_local_overlap": kernel_cmp.overlap,
            "prep_vs_local_ks_p": kernel_cmp.ks_pvalue,
            "equilibrium_energy_per_spin": float(np.mean(etrace[len(etrace) // 2:])),
        }
        print(f"  [{tag}] T={T:.3g}: two-runs indist={two_runs.indistinguishable} "
              f"(overlap={two_runs.overlap:.3f}, KSp={two_runs.ks_pvalue:.3f}); "
              f"prep-vs-local indist={kernel_cmp.indistinguishable} "
              f"(overlap={kernel_cmp.overlap:.3f})")
        _plot_equilibration(tag, T, sweeps, etrace, two_runs, run_dir / f"equilibration_{tag}.png")
        manifest.add_output(f"equilibration_{tag}", run_dir / f"equilibration_{tag}.png")

    passed = all(
        report[t]["two_runs_indistinguishable"] and report[t]["prep_vs_local_indistinguishable"]
        for t in report
    )
    report["passed"] = passed
    print(f"  Phase 1 verdict: {'PASS' if passed else 'REVIEW NEEDED'}")
    return report


def _plot_equilibration(tag, T, sweeps, etrace, two_runs, path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(sweeps, etrace, "-o", ms=3)
    ax[0].set_xlabel("preparation sweeps")
    ax[0].set_ylabel("energy / spin")
    ax[0].set_title(f"equilibration trace, T={T:.3g}")
    ax[1].axis("off")
    txt = (f"two independent runs\n"
           f"  overlap = {two_runs.overlap:.3f}\n"
           f"  KS p    = {two_runs.ks_pvalue:.3f}\n"
           f"  mean_a  = {two_runs.mean_a:.4f}\n"
           f"  mean_b  = {two_runs.mean_b:.4f}\n"
           f"  same?   = {two_runs.indistinguishable}")
    ax[1].text(0.05, 0.5, txt, family="monospace", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Phase 2 — pilot seed sweep                                                    #
# --------------------------------------------------------------------------- #


def phase2_pilot(n, M, T_i, T_f, prep_kernel, prep_sweeps, schedule, base_seed,
                 run_dir, manifest, n_pilot=3):
    print(f"\n=== Phase 2: pilot seed sweep ({n_pilot} seeds, T_f={T_f:.3g}) ===")
    rngs = spawn_rngs(base_seed + 100, n_pilot)
    pilots = []
    for k, rng in enumerate(rngs):
        prep = protocols.prepare_initial_state(n, T_i, M, rng, kernel=prep_kernel, n_sweeps=prep_sweeps)
        traj = protocols.coarsening_trajectory(prep.lattice, T_f, schedule, rng, record_sk=False)
        pilots.append(traj.rows)
    sweeps = np.array([r["sweep"] for r in pilots[0]], float)
    L_S = np.array([[r["L_S"] for r in rows] for rows in pilots])
    E = np.array([[r["energy_per_spin"] for r in rows] for rows in pilots])

    # Stability: relative spread of L_S across seeds at the final time.
    final = L_S[:, -1]
    rel_spread = float(np.std(final) / np.mean(final)) if np.mean(final) else float("nan")
    # Metastability flag: any seed whose final L_S is a gross outlier (> 3 robust sigma).
    med = np.median(final)
    mad = np.median(np.abs(final - med)) + 1e-9
    outliers = int(np.sum(np.abs(final - med) > 4.0 * 1.4826 * mad))
    stable = rel_spread < 0.25 and outliers == 0
    print(f"  final L_S per seed: {np.round(final,2)}  (rel spread={rel_spread:.3f}, outliers={outliers})")
    print(f"  Phase 2 verdict: {'STABLE — proceed to ensemble' if stable else 'UNSTABLE — investigate'}")

    _plot_pilot(sweeps, E, L_S, run_dir / "pilot_seed_sweep.png")
    manifest.add_output("pilot_seed_sweep", run_dir / "pilot_seed_sweep.png")
    return {"final_L_S": final.tolist(), "rel_spread": rel_spread, "outliers": outliers, "stable": stable}


def _plot_pilot(sweeps, E, L_S, path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    pos = sweeps > 0
    for k in range(E.shape[0]):
        ax[0].plot(sweeps, E[k], "-", alpha=0.8, label=f"seed {k}")
        ax[1].plot(sweeps[pos], L_S[k][pos], "-", alpha=0.8, label=f"seed {k}")
    ax[0].set_xlabel("sweeps"); ax[0].set_ylabel("energy / spin"); ax[0].set_title("E(t) per seed")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("sweeps"); ax[1].set_ylabel("$L_S$"); ax[1].set_title("$L_S(t)$ per seed")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# Phase 3 — ensemble run                                                        #
# --------------------------------------------------------------------------- #


def phase3_ensemble(n, M, T_i, T_f, prep_kernel, prep_sweeps, schedule, base_seed,
                    n_real, run_dir, manifest):
    print(f"\n=== Phase 3: ensemble single-quench ({n_real} realisations) ===")
    rngs = spawn_rngs(base_seed, n_real)
    sweeps = np.asarray(schedule, float)
    E = np.empty((n_real, len(sweeps)))
    L_C = np.empty((n_real, len(sweeps)))
    L_S = np.empty((n_real, len(sweeps)))
    sk_accum = None
    sk_k = None
    for k, rng in enumerate(rngs):
        prep = protocols.prepare_initial_state(n, T_i, M, rng, kernel=prep_kernel, n_sweeps=prep_sweeps)
        traj = protocols.coarsening_trajectory(prep.lattice, T_f, schedule, rng, record_sk=True)
        io.write_trajectory_csv(run_dir / f"trajectory_real{k:03d}.csv", traj.rows)
        manifest.add_output(f"trajectory_real{k:03d}", run_dir / f"trajectory_real{k:03d}.csv")
        E[k] = [r["energy_per_spin"] for r in traj.rows]
        L_C[k] = [r["L_C"] for r in traj.rows]
        L_S[k] = [r["L_S"] for r in traj.rows]
        sk_k = traj.sk_k
        sk_accum = traj.sk if sk_accum is None else sk_accum + traj.sk
    sk_mean = sk_accum / n_real

    # E_infinity working estimate: long non-local-assisted equilibration at T_f
    # (horizon-limited reference; its N/horizon dependence is reported, not hidden).
    e_inf, e_inf_sd = _estimate_e_inf(n, M, T_f, base_seed + 31337)
    print(f"  E_inf(T_f) working estimate = {e_inf:.4f} +/- {e_inf_sd:.4f} per spin (horizon-limited)")

    st_E = analysis.ensemble_stats(E)
    st_LC = analysis.ensemble_stats(L_C)
    st_LS = analysis.ensemble_stats(L_S)

    # ensemble CSV
    ens_rows = []
    for j in range(len(sweeps)):
        ens_rows.append({
            "sweep": int(sweeps[j]),
            "E_per_spin_mean": st_E.mean[j], "E_per_spin_lo": st_E.ci_low[j], "E_per_spin_hi": st_E.ci_high[j],
            "L_C_mean": st_LC.mean[j], "L_C_lo": st_LC.ci_low[j], "L_C_hi": st_LC.ci_high[j],
            "L_S_mean": st_LS.mean[j], "L_S_lo": st_LS.ci_low[j], "L_S_hi": st_LS.ci_high[j],
        })
    ens_cols = ("sweep", "E_per_spin_mean", "E_per_spin_lo", "E_per_spin_hi",
                "L_C_mean", "L_C_lo", "L_C_hi", "L_S_mean", "L_S_lo", "L_S_hi")
    io.write_trajectory_csv(run_dir / "ensemble_trajectory.csv", ens_rows, columns=ens_cols)
    manifest.add_output("ensemble_trajectory", run_dir / "ensemble_trajectory.csv")

    np.save(run_dir / "Skt_k.npy", sk_k)
    np.save(run_dir / "Skt_mean.npy", sk_mean)
    np.save(run_dir / "Skt_sweeps.npy", sweeps)
    for label in ("Skt_k", "Skt_mean", "Skt_sweeps"):
        manifest.add_output(label, run_dir / f"{label}.npy")

    # effective exponent of ensemble-mean L_S in the pre-saturation window
    t_eff, alpha = analysis.effective_exponent(sweeps, st_LS.mean)
    eff_window = (t_eff > 100) & (t_eff < 0.6 * sweeps[-1])
    alpha_med = float(np.nanmedian(alpha[eff_window])) if eff_window.any() else float("nan")
    print(f"  raw median effective exponent of L_S in [100, {0.6*sweeps[-1]:.0f}] sweeps: "
          f"{alpha_med:.3f} (offset-suppressed)")

    # Offset control (pre-registered): fit L = R0 + (lambda t)^(1/3), exponent
    # fixed at 1/3, over t > lower_cutoff. The raw effective exponent is below
    # 1/3 only because of the additive offset R0; after subtracting it the data
    # should recover the diffusive 1/3 law.
    cutoff = 100  # configs/preregistration_m5.yaml: lower_cutoff_sweeps
    fits = {}
    for name, st in (("L_C", st_LC), ("L_S", st_LS)):
        m = sweeps > cutoff
        fit = analysis.fit_offset_growth(sweeps[m], st.mean[m])
        tcorr, acorr = analysis.effective_exponent(sweeps[m], st.mean[m] - fit.R0)
        fits[name] = {
            "R0": fit.R0, "R0_err": fit.R0_err, "lambda": fit.lam,
            "r_squared": fit.r_squared,
            "offset_corrected_exponent_median": float(np.nanmedian(acorr)),
        }
        print(f"  {name}: offset fit R0={fit.R0:.3f} lambda={fit.lam:.4g} R^2={fit.r_squared:.4f} "
              f"-> offset-corrected exponent={np.nanmedian(acorr):.3f} (target 1/3)")

    _plot_ensemble(sweeps, st_E, st_LC, st_LS, e_inf, sk_k, sk_mean, t_eff, alpha,
                   fits, cutoff, run_dir / "ensemble_benchmark.png")
    manifest.add_output("ensemble_benchmark", run_dir / "ensemble_benchmark.png")

    return {
        "n_realisations": n_real,
        "e_inf_estimate": e_inf, "e_inf_sd": e_inf_sd,
        "final_L_C_mean": float(st_LC.mean[-1]), "final_L_S_mean": float(st_LS.mean[-1]),
        "raw_effective_exponent_L_S_median": alpha_med,
        "offset_fits": fits,
        "lower_cutoff_sweeps": cutoff,
    }


def _estimate_e_inf(n, M, T_f, seed, sweeps_burn=8000, sample_every=80, samples=40):
    """Horizon-limited E_inf(T_f) estimate via long non-local-assisted equilibration.

    Non-local opposite-spin exchange satisfies detailed balance over the fixed-M
    sector, so it samples the same canonical equilibrium as local Kawasaki but
    mixes far faster below T_c. We burn in, then average the per-spin energy over
    decorrelated samples. This is a *working* reference whose horizon dependence
    is reported, never the sole diagnostic (Mpemba-claim gate).
    """
    from kawasaki2d.lattice import init_lattice, total_energy

    rng = make_rng(seed)
    lat = init_lattice(n, M, rng=rng)
    dynamics.run_nonlocal_exchange(lat, T_f, sweeps_burn, rng)
    es = np.empty(samples)
    for i in range(samples):
        dynamics.run_nonlocal_exchange(lat, T_f, sample_every, rng)
        es[i] = total_energy(lat) / lat.size
    return float(es.mean()), float(es.std(ddof=1))


def _plot_ensemble(sweeps, st_E, st_LC, st_LS, e_inf, sk_k, sk_mean, t_eff, alpha,
                   fits, cutoff, path):
    pos = sweeps > 0
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    a = ax[0, 0]
    a.fill_between(sweeps, st_E.ci_low, st_E.ci_high, alpha=0.3)
    a.plot(sweeps, st_E.mean, "-o", ms=3)
    a.axhline(e_inf, color="r", ls="--", lw=1, label=f"$E_\\infty\\approx${e_inf:.3f}")
    a.set_xlabel("sweeps"); a.set_ylabel("energy / spin"); a.set_title("ensemble $E(t)$ (95% CI)"); a.legend()

    # Length scales with the offset-fit overlaid (exponent fixed at 1/3).
    b = ax[0, 1]
    b.fill_between(sweeps[pos], st_LC.ci_low[pos], st_LC.ci_high[pos], alpha=0.25)
    b.plot(sweeps[pos], st_LC.mean[pos], "o", ms=3, label="$L_C$")
    b.fill_between(sweeps[pos], st_LS.ci_low[pos], st_LS.ci_high[pos], alpha=0.25)
    b.plot(sweeps[pos], st_LS.mean[pos], "s", ms=3, label="$L_S$")
    tf = sweeps[sweeps > cutoff]
    for name, st in (("L_C", st_LC), ("L_S", st_LS)):
        R0, lam = fits[name]["R0"], fits[name]["lambda"]
        b.plot(tf, R0 + np.cbrt(lam * tf), "k-", lw=1,
               label=f"{name}: $R_0$+($\\lambda t)^{{1/3}}$ ($R^2$={fits[name]['r_squared']:.3f})")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel("sweeps"); b.set_ylabel("L"); b.set_title("length scales + offset fit"); b.legend(fontsize=7)

    c = ax[1, 0]
    idxs = np.linspace(1, len(sweeps) - 1, 5).astype(int)
    for j in idxs:
        c.plot(sk_k, sk_mean[j], "-", label=f"t={int(sweeps[j])}")
    c.set_xlabel("k"); c.set_ylabel("S(k,t)"); c.set_title("structure factor evolution"); c.legend(fontsize=8)

    # Raw vs offset-corrected effective exponent.
    d = ax[1, 1]
    d.plot(t_eff, alpha, "-o", ms=3, label=r"raw $L_S$")
    for name, st in (("L_C", st_LC), ("L_S", st_LS)):
        m = sweeps > cutoff
        tc, ac = analysis.effective_exponent(sweeps[m], st.mean[m] - fits[name]["R0"])
        d.plot(tc, ac, "-^", ms=3, label=f"{name}$-R_0$")
    d.axhline(1 / 3, color="r", ls="--", lw=1, label="1/3")
    d.set_xscale("log"); d.set_ylim(0, 0.45)
    d.set_xlabel("sweeps"); d.set_ylabel(r"$d\ln L/d\ln t$")
    d.set_title("effective exponent: raw vs offset-corrected"); d.legend(fontsize=7)

    fig.suptitle("Milestone 2 — single-quench coarsening benchmark (offset-controlled)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #


def main(config_path: str) -> Path:
    config = io.load_config(config_path)
    run_id = config["run_id"]
    n = int(config["model"]["N"])
    M = int(config["model"].get("magnetisation", 0))
    prep = config["preparation"]
    T_i = float(prep["T_i"])
    prep_kernel = prep.get("kernel", "nonlocal")
    prep_sweeps = int(prep["n_sweeps"])
    T_f = float(config["quench"]["T_f"])
    schedule = protocols.log_schedule(
        t_max=int(config["quench"]["schedule"]["t_max"]),
        n_points=int(config["quench"]["schedule"]["n_points"]),
    )
    n_real = int(config["ensemble"].get("n_realisations", 8))
    base_seed = int(config["ensemble"].get("seed", 0))

    run_dir = io.new_run_directory(REPO_ROOT / "results", run_id)
    io.dump_config(config, run_dir / "config.yaml")
    manifest = provenance.Manifest.build(
        run_id=run_id, timestamp=_utc(), config=config,
        seeds={"base": base_seed, "n_realisations": n_real},
        repo_root=REPO_ROOT, notes="Milestone 2: single-quench benchmark (3 phases)",
    )

    print(f"Milestone 2: N={n}, M={M}, T_i={T_i}, T_f={T_f:.3g} (T_f/T_c={T_f/T_C:.2f})")
    report = {"config": config}
    report["phase1_equilibration"] = phase1_equilibration(
        n, M, T_i, prep_kernel, prep_sweeps, base_seed, run_dir, manifest)
    report["phase2_pilot"] = phase2_pilot(
        n, M, T_i, T_f, prep_kernel, prep_sweeps, schedule, base_seed, run_dir, manifest)
    report["phase3_ensemble"] = phase3_ensemble(
        n, M, T_i, T_f, prep_kernel, prep_sweeps, schedule, base_seed, n_real, run_dir, manifest)

    (run_dir / "m2_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    manifest.add_output("m2_report", run_dir / "m2_report.json")
    manifest.write(run_dir / "manifest.json")
    print(f"\nwrote {run_dir}")
    return run_dir


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "configs" / "single_quench.yaml")
    main(cfg)
