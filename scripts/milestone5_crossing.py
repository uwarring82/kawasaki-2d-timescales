#!/usr/bin/env python
"""Milestone 5 — pre-registered crossing search (primary pair).

Locked to configs/preregistration_m5.yaml + the M5 protocol amendment. Three
stages (run in order):

  --gate     : N=128 equilibration gate for both primary T_i (tau_E measured
               directly, not scaled from N=64). Writes gated budgets.
  --sweep    : production ensembles for both legs on validated initial states;
               saves per-realisation L_C/L_S/E arrays + E_inf. Reports the
               pilot-based power (seed budget).
  --analyse  : raw AND offset-corrected difference bootstrap for L_C/L_S/L_E,
               two-of-three agreement, BH-FDR across time points, saturation
               check, and the four-outcome verdict. Writes figures + report.

Usage:
    python scripts/milestone5_crossing.py [config] --gate
    python scripts/milestone5_crossing.py [config] --sweep
    python scripts/milestone5_crossing.py [config] --analyse
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

from kawasaki2d import T_C, analysis, equilibration as eq, io, observables as obs, protocols, provenance  # noqa: E402
from kawasaki2d.rng import make_rng, spawn_rngs  # noqa: E402


def _utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _T_f(cfg):
    return float(cfg["primary_pair"]["T_f_over_Tc"]) * T_C


# --------------------------------------------------------------------------- #
# Stage 1 — gate                                                               #
# --------------------------------------------------------------------------- #


def stage_gate(cfg, run_dir, manifest):
    N = int(cfg["model"]["N"]); M = int(cfg["model"]["magnetisation"])
    kernel = cfg["preparation"]["kernel"]; g = cfg["gate"]; base = int(cfg["ensemble"]["seed"])
    legs = {}
    for li, leg in enumerate(g["legs"]):
        res = eq.calibrate_prep_budget(
            N, float(leg["T_i"]), M, candidates=leg["candidates"], ref_burn=int(leg["ref_burn"]),
            base_seed=base + 200 * (li + 1), kernel=kernel,
            n_test=int(g["n_test"]), n_ref=int(g["n_ref"]),
            mean_sigma=float(g["mean_sigma"]), safety_tau_mult=float(g["safety_tau_mult"]))
        legs[str(float(leg["T_i"]))] = res
        print(f"  T_i={leg['T_i']:5.2f}: tau_E={res['tau_E_sweeps']:6.1f} sw | "
              f"smallest-converged={res['smallest_converged_budget']} | GATED={res['gated_budget']} | "
              f"passed={res['gate_passed']}")
        for r in res["candidates"]:
            print(f"       B={r['budget']:6d}: |Δ⟨e⟩|={r['energy_gap_sem']:.1f}σ "
                  f"|Δ⟨L_C⟩|={r['LC_gap_sem']:.1f}σ -> {'converged' if r['converged'] else 'no'}")
    budgets = {t: legs[t]["gated_budget"] for t in legs}
    (run_dir / "budgets.json").write_text(json.dumps({"budgets": budgets, "legs": legs}, indent=2, default=str) + "\n")
    manifest.add_output("budgets", run_dir / "budgets.json")
    return legs


# --------------------------------------------------------------------------- #
# Stage 2 — sweep (production ensembles)                                        #
# --------------------------------------------------------------------------- #


def stage_sweep(cfg, run_dir, manifest):
    N = int(cfg["model"]["N"]); M = int(cfg["model"]["magnetisation"])
    kernel = cfg["preparation"]["kernel"]; T_f = _T_f(cfg)
    schedule = protocols.log_schedule(int(cfg["quench"]["schedule"]["t_max"]),
                                      int(cfg["quench"]["schedule"]["n_points"]))
    sweeps = np.asarray(schedule, float)
    n_real = int(cfg["ensemble"]["max_realisations"]); base = int(cfg["ensemble"]["seed"])
    pp = cfg["primary_pair"]
    gate = json.loads((REPO_ROOT / "results" / cfg["run_id_gate"] / "budgets.json").read_text())["budgets"]

    ei = cfg["einf"]
    einf = eq.estimate_equilibrium_energy(N, T_f, M, make_rng(base + 7), kernel=ei["kernel"],
                                          sweeps_burn=int(ei["sweeps_burn"]), sample_every=int(ei["sample_every"]),
                                          n_samples=int(ei["n_samples"]))
    print(f"  E_inf(T_f) = {einf.mean:.4f} +/- {einf.sd:.4f} /spin")

    legs = {"hot": float(pp["T_i_hot"]), "cold": float(pp["T_i_cold"])}
    meta = {"N": N, "T_f": T_f, "T_f_over_Tc": float(pp["T_f_over_Tc"]), "n_realisations": n_real,
            "sweeps": sweeps.tolist(), "e_inf": einf.mean, "e_inf_sd": einf.sd,
            "budgets": {k: int(gate[str(legs[k])]) for k in legs}}
    np.save(run_dir / "sweeps.npy", sweeps)
    for li, (label, T_i) in enumerate(legs.items()):
        budget = int(gate[str(T_i)])
        rngs = spawn_rngs(base + 1000 * (li + 1), n_real)
        E = np.empty((n_real, len(sweeps))); LC = np.empty_like(E); LS = np.empty_like(E)
        for k, rng in enumerate(rngs):
            prep = protocols.prepare_initial_state(N, T_i, M, rng, kernel=kernel, n_sweeps=budget)
            traj = protocols.coarsening_trajectory(prep.lattice, T_f, schedule, rng,
                                                   with_clusters=False, record_sk=False)
            E[k] = [r["energy_per_spin"] for r in traj.rows]
            LC[k] = [r["L_C"] for r in traj.rows]
            LS[k] = [r["L_S"] for r in traj.rows]
        np.save(run_dir / f"{label}_E.npy", E)
        np.save(run_dir / f"{label}_LC.npy", LC)
        np.save(run_dir / f"{label}_LS.npy", LS)
        for s in ("E", "LC", "LS"):
            manifest.add_output(f"{label}_{s}", run_dir / f"{label}_{s}.npy")
        print(f"  {label} (T_i={T_i}, budget {budget}): final L_S={np.nanmean(LS[:,-1]):.2f}  "
              f"max L_S/N={np.nanmax(np.nanmean(LS,0))/N:.3f}")
    (run_dir / "sweep_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    manifest.add_output("sweep_meta", run_dir / "sweep_meta.json")
    return meta


# --------------------------------------------------------------------------- #
# Stage 3 — analyse + verdict                                                   #
# --------------------------------------------------------------------------- #


def _load_sweep(cfg):
    d = REPO_ROOT / "results" / cfg["run_id_sweep"]
    meta = json.loads((d / "sweep_meta.json").read_text())
    sweeps = np.load(d / "sweeps.npy")
    data = {}
    for label in ("hot", "cold"):
        data[label] = {s: np.load(d / f"{label}_{s}.npy") for s in ("E", "LC", "LS")}
    return meta, sweeps, data


def _estimators(data, e_inf):
    """Return per-leg dict of estimator arrays (n_real, n_times): L_C, L_S, L_E."""
    out = {}
    for label in ("hot", "cold"):
        excess = data[label]["E"] - e_inf
        with np.errstate(divide="ignore", invalid="ignore"):
            LE = np.where(excess > 0, 1.0 / excess, np.nan)
        out[label] = {"L_C": data[label]["LC"], "L_S": data[label]["LS"], "L_E": LE}
    return out


def stage_analyse(cfg, run_dir, manifest):
    N = int(cfg["model"]["N"]); an = cfg["analysis"]
    cutoff = int(an["lower_cutoff_sweeps"]); fs_frac = float(an["finite_size_fraction"])
    n_boot = int(an["n_boot"]); ci = float(an["ci"]); fdr_alpha = float(an["fdr_alpha"])
    pilot = int(cfg["ensemble"]["pilot_realisations"]); power = float(cfg["ensemble"]["target_power"])
    alpha = float(cfg["ensemble"]["alpha"])
    meta, sweeps, data = _load_sweep(cfg)
    e_inf = meta["e_inf"]
    est = _estimators(data, e_inf)
    rng = make_rng(int(cfg["ensemble"]["seed"]) + 99)

    # --- saturation guard: window upper where BOTH legs' mean L_S < fs_frac*N ---
    ls_hot = np.nanmean(est["hot"]["L_S"], 0); ls_cold = np.nanmean(est["cold"]["L_S"], 0)
    below = (ls_hot < fs_frac * N) & (ls_cold < fs_frac * N)
    upper = float(sweeps[below][-1]) if below.any() else float(sweeps[-1])
    print(f"  saturation guard: window upper={upper:.0f} sweeps "
          f"(max L_S/N: hot={np.nanmax(ls_hot)/N:.3f}, cold={np.nanmax(ls_cold)/N:.3f})")

    win = (sweeps > cutoff) & (sweeps <= upper)
    results = {"meta": meta, "window": [cutoff, upper], "estimators": {}}
    for name in ("L_C", "L_S", "L_E"):
        hot = est["hot"][name]; cold = est["cold"][name]
        # raw difference bootstrap + pre-registered crossing rule
        raw = analysis.difference_bootstrap(hot[:, win], cold[:, win], sweeps[win], rng,
                                            n_boot=n_boot, ci=ci)
        raw_cross = analysis.crossing_test(raw)
        # offset-corrected difference bootstrap (verdict quantity)
        oc, R0h, R0c = analysis.offset_corrected_difference_bootstrap(
            hot, cold, sweeps[(sweeps <= upper)], rng, cutoff=cutoff, n_boot=n_boot, ci=ci)
        oc_cross = analysis.crossing_test(oc)
        # BH-FDR across time points on the offset-corrected p-values
        bh = analysis.benjamini_hochberg(oc.pvalue, alpha=fdr_alpha)
        # late-window robust sign (FDR-significant points in the last third)
        late = oc.times > (cutoff + 0.66 * (upper - cutoff))
        late_sig = bh.rejected & late
        late_sign = int(np.sign(np.nanmean(oc.diff_mean[late]))) if late.any() else 0
        results["estimators"][name] = {
            "R0_hot": R0h, "R0_cold": R0c,
            "raw_crossing": raw_cross.crossed, "raw_detail": raw_cross.detail,
            "raw_final_sign": int(raw.sign[-1]),
            "oc_any_fdr_significant": bool(bh.rejected.any()),
            "oc_late_fdr_significant": bool(late_sig.any()),
            "oc_late_sign": late_sign,
            "oc_final_diff": float(oc.diff_mean[-1]),
            "oc_final_ci": [float(oc.ci_low[-1]), float(oc.ci_high[-1])],
        }
        print(f"  {name}: R0_hot={R0h:.2f} R0_cold={R0c:.2f} | raw crossing={raw_cross.crossed} "
              f"(final sign {int(raw.sign[-1])}) | offset-corr late sign={late_sign} "
              f"FDR-sig late={bool(late_sig.any())} | D_final={oc.diff_mean[-1]:+.3f} "
              f"CI[{oc.ci_low[-1]:+.3f},{oc.ci_high[-1]:+.3f}]")
        _store_oc(results["estimators"][name], oc, bh)

    # --- seed-budget / power (pilot variance at the target time) ---
    tgt = -1  # last in-window point
    win_idx = np.where(win)[0]
    j = win_idx[-1]
    var_hot = float(np.nanvar(est["hot"]["L_S"][:pilot, j], ddof=1))
    var_cold = float(np.nanvar(est["cold"]["L_S"][:pilot, j], ddof=1))
    # signal: |offset-corrected D| for L_S at the target (use the point estimate)
    signal = abs(results["estimators"]["L_S"]["oc_final_diff"]) or 1e-6
    need = analysis.required_ensemble_size(var_hot, var_cold, signal, power=power, alpha=alpha)
    se_diff = float(np.sqrt(var_hot / meta["n_realisations"] + var_cold / meta["n_realisations"]))
    results["power"] = {"pilot_var_hot": var_hot, "pilot_var_cold": var_cold,
                        "signal_LS_oc": signal, "required_n": need,
                        "achieved_n": meta["n_realisations"], "se_diff_LS": se_diff,
                        "se_to_signal": se_diff / signal}
    print(f"  power: signal(L_S oc)={signal:.3f}, SE_diff={se_diff:.3f}, "
          f"SE/signal={se_diff/signal:.2f}, required_n~{need} (have {meta['n_realisations']})")

    # --- four-outcome verdict ---
    verdict = _verdict(results)
    results["verdict"] = verdict
    print(f"\n  VERDICT: {verdict['outcome']}\n    {verdict['rationale']}")

    (run_dir / "m5_report.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    manifest.add_output("m5_report", run_dir / "m5_report.json")
    _plot_verdict(cfg, sweeps, est, e_inf, cutoff, upper, results, run_dir / "m5_verdict.png")
    manifest.add_output("m5_verdict_fig", run_dir / "m5_verdict.png")
    return results


def _store_oc(slot, oc, bh):
    slot["oc_times"] = oc.times.tolist()
    slot["oc_diff"] = oc.diff_mean.tolist()
    slot["oc_ci_low"] = oc.ci_low.tolist()
    slot["oc_ci_high"] = oc.ci_high.tolist()
    slot["oc_pvalue"] = oc.pvalue.tolist()
    slot["oc_fdr_rejected"] = bh.rejected.tolist()


def _verdict(results):
    """Four-outcome verdict (amendment A1: judged on the offset-corrected diff)."""
    ests = results["estimators"]
    # raw overtaking crossing (directional: hot overtakes cold => sign change to +)
    raw_cross = [n for n, e in ests.items() if e["raw_crossing"]]
    # offset-corrected: estimators with an FDR-significant late difference, by sign
    oc_pos = [n for n, e in ests.items() if e["oc_late_fdr_significant"] and e["oc_late_sign"] > 0]
    oc_neg = [n for n, e in ests.items() if e["oc_late_fdr_significant"] and e["oc_late_sign"] < 0]
    pw = results["power"]
    underpowered = pw["se_to_signal"] > 0.5  # SE more than half the signal => can't resolve

    if len(raw_cross) >= 2:
        return {"outcome": "coarsening_route_inversion",
                "rationale": f"raw L_hot overtakes L_cold (crossing) in {raw_cross} (>=2 estimators), "
                             "surviving the difference-bootstrap crossing rule."}
    if len(oc_pos) >= 2:
        return {"outcome": "coarsening_route_inversion",
                "rationale": f"no raw overtaking, but the offset-corrected difference is robustly "
                             f"POSITIVE (hotter coarsens faster after removing the head-start) in "
                             f"{oc_pos} with BH-FDR control — a rate inversion consistent with the "
                             "directional hypothesis, though the colder prep still leads in raw L."}
    if len(oc_neg) >= 2:
        return {"outcome": "no_supported_inversion",
                "rationale": f"the offset-corrected difference is robustly NEGATIVE in {oc_neg} "
                             "(colder coarsens at least as fast after head-start removal): the raw "
                             "lead is not an offset artefact hiding a hot-overtake. No inversion."}
    if underpowered:
        return {"outcome": "underdetermined",
                "rationale": f"no >=2-estimator effect survives, and SE/signal={pw['se_to_signal']:.2f} "
                             f">0.5 (required n~{pw['required_n']} vs achieved {pw['achieved_n']}): "
                             "statistics insufficient to separate the outcomes."}
    return {"outcome": "no_supported_inversion",
            "rationale": "no raw overtaking crossing and no >=2-estimator offset-corrected difference "
                         "survives uncertainty/FDR; the colder preparation leads throughout and the "
                         "offset-corrected difference is consistent with no genuine rate inversion."}


def _plot_verdict(cfg, sweeps, est, e_inf, cutoff, upper, results, path):
    pp = cfg["primary_pair"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    pos = sweeps > 0
    # raw L_S
    for label, c in (("cold", "navy"), ("hot", "crimson")):
        m = np.nanmean(est[label]["L_S"], 0); s = np.nanstd(est[label]["L_S"], 0, ddof=1) / np.sqrt(est[label]["L_S"].shape[0])
        ax[0].plot(sweeps[pos], m[pos], "-", color=c, label=f"{label} (T_i={pp['T_i_'+label]})")
        ax[0].fill_between(sweeps[pos], (m - 1.96 * s)[pos], (m + 1.96 * s)[pos], color=c, alpha=0.2)
    ax[0].axvline(upper, color="gray", ls=":", lw=1, label="sat. guard")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("sweeps"); ax[0].set_ylabel("$L_S$")
    ax[0].set_title("raw $L_S(t)$ (95% CI)"); ax[0].legend(fontsize=8)
    # offset-corrected difference for each estimator
    for name, c in (("L_C", "tab:green"), ("L_S", "tab:orange"), ("L_E", "tab:purple")):
        e = results["estimators"][name]
        t = np.array(e["oc_times"]); d = np.array(e["oc_diff"])
        lo = np.array(e["oc_ci_low"]); hi = np.array(e["oc_ci_high"])
        ax[1].plot(t, d, "-", color=c, label=name)
        ax[1].fill_between(t, lo, hi, color=c, alpha=0.18)
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xscale("log"); ax[1].set_xlabel("sweeps")
    ax[1].set_ylabel(r"$(L-R_0)_{hot}-(L-R_0)_{cold}$")
    ax[1].set_title("offset-corrected difference (verdict quantity)"); ax[1].legend(fontsize=8)
    # verdict text
    ax[2].axis("off")
    v = results["verdict"]; pw = results["power"]
    txt = (f"PRIMARY PAIR  T_i: {pp['T_i_hot']} (hot) vs {pp['T_i_cold']} (cold)\n"
           f"T_f={pp['T_f_over_Tc']} T_c,  N={cfg['model']['N']},  n={results['meta']['n_realisations']}\n"
           f"window: ({cutoff}, {int(upper)}] sweeps\n\n"
           f"VERDICT:\n  {v['outcome'].upper()}\n\n"
           f"SE/signal (L_S) = {pw['se_to_signal']:.2f}\n"
           f"required n ~ {pw['required_n']}\n\n"
           + "\n".join(f"{n}: rawX={results['estimators'][n]['raw_crossing']}, "
                       f"oc_late_sign={results['estimators'][n]['oc_late_sign']}, "
                       f"FDR={results['estimators'][n]['oc_late_fdr_significant']}"
                       for n in ("L_C", "L_S", "L_E")))
    ax[2].text(0.0, 0.5, txt, family="monospace", va="center", fontsize=9)
    fig.suptitle("Milestone 5 — pre-registered crossing search (primary pair)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #


def main(config_path, stage):
    cfg = io.load_config(config_path)
    run_id = {"gate": cfg["run_id_gate"], "sweep": cfg["run_id_sweep"], "analyse": cfg["run_id_verdict"]}[stage]
    note = {"gate": "M5 N=128 equilibration gate", "sweep": "M5 primary-pair ensembles",
            "analyse": "M5 crossing analysis + verdict"}[stage]
    print(f"Milestone 5 [{stage}]: N={cfg['model']['N']}, primary pair "
          f"{cfg['primary_pair']['T_i_hot']} vs {cfg['primary_pair']['T_i_cold']}, "
          f"T_f={cfg['primary_pair']['T_f_over_Tc']} T_c")
    run_dir = io.new_run_directory(REPO_ROOT / "results", run_id)
    io.dump_config(cfg, run_dir / "config.yaml")
    manifest = provenance.Manifest.build(run_id=run_id, timestamp=_utc(), config=cfg,
                                         seeds={"base": int(cfg["ensemble"]["seed"])},
                                         repo_root=REPO_ROOT, notes=note)
    {"gate": stage_gate, "sweep": stage_sweep, "analyse": stage_analyse}[stage](cfg, run_dir, manifest)
    manifest.write(run_dir / "manifest.json")
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    stage = "gate" if "--gate" in sys.argv else "sweep" if "--sweep" in sys.argv else "analyse" if "--analyse" in sys.argv else None
    if stage is None:
        raise SystemExit("specify a stage: --gate | --sweep | --analyse")
    cfg_path = argv[0] if argv else str(REPO_ROOT / "configs" / "milestone5_primary.yaml")
    main(cfg_path, stage)
