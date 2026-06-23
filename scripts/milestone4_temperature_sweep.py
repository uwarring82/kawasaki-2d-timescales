#!/usr/bin/env python
"""Milestone 4 — initial-temperature sweep at common T_f.

Phase A (equilibration gate): for each T_i, calibrate the preparation-sweep
budget with an *independent-chains* test. K chains are run from independent
random starts for a candidate budget B; their single (fully independent) final
samples of the energy AND the correlation length L_C give means compared against
a long, decorrelated reference equilibrium distribution. A budget converges when
both means are within `mean_sigma` combined SEM of the reference (mean stability
is robust; the two-sample KS p-values are noisy at modest K near T_c and are
reported only as supporting evidence). The gated budget is the smallest
converged candidate, raised if necessary to comfortably cover the measured
energy autocorrelation time (>= safety_tau_mult * tau). M = 0 is conserved
exactly, so there is no magnetisation distribution to equilibrate (it is a delta
at 0); we assert it.

Phase B (sweep): common protocol — fixed T_f, N, schedule, and local-Kawasaki
kinetic kernel — varying only T_i and its gated prep budget. Ensemble post-quench
trajectories E(t), L_C(t), L_S(t), L_E(t); the manifest records the gated budget
per T_i.

Usage:
    python scripts/milestone4_temperature_sweep.py [config]            # Phase A (gate)
    python scripts/milestone4_temperature_sweep.py [config] --sweep    # Phase B (sweep)
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

from kawasaki2d import T_C, analysis, dynamics, equilibration as eq, io, observables as obs, protocols, provenance  # noqa: E402
from kawasaki2d.lattice import init_lattice, magnetisation, total_energy  # noqa: E402
from kawasaki2d.rng import make_rng, spawn_rngs  # noqa: E402

try:
    from scipy.stats import ks_2samp
except Exception:  # pragma: no cover
    ks_2samp = None


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Phase A — equilibration gate                                                 #
# --------------------------------------------------------------------------- #


def _independent_final(N, M, T, budget, K, seed, kernel):
    """K independent chains, each run `budget` sweeps; return per-chain (e/spin, L_C)."""
    run = eq._KERNELS[kernel]
    E = np.empty(K)
    LC = np.empty(K)
    for i, rng in enumerate(spawn_rngs(seed, K)):
        lat = init_lattice(N, M, rng=rng)
        run(lat, T, budget, rng)
        assert magnetisation(lat) == M  # M conserved exactly
        E[i] = total_energy(lat) / lat.size
        LC[i] = obs.length_from_correlation(lat)
    return E, LC


def _reference_dist(N, M, T, burn, n_samples, spacing, seed, kernel):
    """Decorrelated equilibrium samples (e/spin, L_C) from one long chain."""
    run = eq._KERNELS[kernel]
    rng = make_rng(seed)
    lat = init_lattice(N, M, rng=rng)
    run(lat, T, burn, rng)
    E = np.empty(n_samples)
    LC = np.empty(n_samples)
    for i in range(n_samples):
        run(lat, T, spacing, rng)
        E[i] = total_energy(lat) / lat.size
        LC[i] = obs.length_from_correlation(lat)
    return E, LC


def calibrate_leg(N, M, leg, cal, kernel, base_seed):
    T = float(leg["T_i"])
    candidates = [int(c) for c in leg["candidates"]]
    ref_burn = int(leg["ref_burn"])
    K = int(cal["n_test_chains"])
    n_ref = int(cal["n_ref_samples"])
    alpha = float(cal["ks_alpha"])

    mean_sigma = float(cal.get("mean_sigma", 4.0))

    # energy autocorrelation time (in sweeps) for context + safety floor
    tau, _ = eq.measure_autocorrelation_sweeps(
        N, T, M, make_rng(base_seed + 1), kernel=kernel,
        burn=min(ref_burn, 8000), n_samples=4000, sample_every=2,
    )
    spacing = max(2, int(round(float(cal["ref_spacing_tau_mult"]) * tau)))
    refE, refLC = _reference_dist(N, M, T, ref_burn, n_ref, spacing, base_seed + 2, kernel)
    refE_m, refLC_m = float(refE.mean()), float(refLC.mean())
    refE_sem = float(refE.std(ddof=1) / np.sqrt(len(refE)))
    refLC_sem = float(refLC.std(ddof=1) / np.sqrt(len(refLC)))

    # Convergence is gated on MEAN STABILITY (robust) — |mean - ref| within
    # `mean_sigma` combined SEM for BOTH energy and L_C — with the KS p-values
    # reported as supporting evidence (noisy at modest K near criticality).
    rows = []
    for b in candidates:
        E, LC = _independent_final(N, M, T, b, K, base_seed + 1000 + b, kernel)
        eM, lM = float(E.mean()), float(LC.mean())
        eS = float(E.std(ddof=1) / np.sqrt(len(E)))
        lS = float(LC.std(ddof=1) / np.sqrt(len(LC)))
        dE = abs(eM - refE_m); dLC = abs(lM - refLC_m)
        conv = bool(dE <= mean_sigma * np.hypot(eS, refE_sem)
                    and dLC <= mean_sigma * np.hypot(lS, refLC_sem))
        rows.append({
            "budget": b, "mean_energy": eM, "sem_energy": eS, "mean_LC": lM, "sem_LC": lS,
            "ks_p_energy": float(ks_2samp(E, refE).pvalue),
            "ks_p_LC": float(ks_2samp(LC, refLC).pvalue),
            "energy_gap_sem": dE / np.hypot(eS, refE_sem),
            "LC_gap_sem": dLC / np.hypot(lS, refLC_sem),
            "converged": conv,
        })

    # smallest converged candidate (means stable vs the reference)
    converged_budgets = [r["budget"] for r in rows if r["converged"]]
    smallest_conv = converged_budgets[0] if converged_budgets else None
    tau_floor = int(np.ceil(float(cal["safety_tau_mult"]) * tau))
    if smallest_conv is None:
        gated = int(candidates[-1])  # nothing converged: use largest, flag below
    else:
        need = max(smallest_conv, tau_floor)
        # round up to the smallest candidate >= need (keep budget among tested values)
        ge = [b for b in candidates if b >= need]
        gated = int(ge[0]) if ge else int(candidates[-1])
    return {
        "T_i": T, "tau_E_sweeps": tau, "ref_spacing_sweeps": spacing,
        "ref_mean_energy": refE_m, "ref_sem_energy": refE_sem,
        "ref_mean_LC": refLC_m, "ref_sem_LC": refLC_sem,
        "candidates": rows, "smallest_converged_budget": smallest_conv,
        "tau_floor_budget": tau_floor, "gated_budget": gated,
        "gate_passed": smallest_conv is not None,
    }


def phase_a(cfg, run_dir, manifest):
    N = int(cfg["model"]["N"])
    M = int(cfg["model"]["magnetisation"])
    kernel = cfg["preparation"]["kernel"]
    cal = cfg["calibration"]
    base = int(cfg["ensemble"]["seed"])

    # high-T sanity reference: at T_i=10, e/spin should match the high-T expansion -2/T.
    legs_out = []
    for li, leg in enumerate(cfg["T_i_legs"]):
        res = calibrate_leg(N, M, leg, cal, kernel, base + 100 * (li + 1))
        if abs(res["T_i"] - 10.0) < 1e-9:
            res["high_T_expectation_minus2_over_T"] = -2.0 / res["T_i"]
        legs_out.append(res)
        flag = "" if res["gate_passed"] else "  [NO candidate converged — using largest]"
        print(f"  T_i={res['T_i']:5.2f}: tau_E={res['tau_E_sweeps']:5.1f} sw | "
              f"smallest-converged={res['smallest_converged_budget']} | "
              f"tau-floor={res['tau_floor_budget']} | GATED={res['gated_budget']} sweeps{flag}")
        for r in res["candidates"]:
            print(f"       B={r['budget']:6d}: |Δ⟨e⟩|={r['energy_gap_sem']:.1f}σ "
                  f"|Δ⟨L_C⟩|={r['LC_gap_sem']:.1f}σ  KS_p(E)={r['ks_p_energy']:.2f} "
                  f"KS_p(L_C)={r['ks_p_LC']:.2f} -> {'converged' if r['converged'] else 'no'}")

    budgets = {str(r["T_i"]): r["gated_budget"] for r in legs_out}
    (run_dir / "budgets.json").write_text(json.dumps({"budgets": budgets, "legs": legs_out},
                                                     indent=2, default=str) + "\n")
    manifest.add_output("budgets", run_dir / "budgets.json")
    _plot_calibration(legs_out, float(cfg["calibration"].get("mean_sigma", 4.0)),
                      run_dir / "equilibration_gate.png")
    manifest.add_output("equilibration_gate", run_dir / "equilibration_gate.png")
    return legs_out


def _plot_calibration(legs, mean_sigma, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for res in legs:
        b = [r["budget"] for r in res["candidates"]]
        ge = [r["energy_gap_sem"] for r in res["candidates"]]
        gl = [r["LC_gap_sem"] for r in res["candidates"]]
        lbl = f"T_i={res['T_i']} (gated {res['gated_budget']})"
        ax[0].plot(b, ge, "-o", ms=4, label=lbl)
        ax[1].plot(b, gl, "-o", ms=4, label=lbl)
    for a, title in ((ax[0], r"energy mean gap $|\langle e\rangle-\langle e\rangle_{ref}|$"),
                     (ax[1], r"$L_C$ mean gap $|\langle L_C\rangle-\langle L_C\rangle_{ref}|$")):
        a.axhline(mean_sigma, color="r", ls="--", lw=1, label=f"gate = {mean_sigma}σ")
        a.set_xscale("log"); a.set_xlabel("prep sweeps (budget)")
        a.set_ylabel("gap (combined SEM units)"); a.set_title(title); a.legend(fontsize=7)
    fig.suptitle("M4 Phase A — equilibration gate: mean stability vs long reference (converged = below "
                 f"{mean_sigma}σ)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# Phase B — common-protocol sweep                                              #
# --------------------------------------------------------------------------- #


def phase_b(cfg, budgets, run_dir, manifest):
    N = int(cfg["model"]["N"])
    M = int(cfg["model"]["magnetisation"])
    kernel = cfg["preparation"]["kernel"]
    T_f = float(cfg["quench"]["T_f"])
    schedule = protocols.log_schedule(int(cfg["quench"]["schedule"]["t_max"]),
                                      int(cfg["quench"]["schedule"]["n_points"]))
    sweeps = np.asarray(schedule, float)
    n_real = int(cfg["ensemble"]["n_realisations"])
    base = int(cfg["ensemble"]["seed"])
    T_grid = [float(leg["T_i"]) for leg in cfg["T_i_legs"]]

    # common E_inf(T_f) (same for every T_i)
    ei = cfg["einf"]
    einf = eq.estimate_equilibrium_energy(N, T_f, M, make_rng(base + 7), kernel=ei["kernel"],
                                          sweeps_burn=int(ei["sweeps_burn"]),
                                          sample_every=int(ei["sample_every"]),
                                          n_samples=int(ei["n_samples"]))
    print(f"  common E_inf(T_f) = {einf.mean:.4f} +/- {einf.sd:.4f} /spin")

    per_T = {}
    for ti, T_i in enumerate(T_grid):
        budget = int(budgets[str(T_i)])
        rngs = spawn_rngs(base + 1000 * (ti + 1), n_real)
        E = np.empty((n_real, len(sweeps)))
        LC = np.empty((n_real, len(sweeps)))
        LS = np.empty((n_real, len(sweeps)))
        init_LC = np.empty(n_real)
        for k, rng in enumerate(rngs):
            prep = protocols.prepare_initial_state(N, T_i, M, rng, kernel=kernel, n_sweeps=budget)
            init_LC[k] = obs.length_from_correlation(prep.lattice)
            traj = protocols.coarsening_trajectory(prep.lattice, T_f, schedule, rng,
                                                   with_clusters=False, record_sk=False)
            E[k] = [r["energy_per_spin"] for r in traj.rows]
            LC[k] = [r["L_C"] for r in traj.rows]
            LS[k] = [r["L_S"] for r in traj.rows]
        stE, stLC, stLS = (analysis.ensemble_stats(a) for a in (E, LC, LS))
        excess = E - einf.mean
        with np.errstate(divide="ignore", invalid="ignore"):
            LE = np.where(excess > 0, 1.0 / excess, np.nan)
        stLE = analysis.ensemble_stats(LE)

        rows = [{"sweep": int(sweeps[j]),
                 "E_per_spin_mean": stE.mean[j],
                 "L_C_mean": stLC.mean[j], "L_C_sem": stLC.sem[j],
                 "L_S_mean": stLS.mean[j], "L_S_sem": stLS.sem[j],
                 "L_E_mean": stLE.mean[j], "L_E_sem": stLE.sem[j]} for j in range(len(sweeps))]
        cols = ("sweep", "E_per_spin_mean", "L_C_mean", "L_C_sem", "L_S_mean", "L_S_sem",
                "L_E_mean", "L_E_sem")
        io.write_trajectory_csv(run_dir / f"trajectory_Ti{T_i:04.1f}.csv", rows, columns=cols)
        manifest.add_output(f"trajectory_Ti{T_i}", run_dir / f"trajectory_Ti{T_i:04.1f}.csv")
        per_T[T_i] = {"prep_budget": budget, "initial_L_C_mean": float(init_LC.mean()),
                      "initial_L_C_sem": float(init_LC.std(ddof=1) / np.sqrt(n_real)),
                      "stE": stE, "stLC": stLC, "stLS": stLS, "sweeps": sweeps}
        print(f"  T_i={T_i:5.2f} (budget {budget}): initial L_C={init_LC.mean():.2f}, "
              f"final L_S={stLS.mean[-1]:.2f}")

    _plot_sweep(per_T, T_grid, einf.mean, run_dir / "temperature_sweep.png")
    manifest.add_output("temperature_sweep", run_dir / "temperature_sweep.png")
    summary = {"T_f": T_f, "T_f_over_Tc": T_f / T_C, "N": N, "n_realisations": n_real,
               "e_inf": einf.mean, "e_inf_sd": einf.sd,
               "prep_budgets": {str(t): int(budgets[str(t)]) for t in T_grid},
               "initial_L_C": {str(t): per_T[t]["initial_L_C_mean"] for t in T_grid},
               "final_L_S": {str(t): float(per_T[t]["stLS"].mean[-1]) for t in T_grid}}
    (run_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    manifest.add_output("sweep_summary", run_dir / "sweep_summary.json")
    return summary


def _plot_sweep(per_T, T_grid, e_inf, path):
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(min(T_grid), max(T_grid))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for T_i in T_grid:
        d = per_T[T_i]; sw = d["sweeps"]; pos = sw > 0; c = cmap(norm(T_i))
        ax[0].plot(sw, d["stE"].mean, "-", color=c, label=f"$T_i$={T_i}")
        ax[1].plot(sw[pos], d["stLS"].mean[pos], "-", color=c, label=f"$T_i$={T_i}")
        ax[1].fill_between(sw[pos], d["stLS"].ci_low[pos], d["stLS"].ci_high[pos], color=c, alpha=0.15)
    ax[0].axhline(e_inf, color="k", ls=":", lw=1, label="$E_\\infty$")
    ax[0].set_xlabel("sweeps"); ax[0].set_ylabel("energy / spin"); ax[0].set_title("E(t) by $T_i$"); ax[0].legend(fontsize=7)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("sweeps"); ax[1].set_ylabel("$L_S$"); ax[1].set_title("$L_S(t)$ by $T_i$ (95% CI)"); ax[1].legend(fontsize=7)
    # initial correlation length per T_i ("starting ahead" structure)
    iLC = [per_T[T]["initial_L_C_mean"] for T in T_grid]
    iLCe = [per_T[T]["initial_L_C_sem"] for T in T_grid]
    ax[2].errorbar(T_grid, iLC, yerr=iLCe, fmt="o-", capsize=3)
    ax[2].axvline(T_C, color="r", ls="--", lw=1, label="$T_c$")
    ax[2].set_xlabel("$T_i$"); ax[2].set_ylabel("initial $L_C$ (post-prep)")
    ax[2].set_title("initial correlation length vs $T_i$"); ax[2].legend(fontsize=8)
    fig.suptitle("Milestone 4 — initial-temperature sweep at common $T_f$")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #


def main(config_path, do_sweep=False):
    cfg = io.load_config(config_path)
    print(f"Milestone 4 ({'SWEEP' if do_sweep else 'GATE'}): N={cfg['model']['N']}, "
          f"T_f={cfg['quench']['T_f']}, T_i grid={[l['T_i'] for l in cfg['T_i_legs']]}")
    if not do_sweep:
        run_dir = io.new_run_directory(REPO_ROOT / "results", cfg["run_id_calibration"])
        io.dump_config(cfg, run_dir / "config.yaml")
        manifest = provenance.Manifest.build(
            run_id=cfg["run_id_calibration"], timestamp=_utc(), config=cfg,
            seeds={"base": int(cfg["ensemble"]["seed"])}, repo_root=REPO_ROOT,
            notes="M4 Phase A: independent-chains equilibration gate")
        print("\n=== Phase A: equilibration gate ===")
        phase_a(cfg, run_dir, manifest)
        manifest.write(run_dir / "manifest.json")
        print(f"\nwrote {run_dir}  (review budgets.json, then re-run with --sweep)")
    else:
        cal_dir = REPO_ROOT / "results" / cfg["run_id_calibration"]
        budgets = json.loads((cal_dir / "budgets.json").read_text())["budgets"]
        run_dir = io.new_run_directory(REPO_ROOT / "results", cfg["run_id_sweep"])
        io.dump_config(cfg, run_dir / "config.yaml")
        manifest = provenance.Manifest.build(
            run_id=cfg["run_id_sweep"], timestamp=_utc(), config=cfg,
            seeds={"base": int(cfg["ensemble"]["seed"]), "prep_budgets": budgets},
            repo_root=REPO_ROOT, notes="M4 Phase B: common-protocol initial-temperature sweep")
        print("\n=== Phase B: common-protocol sweep ===")
        phase_b(cfg, budgets, run_dir, manifest)
        manifest.write(run_dir / "manifest.json")
        print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_sweep = "--sweep" in sys.argv
    cfg_path = argv[0] if argv else str(REPO_ROOT / "configs" / "temperature_sweep_m4.yaml")
    main(cfg_path, do_sweep=do_sweep)
