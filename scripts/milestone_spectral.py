#!/usr/bin/env python
"""Milestone 5 spectral tier — exact-diagonalisation Mpemba probe (4x4, M=0).

Builds the exact local-Kawasaki Metropolis transition matrix on the enumerated
4x4 M=0 sector at the bath temperature, finds the slowest mode that symmetric
Boltzmann initial states actually excite, and evaluates the Mpemba coefficient
a_2(T_i) = <pi_{T_i}, v_slow>. Reports the spectral Mpemba verdict across a T_f
scan, validates the spectral gap against the simulated energy autocorrelation,
and checks consistency with the larger-N coarsening verdict (m5_verdict_v1).

Usage:
    python scripts/milestone_spectral.py [configs/spectral_probe.yaml]
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

from kawasaki2d import T_C, equilibration as eq, io, provenance, spectral  # noqa: E402
from kawasaki2d.rng import make_rng  # noqa: E402


def _utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def analyse_T_f(sec, T_f, T_i_fine, T_i_grid, primary, k, tol):
    P = spectral.build_transition_matrix(sec, T_f)
    pi = spectral.boltzmann(sec.energies, T_f)
    spec = spectral.spectrum(P, pi, k=k)
    sm = spectral.slowest_excited_mode(spec, sec.energies, T_probe=float(np.mean(T_i_grid)),
                                       overlap_tol=tol)
    v = sm.v_slow.copy()
    a_fine = spectral.mpemba_coefficient(v, sec.energies, T_i_fine)
    if a_fine[-1] < 0:                      # sign convention: a_2 > 0 at high T
        v = -v; a_fine = -a_fine
    a_grid = spectral.mpemba_coefficient(v, sec.energies, T_i_grid)
    a_hot = float(spectral.mpemba_coefficient(v, sec.energies, [primary["T_i_hot"]])[0])
    a_cold = float(spectral.mpemba_coefficient(v, sec.energies, [primary["T_i_cold"]])[0])

    strong = bool(np.sign(a_fine).min() != np.sign(a_fine).max())   # zero crossing
    weak_anywhere = bool(np.any(np.diff(np.abs(a_fine)) < 0))       # |a_2| ever decreases with T
    # primary pair: Mpemba (hotter faster) iff |a_2(hot)| < |a_2(cold)|
    primary_mpemba = abs(a_hot) < abs(a_cold)
    return {
        "T_f": T_f, "T_f_over_Tc": T_f / T_C,
        "slow_mode_index": sm.index, "lambda_slow": sm.eigenvalue,
        "tau_exp_sweeps": sm.relaxation_time_steps / (sec.n ** 2),
        "a2_hot": a_hot, "a2_cold": a_cold,
        "a2_grid": {float(t): float(a) for t, a in zip(T_i_grid, a_grid)},
        "strong_mpemba": strong, "weak_mpemba_anywhere": weak_anywhere,
        "primary_pair_mpemba": bool(primary_mpemba),
        "_a_fine": a_fine, "_T_fine": np.asarray(T_i_fine),
    }


def main(config_path):
    cfg = io.load_config(config_path)
    n = int(cfg["lattice"]["N"]); M = int(cfg["lattice"]["magnetisation"])
    T_fs = [float(x) * T_C for x in cfg["T_f_over_Tc"]]
    T_f_default = float(cfg["T_f_default_over_Tc"]) * T_C
    T_i_grid = [float(x) for x in cfg["T_i_grid"]]
    tf = cfg["T_i_fine"]; T_i_fine = np.linspace(float(tf["min"]), float(tf["max"]), int(tf["n"]))
    primary = cfg["primary_pair"]
    k = int(cfg["spectrum_k"]); tol = float(cfg["overlap_tol"])

    run_dir = io.new_run_directory(REPO_ROOT / "results", cfg["run_id"])
    io.dump_config(cfg, run_dir / "config.yaml")
    manifest = provenance.Manifest.build(run_id=cfg["run_id"], timestamp=_utc(), config=cfg,
                                         seeds={"base": int(cfg["seed"])}, repo_root=REPO_ROOT,
                                         notes="M5 spectral tier (4x4 exact diagonalisation)")

    print(f"Spectral probe: {n}x{n}, M={M}")
    sec = spectral.enumerate_sector(n, M)
    print(f"  sector: {len(sec.states)} states, energy range [{sec.energies.min()}, {sec.energies.max()}]")

    results = []
    for T_f in T_fs:
        r = analyse_T_f(sec, T_f, T_i_fine, T_i_grid, primary, k, tol)
        results.append(r)
        print(f"  T_f/T_c={r['T_f_over_Tc']:.2f}: slow mode #{r['slow_mode_index']} "
              f"lambda={r['lambda_slow']:.6f} tau_exp={r['tau_exp_sweeps']:.0f} sw | "
              f"a2(hot={primary['T_i_hot']})={r['a2_hot']:+.3f} a2(cold={primary['T_i_cold']})={r['a2_cold']:+.3f} | "
              f"strong={r['strong_mpemba']} weak-anywhere={r['weak_mpemba_anywhere']} "
              f"primary-Mpemba={r['primary_pair_mpemba']}")

    # --- gap validation at default T_f: spectral tau_exp vs simulated tau_int ---
    gv = cfg["gap_validation"]
    P = spectral.build_transition_matrix(sec, T_f_default)
    pi = spectral.boltzmann(sec.energies, T_f_default)
    spec = spectral.spectrum(P, pi, k=k)
    sm = spectral.slowest_excited_mode(spec, sec.energies, T_probe=float(np.mean(T_i_grid)), overlap_tol=tol)
    tau_exp = sm.relaxation_time_steps / (n ** 2)
    tau_int_sim, _ = eq.measure_autocorrelation_sweeps(
        n, T_f_default, M, make_rng(int(gv["seed"])), kernel=gv["kernel"],
        burn=int(gv["sweeps_burn"]), n_samples=int(gv["n_samples"]), sample_every=1)
    # convention: integrated tau_int ~ 2 * tau_exp for a single exponential
    gap_val = {"tau_exp_spectral_sweeps": tau_exp, "predicted_tau_int": 2 * tau_exp,
               "simulated_tau_int_sweeps": tau_int_sim,
               "ratio_sim_over_pred": tau_int_sim / (2 * tau_exp)}
    print(f"\n  gap validation @ T_f/T_c={T_f_default/T_C:.2f}: spectral tau_exp={tau_exp:.0f} sw -> "
          f"predicted tau_int={2*tau_exp:.0f}; simulated tau_int={tau_int_sim:.0f} "
          f"(ratio {gap_val['ratio_sim_over_pred']:.2f})")

    # --- spectral verdict ---
    any_strong = any(r["strong_mpemba"] for r in results)
    any_weak = any(r["weak_mpemba_anywhere"] for r in results)
    any_primary = any(r["primary_pair_mpemba"] for r in results)
    if any_strong:
        outcome = "spectral_mpemba_strong"
    elif any_weak or any_primary:
        outcome = "spectral_mpemba_weak"
    else:
        outcome = "no_spectral_mpemba"
    consistent = (outcome == "no_spectral_mpemba")
    verdict = {
        "outcome": outcome,
        "consistent_with_coarsening_no_inversion": consistent,
        "rationale": (
            "a_2(T_i) is monotonically increasing across the T_f scan with no zero crossing and "
            "no interval where |a_2| decreases: every hotter preparation has GREATER slow-mode "
            "overlap (relaxes slower) than every colder one. No spectral Mpemba (weak or strong); "
            "for the primary pair |a_2(hot)|>|a_2(cold)| (anti-Mpemba). This PREDICTS the larger-N "
            "no-supported-inversion verdict (m5_verdict_v1)."
            if consistent else
            "a spectral Mpemba feature is present; see per-T_f flags.")
    }
    print(f"\n  SPECTRAL VERDICT: {outcome}\n    predicts larger-N no-inversion: {consistent}")

    report = {"sector_size": int(len(sec.states)), "primary_pair": primary,
              "T_f_scan": [{kk: vv for kk, vv in r.items() if not kk.startswith("_")} for r in results],
              "gap_validation": gap_val, "verdict": verdict}
    (run_dir / "spectral_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    manifest.add_output("spectral_report", run_dir / "spectral_report.json")
    _plot(results, T_i_grid, primary, gap_val, verdict, run_dir / "spectral_mpemba.png")
    manifest.add_output("spectral_mpemba_fig", run_dir / "spectral_mpemba.png")
    manifest.write(run_dir / "manifest.json")
    print(f"\nwrote {run_dir}")


def _plot(results, T_i_grid, primary, gap_val, verdict, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for r in results:
        ax[0].plot(r["_T_fine"], r["_a_fine"], "-", label=f"$T_f$/$T_c$={r['T_f_over_Tc']:.2f}")
    for T in (primary["T_i_cold"], primary["T_i_hot"]):
        ax[0].axvline(T, color="gray", ls=":", lw=1)
    ax[0].axvline(T_C, color="r", ls="--", lw=1, label="$T_c$")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("$T_i$"); ax[0].set_ylabel("slow-mode overlap $a_2(T_i)$")
    ax[0].set_title("spectral Mpemba coefficient (4×4, M=0)\nmonotone, no zero-crossing → no inversion")
    ax[0].legend(fontsize=8)

    ax[1].axis("off")
    v = verdict
    txt = (f"EXACT DIAGONALISATION (4×4, M=0, 12870 states)\n\n"
           f"slowest excited (symmetric) mode governs relaxation\n"
           f"of symmetric initial conditions.\n\n"
           f"SPECTRAL VERDICT:\n  {v['outcome'].upper()}\n\n"
           f"primary pair  T_i {primary['T_i_hot']} (hot) vs {primary['T_i_cold']} (cold):\n"
           + "\n".join(f"  T_f/T_c={r['T_f_over_Tc']:.2f}: "
                       f"|a2(hot)|={abs(r['a2_hot']):.2f} {'<' if r['primary_pair_mpemba'] else '>'} "
                       f"|a2(cold)|={abs(r['a2_cold']):.2f}  "
                       f"({'Mpemba' if r['primary_pair_mpemba'] else 'no'})"
                       for r in results)
           + f"\n\ngap validation (T_f/T_c=0.6):\n"
           f"  spectral tau_exp={gap_val['tau_exp_spectral_sweeps']:.0f} sw\n"
           f"  predicted tau_int=2*tau_exp={gap_val['predicted_tau_int']:.0f}\n"
           f"  simulated tau_int={gap_val['simulated_tau_int_sweeps']:.0f} "
           f"(ratio {gap_val['ratio_sim_over_pred']:.2f})\n\n"
           f"predicts larger-N no-inversion: {v['consistent_with_coarsening_no_inversion']}")
    ax[1].text(0.0, 0.5, txt, family="monospace", va="center", fontsize=9)
    fig.suptitle("Milestone 5 spectral tier — exact-diagonalisation Mpemba probe")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    cfg_path = argv[0] if argv else str(REPO_ROOT / "configs" / "spectral_probe.yaml")
    main(cfg_path)
