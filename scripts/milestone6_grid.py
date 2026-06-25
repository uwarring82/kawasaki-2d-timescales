#!/usr/bin/env python
"""Milestone 6 — grid crossing search across (T_i, T_f, N).

Broadens the M5 primary-pair verdict to the wider pre-registered grid (the
across-pairs FDR deferred in M5 amendment A2). For each (N, T_f) cell, ensembles
are run for every T_i; every hot>cold pair is tested for a *directional*
offset-corrected inversion (hotter overtakes colder after the head-start is
removed), and Benjamini-Hochberg FDR is applied across the whole grid of per-pair
tests.

Per pair, per estimator (L_C, L_S, L_E): the offset-corrected difference
D = (L_hot-R0_hot) - (L_cold-R0_cold) (fixed-1/3, pre-registered) over the
saturation-guarded window, with a one-sided "hot ahead" bootstrap p-value. The
per-pair significance is the 2nd-smallest of the three one-sided p-values (the
level at which the two-of-three rule is met). BH-FDR over all per-pair p-values.

Stages:
    python scripts/milestone6_grid.py [config] --sweep     # run ensembles (slow)
    python scripts/milestone6_grid.py [config] --analyse   # pairwise verdict (fast)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import combinations
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


def _key(N, tfr, T_i):
    return f"N{N:03d}_Tf{tfr:.2f}_Ti{T_i:04.1f}"


# --------------------------------------------------------------------------- #
# Sweep                                                                        #
# --------------------------------------------------------------------------- #


def stage_sweep(cfg, run_dir, manifest):
    M = int(cfg["model"]["magnetisation"])
    kernel = cfg["preparation"]["kernel"]
    budgets = {str(k): int(v) for k, v in cfg["preparation"]["budgets"].items()}
    Ns = [int(n) for n in cfg["grid"]["N"]]
    Tfs = [float(x) for x in cfg["grid"]["T_f_over_Tc"]]
    Tis = [float(x) for x in cfg["grid"]["T_i"]]
    n_real = int(cfg["ensemble"]["n_realisations"])
    base = int(cfg["ensemble"]["seed"])
    ei = cfg["einf"]

    meta = {"Ns": Ns, "Tfs": Tfs, "Tis": Tis, "n_realisations": n_real, "e_inf": {}, "schedule": {}}
    for N in Ns:
        sch = cfg["schedule"][str(N)]
        sweeps = protocols.log_schedule(int(sch["t_max"]), int(sch["n_points"]))
        np.save(run_dir / f"sweeps_N{N:03d}.npy", sweeps)
        meta["schedule"][str(N)] = sweeps.tolist()
        for tfi, tfr in enumerate(Tfs):
            T_f = tfr * T_C
            einf = eq.estimate_equilibrium_energy(N, T_f, M, make_rng(base + 9000 + N * 13 + tfi),
                                                  kernel=ei["kernel"], sweeps_burn=int(ei["sweeps_burn"]),
                                                  sample_every=int(ei["sample_every"]), n_samples=int(ei["n_samples"]))
            meta["e_inf"][f"N{N:03d}_Tf{tfr:.2f}"] = einf.mean
            for tii, T_i in enumerate(Tis):
                budget = budgets[str(T_i)]
                cell_seed = base + 100000 * N + 1000 * tfi + 10 * tii
                rngs = spawn_rngs(cell_seed, n_real)
                E = np.empty((n_real, len(sweeps))); LC = np.empty_like(E); LS = np.empty_like(E)
                for k, rng in enumerate(rngs):
                    prep = protocols.prepare_initial_state(N, T_i, M, rng, kernel=kernel, n_sweeps=budget)
                    traj = protocols.coarsening_trajectory(prep.lattice, T_f, sweeps, rng,
                                                           with_clusters=False, record_sk=False)
                    E[k] = [r["energy_per_spin"] for r in traj.rows]
                    LC[k] = [r["L_C"] for r in traj.rows]
                    LS[k] = [r["L_S"] for r in traj.rows]
                key = _key(N, tfr, T_i)
                for nm, arr in (("E", E), ("LC", LC), ("LS", LS)):
                    np.save(run_dir / f"{key}_{nm}.npy", arr)
                print(f"  {key}: final L_S={np.nanmean(LS[:,-1]):5.2f}  max L_S/N={np.nanmax(np.nanmean(LS,0))/N:.3f}")
    (run_dir / "grid_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    manifest.add_output("grid_meta", run_dir / "grid_meta.json")
    return meta


# --------------------------------------------------------------------------- #
# Analyse                                                                      #
# --------------------------------------------------------------------------- #


def _load_cell(d, N, tfr, T_i, e_inf):
    key = _key(N, tfr, T_i)
    E = np.load(d / f"{key}_E.npy")
    with np.errstate(divide="ignore", invalid="ignore"):
        LE = np.where(E - e_inf > 0, 1.0 / (E - e_inf), np.nan)
    return {"L_C": np.load(d / f"{key}_LC.npy"), "L_S": np.load(d / f"{key}_LS.npy"), "L_E": LE}


def stage_analyse(cfg, run_dir, manifest):
    an = cfg["analysis"]
    cutoff = int(an["lower_cutoff_sweeps"]); fs = float(an["finite_size_fraction"])
    n_boot = int(an["n_boot"]); ci = float(an["ci"]); fdr_alpha = float(an["fdr_alpha"])
    sweep_dir = REPO_ROOT / "results" / cfg["run_id_sweep"]
    meta = json.loads((sweep_dir / "grid_meta.json").read_text())
    rng = make_rng(int(cfg["ensemble"]["seed"]) + 777)

    pair_records = []
    skipped = []
    for N in meta["Ns"]:
        sweeps = np.load(sweep_dir / f"sweeps_N{N:03d}.npy")
        for tfr in meta["Tfs"]:
            e_inf = meta["e_inf"][f"N{N:03d}_Tf{tfr:.2f}"]
            cells = {T_i: _load_cell(sweep_dir, N, tfr, T_i, e_inf) for T_i in meta["Tis"]}
            for hot, cold in [(max(a, b), min(a, b)) for a, b in combinations(meta["Tis"], 2)]:
                # per-pair saturation window: BOTH legs below the finite-size guard
                below = (sweeps > cutoff)
                below &= (np.nanmean(cells[hot]["L_S"], 0) < fs * N)
                below &= (np.nanmean(cells[cold]["L_S"], 0) < fs * N)
                if below.sum() < 4:
                    # no pre-saturation window for this pair (e.g. a near-critical
                    # cold leg already spans a small lattice) — recorded, not tested.
                    skipped.append({"N": N, "T_f_over_Tc": tfr, "T_i_hot": hot, "T_i_cold": cold})
                    continue
                upper = float(sweeps[below][-1])
                keep = sweeps <= upper
                ests = {}
                for nm in ("L_C", "L_S", "L_E"):
                    oc, R0h, R0c = analysis.offset_corrected_difference_bootstrap(
                        cells[hot][nm][:, keep], cells[cold][nm][:, keep], sweeps[keep], rng,
                        cutoff=cutoff, n_boot=n_boot, ci=ci)
                    D = float(oc.diff_mean[-1]); p2 = float(oc.pvalue[-1])
                    p_hot = (p2 / 2.0) if D > 0 else (1.0 - p2 / 2.0)  # one-sided: hot ahead
                    ests[nm] = {"D_final": D, "p_hot": p_hot,
                                "excl_zero": bool(oc.excludes_zero[-1]), "sign": int(np.sign(D))}
                p_pair = sorted(ests[nm]["p_hot"] for nm in ests)[1]  # two-of-three level
                n_hot_sig = sum(1 for nm in ests if ests[nm]["excl_zero"] and ests[nm]["sign"] > 0)
                pair_records.append({"N": N, "T_f_over_Tc": tfr, "T_i_hot": hot, "T_i_cold": cold,
                                     "upper": upper, "p_pair": p_pair, "n_hot_significant": n_hot_sig,
                                     "L_S_D": ests["L_S"]["D_final"], "L_C_D": ests["L_C"]["D_final"],
                                     "L_E_D": ests["L_E"]["D_final"]})

    if pair_records:
        p_all = np.array([r["p_pair"] for r in pair_records])
        bh = analysis.benjamini_hochberg(p_all, alpha=fdr_alpha)
        for r, rej in zip(pair_records, bh.rejected):
            r["fdr_supported_inversion"] = bool(rej)
        n_supported = int(bh.rejected.sum())
        bh_threshold = float(bh.threshold)
    else:
        n_supported, bh_threshold = 0, 0.0

    n_two_of_three_raw = sum(1 for r in pair_records if r["n_hot_significant"] >= 2)
    verdict = {
        "n_cells": len({(r["N"], r["T_f_over_Tc"]) for r in pair_records}),
        "n_pairs_tested": len(pair_records),
        "n_pairs_skipped_saturated": len(skipped),
        "fdr_alpha": fdr_alpha, "bh_threshold": bh_threshold,
        "n_pairs_two_of_three_raw": n_two_of_three_raw,
        "n_pairs_fdr_supported_inversion": n_supported,
        "grid_verdict": ("no_supported_inversion_across_grid" if n_supported == 0
                         else f"{n_supported}_pairs_show_supported_inversion"),
        "skipped_pairs": skipped,
    }
    print(f"\n  GRID: {verdict['n_pairs_tested']} pairs tested over {verdict['n_cells']} (N,T_f) cells "
          f"({len(skipped)} pairs skipped — no pre-saturation window)")
    print(f"  pairs meeting two-of-three hot-inversion (pre-FDR): {n_two_of_three_raw}")
    print(f"  pairs with BH-FDR-supported inversion (alpha={fdr_alpha}): {n_supported}")
    print(f"  GRID VERDICT: {verdict['grid_verdict']}")

    io.write_trajectory_csv(run_dir / "grid_pairs.csv", pair_records,
                            columns=("N", "T_f_over_Tc", "T_i_hot", "T_i_cold", "upper",
                                     "L_C_D", "L_S_D", "L_E_D", "n_hot_significant", "p_pair",
                                     "fdr_supported_inversion"))
    (run_dir / "grid_verdict.json").write_text(
        json.dumps({"verdict": verdict, "pairs": pair_records}, indent=2, default=str) + "\n")
    manifest.add_output("grid_pairs", run_dir / "grid_pairs.csv")
    manifest.add_output("grid_verdict", run_dir / "grid_verdict.json")
    _plot(meta, pair_records, run_dir / "grid_map.png")
    manifest.add_output("grid_map", run_dir / "grid_map.png")
    return verdict


def _plot(meta, pairs, path):
    Ns = meta["Ns"]; Tfs = meta["Tfs"]; Tis = meta["Tis"]
    pair_list = [(max(a, b), min(a, b)) for a, b in combinations(Tis, 2)]
    labels = [f"{h}>{c}" for h, c in pair_list]
    fig, axes = plt.subplots(1, len(Ns), figsize=(6.5 * len(Ns), 5), squeeze=False)
    vmax = max((abs(r["L_S_D"]) for r in pairs), default=1.0) or 1.0
    for ax, N in zip(axes[0], Ns):
        grid = np.full((len(pair_list), len(Tfs)), np.nan)
        for r in pairs:
            if r["N"] != N:
                continue
            i = pair_list.index((r["T_i_hot"], r["T_i_cold"])); j = Tfs.index(r["T_f_over_Tc"])
            grid[i, j] = r["L_S_D"]
        im = ax.imshow(grid, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(Tfs))); ax.set_xticklabels([f"{t:.2f}" for t in Tfs])
        ax.set_yticks(range(len(pair_list))); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("$T_f/T_c$"); ax.set_title(f"N={N}: offset-corrected $L_S$ D (red=hot ahead)")
        # mark FDR-supported inversions
        for r in pairs:
            if r["N"] == N and r.get("fdr_supported_inversion"):
                i = pair_list.index((r["T_i_hot"], r["T_i_cold"])); j = Tfs.index(r["T_f_over_Tc"])
                ax.plot(j, i, "k*", ms=12)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Milestone 6 — grid crossing search: offset-corrected $L_S$ difference per (pair, $T_f$, N)\n"
                 "positive (red) = hotter ahead; ★ = BH-FDR-supported inversion (none expected)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #


def main(config_path, stage):
    cfg = io.load_config(config_path)
    run_id = cfg["run_id_sweep"] if stage == "sweep" else cfg["run_id_verdict"]
    print(f"Milestone 6 grid [{stage}]: N={cfg['grid']['N']}, T_i={cfg['grid']['T_i']}, "
          f"T_f/T_c={cfg['grid']['T_f_over_Tc']}")
    run_dir = io.new_run_directory(REPO_ROOT / "results", run_id)
    io.dump_config(cfg, run_dir / "config.yaml")
    manifest = provenance.Manifest.build(run_id=run_id, timestamp=_utc(), config=cfg,
                                         seeds={"base": int(cfg["ensemble"]["seed"])},
                                         repo_root=REPO_ROOT, notes=f"M6 grid {stage}")
    (stage_sweep if stage == "sweep" else stage_analyse)(cfg, run_dir, manifest)
    manifest.write(run_dir / "manifest.json")
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    stage = "sweep" if "--sweep" in sys.argv else "analyse" if "--analyse" in sys.argv else None
    if stage is None:
        raise SystemExit("specify --sweep or --analyse")
    cfg_path = argv[0] if argv else str(REPO_ROOT / "configs" / "grid_m6.yaml")
    main(cfg_path, stage)
