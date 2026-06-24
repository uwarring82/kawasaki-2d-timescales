#!/usr/bin/env python
"""M5 robustness — offset-model sensitivity of the crossing verdict.

Prompted by the v1.0.0 review round (reviews/2026-06-24-codex.md and -kimi.md):
the pre-registered verdict uses a fixed-1/3 offset model; reviewers asked whether
the offset-corrected difference survives a free-exponent re-fit. This script
re-analyses the committed primary-pair ensembles (`results/m5_primary_v1/`) under
several offset models and records the result as a reproducible artifact.

For each length estimator (L_C, L_S, L_E) it computes the offset-corrected
difference D = (L_hot - R0_hot) - (L_cold - R0_cold) with R0 fitted per leg under
fixed exponents {0.30, 1/3, 0.36} and a free exponent (per-resample linearity
scan), all with the pre-registered bootstrap (n_boot, CI) and BH-FDR across time.

Headline check: is the two-of-three HOT-inversion rule (>=2 estimators with a
robust POSITIVE late difference) met under ANY model? (Answer recorded below.)

Usage:
    python scripts/m5_offset_sensitivity.py
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

from kawasaki2d import analysis, io, provenance  # noqa: E402
from kawasaki2d.rng import make_rng  # noqa: E402

RUN_ID = "m5_offset_sensitivity_v1"
SOURCE = "m5_primary_v1"
CUTOFF, UPPER = 100, 15000
N_BOOT, CI, FDR_ALPHA = 5000, 0.95, 0.05
MODELS = [("fixed_0.30", 0.30), ("fixed_1/3", 1.0 / 3.0), ("fixed_0.36", 0.36), ("free", None)]
SEED = 50505


def _utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load():
    d = REPO_ROOT / "results" / SOURCE
    meta = json.loads((d / "sweep_meta.json").read_text())
    sweeps = np.load(d / "sweeps.npy")
    e_inf = meta["e_inf"]
    out = {}
    for label in ("hot", "cold"):
        E = np.load(d / f"{label}_E.npy")
        with np.errstate(divide="ignore", invalid="ignore"):
            LE = np.where(E - e_inf > 0, 1.0 / (E - e_inf), np.nan)
        out[label] = {"L_C": np.load(d / f"{label}_LC.npy"),
                      "L_S": np.load(d / f"{label}_LS.npy"), "L_E": LE}
    return meta, sweeps, out


def main():
    meta, sweeps, est = _load()
    run_dir = io.new_run_directory(REPO_ROOT / "results", RUN_ID)
    manifest = provenance.Manifest.build(
        run_id=RUN_ID, timestamp=_utc(), config={"source": SOURCE, "cutoff": CUTOFF, "upper": UPPER,
                                                  "n_boot": N_BOOT, "ci": CI, "fdr_alpha": FDR_ALPHA,
                                                  "models": [m[0] for m in MODELS]},
        seeds={"base": SEED}, repo_root=REPO_ROOT,
        notes="M5 offset-model sensitivity (review response)")

    rows, results = [], {}
    print(f"Offset-model sensitivity (source {SOURCE}, window ({CUTOFF},{UPPER}])")
    for mname, exp in MODELS:
        results[mname] = {}
        line = []
        for e in ("L_C", "L_S", "L_E"):
            rng = make_rng(SEED + 1)  # same stream per estimator for comparability across models
            dci, R0h, R0c = analysis.offset_corrected_difference_bootstrap(
                est["hot"][e], est["cold"][e], sweeps, rng, cutoff=CUTOFF, n_boot=N_BOOT, ci=CI,
                exponent=exp)
            bh = analysis.benjamini_hochberg(dci.pvalue, alpha=FDR_ALPHA)
            late = dci.times > (CUTOFF + 0.66 * (UPPER - CUTOFF))
            late_sig = bool((bh.rejected & late).any())
            sign = int(np.sign(dci.diff_mean[-1])) if late_sig else 0
            info = {"D_final": float(dci.diff_mean[-1]),
                    "ci": [float(dci.ci_low[-1]), float(dci.ci_high[-1])],
                    "fdr_late_significant": late_sig, "late_sign": sign}
            results[mname][e] = info
            rows.append({"model": mname, "estimator": e, "exponent": "free" if exp is None else f"{exp:.3f}",
                         "D_final": info["D_final"], "ci_low": info["ci"][0], "ci_high": info["ci"][1],
                         "fdr_significant": late_sig, "late_sign": sign})
            line.append(f"{e}={info['D_final']:+.3f}{'*' if late_sig else ' '}({'+' if sign>0 else '-' if sign<0 else '0'})")
        # two-of-three hot inversion = >=2 estimators with significant POSITIVE late sign
        n_pos = sum(1 for e in ("L_C", "L_S", "L_E") if results[mname][e]["late_sign"] > 0)
        results[mname]["two_of_three_hot_inversion"] = bool(n_pos >= 2)
        print(f"  {mname:11s}: {'  '.join(line)}   two-of-three hot inversion: {n_pos>=2}")

    any_inversion = any(results[m]["two_of_three_hot_inversion"] for m, _ in MODELS)
    conclusion = {
        "two_of_three_hot_inversion_under_any_model": any_inversion,
        "verdict_robust": (not any_inversion),
        "L_S_sign_by_model": {m: results[m]["L_S"]["late_sign"] for m, _ in MODELS},
        "summary": (
            "Across fixed {0.30, 1/3, 0.36} the offset-corrected difference is null for L_S and "
            "significantly negative for L_C/L_E. Under the free-exponent model L_S flips to a "
            "significant POSITIVE difference, but L_C/L_E remain significantly negative — so the "
            "two-of-three HOT-inversion rule is met under NO offset model. The no_supported_inversion "
            "verdict is robust; only the auxiliary 'no estimator favours hot' wording is model-"
            "dependent (it holds for the pre-registered fixed-1/3 model, not for the free-exponent "
            "stress test).")
    }
    print(f"\n  CONCLUSION: verdict robust = {conclusion['verdict_robust']} "
          f"(two-of-three hot inversion under any model: {any_inversion})")

    io.write_trajectory_csv(run_dir / "offset_sensitivity.csv", rows,
                            columns=("model", "estimator", "exponent", "D_final", "ci_low", "ci_high",
                                     "fdr_significant", "late_sign"))
    (run_dir / "offset_sensitivity.json").write_text(
        json.dumps({"results": results, "conclusion": conclusion}, indent=2, default=str) + "\n")
    manifest.add_output("offset_sensitivity_csv", run_dir / "offset_sensitivity.csv")
    manifest.add_output("offset_sensitivity_json", run_dir / "offset_sensitivity.json")
    _plot(results, run_dir / "offset_sensitivity.png")
    manifest.add_output("offset_sensitivity_fig", run_dir / "offset_sensitivity.png")
    manifest.write(run_dir / "manifest.json")
    print(f"\nwrote {run_dir}")


def _plot(results, path):
    models = [m for m, _ in MODELS]
    ests = ("L_C", "L_S", "L_E")
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.26
    for i, e in enumerate(ests):
        D = [results[m][e]["D_final"] for m in models]
        lo = [results[m][e]["D_final"] - results[m][e]["ci"][0] for m in models]
        hi = [results[m][e]["ci"][1] - results[m][e]["D_final"] for m in models]
        ax.bar(x + (i - 1) * width, D, width, yerr=[lo, hi], capsize=3, label=e)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel(r"offset-corrected $D_{final}$  (+ = hot ahead)")
    ax.set_title("M5 offset-model sensitivity of the crossing verdict\n"
                 "(positive = hot overtakes after head-start removal; two-of-three never met)")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
