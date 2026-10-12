"""
Phase 8 final aggregation -- pulls Dice (from the training result JSONs) and
JDCR/ECE (from the Rule Evaluator JSONs, src/eval/run_phase8_eval.py) for
all 21 real checkpoints, computes mean +/- SD per model, and checks the
pass marks fixed in PHASE8_PLAN.md BEFORE this run started:

    Primary claim: B2's mean strict JDCR > B0's mean strict JDCR
    Guard rail:    B2's mean Dice within 0.02 of B0's mean Dice

Two of B0's 9 runs (fold0/seed1, fold1/seed1) collapsed to predicting no
lesion at all (JDCR=0, Dice~0.02) -- reported explicitly, not silently
dropped. Every aggregate below is computed BOTH ways (all 9 runs, and the
7 non-collapsed runs) so the collapse's effect on the headline numbers is
visible rather than hidden either way.

Usage
-----
    python -m src.eval.phase8_aggregate
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

OUT_TAB = pathlib.Path("outputs/tables")


def dice_file(model: str, fold: int, seed: int) -> str:
    suffix = "" if seed == 0 else f"_seed{seed}"
    if model == "b0":
        return f"baseline_3fold_fold{fold}{suffix}.json"
    if model == "b1":
        return f"b1_graph_seg_fold{fold}{suffix}.json"
    return f"b2_graph_seg_consistency_fold{fold}{suffix}.json"


def jdcr_file(model: str, fold: int, seed: int) -> str:
    return f"jdcr_{model}_fold{fold}_seed{seed}.json"


def load_run(model: str, fold: int, seed: int) -> dict:
    d = json.loads((OUT_TAB / dice_file(model, fold, seed)).read_text())
    j = json.loads((OUT_TAB / jdcr_file(model, fold, seed)).read_text())
    dice_key = "final_mean" if "final_mean" in d else None
    final_dice = (d["final_mean"]["dice"] if dice_key
                  else d["final_segmentation"]["mean"]["dice"])
    return {
        "model": model, "fold": fold, "seed": seed,
        "dice_best": d["best_val_dice"], "dice_final": final_dice,
        "jdcr_strict": j["jdcr_strict"]["jdcr"], "coverage": j["jdcr_strict"]["coverage"],
        "ece": j["ece_mean"],
        "collapsed": j["jdcr_strict"]["coverage"] == 0.0,
    }


def summarize(runs: list[dict], label: str) -> dict:
    def ms(key):
        vals = [r[key] for r in runs if r[key] is not None]
        return {"mean": round(float(np.mean(vals)), 4), "sd": round(float(np.std(vals)), 4),
                "n": len(vals)} if vals else {"mean": None, "sd": None, "n": 0}
    return {"label": label, "n_runs": len(runs),
            "dice_final": ms("dice_final"), "dice_best": ms("dice_best"),
            "jdcr_strict": ms("jdcr_strict"), "ece": ms("ece")}


def main():
    combos = {
        "b0": [(f, s) for f in (0, 1, 2) for s in (0, 1, 2)],
        "b1": [(0, s) for s in (0, 1, 2)],
        "b2": [(f, s) for f in (0, 1, 2) for s in (0, 1, 2)],
    }
    runs = {m: [load_run(m, f, s) for f, s in fs] for m, fs in combos.items()}

    print("Per-run detail:")
    for m in ("b0", "b1", "b2"):
        for r in runs[m]:
            flag = "  <-- COLLAPSED" if r["collapsed"] else ""
            print(f"  {m} fold{r['fold']} seed{r['seed']}: dice_final={r['dice_final']:.4f} "
                  f"jdcr={r['jdcr_strict']:.4f} ece={r['ece']:.4f}{flag}")

    b0_collapsed = [r for r in runs["b0"] if r["collapsed"]]
    b0_clean = [r for r in runs["b0"] if not r["collapsed"]]
    print(f"\nB0 collapse count: {len(b0_collapsed)}/9 -- "
          f"{[(r['fold'], r['seed']) for r in b0_collapsed]}")
    print(f"B1/B2 collapse count: "
          f"{sum(r['collapsed'] for r in runs['b1'])}/3, "
          f"{sum(r['collapsed'] for r in runs['b2'])}/9")

    summary = {
        "b0_all_9": summarize(runs["b0"], "B0, all 9 runs (includes 2 collapsed)"),
        "b0_clean_7": summarize(b0_clean, "B0, 7 non-collapsed runs only"),
        "b1_all_3": summarize(runs["b1"], "B1, 3 runs (fold 0 only)"),
        "b2_all_9": summarize(runs["b2"], "B2, all 9 runs"),
        "b0_collapsed_runs": [(r["fold"], r["seed"]) for r in b0_collapsed],
    }

    print("\n" + "=" * 70)
    print("AGGREGATE (mean +/- SD)")
    print("=" * 70)
    for key in ("b0_all_9", "b0_clean_7", "b1_all_3", "b2_all_9"):
        s = summary[key]
        print(f"\n{s['label']} (n={s['n_runs']}):")
        print(f"  Dice (final):  {s['dice_final']['mean']:.4f} +/- {s['dice_final']['sd']:.4f}")
        print(f"  Dice (best):   {s['dice_best']['mean']:.4f} +/- {s['dice_best']['sd']:.4f}")
        print(f"  JDCR (strict): {s['jdcr_strict']['mean']:.4f} +/- {s['jdcr_strict']['sd']:.4f}")
        print(f"  ECE:           {s['ece']['mean']:.4f} +/- {s['ece']['sd']:.4f}")

    print("\n" + "=" * 70)
    print("PRE-REGISTERED PASS MARKS (PHASE8_PLAN.md, fixed before this run)")
    print("=" * 70)
    for b0_key, note in (("b0_all_9", "including collapses"), ("b0_clean_7", "excluding collapses")):
        b0, b2 = summary[b0_key], summary["b2_all_9"]
        jdcr_pass = b2["jdcr_strict"]["mean"] > b0["jdcr_strict"]["mean"]
        dice_gap = b0["dice_final"]["mean"] - b2["dice_final"]["mean"]
        guard_pass = dice_gap <= 0.02
        print(f"\nUsing B0 {note} (n={b0['n_runs']}):")
        print(f"  PRIMARY  B2 JDCR ({b2['jdcr_strict']['mean']:.4f}) > "
              f"B0 JDCR ({b0['jdcr_strict']['mean']:.4f})? "
              f"{'PASS' if jdcr_pass else 'FAIL'}")
        print(f"  GUARD    B0 dice - B2 dice = {dice_gap:.4f} <= 0.02? "
              f"{'PASS' if guard_pass else 'FAIL'}")
        summary[f"passmarks_vs_{b0_key}"] = {"jdcr_primary_pass": jdcr_pass,
                                              "dice_gap": round(dice_gap, 4),
                                              "guard_pass": guard_pass}

    summary["all_runs"] = {m: runs[m] for m in runs}
    out = OUT_TAB / "phase8_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
