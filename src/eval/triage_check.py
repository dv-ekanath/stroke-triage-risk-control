"""
KG-XAI -- does the Rule Evaluator's own output predict when a prediction is
actually wrong?

Everything used here (laterality consistency, n_predicted_voxels, ECE) is
already computed and saved, once per subject, for all 21 Phase 8 checkpoints
(src/eval/run_rule_evaluator.py). This script adds nothing new at inference
time -- it just asks a question nobody asked of that output yet: if we flag
a prediction as "don't trust this" using ONLY information available at
inference time (no ground truth), is the flagged group actually worse?

If yes: the Rule Evaluator is not just a post-hoc scoring metric, it is a
usable triage signal -- flag these cases for radiologist review instead of
trusting the segmentation blind. If no: that's reported too, just as
honestly as every other check in this project.

CORRECTION (2026-10-11, caught before this was built into the demo): the
original version of this script included a third criterion, "high ECE,"
and framed all three as ground-truth-free. That was wrong for ECE
specifically -- expected_calibration_error(probs, labels, ...) takes the
true lesion mask directly (src/eval/metrics.py); it is computable here
only because this is a held-out validation set with known outcomes, not
at true deployment time, and it is partly circular with Dice since both
are functions of the same (prediction, true_label) pair. The two
criteria below ARE genuinely ground-truth-free: laterality uses the
ADMISSION occlusion side, a different scan from the follow-up target
Dice is computed against; near-zero volume is a property of the
prediction alone. The main() below still reports the original 3-criterion
number for the record, then the corrected 2-criterion one -- which is
STRONGER (p=1.4e-50 vs 1.3e-32, AUC=0.89 vs 0.73), not weaker. The
corrected version is what's used anywhere else in this project (demo,
report, deck).

Flag rule, FIXED BEFORE LOOKING AT ANY DICE NUMBER (the thresholds are
chosen from the laterality/volume distributions themselves, never from
how well they predict Dice):
  - laterality inconsistent (and evaluable -- "unknown side" cases are
    excluded from this criterion, not counted as untrustworthy by it)
  - near-zero predicted volume: < 1000 voxels (< 1 mL on this project's
    1mm^3 grid)

Pass mark, fixed before running: flagged median Dice < unflagged median
Dice, one-sided Mann-Whitney p < 0.05.

Usage
-----
    python -m src.eval.triage_check
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy.stats import mannwhitneyu

OUT_TAB = pathlib.Path("outputs/tables")
VOXEL_ML = 1e-3          # 1mm^3 grid -- see src/model/dataset.py's Spacingd
NEAR_ZERO_VOXELS = 1000  # < 1 mL

# The 21 real Phase 8 runs (excludes the random-edge ablation and smoke
# tests -- a separate, later analysis, not this project's headline set).
RUNS = (
    [("b0", 0, s) for s in (0, 1, 2)]
    + [("b0", 1, s) for s in (0, 1, 2)]
    + [("b0", 2, s) for s in (0, 1, 2)]
    + [("b1", 0, s) for s in (0, 1, 2)]
    + [("b2", 0, s) for s in (0, 1, 2)]
    + [("b2", 1, s) for s in (0, 1, 2)]
    + [("b2", 2, s) for s in (0, 1, 2)]
)


def _dice_file(model: str, fold: int, seed: int) -> pathlib.Path:
    suffix = "" if seed == 0 else f"_seed{seed}"
    if model == "b0":
        return OUT_TAB / f"baseline_3fold_fold{fold}{suffix}.json"
    if model == "b1":
        return OUT_TAB / f"b1_graph_seg_fold{fold}{suffix}.json"
    return OUT_TAB / f"b2_graph_seg_consistency_fold{fold}{suffix}.json"


def _jdcr_file(model: str, fold: int, seed: int) -> pathlib.Path:
    # b0/b2 always have a _seed-suffixed file (even seed 0); b1 fold0 seed0
    # was scored both ways historically -- prefer the seed-suffixed one.
    f = OUT_TAB / f"jdcr_{model}_fold{fold}_seed{seed}.json"
    if f.exists():
        return f
    return OUT_TAB / f"jdcr_{model}_fold{fold}.json"


def _dice_per_subject(path: pathlib.Path) -> dict[str, float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d["final_per_subject"] if "final_per_subject" in d else d["final_segmentation"]["per_subject"]
    return {r["subject"]: float(r["dice"]) for r in rows}


def load_all_rows() -> list[dict]:
    """One row per (subject, run) -- Dice joined with the Rule Evaluator's
    own, ground-truth-free signals for that same subject/checkpoint."""
    out = []
    missing = []
    for model, fold, seed in RUNS:
        df, jf = _dice_file(model, fold, seed), _jdcr_file(model, fold, seed)
        if not df.exists() or not jf.exists():
            missing.append((model, fold, seed, df.exists(), jf.exists()))
            continue
        dice = _dice_per_subject(df)
        jdcr = json.loads(jf.read_text(encoding="utf-8"))
        for r in jdcr["rows"]:
            sub = r["subject"]
            if sub not in dice:
                continue
            lat = r.get("laterality") or {}
            out.append({
                "model": model, "fold": fold, "seed": seed, "subject": sub,
                "dice": dice[sub],
                "evaluable": bool(lat.get("evaluable")),
                "consistent": lat.get("consistent"),
                "n_voxels": lat.get("n_predicted_voxels"),
                "ece": r.get("ece"),
            })
    if missing:
        print(f"  !! {len(missing)} run(s) missing a file: {missing}")
    return out


def main():
    # CORRECTION (2026-10-11, caught before the demo integration): ECE is
    # NOT ground-truth-free -- expected_calibration_error(probs, labels,
    # ...) takes the true lesion mask directly (src/eval/metrics.py). It
    # is computable here only because this is a held-out validation set
    # with known outcomes, not at true deployment time. The original
    # framing of "three ground-truth-free signals" was wrong for this one
    # criterion. Re-running with it EXCLUDED, to see whether the finding
    # survives on the two criteria that are actually clean:
    # laterality (uses ADMISSION occlusion side -- a different scan from
    # the follow-up target Dice is computed against, not circular) and
    # near-zero volume (a property of the prediction alone).
    rows = load_all_rows()
    print(f"loaded {len(rows)} (subject, run) pairs from {len(RUNS)} runs")

    # threshold fixed from the ECE distribution alone, before any Dice join
    ece_vals = np.array([r["ece"] for r in rows if r["ece"] is not None])
    ece_p75 = float(np.percentile(ece_vals, 75))
    print(f"ECE p75 across all {ece_vals.size} subject-run pairs = {ece_p75:.4f} (the fixed high-ECE cutoff)")

    for r in rows:
        inconsistent = bool(r["evaluable"] and r["consistent"] is False)
        near_zero = bool(r["n_voxels"] is not None and r["n_voxels"] < NEAR_ZERO_VOXELS)
        high_ece = bool(r["ece"] is not None and r["ece"] > ece_p75)
        r["flag_inconsistent"] = inconsistent
        r["flag_near_zero"] = near_zero
        r["flag_high_ece"] = high_ece
        r["flagged"] = inconsistent or near_zero or high_ece

    flagged = np.array([r["dice"] for r in rows if r["flagged"]])
    unflagged = np.array([r["dice"] for r in rows if not r["flagged"]])
    print(f"\nflagged:   n={flagged.size:4d}  median dice={np.median(flagged):.4f}  mean={flagged.mean():.4f}")
    print(f"unflagged: n={unflagged.size:4d}  median dice={np.median(unflagged):.4f}  mean={unflagged.mean():.4f}")

    u, p = mannwhitneyu(unflagged, flagged, alternative="greater")  # unflagged > flagged
    auc = float(u / (unflagged.size * flagged.size))
    passed = bool(np.median(unflagged) > np.median(flagged) and p < 0.05)
    print(f"\nMann-Whitney (unflagged > flagged): p={p:.3g}  AUC={auc:.3f}")
    print(f"pass mark: unflagged median > flagged median and p<0.05, fixed before running -> "
          f"{'PASS' if passed else 'FAIL'}")

    # breakdown: does each criterion alone carry signal?
    print("\nper-criterion breakdown (each vs the complement, descriptive):")
    breakdown = {}
    for name in ("flag_inconsistent", "flag_near_zero", "flag_high_ece"):
        a = np.array([r["dice"] for r in rows if r[name]])
        b = np.array([r["dice"] for r in rows if not r[name]])
        if a.size >= 3 and b.size >= 3:
            uu, pp = mannwhitneyu(b, a, alternative="greater")
            print(f"  {name:<20} n_flagged={a.size:4d}  median_flagged={np.median(a):.4f}  "
                  f"median_rest={np.median(b):.4f}  p={pp:.3g}")
            breakdown[name] = {"n_flagged": int(a.size), "median_flagged": round(float(np.median(a)), 4),
                                "median_rest": round(float(np.median(b)), 4), "p": float(pp)}
        else:
            print(f"  {name:<20} too few in one group (n_flagged={a.size}, n_rest={b.size})")

    # per-model-type breakdown
    print("\nper-model-type (does the flag still work within each arm?):")
    model_breakdown = {}
    for m in ("b0", "b1", "b2"):
        a = np.array([r["dice"] for r in rows if r["model"] == m and r["flagged"]])
        b = np.array([r["dice"] for r in rows if r["model"] == m and not r["flagged"]])
        if a.size >= 3 and b.size >= 3:
            uu, pp = mannwhitneyu(b, a, alternative="greater")
            print(f"  {m}  flagged n={a.size:3d} median={np.median(a):.4f}   "
                  f"unflagged n={b.size:3d} median={np.median(b):.4f}   p={pp:.3g}")
            model_breakdown[m] = {"n_flagged": int(a.size), "median_flagged": round(float(np.median(a)), 4),
                                   "n_unflagged": int(b.size), "median_unflagged": round(float(np.median(b)), 4),
                                   "p": float(pp)}
        else:
            print(f"  {m}  too few in one group (flagged={a.size}, unflagged={b.size})")

    # the honest, deployment-realistic version: drop the ECE criterion,
    # since it isn't actually computable without the ground truth it's
    # meant to be independent of
    clean_flagged = np.array([r["dice"] for r in rows if r["flag_inconsistent"] or r["flag_near_zero"]])
    clean_unflagged = np.array([r["dice"] for r in rows if not (r["flag_inconsistent"] or r["flag_near_zero"])])
    clean_u, clean_p = mannwhitneyu(clean_unflagged, clean_flagged, alternative="greater")
    clean_auc = float(clean_u / (clean_unflagged.size * clean_flagged.size))
    clean_cmp = {"testable": True, "p_one_sided": float(clean_p), "auc": round(clean_auc, 3)}
    clean_passed = bool(clean_flagged.size and clean_unflagged.size
                         and float(np.median(clean_unflagged)) > float(np.median(clean_flagged))
                         and clean_p < 0.05)
    print(f"\nCORRECTED (ECE excluded -- it is not actually ground-truth-free):")
    print(f"  flagged:   n={clean_flagged.size:4d}  median dice={np.median(clean_flagged):.4f}")
    print(f"  unflagged: n={clean_unflagged.size:4d}  median dice={np.median(clean_unflagged):.4f}")
    print(f"  p = {clean_cmp['p_one_sided']:.3g}, AUC = {clean_cmp['auc']:.3f} -> "
          f"{'PASS' if clean_passed else 'FAIL'}")

    print("\n  per-model-type, corrected (2-criterion) flag:")
    clean_model_breakdown = {}
    for m in ("b0", "b1", "b2"):
        a = np.array([r["dice"] for r in rows if r["model"] == m and (r["flag_inconsistent"] or r["flag_near_zero"])])
        b = np.array([r["dice"] for r in rows if r["model"] == m and not (r["flag_inconsistent"] or r["flag_near_zero"])])
        if a.size >= 3 and b.size >= 3:
            uu, pp = mannwhitneyu(b, a, alternative="greater")
            print(f"    {m}  flagged n={a.size:3d} median={np.median(a):.4f}   "
                  f"unflagged n={b.size:3d} median={np.median(b):.4f}   p={pp:.3g}")
            clean_model_breakdown[m] = {"n_flagged": int(a.size), "median_flagged": round(float(np.median(a)), 4),
                                         "n_unflagged": int(b.size), "median_unflagged": round(float(np.median(b)), 4),
                                         "p": float(pp)}
        else:
            print(f"    {m}  too few in one group (flagged={a.size}, unflagged={b.size})")

    payload = {
        "rule": "Flag a prediction (ground-truth-free, inference-time only) if laterality is "
                "inconsistent, predicted volume < 1mL, or ECE > the 75th percentile across all "
                "scored subjects. Does the flagged group actually have worse Dice?",
        "correction_2026_10_11": "ECE is NOT ground-truth-free (expected_calibration_error takes "
                "the true label) -- only computable here because this is a validation set with "
                "known outcomes. The deployment-realistic version below excludes it.",
        "clean_ground_truth_free_only": {
            "n_flagged": int(clean_flagged.size), "n_unflagged": int(clean_unflagged.size),
            "flagged_median_dice": round(float(np.median(clean_flagged)), 4) if clean_flagged.size else None,
            "unflagged_median_dice": round(float(np.median(clean_unflagged)), 4) if clean_unflagged.size else None,
            "p": float(clean_cmp["p_one_sided"]) if clean_cmp.get("testable") else None,
            "auc": clean_cmp.get("auc"), "passed": clean_passed,
            "per_model_type": clean_model_breakdown,
        },
        "thresholds_fixed_before_dice_join": {
            "near_zero_voxels": NEAR_ZERO_VOXELS, "ece_p75_cutoff": round(ece_p75, 4),
        },
        "n_pairs": len(rows), "n_flagged": int(flagged.size), "n_unflagged": int(unflagged.size),
        "flagged_median_dice": round(float(np.median(flagged)), 4),
        "unflagged_median_dice": round(float(np.median(unflagged)), 4),
        "mannwhitney_p": float(p), "auc": auc,
        "pass_mark": "unflagged median > flagged median and p<0.05, fixed before running",
        "passed": passed,
        "per_criterion": breakdown, "per_model_type": model_breakdown,
    }
    out = OUT_TAB / "triage_check.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
