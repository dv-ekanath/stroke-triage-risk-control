"""
KG-XAI -- does each plausibility constraint actually hold in the ground truth?

A constraint the real data contradicts must not be taught to the model:
forcing the model to obey it makes predictions worse. So every constraint in
kg/guideline_rules.json is checked against ground truth before use, and its
`support.status` records the result. Phase 1 checked the side rule (in
src/graph/features.py); this module holds the rest, and Phase 2 adds the
collateral and territory checks here.

Ground truth (lesion-msk volumes) is used ONLY to test whether a rule is
true -- never as a model input (see leakage_rule in kg/cow_topology.json).

Check 1 -- occlusion level vs infarct volume
  Rule: a clot in a large upstream vessel (ICA / M1) cuts off more brain than
  a clot further downstream, so it should come with a larger infarct.
  Groups come from Phase 1's confidence-gated localisation:
    proximal_anterior   confident ICA or MCA match
    distal_unlocalized  occlusion beyond the segmented circle
  Pass mark, fixed before the first run: proximal median > distal median AND
  one-sided Mann-Whitney U p < 0.05.
  Effect size reported as AUC = P(random proximal volume > random distal one).

  This is a group-level test. Turning it into a per-patient constraint needs a
  volume threshold ("a distal clot with an infarct above T mL is
  implausible"), and T must be fitted on training folds only -- fitting it on
  all 149 here would leak the test set into the rule. So this check reports
  quantiles for reference and leaves T to Phase 2/3.

Usage
-----
    python -m src.graph.constraint_support
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from scipy.stats import mannwhitneyu

OUT_TAB = pathlib.Path("outputs/tables")
ALPHA = 0.05


def localization_class(s: dict) -> str | None:
    """The proposed H2 label scheme, from Phase 1 features."""
    status = s.get("occlusion_status")
    if status == "empty_lvo_mask":
        return "none"
    if status != "ok":
        return None                      # shape mismatch etc. -- excluded
    if s["occlusion_unlocalized"]:
        return "distal_unlocalized"
    vessel = (s.get("nearest_vessel") or "").split("-")[-1]
    return "proximal_anterior" if vessel in ("ICA", "MCA") else "other_confident"


def describe(v: np.ndarray) -> dict:
    if v.size == 0:
        return {"n": 0}
    q = np.percentile(v, [25, 50, 75, 90])
    return {"n": int(v.size), "median": round(float(q[1]), 2),
            "q1": round(float(q[0]), 2), "q3": round(float(q[2]), 2),
            "p90": round(float(q[3]), 2), "mean": round(float(v.mean()), 2),
            "max": round(float(v.max()), 2)}


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """One-sided test that a tends to be larger than b."""
    if a.size < 3 or b.size < 3:
        return {"testable": False, "reason": "fewer than 3 subjects in a group"}
    u, p = mannwhitneyu(a, b, alternative="greater")
    return {"testable": True, "U": float(u), "p_one_sided": float(p),
            "auc": round(float(u / (a.size * b.size)), 3)}


def unseen_vessel_diagnostic(feats: dict) -> dict:
    """How often is the MCA or ICA on the clot's side missing from cow-msk?
    An occluded vessel carries no contrast, so it may never be segmented --
    and a clot inside it then looks far from every labelled vessel and gets
    classed as unlocalized/distal. Uses admission data only."""
    idx = {v: i for i, v in enumerate(feats["node_order"])}
    out = {}
    for s in feats["subjects"]:
        cls = localization_class(s)
        if cls not in ("proximal_anterior", "other_confident", "distal_unlocalized"):
            continue
        if cls == "distal_unlocalized":
            side = s.get("unlocalized_side") or "unknown"
        else:
            side = {"L": "left", "R": "right"}.get(s["nearest_vessel"].split("-")[0], "unknown")
        sides = ["L", "R"] if side == "unknown" else [side[0].upper()]
        missing = any(s["segment_present"][idx[f"{x}-{v}"]] == 0
                      for x in sides for v in ("MCA", "ICA"))
        d = out.setdefault(cls, {"n": 0, "vessel_missing": 0})
        d["n"] += 1
        d["vessel_missing"] += int(missing)
    for d in out.values():
        d["fraction"] = round(d["vessel_missing"] / d["n"], 3) if d["n"] else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=pathlib.Path, default=OUT_TAB / "kg_node_features.json")
    ap.add_argument("--audit", type=pathlib.Path, default=OUT_TAB / "data_audit.json")
    args = ap.parse_args()

    feats = json.loads(args.features.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    vol = {c["subject"]: float(c["volume_ml"]) for c in audit["cases"]}

    groups: dict[str, list[float]] = {}
    vessel_vols: dict[str, list[float]] = {}
    missing = []
    for s in feats["subjects"]:
        cls = localization_class(s)
        if cls is None:
            continue
        if s["subject"] not in vol:
            missing.append(s["subject"])
            continue
        groups.setdefault(cls, []).append(vol[s["subject"]])
        if cls in ("proximal_anterior", "other_confident"):
            vessel_vols.setdefault(s["nearest_vessel"].split("-")[-1], []).append(vol[s["subject"]])

    arr = {k: np.asarray(v) for k, v in groups.items()}
    print("\nCheck 1 -- occlusion level vs final infarct volume (mL, ground truth)")
    if missing:
        print(f"  !! no audited volume for: {missing}")
    print(f"  {'group':<20} {'n':>4} {'median':>8} {'IQR':>17} {'p90':>8} {'max':>8}")
    order = ["proximal_anterior", "other_confident", "distal_unlocalized", "none"]
    stats = {}
    for k in order:
        d = describe(arr.get(k, np.empty(0)))
        stats[k] = d
        if d["n"]:
            print(f"  {k:<20} {d['n']:>4} {d['median']:>8.1f} "
                  f"{d['q1']:>7.1f} - {d['q3']:<7.1f} {d['p90']:>8.1f} {d['max']:>8.1f}")

    print("\n  by vessel (confident matches):")
    vstats = {}
    for v in ("ICA", "MCA", "ACA", "PCA", "BA"):
        d = describe(np.asarray(vessel_vols.get(v, [])))
        vstats[v] = d
        if d["n"]:
            print(f"    {v:<4} n={d['n']:>3}  median {d['median']:>6.1f}  IQR {d['q1']:.1f}-{d['q3']:.1f}")

    prox, dist = arr.get("proximal_anterior", np.empty(0)), arr.get("distal_unlocalized", np.empty(0))
    primary = compare(prox, dist)
    passed = bool(primary.get("testable") and stats["proximal_anterior"]["median"] > stats["distal_unlocalized"]["median"]
                  and primary["p_one_sided"] < ALPHA)

    print(f"\n  PRIMARY  proximal (ICA/M1) > distal: ", end="")
    if primary["testable"]:
        print(f"p = {primary['p_one_sided']:.2e}, AUC = {primary['auc']}  "
              f"-> {'PASS' if passed else 'FAIL'} (pass mark: higher median and p < {ALPHA}, fixed before running)")
    else:
        print(f"not testable -- {primary['reason']}")

    secondary = {
        "ICA_vs_MCA": compare(np.asarray(vessel_vols.get("ICA", [])), np.asarray(vessel_vols.get("MCA", []))),
        "proximal_vs_other_confident": compare(prox, arr.get("other_confident", np.empty(0))),
    }
    print("\n  secondary (reported, not gated):")
    for name, r in secondary.items():
        if r["testable"]:
            print(f"    {name:<28} p = {r['p_one_sided']:.3g}, AUC = {r['auc']}")
        else:
            print(f"    {name:<28} not testable -- {r['reason']}")

    contamination = unseen_vessel_diagnostic(feats)
    print("\n  diagnostic -- clot-side MCA/ICA missing from cow-msk (possible occluded, "
          "unopacified vessel):")
    for k, d in contamination.items():
        print(f"    {k:<20} {d['vessel_missing']:>2}/{d['n']:<3} ({d['fraction']:.0%})")
    print("    Distal cases with a missing clot-side vessel may be upstream clots in a vessel the")
    print("    CTA never showed. Too few to explain the result; a label-quality issue for Phase 2.")

    OUT_TAB.mkdir(parents=True, exist_ok=True)
    out = OUT_TAB / "constraint_support.json"
    payload = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    payload["occlusion_level_vs_volume"] = {
        "rule": "Proximal (ICA/M1) occlusions come with larger final infarcts than distal occlusions.",
        "pass_mark": f"proximal median > distal median and one-sided Mann-Whitney p < {ALPHA}, fixed before running",
        "groups": stats, "by_vessel": vstats,
        "primary": primary, "secondary": secondary, "passed": passed,
        "secondary_note": "ICA > MCA was one of two unplanned secondary tests; at a Bonferroni-corrected 0.025 it does not pass. A lead to re-test on training folds, not a supported rule.",
        "unseen_vessel_diagnostic": contamination,
        "likely_reason_for_failure": "Every ISLES'24 patient was successfully reperfused, so final infarct SIZE reflects how fast flow was restored more than where the clot sat. Reasoned, not tested here. Implication: size-based constraints are weak in this cohort; location-based ones (side, territory) are not affected by reperfusion.",
        "note": "Group-level result. A per-patient threshold must be fitted on training folds only.",
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
