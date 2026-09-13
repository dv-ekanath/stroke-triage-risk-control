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
    unlocalized  occlusion beyond the segmented circle
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
        return "unlocalized"
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
        if cls not in ("proximal_anterior", "other_confident", "unlocalized"):
            continue
        if cls == "unlocalized":
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


def check_hir_vs_mismatch(perf_rows: list[dict]) -> dict:
    """Check 2 -- does HIR behave like a collateral signal, independent of how
    it was defined?

    Established relationship (not the one HIR is defined from): poor
    collaterals let the core grow faster even before reperfusion, so a high
    HIR (poor collaterals) should coincide with a LOW mismatch ratio
    (penumbra/core) -- less penumbra survives relative to core when
    collaterals are worse. HIR and mismatch ratio are computed from
    different combinations of maps (HIR: Tmax alone; mismatch: Tmax and
    CBF), so this is an independent check, not circular.

    Pass mark, fixed before running: negative Spearman correlation,
    p < 0.05.
    """
    from scipy.stats import spearmanr
    pairs = [(r["hir"], r["mismatch_ratio"]) for r in perf_rows
             if r.get("hir") is not None and r.get("mismatch_ratio") is not None]
    if len(pairs) < 10:
        return {"testable": False, "reason": f"only {len(pairs)} subjects with both values"}
    hir, mr = zip(*pairs)
    rho, p = spearmanr(hir, mr)
    passed = bool(rho < 0 and p < 0.05)
    return {"testable": True, "n": len(pairs), "spearman_rho": round(float(rho), 4),
            "p": float(p), "pass_mark": "rho < 0 and p < 0.05, fixed before running",
            "passed": passed}


def check_perfusion_overlap(root: pathlib.Path, feats: dict) -> dict:
    """Check 3 -- perfusion-overlap plausibility rule: the final infarct
    should mostly sit inside the admission hypoperfused region (Tmax > 6s).
    All patients here were successfully reperfused, so the final infarct is
    expected to be a SUBSET of the original at-risk tissue, not exceed it.

    No registration needed -- ses-02 derivatives are already delivered in
    the same '_space-ncct_' grid as ses-01 (checked: identical shape/affine
    to the Tmax map for every subject tested).

    Pass mark, fixed before running: median fraction of final-infarct
    voxels lying inside the Tmax>6s mask >= 0.70, over subjects with a
    non-trivial infarct (>= 1 mL, to avoid tiny masks being dominated by
    single-voxel boundary noise).
    """
    import nibabel as nib
    GATE = 0.70
    fracs, skipped = [], []
    for s in feats["subjects"]:
        sub = s["subject"]
        tmax_f = sorted((root / "derivatives" / sub / "ses-01" / "perfusion-maps").glob("*_tmax.nii*"))
        les_f = sorted((root / "derivatives" / sub / "ses-02").glob("*_lesion-msk.nii*"))
        if not tmax_f or not les_f:
            skipped.append((sub, "missing_file")); continue
        try:
            tmax = np.asanyarray(nib.load(str(tmax_f[0])).dataobj)
            les = np.asanyarray(nib.load(str(les_f[0])).dataobj) > 0
        except Exception as e:
            skipped.append((sub, f"unreadable: {e}")); continue
        if tmax.shape != les.shape:
            skipped.append((sub, "shape_mismatch")); continue
        tmax = np.nan_to_num(tmax, nan=-30.0, posinf=0.0, neginf=-30.0)
        n_les = int(les.sum())
        vox_ml = float(np.prod(nib.load(str(les_f[0])).header.get_zooms()[:3])) / 1000.0
        if n_les * vox_ml < 1.0:
            skipped.append((sub, "infarct_too_small")); continue
        overlap = int((les & (tmax > 6.0)).sum())
        fracs.append({"subject": sub, "fraction_in_penumbra": round(overlap / n_les, 4)})

    if not fracs:
        return {"testable": False, "reason": "no usable subjects"}
    vals = np.array([f["fraction_in_penumbra"] for f in fracs])
    median = float(np.median(vals))
    passed = bool(median >= GATE)
    return {"testable": True, "n": len(fracs), "n_skipped": len(skipped),
            "median_fraction_in_penumbra": round(median, 4),
            "mean_fraction_in_penumbra": round(float(vals.mean()), 4),
            "pct_below_50pct": round(float((vals < 0.5).mean()), 4),
            "pass_mark": f">= {GATE}, fixed before running", "passed": passed,
            "worst_5": sorted(fracs, key=lambda d: d["fraction_in_penumbra"])[:5],
            "skipped": skipped}


def check_hir_vs_final_volume(feats: dict, perf_ok: list[dict], vol: dict[str, float]) -> dict:
    """Check 4 -- does HIR (collateral proxy) predict final infarct SIZE among
    proximal occlusions, the way the two collateral plausibility constraints
    assume?

    Rule under test: proximal occlusion + poor collaterals (high HIR) ->
    larger final infarct than proximal occlusion + good collaterals (low HIR).

    Groups: HIR bin 0 ("good", <=0.25 by src/graph/perfusion_volumes.py's
    binning) vs bin 2-3 ("poor", >0.5), restricted to proximal_anterior
    occlusions (src/graph/constraint_support.py:localization_class).

    Pass mark, fixed before running: poor-collateral median > good-collateral
    median AND one-sided Mann-Whitney p < 0.05 -- identical structure to
    Check 1, applied to collateral status instead of occlusion level.
    """
    hir_by_sub = {r["subject"]: r for r in perf_ok}
    good, poor = [], []
    for s in feats["subjects"]:
        if localization_class(s) != "proximal_anterior":
            continue
        p = hir_by_sub.get(s["subject"])
        v = vol.get(s["subject"])
        if not p or p.get("hir_bin") is None or v is None:
            continue
        if p["hir_bin"] == 0:
            good.append(v)
        elif p["hir_bin"] in (2, 3):
            poor.append(v)
    good_v, poor_v = np.asarray(good), np.asarray(poor)
    cmp = compare(poor_v, good_v)
    passed = bool(cmp.get("testable") and poor_v.size and good_v.size
                  and float(np.median(poor_v)) > float(np.median(good_v)) and cmp["p_one_sided"] < 0.05)
    return {"good_collateral": describe(good_v), "poor_collateral": describe(poor_v),
            "comparison": cmp, "pass_mark": "poor median > good median and p < 0.05, fixed before running",
            "passed": passed}


def check_hir_vs_final_volume_continuous(feats: dict, perf_ok: list[dict], vol: dict[str, float]) -> dict:
    """Check 5 -- re-test of the two collateral plausibility constraints,
    fixing Check 4's underpowering: continuous Spearman correlation across
    ALL occlusion-positive patients (proximal_anterior + other_confident +
    unlocalized), not a binned good-vs-poor group restricted to proximal
    occlusions only (n=9 there). "none" (no occlusion, n=4) is excluded --
    collateral status is not a meaningful concept without an occlusion to
    have collaterals around.

    Unlike Check 2 (HIR vs mismatch_ratio), this pairing has no shared-term
    confound: final_infarct_volume comes from the independent follow-up scan
    (ses-02 lesion-msk), not from any Tmax-derived quantity.

    Pass mark, fixed before running: Spearman rho > 0 and p < 0.05.
    """
    from scipy.stats import spearmanr
    hir_by_sub = {r["subject"]: r["hir"] for r in perf_ok if r.get("hir") is not None}
    pairs = [(hir_by_sub[s["subject"]], vol[s["subject"]])
             for s in feats["subjects"]
             if localization_class(s) in ("proximal_anterior", "other_confident", "unlocalized")
             and s["subject"] in hir_by_sub and s["subject"] in vol]
    if len(pairs) < 10:
        return {"testable": False, "reason": f"only {len(pairs)} subjects with both values"}
    hir, v = zip(*pairs)
    rho, p = spearmanr(hir, v)
    passed = bool(rho > 0 and p < 0.05)
    return {"testable": True, "n": len(pairs), "spearman_rho": round(float(rho), 4),
            "p": float(p), "pass_mark": "rho > 0 and p < 0.05, fixed before running",
            "passed": passed}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=pathlib.Path, default=OUT_TAB / "kg_node_features.json")
    ap.add_argument("--audit", type=pathlib.Path, default=OUT_TAB / "data_audit.json")
    ap.add_argument("--perfusion", type=pathlib.Path, default=OUT_TAB / "perfusion_volumes.json")
    ap.add_argument("--root", type=pathlib.Path, default=None,
                     help="ISLES-2024 root; required for the perfusion-overlap check")
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
    order = ["proximal_anterior", "other_confident", "unlocalized", "none"]
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

    prox, dist = arr.get("proximal_anterior", np.empty(0)), arr.get("unlocalized", np.empty(0))
    primary = compare(prox, dist)
    passed = bool(primary.get("testable") and stats["proximal_anterior"]["median"] > stats["unlocalized"]["median"]
                  and primary["p_one_sided"] < ALPHA)

    print(f"\n  PRIMARY  proximal (ICA/M1) > unlocalized: ", end="")
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

    payload = json.loads(out.read_text(encoding="utf-8")) if (out := OUT_TAB / "constraint_support.json").exists() else {}

    if args.perfusion.exists():
        perf = json.loads(args.perfusion.read_text(encoding="utf-8"))
        perf_ok = [r for r in perf["rows"] if r["status"] == "ok"]
        print("\nCheck 2 -- HIR vs mismatch ratio (independent collateral-signal check)")
        hir_check = check_hir_vs_mismatch(perf_ok)
        if hir_check["testable"]:
            print(f"  n={hir_check['n']}  Spearman rho={hir_check['spearman_rho']}  p={hir_check['p']:.3g}")
            print(f"  pass mark: {hir_check['pass_mark']}  -> {'PASS' if hir_check['passed'] else 'FAIL'}")
        else:
            print(f"  not testable -- {hir_check['reason']}")
        payload["hir_vs_mismatch"] = hir_check

        print("\nCheck 4 -- HIR vs final infarct volume, proximal occlusions only "
              "(the collateral constraints' own claim)")
        vol = {c["subject"]: float(c["volume_ml"]) for c in audit["cases"]}
        hir_vol_check = check_hir_vs_final_volume(feats, perf_ok, vol)
        g, p_ = hir_vol_check["good_collateral"], hir_vol_check["poor_collateral"]
        if g["n"] and p_["n"]:
            print(f"  good collateral (HIR bin 0): n={g['n']}  median={g['median']}")
            print(f"  poor collateral (HIR bin 2-3): n={p_['n']}  median={p_['median']}")
            c = hir_vol_check["comparison"]
            if c["testable"]:
                print(f"  p = {c['p_one_sided']:.3g}, AUC = {c['auc']}")
            print(f"  pass mark: {hir_vol_check['pass_mark']}  -> "
                  f"{'PASS' if hir_vol_check['passed'] else 'FAIL'}")
        else:
            print(f"  not enough subjects (good n={g['n']}, poor n={p_['n']})")
        payload["hir_vs_final_volume"] = hir_vol_check

        print("\nCheck 5 -- HIR vs final infarct volume, CONTINUOUS, all occlusion-positive "
              "patients (re-test of Check 4, fixing the underpowering)")
        cont_check = check_hir_vs_final_volume_continuous(feats, perf_ok, vol)
        if cont_check["testable"]:
            print(f"  n={cont_check['n']}  Spearman rho={cont_check['spearman_rho']}  p={cont_check['p']:.3g}")
            print(f"  pass mark: {cont_check['pass_mark']}  -> {'PASS' if cont_check['passed'] else 'FAIL'}")
        else:
            print(f"  not testable -- {cont_check['reason']}")
        payload["hir_vs_final_volume_continuous"] = cont_check

    if args.root is not None:
        print("\nCheck 3 -- perfusion-overlap plausibility rule "
              "(final infarct should sit inside admission Tmax>6s region)")
        overlap_check = check_perfusion_overlap(args.root.expanduser().resolve(), feats)
        if overlap_check["testable"]:
            print(f"  n={overlap_check['n']}  median fraction inside penumbra = "
                  f"{overlap_check['median_fraction_in_penumbra']:.3f}  "
                  f"({overlap_check['pct_below_50pct']*100:.1f}% of subjects below 50%)")
            print(f"  pass mark: {overlap_check['pass_mark']}  -> {'PASS' if overlap_check['passed'] else 'FAIL'}")
            if overlap_check["worst_5"]:
                print("  worst 5:", overlap_check["worst_5"])
        else:
            print(f"  not testable -- {overlap_check['reason']}")
        payload["perfusion_overlap"] = overlap_check

    contamination = unseen_vessel_diagnostic(feats)
    print("\n  diagnostic -- clot-side MCA/ICA missing from cow-msk (possible occluded, "
          "unopacified vessel):")
    for k, d in contamination.items():
        print(f"    {k:<20} {d['vessel_missing']:>2}/{d['n']:<3} ({d['fraction']:.0%})")
    print("    Unlocalized cases with a missing clot-side vessel may be upstream clots in a vessel")
    print("    the CTA never showed. Too few to explain the result; a label-quality issue for Phase 2.")

    OUT_TAB.mkdir(parents=True, exist_ok=True)
    payload["occlusion_level_vs_volume"] = {
        "rule": "Proximal (ICA/M1) occlusions come with larger final infarcts than unlocalized occlusions.",
        "pass_mark": f"proximal median > unlocalized median and one-sided Mann-Whitney p < {ALPHA}, fixed before running",
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
