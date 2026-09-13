"""
KG-XAI Phase 1 -- per-patient node features for the vascular graph, and the
laterality gate that checks them.

Structural features built here (perfusion features per territory are Phase 2):

  segment_present[k]      the subject's cow-msk contains TopCoW label k
  occluded[k]             the occlusion was localised to vessel k by
                          src/data/occlusion_site.py WITH A CONFIDENT MATCH
                          (nearest-vessel distance <= 15 mm). Never set from a
                          low-confidence match -- see below.
  occlusion_unlocalized   an occlusion exists but was beyond the segmented
                          circle (low-confidence match): a real, distal
                          occlusion with no node to put it on

Why the gating is non-negotiable: of the 20 subjects whose nearest vessel was
BA, 19 were low-confidence at a median 39 mm. BA is simply the vessel closest
to the middle when the real occlusion is far from everything. Ungated, those
become 19 fabricated basilar occlusions in the GNN's input.

The laterality gate -- an independent check that confident matches land on
the right vessel. A left anterior-circulation occlusion should produce a
left-hemisphere infarct. The infarct comes from a *different* annotation
(ses-02 lesion-msk, follow-up DWI) than the occlusion (ses-01 lvo-msk), and
its side is read from world geometry (RAS+, x increasing toward the patient's
right), not from any vessel label -- so agreement is evidence, not an
algorithm agreeing with itself. The midline is the median x of the midpoints
of every left/right vessel pair present in that subject's cow-msk, which is
symmetric in the labels and so does not depend on their L/R names either.

Pass mark, fixed before the first run: >= 80% side agreement over confident
anterior-circulation (ICA / MCA / ACA) cases.

Usage
-----
    python -m src.graph.features --root D:/ISLES-2024/train
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter

import numpy as np

from src.graph.build import load_graph

KG_DIR = pathlib.Path("kg")
OUT_TAB = pathlib.Path("outputs/tables")
GATE = 0.80
ANTERIOR = {"ICA", "MCA", "ACA"}
PAIRS = ("ICA", "MCA", "ACA", "PCA", "Pcom")


def _load(path):
    import nibabel as nib
    img = nib.load(str(path))
    return np.asanyarray(img.dataobj), img.affine


def _world_x(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    import nibabel as nib
    vox = np.argwhere(mask)
    if vox.size == 0:
        return np.empty(0)
    return nib.affines.apply_affine(affine, vox)[:, 0]


def midline_x(cow: np.ndarray, affine: np.ndarray, label_of: dict[str, int]) -> float | None:
    """Median x of left/right pair midpoints. Symmetric in the L/R names, so a
    mislabelled side would not bias it."""
    mids = []
    for vessel in PAIRS:
        l, r = label_of.get(f"L-{vessel}"), label_of.get(f"R-{vessel}")
        if l is None or r is None:
            continue
        xl, xr = _world_x(cow == l, affine), _world_x(cow == r, affine)
        if xl.size and xr.size:
            mids.append((xl.mean() + xr.mean()) / 2.0)
    return float(np.median(mids)) if mids else None


def _find(root: pathlib.Path, pattern: str) -> pathlib.Path | None:
    hits = sorted(root.glob(pattern))
    return hits[0] if hits else None


def build(root: pathlib.Path, cow_graph: dict, occlusion_rows: dict[str, dict]):
    nodes = cow_graph["nodes"]
    ids = [n["id"] for n in nodes]
    label_of = {n["id"]: n["topcow_label"] for n in nodes}
    deriv = root / "derivatives"

    subjects, labels_seen, validation = [], Counter(), {}
    for sdir in sorted(deriv.glob("sub-stroke*")):
        sub = sdir.name
        cow_f = _find(sdir, "ses-01/*cow-msk.nii*")
        les_f = _find(sdir, "ses-02/*lesion-msk.nii*")
        occ = occlusion_rows.get(sub, {})
        rec = {"subject": sub, "occlusion_status": occ.get("status", "missing")}

        present = [0] * len(ids)
        mid = None
        if cow_f is not None:
            cow, aff = _load(cow_f)
            uniq = set(int(v) for v in np.unique(cow)) - {0}
            labels_seen.update(uniq)
            present = [int(label_of[i] in uniq) for i in ids]
            mid = midline_x(cow, aff, label_of)
        rec["segment_present"] = present

        occluded = [0] * len(ids)
        rec["occlusion_unlocalized"] = 0
        rec["unlocalized_side"] = None
        rec["nearest_vessel"] = occ.get("nearest_cow_name")
        rec["confident"] = bool(occ.get("confident", False))
        if occ.get("status") == "ok":
            if rec["confident"] and rec["nearest_vessel"] in ids:
                occluded[ids.index(rec["nearest_vessel"])] = 1
            else:
                rec["occlusion_unlocalized"] = 1
                # side from the vessel NAME only -- never from lesion-msk,
                # which is the prediction target (see leakage_rule)
                rec["unlocalized_side"] = {"L": "left", "R": "right"}.get(
                    (rec["nearest_vessel"] or "").split("-")[0], "unknown")
        rec["occluded"] = occluded

        # Infarct side, for the laterality gate ONLY. Kept out of `rec` so it
        # can never be picked up as a model input -- lesion-msk is the
        # segmentation target (see leakage_rule in kg/cow_topology.json).
        val = {"lesion_right_fraction": None,
               "midline_x": None if mid is None else round(mid, 3)}
        if les_f is not None and mid is not None:
            les, laff = _load(les_f)
            x = _world_x(les > 0, laff)
            if x.size:
                val["lesion_right_fraction"] = round(float((x > mid).mean()), 4)
        validation[sub] = val
        subjects.append(rec)
    return subjects, labels_seen, validation


def laterality_gate(subjects: list[dict], validation: dict[str, dict]) -> dict:
    """Side agreement between the confidently localised occlusion and the
    follow-up infarct, over anterior-circulation vessels."""
    def side_of_vessel(v):
        return {"L": "left", "R": "right"}.get(v.split("-")[0])

    subjects = [dict(s, lesion_right_fraction=validation[s["subject"]]["lesion_right_fraction"])
                for s in subjects]

    def assess(group):
        agree, total, ambiguous = 0, 0, 0
        for r in group:
            f = r["lesion_right_fraction"]
            lesion_side = "right" if f > 0.5 else "left"
            total += 1
            agree += int(lesion_side == side_of_vessel(r["nearest_vessel"]))
            ambiguous += int(0.35 <= f <= 0.65)
        return {"n": total, "agree": agree,
                "rate": round(agree / total, 4) if total else None,
                "ambiguous_bilateral": ambiguous}

    usable = [r for r in subjects
              if r["occlusion_status"] == "ok" and r["lesion_right_fraction"] is not None
              and r["nearest_vessel"] and side_of_vessel(r["nearest_vessel"])]
    ant = lambda r: r["nearest_vessel"].split("-")[1] in ANTERIOR
    conf_ant = [r for r in usable if r["confident"] and ant(r)]
    conf_post = [r for r in usable if r["confident"] and not ant(r)]
    low_ant = [r for r in usable if not r["confident"] and ant(r)]

    res = {"gate": GATE,
           "confident_anterior": assess(conf_ant),
           "confident_posterior_pca": assess(conf_post),
           "low_confidence_anterior": assess(low_ant),
           "disagreements": [
               {"subject": r["subject"], "vessel": r["nearest_vessel"],
                "lesion_right_fraction": r["lesion_right_fraction"]}
               for r in conf_ant
               if ("right" if r["lesion_right_fraction"] > 0.5 else "left")
               != side_of_vessel(r["nearest_vessel"])]}
    rate = res["confident_anterior"]["rate"]
    res["passed"] = bool(rate is not None and rate >= GATE)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--occlusion-json", type=pathlib.Path,
                    default=OUT_TAB / "occlusion_site.json")
    args = ap.parse_args()

    cow_graph = load_graph(KG_DIR / "cow_topology.json")
    occ = json.loads(args.occlusion_json.read_text(encoding="utf-8"))
    occ_rows = {r["subject"]: r for r in occ["rows"]}

    print(f"\nBuilding vascular-graph node features  root={args.root}")
    subjects, labels_seen, validation = build(args.root.expanduser().resolve(), cow_graph, occ_rows)
    ids = [n["id"] for n in cow_graph["nodes"]]
    graph_labels = {n["topcow_label"] for n in cow_graph["nodes"]}

    print(f"  {len(subjects)} subjects")
    extra = sorted(set(labels_seen) - graph_labels)
    print(f"  cow-msk labels seen across cohort: {sorted(labels_seen)}")
    print("  labels outside the graph: " + (str(extra) if extra else "none"))

    presence = np.array([s["segment_present"] for s in subjects])
    print("\n  vessel present in cow-msk (of %d):" % len(subjects))
    for i, v in enumerate(ids):
        print(f"    {v:>7}: {int(presence[:, i].sum()):>3}")

    status = Counter(s["occlusion_status"] for s in subjects)
    occ_nodes = Counter(ids[int(np.argmax(s["occluded"]))]
                        for s in subjects if sum(s["occluded"]))
    n_unloc = sum(s["occlusion_unlocalized"] for s in subjects)
    print(f"\n  occlusion status: {dict(status)}")
    print(f"  occluded node set (confident only): {sum(occ_nodes.values())} subjects "
          f"-> {dict(occ_nodes.most_common())}")
    print(f"  occlusion_unlocalized flag: {n_unloc} subjects, side: "
          f"{dict(Counter(s['unlocalized_side'] for s in subjects if s['occlusion_unlocalized']))}")

    # internal consistency: every confident ok case sets exactly one node,
    # every low-confidence ok case sets none and raises the flag
    bad = [s["subject"] for s in subjects if s["occlusion_status"] == "ok" and
           (sum(s["occluded"]) + s["occlusion_unlocalized"]) != 1]
    print("  every localised case sets exactly one of {node, unlocalized}: "
          + ("OK" if not bad else f"FAILED {bad}"))

    gate = laterality_gate(subjects, validation)
    ca = gate["confident_anterior"]
    print(f"\nLaterality gate (pass mark {GATE:.0%}, fixed before running)")
    print(f"  confident anterior (ICA/MCA/ACA): {ca['agree']}/{ca['n']} agree "
          f"= {ca['rate']:.1%}   -> {'PASS' if gate['passed'] else 'FAIL'}")
    print(f"    of which bilateral-looking infarcts (35-65% right): {ca['ambiguous_bilateral']}")
    cp, la = gate["confident_posterior_pca"], gate["low_confidence_anterior"]
    if cp["n"]:
        print(f"  confident posterior (PCA), for reference: {cp['agree']}/{cp['n']} = {cp['rate']:.1%}")
    if la["n"]:
        print(f"  low-confidence anterior, for reference: {la['agree']}/{la['n']} = {la['rate']:.1%}")
    if gate["disagreements"]:
        print("  disagreements (confident anterior):")
        for d in gate["disagreements"]:
            print(f"    {d['subject']}  {d['vessel']:>6}  lesion right-fraction {d['lesion_right_fraction']}")

    OUT_TAB.mkdir(parents=True, exist_ok=True)
    out = OUT_TAB / "kg_node_features.json"
    out.write_text(json.dumps({
        "node_order": ids,
        "labels_seen": sorted(labels_seen), "labels_outside_graph": extra,
        "laterality_gate": gate,
        "subjects": subjects,
        "validation_only_do_not_use_as_features": {
            "note": "Derived from lesion-msk (the segmentation target). For the laterality gate only.",
            "per_subject": validation,
        },
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    if bad or not gate["passed"]:
        raise SystemExit("node-feature checks FAILED")


if __name__ == "__main__":
    main()
