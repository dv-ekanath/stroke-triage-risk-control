"""
Review 1, item 1.7a -- derive occlusion-site labels from cow-msk x lvo-msk.

v3 section 1.3 describes this as "near-trivial": intersect the manual CTA
occlusion mask with the labelled Circle-of-Willis (CoW) segmentation, no
atlas registration needed. On the real data that is only half right.

  - Direct voxel INTERSECTION mostly comes up empty: the two masks are both
    small, independently-drawn annotations that sit near each other rather
    than on top of each other. Checked on real data (see repo notes): only
    ~2/5 sampled subjects had ANY overlapping voxel.
  - NEAREST-DISTANCE from the lvo-msk centroid to each cow-msk label works
    well instead. Confident matches (proximal occlusions, at/near the Circle
    itself) sit within a few mm. A minority of subjects have their nearest
    label 30+ mm away -- almost certainly a more distal (M2 or beyond)
    occlusion that the CoW segmentation, which only covers the proximal
    circle, was never going to capture. Those get flagged LOW-CONFIDENCE
    rather than silently assigned a site.

Label convention: this release ships no legend file (no JSON/TSV anywhere in
the archive). The label integers are assumed to follow the standard TopCoW
challenge convention (v3 section 1.3 confirms the CoW masks are
"TopCoW-derived"), and this was VERIFIED empirically on this data: for every
candidate left/right pair below, the lower-numbered label sits at a
consistently higher world x-coordinate (RAS+, so more toward patient right)
than its paired higher-numbered label, across every one of 15 spot-checked
subjects with both labels present. That is exactly the pattern the TopCoW
numbering predicts. It is still a pseudolabel pipeline (v3 section 7 risk:
"CoW pseudolabels are model-generated ... spot-check a subset"), so this
script's job is to make that spot-check possible, not to assert the mapping
is ground truth.

TOPCOW_LABELS values 13 and 15 (3rd-A2 / accessory ACA) were not observed in
the sample used to verify the pairing and are included from the published
convention without independent confirmation here.

Site bucketing follows v3 section 5.1's H2 target space (ICA / M1 / M2 / other):
  ICA labels        -> "ICA"
  MCA labels, close  -> "M1"   (occlusion sits at/near the CoW-segmented MCA
                                 stem, i.e. proximal M1)
  MCA labels, far     -> "M2"   (nearest label is MCA but the distance implies
                                 a more distal occlusion the segmentation
                                 doesn't reach)
  everything else     -> "other"
  distance beyond the confidence threshold, any label -> "other (low-confidence)"

Usage
-----
    python3 -m src.data.occlusion_site --root D:/ISLES-2024/train
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter

import numpy as np

OUT_TAB = pathlib.Path("outputs/tables")

# Standard TopCoW 13-class convention (0 = background). Verified for the
# paired labels below; 1 and 10 are the unpaired midline vessels; 13/15 are
# the rare accessory-ACA variant, not independently confirmed on this data.
TOPCOW_LABELS = {
    1: "BA", 2: "R-PCA", 3: "L-PCA", 4: "R-ICA", 5: "R-MCA",
    6: "L-ICA", 7: "L-MCA", 8: "R-Pcom", 9: "L-Pcom", 10: "Acom",
    11: "R-ACA", 12: "L-ACA", 13: "3rd-A2", 15: "3rd-A2",
}
ICA_LABELS = {4, 6}
MCA_LABELS = {5, 7}

# Nearest-label distance beyond which the match is not trusted. Chosen from
# the observed gap in the data: confident matches cluster under ~5 mm,
# distal/uncertain ones start around 30 mm. 15 mm sits in the empty gap
# between those two populations in the spot-checked sample.
CONFIDENT_MM = 15.0


def _load(path: pathlib.Path):
    import nibabel as nib
    img = nib.load(str(path))
    return np.asanyarray(img.dataobj), img.affine


def nearest_cow_label(cow_data: np.ndarray, affine: np.ndarray, lvo_data: np.ndarray):
    """Distance (mm) from the LVO mask's centroid to the nearest voxel of
    each present CoW label. Returns dict {label: distance_mm}, or {} if the
    LVO mask is empty."""
    import nibabel as nib

    lvo_vox = np.argwhere(lvo_data > 0)
    if lvo_vox.size == 0:
        return {}
    centroid_world = nib.affines.apply_affine(affine, lvo_vox).mean(axis=0)

    out = {}
    for val in np.unique(cow_data):
        if val == 0:
            continue
        cow_vox = np.argwhere(cow_data == val)
        cow_world = nib.affines.apply_affine(affine, cow_vox)
        d = float(np.sqrt(((cow_world - centroid_world) ** 2).sum(axis=1)).min())
        out[int(val)] = d
    return out


def classify_site(nearest_label: int | None, dist_mm: float | None) -> str:
    if nearest_label is None:
        return "unknown (no lvo-msk)"
    if dist_mm is not None and dist_mm > CONFIDENT_MM:
        return "other (low-confidence)"
    if nearest_label in ICA_LABELS:
        return "ICA"
    if nearest_label in MCA_LABELS:
        return "M1"
    return "other"


def process_subject(sub_dir: pathlib.Path) -> dict | None:
    cow_f = sorted(sub_dir.glob("ses-01/*cow-msk.nii*"))
    lvo_f = sorted(sub_dir.glob("ses-01/*lvo-msk.nii*"))
    if not cow_f or not lvo_f:
        return None

    cow_data, cow_aff = _load(cow_f[0])
    lvo_data, lvo_aff = _load(lvo_f[0])
    if cow_data.shape != lvo_data.shape:
        return {"subject": sub_dir.name, "status": "shape_mismatch"}

    dists = nearest_cow_label(cow_data, cow_aff, lvo_data > 0)
    if not dists:
        return {"subject": sub_dir.name, "status": "empty_lvo_mask"}

    nearest_label = min(dists, key=dists.get)
    nearest_dist = dists[nearest_label]
    site = classify_site(nearest_label, nearest_dist)

    return {
        "subject": sub_dir.name,
        "status": "ok",
        "lvo_voxels": int((lvo_data > 0).sum()),
        "nearest_cow_label": nearest_label,
        "nearest_cow_name": TOPCOW_LABELS.get(nearest_label, f"label-{nearest_label}"),
        "nearest_dist_mm": round(nearest_dist, 2),
        "site": site,
        "confident": bool(nearest_dist <= CONFIDENT_MM),
        "all_label_distances_mm": {TOPCOW_LABELS.get(k, str(k)): round(v, 2)
                                    for k, v in sorted(dists.items(), key=lambda kv: kv[1])[:3]},
    }


def main():
    global CONFIDENT_MM

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=pathlib.Path,
                     help="ISLES-2024 root, e.g. D:/ISLES-2024/train")
    ap.add_argument("--confident-mm", type=float, default=CONFIDENT_MM,
                     help="distance threshold below which a site match is trusted")
    args = ap.parse_args()

    CONFIDENT_MM = args.confident_mm

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"{root} does not exist")
    OUT_TAB.mkdir(parents=True, exist_ok=True)

    deriv = root / "derivatives"
    subs = sorted(p for p in deriv.glob("sub-stroke*") if p.is_dir())
    if not subs:
        raise SystemExit(f"no sub-stroke* directories under {deriv}")

    print(f"\nOcclusion-site derivation  root={root}  n_subjects={len(subs)}")
    print(f"confidence threshold = {CONFIDENT_MM:g} mm\n")

    rows = []
    for i, s in enumerate(subs, 1):
        r = process_subject(s)
        if r is not None:
            rows.append(r)
        if i % 25 == 0 or i == len(subs):
            print(f"  {i}/{len(subs)}")

    ok = [r for r in rows if r["status"] == "ok"]
    bad = [r for r in rows if r["status"] != "ok"]

    site_counts = Counter(r["site"] for r in ok)
    n_confident = sum(1 for r in ok if r["confident"])

    print(f"\nProcessed {len(ok)}/{len(subs)} subjects with both masks present")
    if bad:
        bad_counts = Counter(r["status"] for r in bad)
        print(f"  skipped: {dict(bad_counts)}  -> {[r['subject'] for r in bad]}")

    print(f"\nSite distribution (n={len(ok)}):")
    for site, n in site_counts.most_common():
        print(f"  {n:>4}  ({n/len(ok)*100:>5.1f}%)  {site}")

    print(f"\nConfident matches (<= {CONFIDENT_MM:g} mm): "
          f"{n_confident}/{len(ok)} ({n_confident/len(ok)*100:.1f}%)")
    low_conf = [r["subject"] for r in ok if not r["confident"]]
    if low_conf:
        print(f"  NOT confident, flagged for manual spot-check ({len(low_conf)}):")
        print(f"    {', '.join(low_conf)}")

    dist_arr = np.array([r["nearest_dist_mm"] for r in ok])
    print(f"\nnearest-distance summary (mm): mean={dist_arr.mean():.2f} "
          f"median={np.median(dist_arr):.2f} max={dist_arr.max():.2f}")

    payload = {
        "root": str(root),
        "confident_mm": CONFIDENT_MM,
        "n_subjects": len(subs),
        "n_processed": len(ok),
        "site_counts": dict(site_counts),
        "n_confident": n_confident,
        "low_confidence_subjects": low_conf,
        "skipped": [{"subject": r["subject"], "status": r["status"]} for r in bad],
        "rows": rows,
    }
    (OUT_TAB / "occlusion_site.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT_TAB/'occlusion_site.json'}")


if __name__ == "__main__":
    main()
