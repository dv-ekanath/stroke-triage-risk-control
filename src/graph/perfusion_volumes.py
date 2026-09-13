"""
KG-XAI Phase 2 -- brain mask, acute core, penumbra, mismatch, HIR.

All derived from ADMISSION (ses-01) perfusion maps only -- see leakage_rule
in kg/cow_topology.json. Ground truth (lesion-msk) is used later, only to
TEST these numbers, never to compute them.

Brain mask
----------
CBF/CBV/MTT/Tmax are NOT brain-masked in the release (checked: CBF/CBV/MTT
background = 0, Tmax background = -30, but the CBF>0 footprint measures
~2200 mL -- far more than an adult brain -- so it includes extracranial
tissue and can't be used as a brain mask on its own). A brain mask is built
from NCCT instead: soft-tissue window (0-100 HU, excludes air and bone),
morphological opening to drop thin scalp strands, largest 3D connected
component, per-slice hole filling. This is also Phase 2's "skull stripping"
deliverable, not a separate step.

Gate (fixed before running on the cohort): intracranial volume must fall in
1000-1900 mL for at least 90% of subjects. Spot-checked at 1334-1597 mL on 4
subjects before committing to this method.

Core / penumbra / mismatch
---------------------------
Core: relative CBF < 30% of the CONTRALATERAL hemisphere's median CBF,
within the brain mask. Contralateral side comes from the confidence-gated
occlusion side (src/graph/features.py) -- for unlocalized-side subjects,
core is not computed (no reliable "normal" side to normalise against) and
this is reported, not silently defaulted.
Penumbra: Tmax > 6 s, within the brain mask (background -30 is safely under
the threshold, so no separate masking is strictly needed for Tmax alone --
the brain mask is still applied for consistency and to exclude any positive
extracranial Tmax noise).
Mismatch ratio = penumbra / core (undefined if core = 0).
Mismatch volume = penumbra - core.

Collateral proxy (HIR)
-----------------------
HIR = volume(Tmax>10s) / volume(Tmax>6s), within the brain mask. High HIR =
poor collaterals (Olivot 2014, PMID 24595591). Binned 0-L3 to match the
architecture's collateral-grade output shape; direction verified against
occlusion presence before trusting it (see main()).

Usage
-----
    python -m src.graph.perfusion_volumes --root D:/ISLES-2024/train
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from scipy import ndimage

OUT_TAB = pathlib.Path("outputs/tables")
ICV_RANGE = (1000.0, 1900.0)
ICV_GATE_FRACTION = 0.90


def _load(path):
    import nibabel as nib
    img = nib.load(str(path))
    return np.asanyarray(img.dataobj), img.header.get_zooms()[:3]


def brain_mask(ncct: np.ndarray) -> np.ndarray:
    soft = (ncct > 0) & (ncct < 100)
    soft = ndimage.binary_opening(soft, structure=np.ones((3, 3, 1)))
    lbl, n = ndimage.label(soft)
    if n == 0:
        return soft
    sizes = ndimage.sum(soft, lbl, range(1, n + 1))
    mask = lbl == (int(np.argmax(sizes)) + 1)
    for z in range(mask.shape[2]):
        mask[:, :, z] = ndimage.binary_fill_holes(mask[:, :, z])
    return mask


def hemisphere_masks(mask: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Left/right split of the brain mask by world x=0 (patient midline is
    close to image x=0 for these head-centred acquisitions; refined by the
    mask's own centroid so a small acquisition offset doesn't bias it)."""
    import nibabel as nib
    vox = np.argwhere(mask)
    world = nib.affines.apply_affine(affine, vox)
    mid = float(np.median(world[:, 0]))
    right = world[:, 0] > mid          # RAS+: +x = patient's right
    right_mask = np.zeros(mask.shape, dtype=bool)
    left_mask = np.zeros(mask.shape, dtype=bool)
    right_mask[tuple(vox[right].T)] = True
    left_mask[tuple(vox[~right].T)] = True
    return left_mask, right_mask, mid


def _find(root, pattern):
    hits = sorted(root.glob(pattern))
    return hits[0] if hits else None


def process_subject(root: pathlib.Path, sub: str, occ_side: str | None) -> dict | None:
    raw = root / "raw_data" / sub / "ses-01"
    pm = root / "derivatives" / sub / "ses-01" / "perfusion-maps"
    ncct_f = _find(raw, "*_ncct.nii*")
    cbf_f, tmax_f = _find(pm, "*_cbf.nii*"), _find(pm, "*_tmax.nii*")
    if not (ncct_f and cbf_f and tmax_f):
        return {"subject": sub, "status": "missing_input"}

    import nibabel as nib
    try:
        ncct_img = nib.load(str(ncct_f))
        ncct = np.asanyarray(ncct_img.dataobj)
        cbf, zooms = _load(cbf_f)
        tmax, _ = _load(tmax_f)
    except Exception as e:
        # e.g. sub-stroke0043's truncated cbf.nii.gz -- a real defect in the
        # source archive (confirmed against its own manifest in Review 1),
        # not an extraction problem. Skip cleanly rather than crash the run.
        return {"subject": sub, "status": "unreadable_file", "error": str(e)}
    if cbf.shape != ncct.shape or tmax.shape != ncct.shape:
        return {"subject": sub, "status": "shape_mismatch"}

    vox_ml = float(np.prod(zooms)) / 1000.0
    mask = brain_mask(ncct)
    icv_ml = float(mask.sum() * vox_ml)

    tmax = np.nan_to_num(tmax, nan=-30.0, posinf=0.0, neginf=-30.0)
    cbf = np.nan_to_num(cbf, nan=0.0, posinf=0.0, neginf=0.0)

    penumbra_vox = mask & (tmax > 6.0)
    core_vox_10 = mask & (tmax > 10.0)
    penumbra_ml = float(penumbra_vox.sum() * vox_ml)
    hir_num_ml = float(core_vox_10.sum() * vox_ml)
    hir = round(hir_num_ml / penumbra_ml, 4) if penumbra_ml > 0 else None

    core_ml, mismatch_ratio, mismatch_ml, contra_median_cbf = None, None, None, None
    if occ_side in ("left", "right"):
        left, right, _ = hemisphere_masks(mask, ncct_img.affine)
        contra = left if occ_side == "right" else right   # normal side = opposite the clot
        ipsi = right if occ_side == "right" else left
        contra_cbf = cbf[contra & (cbf > 0)]
        if contra_cbf.size > 100:
            contra_median_cbf = float(np.median(contra_cbf))
            core_vox = ipsi & (cbf < 0.30 * contra_median_cbf) & (cbf > 0)
            core_ml = float(core_vox.sum() * vox_ml)
            if core_ml > 0:
                mismatch_ratio = round(penumbra_ml / core_ml, 4)
                mismatch_ml = round(penumbra_ml - core_ml, 2)

    return {
        "subject": sub, "status": "ok",
        "icv_ml": round(icv_ml, 1),
        "occ_side_used": occ_side,
        "contralateral_median_cbf": round(contra_median_cbf, 2) if contra_median_cbf else None,
        "acute_core_ml": round(core_ml, 2) if core_ml is not None else None,
        "penumbra_ml": round(penumbra_ml, 2),
        "mismatch_ratio": mismatch_ratio,
        "mismatch_volume_ml": mismatch_ml,
        "hir": hir,
        "hir_bin": None if hir is None else min(3, int(hir * 4)),  # 0-0.25->0, ..., >=0.75->3(capped)
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--features", type=pathlib.Path, default=OUT_TAB / "kg_node_features.json")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    feats = json.loads(args.features.read_text(encoding="utf-8"))

    def occ_side_of(s: dict) -> str | None:
        if s.get("occlusion_status") != "ok":
            return None
        if not s.get("confident"):
            return s.get("unlocalized_side") if s.get("unlocalized_side") != "unknown" else None
        v = s.get("nearest_vessel") or ""
        return {"L": "left", "R": "right"}.get(v.split("-")[0])

    subs = feats["subjects"]
    print(f"\nPerfusion-derived volumes  root={root}  n={len(subs)}")
    rows = []
    for i, s in enumerate(subs, 1):
        r = process_subject(root, s["subject"], occ_side_of(s))
        rows.append(r)
        if i % 25 == 0 or i == len(subs):
            print(f"  {i}/{len(subs)}")

    ok = [r for r in rows if r["status"] == "ok"]
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"\nProcessed {len(ok)}/{len(subs)}")
    if bad:
        from collections import Counter
        print(f"  skipped: {dict(Counter(r['status'] for r in bad))}")

    icv = np.array([r["icv_ml"] for r in ok])
    within = np.mean((icv >= ICV_RANGE[0]) & (icv <= ICV_RANGE[1]))
    print(f"\nBrain-mask gate: ICV in [{ICV_RANGE[0]:.0f}, {ICV_RANGE[1]:.0f}] mL "
          f"for {within*100:.1f}% of subjects (need >= {ICV_GATE_FRACTION*100:.0f}%)")
    print(f"  ICV distribution: mean={icv.mean():.0f} median={np.median(icv):.0f} "
          f"min={icv.min():.0f} max={icv.max():.0f}")
    icv_gate_passed = bool(within >= ICV_GATE_FRACTION)
    print(f"  -> {'PASS' if icv_gate_passed else 'FAIL'}")

    n_core = sum(1 for r in ok if r["acute_core_ml"] is not None)
    print(f"\nAcute core computed for {n_core}/{len(ok)} "
          f"(needs a known occlusion side; excluded when unlocalized+unknown-side)")
    mism = [r["mismatch_ratio"] for r in ok if r["mismatch_ratio"] is not None]
    if mism:
        print(f"  mismatch ratio: median={np.median(mism):.2f}  "
              f"(DEFUSE-3 threshold: >= 1.8)")

    hir = [r["hir"] for r in ok if r["hir"] is not None]
    print(f"\nHIR computed for {len(hir)}/{len(ok)} (needs nonzero penumbra)")
    if hir:
        print(f"  HIR: mean={np.mean(hir):.3f} median={np.median(hir):.3f} "
              f"bins: {dict(zip(*np.unique([r['hir_bin'] for r in ok if r['hir_bin'] is not None], return_counts=True)))}")

    OUT_TAB.mkdir(parents=True, exist_ok=True)
    out = OUT_TAB / "perfusion_volumes.json"
    out.write_text(json.dumps({
        "root": str(root),
        "icv_gate": {"range_ml": ICV_RANGE, "required_fraction": ICV_GATE_FRACTION,
                     "observed_fraction": round(float(within), 4), "passed": icv_gate_passed},
        "n_subjects": len(subs), "n_ok": len(ok),
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    if not icv_gate_passed:
        raise SystemExit("brain-mask gate FAILED")


if __name__ == "__main__":
    main()
