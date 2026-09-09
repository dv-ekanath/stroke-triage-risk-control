"""
Review 1, item 1.7b -- construct the mRS-shift target (H4).

v3 section 5.1: H4 predicts mRS SHIFT (3-month minus premorbid), not absolute
3-month mRS.  Premorbid mRS averages 0.9 +/- 1.2 in the training set and is a
strong determinant of the absolute outcome, so predicting the absolute score
lets the model score well by learning baseline disability from tabular data
alone; shift removes that shortcut.  Absolute mRS is kept alongside for
comparability with the outcome-prediction literature (v3 section 6).

This release ships no single clinical table (see src/data/center_labels.py):
premorbid mRS lives in each subject's
    phenotype/<sub>/ses-01/<sub>_ses-01_demographic_baseline.csv  ("mRS premorbid")
and 3-month mRS in
    phenotype/<sub>/ses-02/<sub>_ses-02_outcome.csv                ("mRS 3 months")

Usage
-----
    python3 -m src.data.mrs_shift --root D:/ISLES-2024/train
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np

OUT_TAB = pathlib.Path("outputs/tables")

BASELINE_COL = "mRS premorbid"
OUTCOME_COL = "mRS 3 months"
TICI_COL = "TICI postinterventional"


def _read_one_row(path: pathlib.Path) -> dict:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    return row or {}


def _to_float(x) -> float | None:
    """None for missing.  Guards against the literal string 'nan' -- some
    columns in this release were exported from pandas without na_rep, so a
    missing value shows up as the four characters "nan", not an empty field.
    float("nan") does NOT raise, so without this check a missing value would
    silently become a real NaN, not None, and corrupt any mean/std computed
    over it without a nan-aware reduction."""
    x = (x or "").strip()
    if x == "" or x.lower() == "nan":
        return None
    try:
        v = float(x)
    except ValueError:
        return None
    return None if np.isnan(v) else v


def _clean_str(x) -> str | None:
    """String field, treating '' and the literal 'nan' as missing."""
    x = (x or "").strip()
    return None if x == "" or x.lower() == "nan" else x


def process_subject(sub_dir: pathlib.Path) -> dict | None:
    base_f = sorted(sub_dir.glob("ses-01/*demographic_baseline.csv"))
    out_f = sorted(sub_dir.glob("ses-02/*outcome.csv"))
    if not base_f or not out_f:
        return None

    base = _read_one_row(base_f[0])
    outc = _read_one_row(out_f[0])

    premorbid = _to_float(base.get(BASELINE_COL))
    three_mo = _to_float(outc.get(OUTCOME_COL))
    tici = _clean_str(outc.get(TICI_COL))

    shift = None
    if premorbid is not None and three_mo is not None:
        shift = three_mo - premorbid

    return {
        "subject": sub_dir.name,
        "mrs_premorbid": premorbid,
        "mrs_3mo": three_mo,
        "mrs_shift": shift,
        "tici_post": tici,
        "status": "ok" if shift is not None else "missing_value",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=pathlib.Path,
                     help="ISLES-2024 root, e.g. D:/ISLES-2024/train")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"{root} does not exist")
    OUT_TAB.mkdir(parents=True, exist_ok=True)

    pheno = root / "phenotype"
    subs = sorted(p for p in pheno.glob("sub-stroke*") if p.is_dir())
    if not subs:
        raise SystemExit(f"no sub-stroke* directories under {pheno}")

    print(f"\nmRS-shift target construction  root={root}  n_subjects={len(subs)}")

    rows = [r for r in (process_subject(s) for s in subs) if r is not None]
    ok = [r for r in rows if r["status"] == "ok"]
    missing = [r for r in rows if r["status"] != "ok"]

    print(f"\nComplete (premorbid + 3mo) pairs: {len(ok)}/{len(rows)}")
    if missing:
        no_pre = sum(1 for r in missing if r["mrs_premorbid"] is None)
        no_3mo = sum(1 for r in missing if r["mrs_3mo"] is None)
        print(f"  missing premorbid: {no_pre}   missing 3mo: {no_3mo}")
        print(f"  subjects: {', '.join(r['subject'] for r in missing)}")

    if ok:
        pre = np.array([r["mrs_premorbid"] for r in ok])
        post = np.array([r["mrs_3mo"] for r in ok])
        shift = np.array([r["mrs_shift"] for r in ok])
        print(f"\nmRS premorbid : mean={pre.mean():.2f} +/- {pre.std(ddof=1):.2f}"
              f"   (v3 reference: 0.9 +/- 1.2)")
        print(f"mRS 3-month   : mean={post.mean():.2f} +/- {post.std(ddof=1):.2f}")
        print(f"mRS shift     : mean={shift.mean():.2f} +/- {shift.std(ddof=1):.2f}"
              f"   range [{shift.min():.0f}, {shift.max():.0f}]")
        print(f"good outcome (absolute mRS <= 2 at 3mo): "
              f"{int((post <= 2).sum())}/{len(post)} "
              f"({(post <= 2).mean()*100:.1f}%)")

    n_tici_all = sum(1 for r in rows if r["tici_post"])
    print(f"\nmTICI recorded (whole cohort): {n_tici_all}/{len(rows)} "
          f"(v3 reference: 97/149 in the training set)")

    payload = {
        "root": str(root),
        "n_subjects": len(subs),
        "n_complete": len(ok),
        "n_missing": len(missing),
        "n_tici_recorded": n_tici_all,
        "summary": {
            "mrs_premorbid_mean": float(pre.mean()) if ok else None,
            "mrs_premorbid_sd": float(pre.std(ddof=1)) if ok else None,
            "mrs_3mo_mean": float(post.mean()) if ok else None,
            "mrs_3mo_sd": float(post.std(ddof=1)) if ok else None,
            "mrs_shift_mean": float(shift.mean()) if ok else None,
            "mrs_shift_sd": float(shift.std(ddof=1)) if ok else None,
            "good_outcome_rate": float((post <= 2).mean()) if ok else None,
        } if ok else {},
        "rows": rows,
    }
    (OUT_TAB / "mrs_shift.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT_TAB/'mrs_shift.json'}")


if __name__ == "__main__":
    main()
