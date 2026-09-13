"""
KG-XAI Phase 1 -- which clinical variables can the guideline graph actually use?

The rule graph (kg/guideline_rules.json) conditions on age, NIHSS, premorbid
mRS and time from onset. Whether a rule can be evaluated on ISLES'24 is a
question about the data, not about the rule -- so it gets measured here, per
column, rather than assumed.

The headline result this script exists to record: every timing column in the
public release (onset-to-door, alert-to-door, door-to-imaging, door-to-groin,
...) is empty for all 149 subjects. No time-window criterion from DAWN,
DEFUSE-3 or the AHA/ASA guideline can be evaluated on this dataset. Rules keep
their time conditions for fidelity to the source, marked non-evaluable.

Missingness convention: '' and the literal string 'nan' both mean missing --
this release exports some columns from pandas without na_rep (see
src/data/mrs_shift.py for the bug that caused).

Usage
-----
    python -m src.data.clinical_audit --root D:/ISLES-2024/train
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

OUT_TAB = pathlib.Path("outputs/tables")

# Columns the guideline graph conditions on, mapped to the concept they feed.
GRAPH_COLUMNS = {
    "Age": "age",
    "NIHSS at admission": "nihss",
    "mRS premorbid": "mrs_premorbid",
    "Wake-up": "wake_up",
    "Onset to door": "time_window",
}


def _missing(v) -> bool:
    v = (v or "").strip()
    return v == "" or v.lower() == "nan"


def read_table(root: pathlib.Path, session: str, suffix: str) -> dict[str, dict]:
    rows = {}
    for f in sorted(root.glob(f"phenotype/sub-stroke*/{session}/*_{suffix}.csv")):
        with open(f, newline="", encoding="utf-8", errors="replace") as fh:
            rows[f.parent.parent.name] = next(csv.DictReader(fh), {})
    return rows


def completeness(rows: dict[str, dict]) -> dict[str, dict]:
    cols = sorted({c for r in rows.values() for c in r})
    n = len(rows)
    out = {}
    for c in cols:
        filled = sum(1 for r in rows.values() if not _missing(r.get(c)))
        out[c] = {"filled": filled, "total": n, "fraction": round(filled / n, 3) if n else 0.0}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=pathlib.Path)
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    base = read_table(root, "ses-01", "demographic_baseline")
    outc = read_table(root, "ses-02", "outcome")
    if not base:
        raise SystemExit(f"no phenotype CSVs under {root}/phenotype")

    comp_base = completeness(base)
    comp_out = completeness(outc)

    print(f"\nClinical completeness  root={root}  n={len(base)}")
    print(f"\n  baseline (ses-01)")
    for c, s in sorted(comp_base.items(), key=lambda kv: -kv[1]["filled"]):
        mark = "  EMPTY" if s["filled"] == 0 else ""
        print(f"    {c:>28}: {s['filled']:>3}/{s['total']}{mark}")
    print(f"\n  outcome (ses-02)")
    for c, s in sorted(comp_out.items(), key=lambda kv: -kv[1]["filled"]):
        mark = "  EMPTY" if s["filled"] == 0 else ""
        print(f"    {c:>28}: {s['filled']:>3}/{s['total']}{mark}")

    empty = [c for c, s in comp_base.items() if s["filled"] == 0]
    print(f"\n  completely empty baseline columns ({len(empty)}): {', '.join(empty)}")

    graph_use = {}
    print("\n  concepts the guideline graph needs:")
    for col, concept in GRAPH_COLUMNS.items():
        s = comp_base.get(col, {"filled": 0, "total": len(base)})
        evaluable = s["filled"] > 0
        graph_use[concept] = {"column": col, **s, "evaluable_on_isles24": evaluable}
        print(f"    {concept:>14} <- {col:<20} {s['filled']:>3}/{s['total']}  "
              f"{'usable' if evaluable else 'NOT EVALUABLE'}")

    OUT_TAB.mkdir(parents=True, exist_ok=True)
    out = OUT_TAB / "clinical_completeness.json"
    out.write_text(json.dumps({
        "root": str(root), "n_subjects": len(base),
        "baseline": comp_base, "outcome": comp_out,
        "empty_baseline_columns": empty,
        "graph_concepts": graph_use,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
