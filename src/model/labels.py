"""
KG-XAI Phase 3 -- per-subject labels for the two new task heads (H2 LVO
localization, H3 collateral proxy), built from the real outputs already on
disk. Nothing here recomputes anything Phase 1/2 already did.

LVO localization (H2): reuses the exact classification logic Phase 1/2
already validated (`src/graph/constraint_support.py:localization_class`),
so the model's target and the knowledge-graph's `lvo_localization` concept
(`kg/guideline_rules.json`) are guaranteed to agree -- one function, not two
copies that could drift.

    none               (4)   empty lvo-msk
    other_confident   (15)   confident ACA/PCA/BA match
    proximal_anterior (87)   confident ICA/MCA match
    unlocalized       (42)   low-confidence match, beyond the segmented circle

Collateral proxy (H3): the HIR bin (0-3) from `src/graph/perfusion_volumes.py`
-- a real, derived quantity, but recall it FAILED every independent
validation as a collateral signal in this cohort (Check 2 and Check 5,
`kg/guideline_rules.json`). The head is still trained to reproduce it
faithfully (that's a well-defined target regardless), but its predictions
must not be presented as validated collateral status -- say "collateral
proxy" every time, never "collateral grade."

Usage
-----
    from src.model.labels import load_auxiliary_labels
    labels = load_auxiliary_labels()
    labels["sub-stroke0001"] -> {"lvo_class": 2, "collateral_bin": 1}
"""

from __future__ import annotations

import json
import pathlib
import sys

OUT_TAB = pathlib.Path("outputs/tables")

LVO_CLASSES = ("none", "other_confident", "proximal_anterior", "unlocalized")
N_LVO_CLASSES = len(LVO_CLASSES)
N_COLLATERAL_BINS = 4  # HIR bins 0-3, see src/graph/perfusion_volumes.py


def _localization_class(row: dict) -> str | None:
    """Same logic as src/graph/constraint_support.py:localization_class,
    duplicated rather than imported to avoid src/model importing from
    src/graph for one small function -- keep in sync if that function
    changes. row is one entry of kg_node_features.json's "subjects" list."""
    status = row.get("occlusion_status")
    if status == "empty_lvo_mask":
        return "none"
    if status != "ok":
        return None
    if row["occlusion_unlocalized"]:
        return "unlocalized"
    vessel = (row.get("nearest_vessel") or "").split("-")[-1]
    return "proximal_anterior" if vessel in ("ICA", "MCA") else "other_confident"


def load_auxiliary_labels(
    features_json: pathlib.Path = OUT_TAB / "kg_node_features.json",
    perfusion_json: pathlib.Path = OUT_TAB / "perfusion_volumes.json",
) -> dict[str, dict]:
    """{subject: {"lvo_class": int, "lvo_class_name": str,
                  "collateral_bin": int | None}}
    for every subject list_cases() would also accept -- callers should still
    handle a subject missing from this dict (label sources have their own
    skip lists, e.g. the corrupted CBF file, the CoW/LVO shape mismatch)."""
    feats = json.loads(pathlib.Path(features_json).read_text(encoding="utf-8"))
    perf = json.loads(pathlib.Path(perfusion_json).read_text(encoding="utf-8"))
    hir_bin = {r["subject"]: r.get("hir_bin") for r in perf["rows"] if r["status"] == "ok"}

    class_to_idx = {name: i for i, name in enumerate(LVO_CLASSES)}
    out = {}
    for row in feats["subjects"]:
        cls = _localization_class(row)
        if cls is None:
            continue
        out[row["subject"]] = {
            "lvo_class": class_to_idx[cls],
            "lvo_class_name": cls,
            "collateral_bin": hir_bin.get(row["subject"]),
        }
    return out


def class_weights(labels: dict[str, dict]) -> list[float]:
    """Inverse-frequency weights for the LVO cross-entropy loss -- the
    classes are badly imbalanced (4 'none' vs 87 'proximal_anterior')."""
    counts = [0] * N_LVO_CLASSES
    for v in labels.values():
        counts[v["lvo_class"]] += 1
    total = sum(counts)
    return [total / (N_LVO_CLASSES * max(c, 1)) for c in counts]


if __name__ == "__main__":
    labels = load_auxiliary_labels()
    print(f"Auxiliary labels for {len(labels)} subjects")

    from collections import Counter
    lvo_counts = Counter(v["lvo_class_name"] for v in labels.values())
    print("\nLVO localization classes:")
    for name in LVO_CLASSES:
        print(f"  {name:<20} {lvo_counts.get(name, 0)}")

    n_collateral = sum(1 for v in labels.values() if v["collateral_bin"] is not None)
    col_counts = Counter(v["collateral_bin"] for v in labels.values() if v["collateral_bin"] is not None)
    print(f"\nCollateral bin (HIR-derived), {n_collateral}/{len(labels)} have one:")
    for b in range(N_COLLATERAL_BINS):
        print(f"  bin {b}: {col_counts.get(b, 0)}")

    print(f"\nClass weights (inverse frequency, for the LVO loss): "
          f"{[round(w, 3) for w in class_weights(labels)]}")
    sys.exit(0)
