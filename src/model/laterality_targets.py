"""
KG-XAI Phase 6 -- per-subject expected infarct side, for the laterality
consistency loss (src/model/consistency_loss.py). This is the ONE
plausibility constraint that's actually `supported`
(kg/guideline_rules.json: `infarct_side_matches_occlusion_side`,
94.7%/6-6/17-17 real-data agreement, `strength: "soft"` -- real data
violates it 5.3% of the time, so the loss must stay a soft nudge, never a
hard requirement).

Expected side comes from admission-only data (kg_node_features.json's
`nearest_vessel` for confident matches, `unlocalized_side` for low-confidence
ones -- both computed in Phase 1 from cow-msk x lvo-msk, ses-01 only). No
lesion-msk involved in computing the TARGET; lesion-msk only ever appears as
what the model is trained to predict, same as every other loss in this
project (see leakage_rule in kg/cow_topology.json).

Usage
-----
    from src.model.laterality_targets import load_laterality_targets
    targets = load_laterality_targets()
    targets["sub-stroke0001"] -> {"expected_side": "right"} | not present (excluded)
"""

from __future__ import annotations

import json
import pathlib

OUT_TAB = pathlib.Path("outputs/tables")


def _side_of_vessel(vessel_id: str | None) -> str | None:
    if not vessel_id:
        return None
    return {"L": "left", "R": "right"}.get(vessel_id.split("-")[0])


def load_laterality_targets(features_json: pathlib.Path = OUT_TAB / "kg_node_features.json") -> dict[str, dict]:
    """{subject: {"expected_side": "left"|"right"}} for every subject with a
    known, lateral occlusion side -- confident matches (any vessel except
    the midline BA) plus low-confidence/unlocalized cases (side from vessel
    NAME only, per Phase 1). Subjects with no occlusion, a midline (BA)
    occlusion, or unreadable data are simply absent -- callers mask them
    out via the same -1-sentinel pattern as src/model/labels.py."""
    feats = json.loads(pathlib.Path(features_json).read_text(encoding="utf-8"))
    out = {}
    for row in feats["subjects"]:
        if row.get("occlusion_status") != "ok":
            continue
        if row.get("confident") and row.get("nearest_vessel"):
            side = _side_of_vessel(row["nearest_vessel"])
        elif row.get("occlusion_unlocalized"):
            side = row.get("unlocalized_side")
            side = side if side != "unknown" else None
        else:
            side = None
        if side is None:
            continue
        out[row["subject"]] = {"expected_side": side}
    return out


if __name__ == "__main__":
    from collections import Counter
    targets = load_laterality_targets()
    print(f"{len(targets)} subjects with a known expected infarct side")
    print(Counter(v["expected_side"] for v in targets.values()))
