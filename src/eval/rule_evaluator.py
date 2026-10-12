"""
KG-XAI Phase 7 -- Rule Evaluator: checks a model's ACTUAL predictions
against the plausibility constraint(s) real data supports, reports the
Joint-Decision Consistency Rate (JDCR), and generates a deterministic,
traceable per-patient rationale. No LLM involved -- every sentence is
assembled from a real computed number or a real field already on disk.

Scope, stated honestly: JDCR here is built on exactly the ONE plausibility
constraint Phase 2 found `supported` by real data
(`infarct_side_matches_occlusion_side`, kg/guideline_rules.json) -- the
other 6 were `not_supported` or untestable and are correctly excluded from
both the consistency loss (Phase 6) and JDCR here: a constraint the data
contradicts must not be taught OR checked against
(kg/guideline_rules.json's own discipline). The eligibility rules
(DEFUSE-3/DAWN/etc.) remain unevaluable end-to-end on ISLES'24 (no timing
data, PROJECT_PLAN.md), so they are not part of JDCR either.

Usage
-----
    from src.eval.rule_evaluator import evaluate_laterality, compute_jdcr, generate_rationale
"""

from __future__ import annotations

import numpy as np


def evaluate_laterality(pred_mask: np.ndarray, expected_side: str | None,
                         midline_voxel_x: float) -> dict:
    """pred_mask: [S0, S1, S2] binary array (1 = predicted lesion), in the
    SAME voxel-index space as midline_voxel_x -- for Phase 8 this MUST be
    the model's real, full-resolution (sliding-window) prediction with the
    NATIVE-grid midline (`native_midline_voxel_x`, see
    src/model/dataset.py:ExtractVesselCoordsd's `native_midline` option),
    not the coarse wb-grid view (CLAUDE.md KNOWN ISSUES #2 -- scoring on a
    grid only one model variant was ever trained to segment made that
    variant look artificially better/worse for the wrong reason).

    expected_side: "left"/"right"/None (None = occlusion side genuinely
    unknown for this subject, e.g. a midline BA match -- not the model's
    fault, and distinct from the model predicting nothing).

    `reason` distinguishes the two ways a subject can be not-evaluable,
    because they mean different things for JDCR (see compute_jdcr):
      "unknown_side"     expected side or midline unknown -- excluded from
                          JDCR under both the lenient and strict definition.
      "empty_prediction" the model predicted zero lesion voxels -- under
                          the strict definition (Phase 8's fix) this counts
                          as NOT consistent, not excluded; a real clinical
                          system that predicts nothing has made a decision.
    """
    n = int(pred_mask.sum())
    side_known = expected_side is not None and np.isfinite(midline_voxel_x)
    if not side_known:
        return {"evaluable": False, "reason": "unknown_side", "predicted_side": None,
                "consistent": None, "right_fraction": None, "n_predicted_voxels": n}
    if n == 0:
        return {"evaluable": False, "reason": "empty_prediction", "predicted_side": None,
                "consistent": None, "right_fraction": None, "n_predicted_voxels": 0}
    idx = np.arange(pred_mask.shape[0]).reshape(-1, 1, 1)
    right_voxels = int((pred_mask * (idx > midline_voxel_x)).sum())
    right_fraction = right_voxels / n
    predicted_side = "right" if right_fraction > 0.5 else "left"
    return {"evaluable": True, "reason": None, "predicted_side": predicted_side,
            "consistent": bool(predicted_side == expected_side),
            "right_fraction": round(right_fraction, 4), "n_predicted_voxels": n}


def compute_jdcr(rows: list[dict], strict: bool = True) -> dict:
    """rows: each a dict with a "laterality" sub-dict from evaluate_laterality.
    Only ONE constraint currently feeds JDCR (see module docstring) --
    JDCR reduces to that constraint's pass rate. A framework for more
    constraints, not a promise more exist yet.

    strict=True (Phase 8's fix, CLAUDE.md KNOWN ISSUES #2, default here):
    a subject the model predicted NOTHING for counts as NOT consistent, not
    excluded -- so a model can't inflate JDCR just by abstaining more. Only
    subjects with a genuinely unknown expected side are excluded.
    strict=False (the original, lenient definition -- kept for comparison /
    reproducing the pre-fix numbers, never as the headline figure): both
    "unknown_side" and "empty_prediction" subjects are excluded, and
    coverage (n_evaluable/n_total) must be read alongside it or a model
    that abstains more looks artificially better."""
    known_side = [r for r in rows if r["laterality"]["reason"] != "unknown_side"]
    if strict:
        consistent = [r for r in known_side if r["laterality"]["consistent"]]
        denom = known_side
    else:
        denom = [r for r in known_side if r["laterality"]["evaluable"]]
        consistent = [r for r in denom if r["laterality"]["consistent"]]
    n_empty = sum(1 for r in known_side if r["laterality"]["reason"] == "empty_prediction")
    coverage = round((len(known_side) - n_empty) / len(known_side), 4) if known_side else None
    return {"n_total": len(rows), "n_unknown_side": len(rows) - len(known_side),
            "n_known_side": len(known_side), "n_empty_prediction": n_empty,
            "n_evaluated": len(denom), "n_consistent": len(consistent),
            "coverage": coverage,
            "jdcr": round(len(consistent) / len(denom), 4) if denom else None,
            "definition": "strict (abstaining counts against the model)" if strict
                          else "lenient (abstaining excluded from denominator -- read with coverage)"}


def generate_rationale(subject: str, occlusion_row: dict, expected_side: str | None,
                        laterality: dict) -> str:
    """Deterministic template -- every clause traces to a real field (the
    occlusion row from outputs/tables/kg_node_features.json, or the
    laterality dict computed by evaluate_laterality above). No LLM."""
    status = occlusion_row.get("occlusion_status")
    if status == "empty_lvo_mask":
        occ_desc = "no large-vessel occlusion was identified on admission imaging"
    elif status == "ok":
        vessel = occlusion_row.get("nearest_vessel") or "an unidentified vessel"
        conf = "a confident" if occlusion_row.get("confident") else "a low-confidence"
        occ_desc = f"{conf} match to an occlusion of {vessel} on admission imaging"
    else:
        occ_desc = f"occlusion status could not be determined ({status!r})"

    lines = [f"Subject {subject}: {occ_desc}."]

    if laterality["reason"] == "unknown_side":
        lines.append("Laterality consistency: not evaluable -- the expected occlusion side "
                      "or midline is unknown for this subject.")
    elif laterality["reason"] == "empty_prediction":
        lines.append(
            "Laterality consistency: INCONSISTENT (by the strict definition) -- the model "
            "predicted no lesion voxels at all, so its laterality cannot match the side "
            "admission imaging points to. Flagged for clinician review: a real prediction of "
            "'no lesion' still needs to be checked against the occlusion evidence, not skipped."
        )
    else:
        verdict = "CONSISTENT" if laterality["consistent"] else "INCONSISTENT"
        # right_fraction is always "share on the right" -- report the share
        # ON THE SIDE NAMED (predicted_side), not right_fraction verbatim,
        # or a left-predicted subject's sentence describes the wrong number
        # (caught by manual review, see PROJECT_PLAN.md Phase 7's gate)
        side_fraction = (laterality["right_fraction"] if laterality["predicted_side"] == "right"
                          else 1.0 - laterality["right_fraction"])
        lines.append(
            f"Laterality consistency: {verdict} -- {side_fraction*100:.1f}% of "
            f"{laterality['n_predicted_voxels']} predicted lesion voxels sit on the "
            f"{laterality['predicted_side']} side of the midline; admission imaging places the "
            f"occlusion on the {expected_side} side."
        )
        if not laterality["consistent"]:
            lines.append("Flagged for clinician review: predicted infarct laterality does not "
                          "match the side the occlusion evidence points to.")
    return " ".join(lines)
