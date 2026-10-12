"""
KG-XAI Phase 6 -- B2: differentiable consistency loss for the ONE
plausibility constraint that's actually `supported` by real data
(kg/guideline_rules.json: `infarct_side_matches_occlusion_side`). Every
other plausibility constraint tested in Phase 2 was `not_supported` or
untestable -- teaching the model a rule the data contradicts would make
predictions worse, so this is deliberately the only one wired in.

`strength: "soft"` in the graph file is load-bearing: ground truth itself
violates the rule 5.3% of the time (near-midline occlusions, one bilateral
infarct). The loss below is inherently soft -- it penalizes the FRACTION of
predicted lesion probability mass on the wrong side of the midline, not a
hard yes/no -- so it nudges, and can never force, itself never reaching
exactly 0 even on a perfect model, by design.

Geometry: works entirely in the whole-brain (wb) grid's own voxel-index
space (axis 0 = the R/L axis after Orientationd's RAS canonicalisation),
matching `midline_voxel_x` from src/model/dataset.py:ExtractVesselCoordsd
and the whole-brain segmentation prediction from
src/model/segresnet_graph.py:GraphInjectedSegResNet.forward_wb_seg_and_graph.
No world-coordinate/affine round-trip needed -- see that transform's
docstring for why.

Usage
-----
    from src.model.consistency_loss import laterality_consistency_loss
    loss = laterality_consistency_loss(wb_seg_logits, expected_side, midline_voxel_x)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def laterality_consistency_loss(wb_seg_logits: torch.Tensor, expected_side: torch.Tensor,
                                  midline_voxel_x: torch.Tensor) -> torch.Tensor:
    """wb_seg_logits: [B, 2, S0, S1, S2] (foreground/background, whole-brain
    grid). expected_side: [B] long, values in {0 (left), 1 (right), -1
    (unknown -- excluded)}. midline_voxel_x: [B] float, NaN where no L/R
    vessel pair was present to compute one (also excluded).

    Returns the mean, over subjects with both a known side AND a computable
    midline, of the fraction of predicted foreground probability mass
    sitting on the WRONG side of the midline. 0 for a batch with no such
    subjects (matches src/model/train_multitask.py:masked_ce's convention:
    return a real zero-valued tensor tied to the graph, not a bare Python
    0.0, so it composes safely into a larger autograd loss)."""
    valid = (expected_side >= 0) & torch.isfinite(midline_voxel_x)
    probs = F.softmax(wb_seg_logits, dim=1)[:, 1]  # [B, S0, S1, S2] foreground prob

    B, S0 = probs.shape[0], probs.shape[1]
    idx = torch.arange(S0, device=probs.device, dtype=probs.dtype).view(1, S0, 1, 1)
    is_right = (idx > midline_voxel_x.view(B, 1, 1, 1)).to(probs.dtype)  # [B, S0, 1, 1], broadcasts

    right_mass = (probs * is_right).sum(dim=(1, 2, 3))
    left_mass = (probs * (1.0 - is_right)).sum(dim=(1, 2, 3))
    total_mass = right_mass + left_mass + 1e-6

    expects_right = (expected_side == 1)
    wrong_mass = torch.where(expects_right, left_mass, right_mass)
    frac_wrong = wrong_mass / total_mass

    if valid.sum() == 0:
        return wb_seg_logits.sum() * 0.0
    return frac_wrong[valid].mean()
