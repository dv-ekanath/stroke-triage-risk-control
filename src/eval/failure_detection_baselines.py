"""
KG-XAI -- does the knowledge-grounded triage flag (laterality + near-zero
volume, src/eval/triage_check.py, p=1.4e-50) actually add something beyond
a standard statistical uncertainty baseline, or is it redundant with what
an ensemble already tells you for free?

No new training. Reuses the 3 already-trained B0 seeds per fold (same
checkpoints as src/eval/ensemble_check.py). For fold0 and fold1 (the two
folds already used there), runs all 3 seeds' full-resolution inference on
every validation subject and computes, per (subject, seed):
  - dice (vs ground truth -- the "outcome" every signal below tries to predict)
  - our knowledge-based triage flag (laterality inconsistent OR predicted
    volume < 1mL -- the exact corrected rule from triage_check.py)
  - a standard ensemble-disagreement signal: 1 - mean pairwise Dice between
    the 3 seeds' own predictions (how much the ensemble disagrees with
    itself, nothing to do with the knowledge graph) -- the most standard,
    most-cited uncertainty baseline in the field (deep ensembles)
  - predicted-volume coefficient of variation across the 3 seeds, a second,
    simpler disagreement proxy

Pre-registered BEFORE running:
  1. Does ensemble disagreement predict dice at all? (Spearman rho<0, p<0.05
     -- a sanity check that the baseline itself is a real signal here,
     not a strawman.)
  2. Head-to-head: AUC of the knowledge-flag vs. AUC of a disagreement-based
     flag (top-quartile-disagreement = flagged) for separating low-dice
     from high-dice cases, on the exact same subjects.
  3. Complementarity: of the cases EITHER method flags, what fraction does
     EACH method catch alone that the other misses? If they're mostly the
     same cases, the knowledge flag is redundant with a free ensemble
     signal; if they diverge, it's catching something different.

Usage
-----
    python -m src.eval.failure_detection_baselines --root D:/ISLES-2024/train
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader

OUT_TAB = pathlib.Path("outputs/tables")
OUT_CKPT = pathlib.Path("outputs/checkpoints")
NEAR_ZERO_VOXELS = 1000  # matches src/eval/triage_check.py


def load_models(fold: int, device):
    from monai.networks.nets import SegResNet
    models = {}
    for seed in (0, 1, 2):
        suf = "" if seed == 0 else f"_seed{seed}"
        ckpt = OUT_CKPT / f"segresnet_3fold_fold{fold}{suf}_best.pt"
        m = SegResNet(spatial_dims=3, in_channels=6, out_channels=2, init_filters=16, dropout_prob=0.2).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        models[seed] = m
    return models


def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else float(2 * inter / denom)


def run_fold(root: pathlib.Path, fold: int, patch_size, device) -> list[dict]:
    from src.model.labels import load_auxiliary_labels
    from src.model.gnn_head import topcow_label_order
    from src.model.train_multitask_gnn import make_gnn_datasets
    from src.eval.rule_evaluator import evaluate_laterality
    from monai.data import list_data_collate
    from monai.inferers import sliding_window_inference

    labels = load_auxiliary_labels()
    topcow_labels = topcow_label_order()
    _, val_ds, val_ids, _ = make_gnn_datasets(
        root, OUT_TAB / "data_audit.json", fold, 3, patch_size, patch_size, 0, None, labels, topcow_labels,
        native_midline=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=list_data_collate)
    print(f"  fold{fold}: {len(val_ds)} val subjects", flush=True)

    models = load_models(fold, device)
    side_of_code = {0: "left", 1: "right"}

    rows = []
    with torch.no_grad():
        for batch in val_loader:
            sub = batch["subject"][0]
            image = batch["image"].to(device)
            label = batch["label"].numpy()[0, 0]
            midline_x = float(np.asarray(batch["native_midline_voxel_x"])[0])
            side_code = int(batch["expected_side_code"][0].item())
            expected_side = side_of_code.get(side_code)

            preds, probs_vol = {}, {}
            for seed, model in models.items():
                with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                    logits = sliding_window_inference(image, roi_size=patch_size, sw_batch_size=1,
                                                        predictor=model, overlap=0.25)
                pred = (torch.softmax(logits.float(), dim=1)[0, 1] > 0.5).cpu().numpy().astype(np.uint8)
                preds[seed] = pred
                probs_vol[seed] = int(pred.sum())

            # ensemble-disagreement signal: 1 - mean pairwise Dice between
            # the 3 seeds' own predictions (nothing to do with ground truth
            # or the knowledge graph -- a standard uncertainty baseline)
            pairs = [(0, 1), (0, 2), (1, 2)]
            pairwise_dice = [dice_score(preds[a], preds[b]) for a, b in pairs]
            disagreement = 1.0 - float(np.mean(pairwise_dice))
            vols = np.array([probs_vol[s] for s in (0, 1, 2)], dtype=float)
            volume_cv = float(vols.std() / vols.mean()) if vols.mean() > 0 else float("nan")

            for seed in (0, 1, 2):
                pred = preds[seed]
                dice = dice_score(pred, label)
                lat = evaluate_laterality(pred, expected_side, midline_x)
                near_zero = bool(pred.sum() < NEAR_ZERO_VOXELS)
                inconsistent = bool(lat.get("evaluable") and lat.get("consistent") is False)
                flagged = bool(near_zero or inconsistent)
                rows.append({
                    "fold": fold, "subject": sub, "seed": seed, "dice": dice,
                    "pred_voxels": probs_vol[seed],
                    "knowledge_flagged": flagged,
                    "ensemble_disagreement": disagreement,
                    "volume_cv": volume_cv,
                })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--patch-size", type=int, nargs=3, default=(128, 128, 64))
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    rows = []
    for fold in args.folds:
        rows += run_fold(args.root, fold, tuple(args.patch_size), device)
        print(f"  fold{fold} done, {len(rows)} total rows so far", flush=True)

    out = OUT_TAB / "failure_detection_raw.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out} ({len(rows)} (subject, seed) rows)")


if __name__ == "__main__":
    main()
