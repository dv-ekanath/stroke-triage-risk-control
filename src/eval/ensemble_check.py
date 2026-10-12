"""
KG-XAI -- does averaging the 3 existing B0 seeds fix the training-collapse
problem (fold0 seed1 and fold1 seed1, each dice=0.02 -- the model predicts
background everywhere)?

No new training. Loads the 3 already-trained B0 checkpoints for a fold,
runs full-resolution sliding-window inference with EACH on the same
validation subjects, averages their softmax probabilities per voxel
(simple unweighted mean -- no retraining, no tuning), argmaxes, and scores
Dice against the real follow-up lesion mask.

Pre-registered BEFORE running: a collapsed seed predicts background
everywhere, which pulls the AVERAGED probability toward background too --
so an unweighted ensemble is not guaranteed to be a clean fix; it could
still pay a real cost for including a broken member. The fair question is
not "does the ensemble beat the collapsed seed alone" (trivially yes) but
"does the ensemble lose much compared to just the 2 healthy seeds."

Pass mark: 3-seed ensemble mean Dice is within 0.02 of the 2-healthy-seed
ensemble mean Dice, in both fold0 and fold1 (the two folds with a real
collapse). Fold2 (no collapse in any of its 3 seeds) is run as a control:
here the pass mark is simply that the 3-seed ensemble does not score worse
than the single best seed by more than 0.02 -- ensembling healthy seeds
should not actively hurt.

Usage
-----
    python -m src.eval.ensemble_check --root D:/ISLES-2024/train
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

# fold -> which seed collapsed (dice=0.02), from baseline_3fold_fold*.json
COLLAPSED_SEED = {0: 1, 1: 1, 2: None}


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


def dice_score(pred: np.ndarray, label: np.ndarray) -> float:
    p, l = pred.astype(bool), label.astype(bool)
    inter = np.logical_and(p, l).sum()
    denom = p.sum() + l.sum()
    return float(1.0) if denom == 0 else float(2 * inter / denom)


def run_fold(root: pathlib.Path, fold: int, patch_size, device) -> dict:
    from src.model.labels import load_auxiliary_labels
    from src.model.gnn_head import topcow_label_order
    from src.model.train_multitask_gnn import make_gnn_datasets
    from monai.data import list_data_collate
    from monai.inferers import sliding_window_inference

    labels = load_auxiliary_labels()
    topcow_labels = topcow_label_order()
    _, val_ds, val_ids, _ = make_gnn_datasets(
        root, OUT_TAB / "data_audit.json", fold, 3, patch_size, patch_size, 0, None, labels, topcow_labels,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=list_data_collate)
    print(f"  fold{fold}: {len(val_ds)} val subjects")

    models = load_models(fold, device)
    collapsed = COLLAPSED_SEED[fold]
    healthy_seeds = [s for s in (0, 1, 2) if s != collapsed] if collapsed is not None else (0, 1, 2)

    rows = []
    with torch.no_grad():
        for batch in val_loader:
            sub = batch["subject"][0]
            image = batch["image"].to(device)
            label = batch["label"].numpy()[0, 0]

            probs = {}
            for seed, model in models.items():
                with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                    logits = sliding_window_inference(image, roi_size=patch_size, sw_batch_size=1,
                                                        predictor=model, overlap=0.25)
                probs[seed] = torch.softmax(logits.float(), dim=1)[0, 1].cpu().numpy()

            row = {"subject": sub}
            for seed in (0, 1, 2):
                row[f"dice_seed{seed}"] = dice_score((probs[seed] > 0.5).astype(np.uint8), label)

            ens3 = np.mean([probs[s] for s in (0, 1, 2)], axis=0)
            row["dice_ensemble3"] = dice_score((ens3 > 0.5).astype(np.uint8), label)

            if collapsed is not None:
                ens_healthy = np.mean([probs[s] for s in healthy_seeds], axis=0)
                row["dice_ensemble_healthy_only"] = dice_score((ens_healthy > 0.5).astype(np.uint8), label)
            rows.append(row)

    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "subject"}
    print(f"    mean dice -- seed0={agg['dice_seed0']:.4f}  seed1={agg['dice_seed1']:.4f}  "
          f"seed2={agg['dice_seed2']:.4f}  3-seed ensemble={agg['dice_ensemble3']:.4f}" +
          (f"  healthy-2 ensemble={agg['dice_ensemble_healthy_only']:.4f}" if collapsed is not None else ""))
    return {"fold": fold, "collapsed_seed": collapsed, "per_subject": rows, "mean": agg}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--patch-size", type=int, nargs=3, default=(128, 128, 64))
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    results = {}
    for fold in args.folds:
        results[fold] = run_fold(args.root, fold, tuple(args.patch_size), device)

    print("\n=== pass marks (fixed before running) ===")
    verdicts = {}
    for fold, r in results.items():
        agg = r["mean"]
        if r["collapsed_seed"] is not None:
            gap = agg["dice_ensemble_healthy_only"] - agg["dice_ensemble3"]
            passed = bool(gap <= 0.02)
            print(f"  fold{fold} (collapsed seed{r['collapsed_seed']}): "
                  f"3-seed={agg['dice_ensemble3']:.4f} vs healthy-2={agg['dice_ensemble_healthy_only']:.4f} "
                  f"(gap={gap:.4f}) -> {'PASS' if passed else 'FAIL'} (pass mark: gap <= 0.02)")
            verdicts[f"fold{fold}"] = {"gap_vs_healthy_only": round(gap, 4), "passed": passed}
        else:
            best_single = max(agg["dice_seed0"], agg["dice_seed1"], agg["dice_seed2"])
            gap = best_single - agg["dice_ensemble3"]
            passed = bool(gap <= 0.02)
            print(f"  fold{fold} (control, no collapse): 3-seed={agg['dice_ensemble3']:.4f} vs "
                  f"best single seed={best_single:.4f} (gap={gap:.4f}) -> "
                  f"{'PASS' if passed else 'FAIL'} (pass mark: gap <= 0.02)")
            verdicts[f"fold{fold}"] = {"gap_vs_best_single": round(gap, 4), "passed": passed}

    out = OUT_TAB / "ensemble_check.json"
    out.write_text(json.dumps({"results": results, "verdicts": verdicts}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
