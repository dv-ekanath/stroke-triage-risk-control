"""
KG-XAI -- a second standard uncertainty baseline for the knowledge-flag
comparison (see failure_detection_baselines.py): MC-dropout.

Scope, disclosed upfront: B0 fold0 seed0 only (the normal, non-collapsed
baseline), not all 3 seeds x 2 folds like the ensemble-disagreement
baseline -- MC-dropout needs N stochastic passes PER subject PER model,
which multiplies the inference cost by N; running it at the same scope as
the ensemble baseline was not a good use of a single unattended GPU
session. One model, one fold, 50 subjects, N=6 passes each (300 total
inferences) is the scope that fits.

Also disclosed: this SegResNet has exactly ONE Dropout3d layer in the
whole network (checked directly, not assumed) -- not dropout at every
layer. MC-dropout's stochastic diversity here is real but limited; this
is a genuine constraint of the architecture already trained, not a choice
made to weaken this baseline.

No new training. Loads the existing segresnet_3fold_fold0_best.pt,
switches ONLY the Dropout3d submodule to train() mode (keeping BatchNorm
and everything else in eval mode -- the standard MC-dropout trick), runs
N stochastic forward passes per validation subject, and computes the
same two uncertainty signals as the ensemble baseline for comparability:
predictive disagreement between the N passes' own argmax predictions, and
predicted-volume coefficient of variation across the N passes.

Usage
-----
    python -m src.eval.mc_dropout_baseline --root D:/ISLES-2024/train --n-passes 6
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


def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else float(2 * inter / denom)


def enable_mc_dropout(model: torch.nn.Module) -> int:
    """model.eval() everywhere EXCEPT the Dropout3d submodule(s), which go
    to train() so they actually sample a mask on each forward call. Returns
    how many dropout modules were switched (a sanity check, not just an
    assumption)."""
    model.eval()
    n = 0
    for mod in model.modules():
        if isinstance(mod, (torch.nn.Dropout, torch.nn.Dropout3d, torch.nn.AlphaDropout)):
            mod.train()
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--patch-size", type=int, nargs=3, default=(128, 128, 64))
    ap.add_argument("--n-passes", type=int, default=6)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    from src.model.labels import load_auxiliary_labels
    from src.model.gnn_head import topcow_label_order
    from src.model.train_multitask_gnn import make_gnn_datasets
    from src.eval.rule_evaluator import evaluate_laterality
    from monai.networks.nets import SegResNet
    from monai.data import list_data_collate
    from monai.inferers import sliding_window_inference

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    labels = load_auxiliary_labels()
    topcow_labels = topcow_label_order()
    _, val_ds, val_ids, _ = make_gnn_datasets(
        args.root, OUT_TAB / "data_audit.json", args.fold, 3, tuple(args.patch_size), tuple(args.patch_size),
        0, None, labels, topcow_labels, native_midline=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=list_data_collate)
    print(f"  fold{args.fold}: {len(val_ds)} val subjects", flush=True)

    model = SegResNet(spatial_dims=3, in_channels=6, out_channels=2, init_filters=16, dropout_prob=0.2).to(device)
    model.load_state_dict(torch.load(OUT_CKPT / f"segresnet_3fold_fold{args.fold}_best.pt", map_location=device))
    n_dropout = enable_mc_dropout(model)
    print(f"  MC-dropout enabled on {n_dropout} module(s), N={args.n_passes} passes/subject", flush=True)

    side_of_code = {0: "left", 1: "right"}
    rows = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            sub = batch["subject"][0]
            image = batch["image"].to(device)
            label = batch["label"].numpy()[0, 0]
            midline_x = float(np.asarray(batch["native_midline_voxel_x"])[0])
            side_code = int(batch["expected_side_code"][0].item())
            expected_side = side_of_code.get(side_code)

            passes = []
            for _ in range(args.n_passes):
                with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                    logits = sliding_window_inference(image, roi_size=tuple(args.patch_size), sw_batch_size=1,
                                                        predictor=model, overlap=0.25)
                pred = (torch.softmax(logits.float(), dim=1)[0, 1] > 0.5).cpu().numpy().astype(np.uint8)
                passes.append(pred)

            mean_pred = np.mean(passes, axis=0)
            consensus = (mean_pred > 0.5).astype(np.uint8)
            dice = dice_score(consensus, label)
            vols = np.array([p.sum() for p in passes], dtype=float)
            volume_cv = float(vols.std() / vols.mean()) if vols.mean() > 0 else float("nan")

            n_pairs = 0
            pairwise_dice = []
            for a in range(len(passes)):
                for b in range(a + 1, len(passes)):
                    pairwise_dice.append(dice_score(passes[a], passes[b]))
                    n_pairs += 1
            disagreement = 1.0 - float(np.mean(pairwise_dice)) if pairwise_dice else float("nan")

            lat = evaluate_laterality(consensus, expected_side, midline_x)
            near_zero = bool(consensus.sum() < 1000)
            inconsistent = bool(lat.get("evaluable") and lat.get("consistent") is False)
            flagged = bool(near_zero or inconsistent)

            rows.append({"fold": args.fold, "subject": sub, "dice": dice,
                          "pred_voxels": int(consensus.sum()),
                          "knowledge_flagged": flagged,
                          "mc_dropout_disagreement": disagreement,
                          "mc_dropout_volume_cv": volume_cv})
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(val_ds)} subjects done", flush=True)

    out = OUT_TAB / "mc_dropout_raw.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out} ({len(rows)} subjects)")


if __name__ == "__main__":
    main()
