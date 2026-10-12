"""
KG-XAI idea 3 (2026-10-11) -- train SegResNetWithAux (segmentation + an
NIHSS-at-admission regression head) and check whether multi-task learning
works on this trunk at all, now that the auxiliary label is real and
98% populated, unlike Phase 3's circular/sparse LVO-localization and
collateral-proxy labels (which collapsed to one constant answer on every
design tried -- see CLAUDE.md).

Target is z-scored using TRAINING-fold mean/std only (no leakage into
val), stored in the output JSON so predictions can be converted back to
raw NIHSS points for reporting.

Pre-registered BEFORE running (same discipline as every other check in
this project):
  PRIMARY: Spearman rho(predicted NIHSS, true NIHSS) > 0 and p < 0.05 on
    validation subjects with a real NIHSS value.
  ANTI-COLLAPSE GUARD: std(predicted NIHSS) > 0.3 * std(true NIHSS) --
    Phase 3's heads "passed" aggregate accuracy while predicting one
    constant answer for every subject; this check exists specifically so
    the same failure mode can't hide behind a correlation computed on a
    near-constant prediction (a near-constant predictor can still get a
    nonzero correlation by chance on small n, but not a std this low).
  GUARD (does the aux task cost us segmentation accuracy): final mean Dice
    should not drop more than 0.02 vs the matching plain B0 run on the
    same fold/seed.

Usage
-----
    python -m src.model.train_seg_nihss --root D:/ISLES-2024/train --fold 0 --epochs 100 \
        --warm-start outputs/checkpoints/segresnet_3fold_fold0_best.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

OUT_TAB = pathlib.Path("outputs/tables")
OUT_CKPT = pathlib.Path("outputs/checkpoints")
MISSING = -1.0  # sentinel for "no NIHSS value" (see train_multitask_gnn.py's attach())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--audit-json", type=pathlib.Path, default=OUT_TAB / "data_audit.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patch-size", type=int, nargs=3, default=(128, 128, 64))
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-interval", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--cache-dir", type=pathlib.Path, default=None)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nihss-weight", type=float, default=1.0)
    ap.add_argument("--warm-start", type=pathlib.Path, default=None)
    args = ap.parse_args()
    seed_suffix = "" if args.seed == 0 else f"_seed{args.seed}"

    from src.model.labels import load_auxiliary_labels, load_nihss_labels
    from src.model.gnn_head import topcow_label_order
    from src.model.segresnet_aux import SegResNetWithAux
    from src.model.train_multitask_gnn import make_gnn_datasets
    from monai.losses import DiceCELoss
    from monai.data import pad_list_data_collate
    from monai.inferers import sliding_window_inference
    from monai.transforms import CenterSpatialCrop, SpatialPad
    from src.eval.metrics import isles24_metrics
    from scipy.stats import spearmanr

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_CKPT.mkdir(parents=True, exist_ok=True)

    labels = load_auxiliary_labels()
    nihss_map = load_nihss_labels(args.root)
    for s, v in nihss_map.items():
        labels.setdefault(s, {"lvo_class": -1, "collateral_bin": None})["nihss"] = v
    topcow_labels = topcow_label_order()

    train_ds, val_ds, val_ids, train_ids = make_gnn_datasets(
        args.root, args.audit_json, args.fold, args.n_folds,
        tuple(args.patch_size), tuple(args.patch_size), 0, args.cache_dir, labels, topcow_labels,
    )
    print(f"  train: {len(train_ds)} subjects   val: {len(val_ds)} subjects")

    # z-score target from TRAINING fold only -- no leakage into val
    train_nihss = np.array([nihss_map[s] for s in train_ids if s in nihss_map])
    nihss_mean, nihss_std = float(train_nihss.mean()), float(train_nihss.std())
    n_train_labeled = train_nihss.size
    n_val_labeled = sum(1 for s in val_ids if s in nihss_map)
    print(f"  NIHSS: {n_train_labeled}/{len(train_ids)} train subjects labeled "
          f"(mean={nihss_mean:.2f}, std={nihss_std:.2f}); {n_val_labeled}/{len(val_ids)} val subjects labeled")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True,
                               collate_fn=pad_list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             collate_fn=pad_list_data_collate)

    model = SegResNetWithAux(in_channels=6).to(device)
    if args.warm_start:
        model.load_trunk_from_baseline(str(args.warm_start), map_location=device)
        print(f"  warm-started trunk from {args.warm_start}")

    seg_loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    # the nihss head only ever sees patch_size-shaped input during training
    # (every training batch is a patch_size crop) -- feeding it a raw,
    # larger full-brain validation volume directly would be both a
    # train/val distribution mismatch for THIS head specifically and a
    # needless memory spike (a full, non-tiled forward pass, unlike
    # segmentation's tiled sliding_window_inference below). Center-crop/pad
    # to patch_size just for this one head; segmentation scoring is
    # unaffected, still full-resolution sliding-window.
    center_crop = CenterSpatialCrop(roi_size=tuple(args.patch_size))
    pad_to_patch = SpatialPad(spatial_size=tuple(args.patch_size))

    def run_validation():
        model.eval()
        seg_preds, seg_labels, subs, nihss_pred_raw, nihss_true_raw = [], [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device)
                sub = batch["subject"][0] if "subject" in batch else "?"
                with torch.autocast(device_type="cuda", enabled=args.amp):
                    def predictor(x_patch):
                        logits, _ = model(x_patch)
                        return logits
                    seg_logits = sliding_window_inference(image, roi_size=tuple(args.patch_size),
                                                            sw_batch_size=1, predictor=predictor, overlap=0.25)
                    nihss_input = pad_to_patch(center_crop(image[0])).unsqueeze(0)
                    _, nihss_z = model(nihss_input)
                seg_preds.append(torch.argmax(seg_logits, dim=1)[0].cpu().numpy())
                seg_labels.append(batch["label"].numpy()[0, 0])
                subs.append(sub)
                true_n = batch["nihss"].item() if "nihss" in batch else MISSING
                if true_n != MISSING:
                    nihss_pred_raw.append(float(nihss_z.item()) * nihss_std + nihss_mean)
                    nihss_true_raw.append(true_n)
        return subs, seg_preds, seg_labels, nihss_pred_raw, nihss_true_raw

    best_dice, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_seg, running_nihss, n_finite, n_nihss_batches = 0.0, 0.0, 0, 0
        for batch in train_loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            nihss_true = batch["nihss"].to(device).float()
            mask = (nihss_true != MISSING).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
                seg_logits, nihss_pred = model(image)
                seg_loss = seg_loss_fn(seg_logits, label)
                nihss_true_z = torch.where(mask > 0, (nihss_true - nihss_mean) / max(nihss_std, 1e-6),
                                            torch.zeros_like(nihss_true))
                sq_err = (nihss_pred - nihss_true_z) ** 2
                n_labeled_in_batch = mask.sum()
                nihss_loss = (sq_err * mask).sum() / n_labeled_in_batch if n_labeled_in_batch > 0 else torch.tensor(0.0, device=device)
                loss = seg_loss + args.nihss_weight * nihss_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if np.isfinite(float(seg_loss.item())):
                running_seg += float(seg_loss.item())
                n_finite += 1
            if n_labeled_in_batch > 0:
                running_nihss += float(nihss_loss.item())
                n_nihss_batches += 1
        scheduler.step()

        dt = time.time() - t0
        mean_seg = running_seg / max(n_finite, 1)
        mean_nihss = running_nihss / max(n_nihss_batches, 1)
        print(f"epoch {epoch+1}/{args.epochs}  seg={mean_seg:.4f}  nihss_z_mse={mean_nihss:.4f}  ({dt:.1f}s)")
        history.append({"epoch": epoch + 1, "seg": mean_seg, "nihss_z_mse": mean_nihss, "seconds": dt})

        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            subs, preds, labs, npred, ntrue = run_validation()
            dices = [isles24_metrics(p, l)["dice"] for p, l in zip(preds, labs)]
            mean_dice = float(np.mean(dices)) if dices else float("nan")
            if len(npred) >= 5:
                rho, p_val = spearmanr(npred, ntrue)
                mae = float(np.mean(np.abs(np.array(npred) - np.array(ntrue))))
                pred_std, true_std = float(np.std(npred)), float(np.std(ntrue))
            else:
                rho, p_val, mae, pred_std, true_std = float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
            print(f"  val: dice={mean_dice:.4f}   nihss: n={len(npred)} rho={rho:.3f} p={p_val:.3g} "
                  f"mae={mae:.2f} pred_std={pred_std:.2f} true_std={true_std:.2f}")
            history[-1].update({"val_dice": mean_dice, "nihss_n": len(npred), "nihss_rho": rho,
                                 "nihss_p": p_val, "nihss_mae": mae, "nihss_pred_std": pred_std,
                                 "nihss_true_std": true_std})
            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), OUT_CKPT / f"segresnet_nihss_fold{args.fold}{seed_suffix}_best.pt")
                print("  new best, checkpoint saved")

    print("\nfinal evaluation:")
    subs, preds, labs, npred, ntrue = run_validation()
    per_subject = []
    for sub, p, l in zip(subs, preds, labs):
        m = isles24_metrics(p, l)
        m["subject"] = sub
        per_subject.append(m)
    agg = {k: float(np.mean([m[k] for m in per_subject]))
           for k in ("dice", "abs_volume_diff_ml", "lesion_f1", "abs_lesion_count_diff")} if per_subject else {}

    rho, p_val = (spearmanr(npred, ntrue) if len(npred) >= 5 else (float("nan"), float("nan")))
    mae = float(np.mean(np.abs(np.array(npred) - np.array(ntrue)))) if npred else float("nan")
    pred_std = float(np.std(npred)) if npred else float("nan")
    true_std = float(np.std(ntrue)) if ntrue else float("nan")
    primary_passed = bool(np.isfinite(rho) and rho > 0 and p_val < 0.05)
    anticollapse_passed = bool(np.isfinite(pred_std) and np.isfinite(true_std) and pred_std > 0.3 * true_std)

    print(f"  segmentation: {agg}")
    print(f"  nihss: n={len(npred)}  rho={rho:.3f}  p={p_val:.3g}  mae={mae:.2f}  "
          f"pred_std={pred_std:.2f}  true_std={true_std:.2f}")
    print(f"  PRIMARY (rho>0 and p<0.05): {'PASS' if primary_passed else 'FAIL'}")
    print(f"  ANTI-COLLAPSE (pred_std > 0.3*true_std): {'PASS' if anticollapse_passed else 'FAIL'}")

    payload = {
        "fold": args.fold, "n_folds": args.n_folds,
        "config": {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()},
        "nihss_train_mean": nihss_mean, "nihss_train_std": nihss_std,
        "n_train_labeled": n_train_labeled, "n_val_labeled": n_val_labeled,
        "history": history, "best_val_dice": best_dice,
        "final_segmentation": {"per_subject": per_subject, "mean": agg},
        "final_nihss": {"n": len(npred), "rho": rho, "p": p_val, "mae": mae,
                          "pred_std": pred_std, "true_std": true_std,
                          "pred": npred, "true": ntrue, "subjects": [s for s in subs if s]},
        "pass_marks": {
            "primary_rho_positive_p05": {"passed": primary_passed, "rho": rho, "p": p_val},
            "anticollapse_std_ratio": {"passed": anticollapse_passed, "pred_std": pred_std, "true_std": true_std},
        },
    }
    out_path = OUT_TAB / f"seg_nihss_fold{args.fold}{seed_suffix}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
