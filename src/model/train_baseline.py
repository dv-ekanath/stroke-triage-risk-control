"""
Review 1, item 1.6 -- baseline segmentation, nnU-Net-class trunk (MONAI SegResNet).

Deliberately conventional (v3 section 5.1): the ISLES'24 challenge showed
transformer variants gave no advantage at this sample size, and the
contribution of this project is the calibration layer (src/conformal/), not
the segmentation backbone. This script's only job is to reproduce a
leaderboard-comparable Dice on the real final-infarct target, evaluated with
all four ISLES'24 metrics (src/eval/metrics.py) -- Review 1's bar, not
Review 2's full multi-task model.

6-channel input (NCCT, CTA, CBF, CBV, MTT, Tmax), 1mm isotropic, patch-based
training sized for an 8GB card. See src/model/dataset.py for why raw 4D CTP
is not one of the channels.

Usage
-----
    python -m src.model.train_baseline --root D:/ISLES-2024/train --fold 0 --epochs 5
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


def build_model(in_channels: int, device):
    from monai.networks.nets import SegResNet
    model = SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=2,          # background, lesion
        init_filters=16,         # kept modest for 8GB VRAM with 6 input channels
        dropout_prob=0.2,
    )
    return model.to(device)


def run_validation(model, val_loader, device, patch_size, amp: bool):
    """Sliding-window inference over each full validation volume; returns
    per-subject predicted masks (as numpy, in the resampled 1mm grid) plus
    the ground-truth labels, for the four ISLES'24 metrics."""
    from monai.inferers import sliding_window_inference

    model.eval()
    preds, labels, subjects = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(device)
            label = batch["label"].numpy()[0, 0]  # (1,1,H,W,D) -> (H,W,D)

            with torch.autocast(device_type="cuda", enabled=amp):
                logits = sliding_window_inference(
                    image, roi_size=patch_size, sw_batch_size=1, predictor=model,
                    overlap=0.25,
                )
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy()  # (H,W,D)

            preds.append(pred)
            labels.append(label)
            subjects.append(batch["subject"][0] if "subject" in batch else "?")
    return subjects, preds, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=pathlib.Path,
                     help="ISLES-2024 root, e.g. D:/ISLES-2024/train")
    ap.add_argument("--audit-json", type=pathlib.Path,
                     default=OUT_TAB / "data_audit.json",
                     help="per-subject volumes for stratified folds (src/data/audit.py output)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5,
                     help="v3 spec is 200 for the full protocol; default here is a "
                          "short smoke-test run, raise this for a real training run")
    ap.add_argument("--patch-size", type=int, nargs=3, default=(128, 128, 64),
                     help="measured at ~1.6GB peak VRAM for batch-size=1 on the "
                          "RTX 5060 (8GB) -- see repo notes; raise if you want more")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-interval", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4,
                     help="requires the if __name__=='__main__' guard on Windows "
                          "(present here); drop to 0 if worker processes misbehave")
    ap.add_argument("--cache-dir", type=pathlib.Path,
                     default=None,
                     help="PersistentDataset cache dir (see src/model/dataset.py) -- "
                          "first epoch pays the full resample cost, every epoch after "
                          "reads cached tensors. Pass e.g. D:/ISLES-2024/cache_1mm. "
                          "Clear it if you change build_transforms(). Omit to disable "
                          "caching (recomputes preprocessing every epoch).")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-val-subjects", type=int, default=None,
                     help="cap validation-fold size for a fast smoke test")
    args = ap.parse_args()

    from src.model.dataset import make_datasets
    from monai.losses import DiceCELoss
    from src.eval.metrics import isles24_metrics

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_CKPT.mkdir(parents=True, exist_ok=True)

    print(f"\nbuilding fold {args.fold}/{args.n_folds} "
          f"(patch={tuple(args.patch_size)})")
    train_ds, val_ds, val_ids = make_datasets(
        args.root, args.audit_json, args.fold, args.n_folds,
        tuple(args.patch_size), seed=args.seed, cache_dir=args.cache_dir,
    )
    if args.max_val_subjects:
        val_ids = val_ids[: args.max_val_subjects]
        val_ds.data = val_ds.data[: args.max_val_subjects]
    print(f"  train: {len(train_ds)} subjects   val: {len(val_ds)} subjects")

    from monai.data import list_data_collate
    # RandCropByPosNegLabeld (used by the train transforms) always returns a
    # list of cropped samples per item, even at num_samples=1 -- needs MONAI's
    # collate to flatten that into a normal batch instead of nesting lists.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True,
                               collate_fn=list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             collate_fn=list_data_collate)

    model = build_model(in_channels=6, device=device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    best_dice, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss, n_finite, n_nonfinite = 0.0, 0, 0
        for batch in train_loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
                logits = model(image)
                loss = loss_fn(logits, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)   # GradScaler already skips the weight
            scaler.update()          # update on a non-finite loss/grad -- expected
                                      # under fp16 AMP, not a training bug by itself.

            loss_val = float(loss.item())
            if np.isfinite(loss_val):
                running_loss += loss_val
                n_finite += 1
            else:
                # Still exclude it from the *logged* mean -- summing a NaN into
                # running_loss would silently poison every subsequent batch's
                # average for the rest of the epoch, even though the weight
                # update itself was correctly skipped by the scaler.
                n_nonfinite += 1
        scheduler.step()

        mean_loss = running_loss / max(n_finite, 1)
        dt = time.time() - t0
        skip_note = f"  ({n_nonfinite} non-finite batch(es) skipped in the average)" \
            if n_nonfinite else ""
        print(f"epoch {epoch+1}/{args.epochs}  loss={mean_loss:.4f}  "
              f"({dt:.1f}s, {n_finite+n_nonfinite} batches){skip_note}")
        history.append({"epoch": epoch + 1, "loss": mean_loss, "seconds": dt,
                         "n_nonfinite_batches": n_nonfinite})

        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            subjects, preds, labels = run_validation(
                model, val_loader, device, tuple(args.patch_size), args.amp)
            dices = [isles24_metrics(p, l)["dice"] for p, l in zip(preds, labels)]
            mean_dice = float(np.mean(dices)) if dices else float("nan")
            print(f"  val Dice (fold {args.fold}): {mean_dice:.4f} "
                  f"over {len(dices)} subjects")
            history[-1]["val_dice"] = mean_dice

            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(),
                           OUT_CKPT / f"segresnet_fold{args.fold}_best.pt")
                print(f"  new best, checkpoint saved")

    # final full evaluation with all four ISLES'24 metrics
    print("\nfinal evaluation, all four ISLES'24 metrics:")
    subjects, preds, labels = run_validation(
        model, val_loader, device, tuple(args.patch_size), args.amp)
    per_subject = []
    for sub, p, l in zip(subjects, preds, labels):
        m = isles24_metrics(p, l)
        m["subject"] = sub
        per_subject.append(m)
        print(f"  {sub}: dice={m['dice']:.3f}  "
              f"avd={m['abs_volume_diff_ml']:.2f}mL  "
              f"lesion_f1={m['lesion_f1']:.3f}  ald={m['abs_lesion_count_diff']}")

    agg = {
        k: float(np.mean([m[k] for m in per_subject]))
        for k in ("dice", "abs_volume_diff_ml", "lesion_f1", "abs_lesion_count_diff")
    } if per_subject else {}
    print(f"\nmean over fold {args.fold}: {agg}")
    print("leaderboard reference (Kurtlab): dice=0.285 avd=21.23mL "
          "lesion_f1=0.144 ald=7.18")

    payload = {
        "fold": args.fold, "n_folds": args.n_folds,
        "config": {k: (str(v) if isinstance(v, pathlib.Path) else v)
                   for k, v in vars(args).items()},
        "history": history,
        "best_val_dice": best_dice,
        "final_per_subject": per_subject,
        "final_mean": agg,
    }
    out_path = OUT_TAB / f"baseline_fold{args.fold}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
