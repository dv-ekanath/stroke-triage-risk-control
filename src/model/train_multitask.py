"""
KG-XAI Phase 3 -- B0: the shared trunk (Review 1 baseline) extended with two
new heads, no knowledge graph yet (that's Phases 4-6, B1/B2).

Three losses:
    L = lambda_seg * (Dice + CE)              segmentation      (H1, unchanged)
  + lambda_lvo * class-weighted CE            LVO localization  (H2, new)
  + lambda_col * CE                           collateral proxy  (H3, new)

LVO localization is a real cross-entropy classification: 4 classes, badly
imbalanced (4 'none' vs 87 'proximal_anterior' in the full cohort) -- uses
inverse-frequency class weights from src/model/labels.py, computed on the
TRAIN split of each fold, not the whole cohort (computing it globally would
leak validation-fold class balance into training).

Collateral proxy is trained to reproduce the HIR bin faithfully. Recall
(kg/guideline_rules.json) HIR failed every independent test as an actual
collateral signal in this cohort -- the head is a real, well-defined
prediction target regardless, but never call its output "collateral grade"
in anything written up; it's a proxy that didn't validate.

Some subjects have no LVO/collateral label (src/model/labels.py; a handful
overlap with the corrupted-file / shape-mismatch exclusions already in
list_cases()). Those subjects still train on segmentation; their LVO/
collateral loss terms are masked out per-sample rather than the whole batch
being skipped.

Usage
-----
    python -m src.model.train_multitask --root D:/ISLES-2024/train --fold 0 --epochs 5 \
        --warm-start outputs/checkpoints/segresnet_fold0_best.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model.labels import LVO_CLASSES, N_LVO_CLASSES, N_COLLATERAL_BINS

OUT_TAB = pathlib.Path("outputs/tables")
OUT_CKPT = pathlib.Path("outputs/checkpoints")


def make_multitask_datasets(root, audit_json, fold, n_folds, patch_size, seed,
                             cache_dir, labels: dict[str, dict]):
    """Same fold construction as src/model/dataset.py, but each item dict
    also carries lvo_class/collateral_bin (or -1 if unavailable for that
    subject) BEFORE the MONAI transform pipeline runs. Dictionary transforms
    only touch the keys they're told to (all_keys = the 6 modalities +
    label) -- these extra scalar keys pass through completely untouched, the
    same way "subject" already does in dataset.py."""
    from monai.data import Dataset, PersistentDataset
    from src.model.dataset import list_cases, load_volumes, make_folds, build_transforms

    cases = list_cases(pathlib.Path(root))
    volumes = load_volumes(audit_json)
    folds = make_folds(sorted(cases.keys()), volumes, n_folds=n_folds, seed=seed)
    if not (0 <= fold < len(folds)):
        raise ValueError(f"fold {fold} out of range [0, {len(folds)})")
    train_ids, val_ids = folds[fold]

    def attach(ids):
        items = []
        for s in ids:
            if s not in cases:
                continue
            item = dict(cases[s])
            aux = labels.get(s)
            item["lvo_class"] = aux["lvo_class"] if aux else -1
            item["collateral_bin"] = (aux["collateral_bin"]
                                       if aux and aux["collateral_bin"] is not None else -1)
            items.append(item)
        return items

    train_items, val_items = attach(train_ids), attach(val_ids)

    if cache_dir is not None:
        cache_dir = pathlib.Path(cache_dir)
        # separate cache subdir from the Review 1 baseline's -- the item
        # dicts now carry extra keys, and PersistentDataset caches the
        # TRANSFORM OUTPUT keyed by input identity, not by dict shape, so
        # reusing the old cache would silently serve entries built before
        # lvo_class/collateral_bin existed for downstream code that reads
        # them (they'd just be missing, not wrong -- but keeping caches
        # separate avoids the question entirely).
        (cache_dir / "train_mt").mkdir(parents=True, exist_ok=True)
        (cache_dir / "val_mt").mkdir(parents=True, exist_ok=True)
        train_ds = PersistentDataset(train_items, transform=build_transforms(patch_size, train=True),
                                      cache_dir=str(cache_dir / "train_mt"))
        val_ds = PersistentDataset(val_items, transform=build_transforms(patch_size, train=False),
                                    cache_dir=str(cache_dir / "val_mt"))
    else:
        train_ds = Dataset(train_items, transform=build_transforms(patch_size, train=True))
        val_ds = Dataset(val_items, transform=build_transforms(patch_size, train=False))
    return train_ds, val_ds, val_ids, train_ids


def masked_ce(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """Cross-entropy over only the samples with a real label (target >= 0).
    Returns 0 (not NaN) if every sample in the batch is unlabeled -- an
    empty-batch loss must not crash training or contaminate the running
    average, same discipline as the NaN-batch handling in train_baseline.py."""
    valid = targets >= 0
    if valid.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], targets[valid], weight=weight)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--audit-json", type=pathlib.Path, default=OUT_TAB / "data_audit.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=3,
                     help="3-fold protocol for this project's revised (2-week) plan, not 5")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patch-size", type=int, nargs=3, default=(128, 128, 64))
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-interval", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--cache-dir", type=pathlib.Path, default=None)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-val-subjects", type=int, default=None)
    ap.add_argument("--warm-start", type=pathlib.Path, default=None,
                     help="Review 1 SegResNet checkpoint to initialize the shared trunk from")
    ap.add_argument("--lambda-seg", type=float, default=1.0)
    ap.add_argument("--lambda-lvo", type=float, default=0.3)
    ap.add_argument("--lambda-col", type=float, default=0.2)
    args = ap.parse_args()

    from src.model.labels import load_auxiliary_labels, class_weights
    from src.model.multitask import MultiTaskSegResNet
    from monai.losses import DiceCELoss
    from monai.data import list_data_collate
    from src.eval.metrics import isles24_metrics

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_CKPT.mkdir(parents=True, exist_ok=True)

    labels = load_auxiliary_labels()
    print(f"\nauxiliary labels available for {len(labels)} subjects")

    train_ds, val_ds, val_ids, train_ids = make_multitask_datasets(
        args.root, args.audit_json, args.fold, args.n_folds,
        tuple(args.patch_size), args.seed, args.cache_dir, labels,
    )
    if args.max_val_subjects:
        val_ids = val_ids[: args.max_val_subjects]
        val_ds.data = val_ds.data[: args.max_val_subjects]
    print(f"  train: {len(train_ds)} subjects   val: {len(val_ds)} subjects")

    # class weights from the TRAIN split's own label distribution only
    train_labels = {s: labels[s] for s in train_ids if s in labels}
    lvo_weight = torch.tensor(class_weights(train_labels), dtype=torch.float32, device=device)
    print(f"  LVO class weights (train-fold only): {[round(w, 2) for w in lvo_weight.tolist()]}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True,
                               collate_fn=list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             collate_fn=list_data_collate)

    model = MultiTaskSegResNet(in_channels=6, n_lvo_classes=N_LVO_CLASSES,
                                n_collateral_bins=N_COLLATERAL_BINS).to(device)
    if args.warm_start:
        model.load_trunk_from_baseline(str(args.warm_start), map_location=device)
        print(f"  warm-started trunk from {args.warm_start}")

    seg_loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    def run_validation():
        from monai.inferers import sliding_window_inference
        model.eval()
        seg_preds, seg_labels, subs = [], [], []
        lvo_correct, lvo_total, col_correct, col_total = 0, 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device)
                label = batch["label"].numpy()[0, 0]
                lvo_t = batch["lvo_class"]
                col_t = batch["collateral_bin"]

                with torch.autocast(device_type="cuda", enabled=args.amp):
                    def predictor(x):
                        seg, _, _ = model(x)
                        return seg
                    seg_logits = sliding_window_inference(image, roi_size=tuple(args.patch_size),
                                                           sw_batch_size=1, predictor=predictor, overlap=0.25)
                    _, lvo_logits, col_logits = model(
                        F.interpolate(image, size=tuple(args.patch_size), mode="trilinear", align_corners=False))

                seg_preds.append(torch.argmax(seg_logits, dim=1)[0].cpu().numpy())
                seg_labels.append(label)
                subs.append(batch["subject"][0] if "subject" in batch else "?")

                if int(lvo_t[0]) >= 0:
                    lvo_total += 1
                    lvo_correct += int(torch.argmax(lvo_logits, dim=1).item() == int(lvo_t[0]))
                if int(col_t[0]) >= 0:
                    col_total += 1
                    col_correct += int(torch.argmax(col_logits, dim=1).item() == int(col_t[0]))
        return subs, seg_preds, seg_labels, lvo_correct, lvo_total, col_correct, col_total

    best_dice, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = {"seg": 0.0, "lvo": 0.0, "col": 0.0}
        n_finite, n_nonfinite = 0, 0
        for batch in train_loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            lvo_t = batch["lvo_class"].to(device)
            col_t = batch["collateral_bin"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
                seg_logits, lvo_logits, col_logits = model(image)
                seg_loss = seg_loss_fn(seg_logits, label)
                lvo_loss = masked_ce(lvo_logits, lvo_t, weight=lvo_weight)
                col_loss = masked_ce(col_logits, col_t)
                loss = args.lambda_seg * seg_loss + args.lambda_lvo * lvo_loss + args.lambda_col * col_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if np.isfinite(float(loss.item())):
                running["seg"] += float(seg_loss.item())
                running["lvo"] += float(lvo_loss.item())
                running["col"] += float(col_loss.item())
                n_finite += 1
            else:
                n_nonfinite += 1
        scheduler.step()

        dt = time.time() - t0
        means = {k: v / max(n_finite, 1) for k, v in running.items()}
        print(f"epoch {epoch+1}/{args.epochs}  seg={means['seg']:.4f}  lvo={means['lvo']:.4f}  "
              f"col={means['col']:.4f}  ({dt:.1f}s, {n_finite+n_nonfinite} batches"
              f"{f', {n_nonfinite} non-finite skipped' if n_nonfinite else ''})")
        history.append({"epoch": epoch + 1, **means, "seconds": dt, "n_nonfinite": n_nonfinite})

        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            subs, preds, labs, lvo_c, lvo_n, col_c, col_n = run_validation()
            dices = [isles24_metrics(p, l)["dice"] for p, l in zip(preds, labs)]
            mean_dice = float(np.mean(dices)) if dices else float("nan")
            lvo_acc = lvo_c / lvo_n if lvo_n else float("nan")
            col_acc = col_c / col_n if col_n else float("nan")
            print(f"  val: dice={mean_dice:.4f}  lvo_acc={lvo_acc:.3f} ({lvo_n} labeled)  "
                  f"collateral_acc={col_acc:.3f} ({col_n} labeled)")
            history[-1].update({"val_dice": mean_dice, "val_lvo_acc": lvo_acc,
                                 "val_collateral_acc": col_acc})
            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), OUT_CKPT / f"multitask_fold{args.fold}_best.pt")
                print("  new best, checkpoint saved")

    print("\nfinal evaluation:")
    subs, preds, labs, lvo_c, lvo_n, col_c, col_n = run_validation()
    per_subject = []
    for sub, p, l in zip(subs, preds, labs):
        m = isles24_metrics(p, l)
        m["subject"] = sub
        per_subject.append(m)
    agg = {k: float(np.mean([m[k] for m in per_subject]))
           for k in ("dice", "abs_volume_diff_ml", "lesion_f1", "abs_lesion_count_diff")} if per_subject else {}
    print(f"  segmentation: {agg}")
    print(f"  LVO localization accuracy: {lvo_c/lvo_n if lvo_n else float('nan'):.3f} ({lvo_n} labeled)")
    print(f"  collateral-proxy accuracy: {col_c/col_n if col_n else float('nan'):.3f} ({col_n} labeled)")
    print("  Review 1 baseline reference (segmentation only, this fold's numbers may differ "
          "since this run uses a 3-fold not 5-fold split): dice=0.215 (fold 0 of 5)")

    payload = {
        "fold": args.fold, "n_folds": args.n_folds,
        "config": {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()},
        "history": history, "best_val_dice": best_dice,
        "final_segmentation": {"per_subject": per_subject, "mean": agg},
        "final_lvo_accuracy": lvo_c / lvo_n if lvo_n else None,
        "final_collateral_accuracy": col_c / col_n if col_n else None,
    }
    out_path = OUT_TAB / f"b0_multitask_fold{args.fold}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
