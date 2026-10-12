"""
KG-XAI Phase 4/5 -- B1: retrain the two failed Phase 3 heads (LVO
localization, collateral proxy) on the vascular GNN (src/model/gnn_head.py)
instead of a global-average-pooled bottleneck vector. Segmentation is
unchanged from B0 (src/model/train_multitask.py) -- same trunk, same crop
pipeline, same loss.

Two independent forward passes per step (see src/model/multitask_gnn.py):
    forward_seg(image)                                 -> seg_logits
    forward_gnn(image_wb, node_coords, node_present)    -> lvo_logits, col_logits

Only subjects with a readable cow-msk file can enter the GNN branch (see
attach_cow_msk in src/model/dataset.py) -- everyone else still trains
segmentation normally, with the LVO/collateral loss masked to 0 for them
exactly as in B0.

Usage
-----
    python -m src.model.train_multitask_gnn --root D:/ISLES-2024/train --fold 0 --epochs 5 \
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

from src.model.labels import N_LVO_CLASSES, N_COLLATERAL_BINS
from src.model.train_multitask import masked_ce

OUT_TAB = pathlib.Path("outputs/tables")
OUT_CKPT = pathlib.Path("outputs/checkpoints")


def make_gnn_datasets(root, audit_json, fold, n_folds, patch_size, wb_size, seed,
                       cache_dir, labels: dict[str, dict], topcow_labels: list[int],
                       native_midline: bool = False):
    from monai.data import Dataset, PersistentDataset
    from src.model.dataset import list_cases, load_volumes, make_folds, attach_cow_msk, build_transforms_gnn

    cases = list_cases(pathlib.Path(root))
    cases = attach_cow_msk(pathlib.Path(root), cases)
    n_before = len(cases)
    cases = {s: c for s, c in cases.items() if "cow_msk" in c}
    print(f"  [gnn-dataset] {len(cases)}/{n_before} subjects have a usable cow-msk "
          f"(the rest are dropped from this run -- they carry no GNN-branch signal)")

    volumes = load_volumes(audit_json)
    folds = make_folds(sorted(cases.keys()), volumes, n_folds=n_folds, seed=seed)
    if not (0 <= fold < len(folds)):
        raise ValueError(f"fold {fold} out of range [0, {len(folds)})")
    train_ids, val_ids = folds[fold]

    # Phase 6's consistency-loss target -- cheap to attach unconditionally;
    # scripts that don't use it (Phase 4/5's shelved heads) just ignore it,
    # same as "subject" already passes through untouched.
    from src.model.laterality_targets import load_laterality_targets
    side_targets = load_laterality_targets()
    side_code = {"left": 0, "right": 1}

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
            # idea 3 (2026-10-11, src/model/train_seg_nihss.py): a real,
            # non-circular label, attached the same cheap/unconditional way
            # as expected_side_code below -- -1.0 sentinel means "missing",
            # masked out of that script's loss, ignored by every other caller
            nihss = aux.get("nihss") if aux else None
            item["nihss"] = float(nihss) if nihss is not None else -1.0
            side = side_targets.get(s)
            item["expected_side_code"] = side_code[side["expected_side"]] if side else -1
            items.append(item)
        return items

    train_items, val_items = attach(train_ids), attach(val_ids)
    transform_train = build_transforms_gnn(patch_size, wb_size, train=True, topcow_labels=topcow_labels,
                                            native_midline=native_midline)
    transform_val = build_transforms_gnn(patch_size, wb_size, train=False, topcow_labels=topcow_labels,
                                          native_midline=native_midline)

    # native_midline changes what the (deterministic, val-side) transform
    # produces -- a separate cache subdir, or a val-side cache built before
    # this flag existed would silently serve stale items missing
    # native_midline_voxel_x (PersistentDataset keys its cache by input
    # identity, not by the transform's own code/args)
    suffix = "_native" if native_midline else ""
    if cache_dir is not None:
        cache_dir = pathlib.Path(cache_dir)
        (cache_dir / "train_gnn").mkdir(parents=True, exist_ok=True)
        (cache_dir / f"val_gnn{suffix}").mkdir(parents=True, exist_ok=True)
        train_ds = PersistentDataset(train_items, transform=transform_train, cache_dir=str(cache_dir / "train_gnn"))
        val_ds = PersistentDataset(val_items, transform=transform_val, cache_dir=str(cache_dir / f"val_gnn{suffix}"))
    else:
        train_ds = Dataset(train_items, transform=transform_train)
        val_ds = Dataset(val_items, transform=transform_val)
    return train_ds, val_ds, val_ids, train_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--audit-json", type=pathlib.Path, default=OUT_TAB / "data_audit.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patch-size", type=int, nargs=3, default=(128, 128, 64))
    ap.add_argument("--wb-size", type=int, nargs=3, default=None,
                     help="whole-brain grid size for the GNN branch; defaults to --patch-size")
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
                     help="Review 1 / B0 SegResNet checkpoint to initialize the shared trunk from")
    ap.add_argument("--freeze-trunk", action="store_true",
                     help="Diagnostic: lock the (warm-started) trunk entirely -- only the GNN "
                          "head's own parameters train. Tests whether the LVO/collateral heads "
                          "are starved of signal because segmentation's loss dominates the "
                          "shared trunk's gradient, vs. the information genuinely not being "
                          "there. Segmentation is then constant by construction, so it's only "
                          "evaluated once, at the final epoch, not at every val checkpoint.")
    ap.add_argument("--gnn-stage", type=int, default=1,
                     help="which encode() skip-connection stage to sample GNN node features from")
    ap.add_argument("--lambda-seg", type=float, default=1.0)
    ap.add_argument("--lambda-lvo", type=float, default=0.3)
    ap.add_argument("--lambda-col", type=float, default=0.2)
    args = ap.parse_args()
    wb_size = tuple(args.wb_size) if args.wb_size else tuple(args.patch_size)

    from src.model.labels import load_auxiliary_labels, class_weights
    from src.model.multitask_gnn import MultiTaskGNN
    from src.model.gnn_head import topcow_label_order
    from monai.losses import DiceCELoss
    from monai.data import pad_list_data_collate
    # pad_list_data_collate not list_data_collate: see train_baseline.py's comment at the same spot -- a real PersistentDataset cache-write bug (not a worker race, not a transform bug) found during Phase 8 fold-2 prep, 2026-09-17.
    from src.eval.metrics import isles24_metrics

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"whole-brain GNN-branch grid size: {wb_size}")

    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_CKPT.mkdir(parents=True, exist_ok=True)

    labels = load_auxiliary_labels()
    print(f"\nauxiliary labels available for {len(labels)} subjects")
    topcow_labels = topcow_label_order()

    train_ds, val_ds, val_ids, train_ids = make_gnn_datasets(
        args.root, args.audit_json, args.fold, args.n_folds,
        tuple(args.patch_size), wb_size, args.seed, args.cache_dir, labels, topcow_labels,
    )
    if args.max_val_subjects:
        val_ids = val_ids[: args.max_val_subjects]
        val_ds.data = val_ds.data[: args.max_val_subjects]
    print(f"  train: {len(train_ds)} subjects   val: {len(val_ds)} subjects")

    train_labels = {s: labels[s] for s in train_ids if s in labels}
    lvo_weight = torch.tensor(class_weights(train_labels), dtype=torch.float32, device=device)
    print(f"  LVO class weights (train-fold only): {[round(w, 2) for w in lvo_weight.tolist()]}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True,
                               collate_fn=pad_list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             collate_fn=pad_list_data_collate)

    model = MultiTaskGNN(in_channels=6, n_lvo_classes=N_LVO_CLASSES,
                          n_collateral_bins=N_COLLATERAL_BINS, gnn_stage=args.gnn_stage).to(device)
    if args.warm_start:
        model.load_trunk_from_baseline(str(args.warm_start), map_location=device)
        print(f"  warm-started trunk from {args.warm_start}")
    if args.freeze_trunk:
        model.trunk.requires_grad_(False)
        n_frozen = sum(p.numel() for p in model.trunk.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  --freeze-trunk: {n_frozen:,} trunk params locked, "
              f"{n_trainable:,} left trainable (GNN head only)")

    seg_loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    def run_validation(compute_seg: bool = True):
        from monai.inferers import sliding_window_inference
        model.eval()
        seg_preds, seg_labels, subs = [], [], []
        lvo_correct, lvo_total, col_correct, col_total = 0, 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                image_wb = batch["image_wb"].to(device)
                node_coords = torch.as_tensor(np.asarray(batch["node_coords"]), dtype=torch.float32).to(device)
                node_present = torch.as_tensor(np.asarray(batch["node_present"]), dtype=torch.float32).to(device)
                lvo_t = batch["lvo_class"]
                col_t = batch["collateral_bin"]

                with torch.autocast(device_type="cuda", enabled=args.amp):
                    if compute_seg:
                        image = batch["image"].to(device)
                        seg_logits = sliding_window_inference(image, roi_size=tuple(args.patch_size),
                                                               sw_batch_size=1, predictor=model.forward_seg, overlap=0.25)
                        seg_preds.append(torch.argmax(seg_logits, dim=1)[0].cpu().numpy())
                        seg_labels.append(batch["label"].numpy()[0, 0])
                    lvo_logits, col_logits = model.forward_gnn(image_wb, node_coords, node_present, wb_size)

                subs.append(batch["subject"][0] if "subject" in batch else "?")
                if int(lvo_t[0]) >= 0:
                    lvo_total += 1
                    lvo_correct += int(torch.argmax(lvo_logits, dim=1).item() == int(lvo_t[0]))
                if int(col_t[0]) >= 0:
                    col_total += 1
                    col_correct += int(torch.argmax(col_logits, dim=1).item() == int(col_t[0]))
        return subs, seg_preds, seg_labels, lvo_correct, lvo_total, col_correct, col_total

    best_dice, best_aux_score, history = -1.0, -1.0, []
    for epoch in range(args.epochs):
        model.train()
        if args.freeze_trunk:
            model.trunk.eval()  # deterministic features for the frozen extractor -- no dropout noise
        t0 = time.time()
        running = {"seg": 0.0, "lvo": 0.0, "col": 0.0}
        n_finite, n_nonfinite = 0, 0
        for batch in train_loader:
            image_wb = batch["image_wb"].to(device)
            node_coords = torch.as_tensor(np.asarray(batch["node_coords"]), dtype=torch.float32).to(device)
            node_present = torch.as_tensor(np.asarray(batch["node_present"]), dtype=torch.float32).to(device)
            lvo_t = batch["lvo_class"].to(device)
            col_t = batch["collateral_bin"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
                lvo_logits, col_logits = model.forward_gnn(image_wb, node_coords, node_present, wb_size)
                lvo_loss = masked_ce(lvo_logits, lvo_t, weight=lvo_weight)
                col_loss = masked_ce(col_logits, col_t)
                if args.freeze_trunk:
                    # trunk locked -- segmentation can't change, skip its
                    # (expensive) forward/backward entirely rather than
                    # compute a loss with nowhere for the gradient to go
                    seg_loss = torch.zeros((), device=device)
                    loss = args.lambda_lvo * lvo_loss + args.lambda_col * col_loss
                else:
                    image = batch["image"].to(device)
                    label = batch["label"].to(device)
                    seg_logits = model.forward_seg(image)
                    seg_loss = seg_loss_fn(seg_logits, label)
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

        is_last = epoch == args.epochs - 1
        if (epoch + 1) % args.val_interval == 0 or is_last:
            # frozen trunk => segmentation is provably unchanged from the
            # warm-start checkpoint; only pay for sliding-window inference
            # once, at the end, instead of at every checkpoint
            compute_seg = (not args.freeze_trunk) or is_last
            subs, preds, labs, lvo_c, lvo_n, col_c, col_n = run_validation(compute_seg=compute_seg)
            dices = [isles24_metrics(p, l)["dice"] for p, l in zip(preds, labs)]
            mean_dice = float(np.mean(dices)) if dices else float("nan")
            lvo_acc = lvo_c / lvo_n if lvo_n else float("nan")
            col_acc = col_c / col_n if col_n else float("nan")
            print(f"  val: dice={mean_dice:.4f}  lvo_acc={lvo_acc:.3f} ({lvo_n} labeled)  "
                  f"collateral_acc={col_acc:.3f} ({col_n} labeled)")
            history[-1].update({"val_dice": mean_dice, "val_lvo_acc": lvo_acc,
                                 "val_collateral_acc": col_acc})
            if args.freeze_trunk:
                # dice isn't a meaningful checkpoint signal here (trunk/decoder
                # frozen -- it can't move); track the two heads' own accuracy
                aux_score = (lvo_acc if lvo_n else 0.0) + (col_acc if col_n else 0.0)
                if aux_score > best_aux_score:
                    best_aux_score = aux_score
                    torch.save(model.state_dict(), OUT_CKPT / f"multitask_gnn_frozen_fold{args.fold}_best.pt")
                    print("  new best (lvo_acc + collateral_acc), checkpoint saved")
            elif mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), OUT_CKPT / f"multitask_gnn_fold{args.fold}_best.pt")
                print("  new best, checkpoint saved")

    print("\nfinal evaluation:")
    subs, preds, labs, lvo_c, lvo_n, col_c, col_n = run_validation(compute_seg=True)
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
    print("  Phase 3 B0 reference (global-avg-pool heads, same fold protocol): "
          "lvo_acc=0.560, collateral_acc=0.600 (both below trivial baseline -- outputs/tables/b0_multitask_fold0.json)")

    payload = {
        "fold": args.fold, "n_folds": args.n_folds,
        "config": {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()},
        "history": history, "best_val_dice": best_dice,
        "final_segmentation": {"per_subject": per_subject, "mean": agg},
        "final_lvo_accuracy": lvo_c / lvo_n if lvo_n else None,
        "final_collateral_accuracy": col_c / col_n if col_n else None,
    }
    suffix = "_frozen_trunk" if args.freeze_trunk else ""
    out_path = OUT_TAB / f"b1_multitask_gnn{suffix}_fold{args.fold}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
