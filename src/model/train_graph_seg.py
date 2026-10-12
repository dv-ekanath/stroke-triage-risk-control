"""
KG-XAI Phase 5 -- B1(seg): train GraphInjectedSegResNet (src/model/segresnet_graph.py)
and compare its Dice against B0's plain trunk (outputs/tables/b0_multitask_fold0.json,
dice=0.247) on the identical fold/split. Only the segmentation loss is used --
the LVO/collateral heads are shelved (see CLAUDE.md).

Reuses Phase 4/5's dataset pipeline unchanged (make_gnn_datasets /
build_transforms_gnn already provide image/label for the segmentation patch
and image_wb/node_coords/node_present for the graph branch -- exactly what
this model needs, lvo_class/collateral_bin just go unused here).

Usage
-----
    python -m src.model.train_graph_seg --root D:/ISLES-2024/train --fold 0 --epochs 100 \
        --warm-start outputs/checkpoints/segresnet_fold0_best.pt
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--audit-json", type=pathlib.Path, default=OUT_TAB / "data_audit.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patch-size", type=int, nargs=3, default=(128, 128, 64))
    ap.add_argument("--wb-size", type=int, nargs=3, default=None,
                     help="whole-brain grid size for the graph branch; defaults to --patch-size")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-interval", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--cache-dir", type=pathlib.Path, default=None)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-val-subjects", type=int, default=None)
    ap.add_argument("--warm-start", type=pathlib.Path, default=None,
                     help="Review 1 / B0 SegResNet checkpoint to initialize the shared trunk from")
    ap.add_argument("--gnn-stage", type=int, default=1)
    ap.add_argument("--edge-mode", choices=["anatomical", "random"], default="anatomical",
                     help="'random' swaps the real CoW anatomy for a same-size random graph "
                          "(the 'is it the real anatomy or just a graph' ablation, CLAUDE.md 2026-10)")
    ap.add_argument("--edge-seed", type=int, default=0,
                     help="seed for the random graph -- held FIXED across --seed 0/1/2 so this arm "
                          "compares training noise, not a different random graph per run")
    args = ap.parse_args()
    wb_size = tuple(args.wb_size) if args.wb_size else tuple(args.patch_size)
    # seed 0 keeps the original un-suffixed name (matches what
    # run_rule_evaluator.py's DEFAULT_CHECKPOINTS already points at); seeds
    # 1+ (Phase 8) get their own name so they don't collide with seed 0 or
    # each other -- defined once, up front, not inside the training loop
    seed_suffix = "" if args.seed == 0 else f"_seed{args.seed}"
    edge_tag = "" if args.edge_mode == "anatomical" else f"_random{args.edge_seed}"

    from src.model.labels import load_auxiliary_labels
    from src.model.gnn_head import topcow_label_order
    from src.model.segresnet_graph import GraphInjectedSegResNet
    from src.model.train_multitask_gnn import make_gnn_datasets
    from monai.losses import DiceCELoss
    from monai.data import pad_list_data_collate
    # pad_list_data_collate not list_data_collate: see train_baseline.py's comment at the same spot -- a real PersistentDataset cache-write bug (not a worker race, not a transform bug) found during Phase 8 fold-2 prep, 2026-09-17.
    from monai.inferers import sliding_window_inference
    from src.eval.metrics import isles24_metrics

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"whole-brain graph-branch grid size: {wb_size}")

    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_CKPT.mkdir(parents=True, exist_ok=True)

    # labels aren't used by this model, but make_gnn_datasets' item dicts
    # carry lvo_class/collateral_bin regardless -- harmless, just ignored
    labels = load_auxiliary_labels()
    topcow_labels = topcow_label_order()

    # fold split pinned to seed=0 always -- see the matching comment in
    # train_baseline.py; args.seed is for training randomness only, not
    # which subjects land in which fold
    train_ds, val_ds, val_ids, train_ids = make_gnn_datasets(
        args.root, args.audit_json, args.fold, args.n_folds,
        tuple(args.patch_size), wb_size, 0, args.cache_dir, labels, topcow_labels,
    )
    if args.max_val_subjects:
        val_ids = val_ids[: args.max_val_subjects]
        val_ds.data = val_ds.data[: args.max_val_subjects]
    print(f"  train: {len(train_ds)} subjects   val: {len(val_ds)} subjects")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True,
                               collate_fn=pad_list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             collate_fn=pad_list_data_collate)

    edge_index = None
    if args.edge_mode == "random":
        from src.model.gnn_head import random_edge_index
        edge_index = random_edge_index(seed=args.edge_seed)
        print(f"  edge-mode: RANDOM (seed={args.edge_seed}) -- {edge_index.shape[1]//2} edges, not the real CoW anatomy")
    model = GraphInjectedSegResNet(in_channels=6, gnn_stage=args.gnn_stage, edge_index=edge_index).to(device)
    if args.warm_start:
        model.load_trunk_from_baseline(str(args.warm_start), map_location=device)
        print(f"  warm-started trunk from {args.warm_start}")

    seg_loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    def run_validation():
        model.eval()
        seg_preds, seg_labels, subs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device)
                image_wb = batch["image_wb"].to(device)
                node_coords = torch.as_tensor(np.asarray(batch["node_coords"]), dtype=torch.float32).to(device)
                node_present = torch.as_tensor(np.asarray(batch["node_present"]), dtype=torch.float32).to(device)

                with torch.autocast(device_type="cuda", enabled=args.amp):
                    # compute the graph embedding ONCE per subject, then reuse
                    # it for every sliding-window patch -- it's a whole-brain
                    # quantity, not something that varies patch to patch
                    node_emb = model.compute_node_embeddings(image_wb, node_coords, node_present, wb_size)

                    def predictor(x_patch):
                        return model.forward_seg(x_patch, node_emb, node_present)

                    seg_logits = sliding_window_inference(image, roi_size=tuple(args.patch_size),
                                                           sw_batch_size=1, predictor=predictor, overlap=0.25)

                seg_preds.append(torch.argmax(seg_logits, dim=1)[0].cpu().numpy())
                seg_labels.append(batch["label"].numpy()[0, 0])
                subs.append(batch["subject"][0] if "subject" in batch else "?")
        return subs, seg_preds, seg_labels

    best_dice, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_seg, n_finite, n_nonfinite = 0.0, 0, 0
        for batch in train_loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            image_wb = batch["image_wb"].to(device)
            node_coords = torch.as_tensor(np.asarray(batch["node_coords"]), dtype=torch.float32).to(device)
            node_present = torch.as_tensor(np.asarray(batch["node_present"]), dtype=torch.float32).to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
                seg_logits = model(image, image_wb, node_coords, node_present, wb_size)
                seg_loss = seg_loss_fn(seg_logits, label)

            scaler.scale(seg_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if np.isfinite(float(seg_loss.item())):
                running_seg += float(seg_loss.item())
                n_finite += 1
            else:
                n_nonfinite += 1
        scheduler.step()

        dt = time.time() - t0
        mean_seg = running_seg / max(n_finite, 1)
        print(f"epoch {epoch+1}/{args.epochs}  seg={mean_seg:.4f}  ({dt:.1f}s, {n_finite+n_nonfinite} batches"
              f"{f', {n_nonfinite} non-finite skipped' if n_nonfinite else ''})")
        history.append({"epoch": epoch + 1, "seg": mean_seg, "seconds": dt, "n_nonfinite": n_nonfinite})

        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            subs, preds, labs = run_validation()
            dices = [isles24_metrics(p, l)["dice"] for p, l in zip(preds, labs)]
            mean_dice = float(np.mean(dices)) if dices else float("nan")
            gates = [round(ca.gate.item(), 4) for ca in model.cross_attn]
            print(f"  val: dice={mean_dice:.4f}   cross-attn gates: {gates}")
            history[-1].update({"val_dice": mean_dice, "gates": gates})
            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(),
                           OUT_CKPT / f"graph_seg{edge_tag}_fold{args.fold}{seed_suffix}_best.pt")
                print("  new best, checkpoint saved")

    print("\nfinal evaluation:")
    subs, preds, labs = run_validation()
    per_subject = []
    for sub, p, l in zip(subs, preds, labs):
        m = isles24_metrics(p, l)
        m["subject"] = sub
        per_subject.append(m)
    agg = {k: float(np.mean([m[k] for m in per_subject]))
           for k in ("dice", "abs_volume_diff_ml", "lesion_f1", "abs_lesion_count_diff")} if per_subject else {}
    print(f"  segmentation: {agg}")
    print("  B0 reference (plain trunk, same fold protocol): dice=0.247 "
          "(outputs/tables/b0_multitask_fold0.json)")

    payload = {
        "fold": args.fold, "n_folds": args.n_folds,
        "config": {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()},
        "history": history, "best_val_dice": best_dice,
        "final_segmentation": {"per_subject": per_subject, "mean": agg},
        "b0_reference_dice": 0.247,
    }
    out_path = OUT_TAB / f"b1_graph_seg{edge_tag}_fold{args.fold}{seed_suffix}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
