"""
KG-XAI Phase 6 -- B2: B1(seg) (src/model/train_graph_seg.py) plus the
laterality consistency loss (src/model/consistency_loss.py), the ONE
plausibility constraint real data actually supports. Same trunk/warm-start,
same dataset pipeline; adds one more loss term and one more forward call
(forward_wb_seg_and_graph, which reuses the SAME encode(x_wb) pass B1
already needed for the graph embeddings, so this is only marginally more
expensive than B1, not a third full pass).

    L = seg_loss(patch)  +  lambda_consistency * laterality_consistency_loss(wb)

Usage
-----
    python -m src.model.train_graph_seg_consistency --root D:/ISLES-2024/train --fold 0 --epochs 100 \
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
    ap.add_argument("--wb-size", type=int, nargs=3, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-interval", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--cache-dir", type=pathlib.Path, default=None)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-val-subjects", type=int, default=None)
    ap.add_argument("--warm-start", type=pathlib.Path, default=None)
    ap.add_argument("--gnn-stage", type=int, default=1)
    ap.add_argument("--lambda-consistency", type=float, default=0.1,
                     help="soft constraint -- kept small deliberately; real data violates "
                          "this rule 5.3% of the time (kg/guideline_rules.json), so it should "
                          "nudge, not dominate, the segmentation loss")
    args = ap.parse_args()
    wb_size = tuple(args.wb_size) if args.wb_size else tuple(args.patch_size)
    # seed 0 keeps the original un-suffixed name (matches what
    # run_rule_evaluator.py's DEFAULT_CHECKPOINTS already points at); seeds
    # 1+ (Phase 8) get their own name so they don't collide with seed 0 or
    # each other
    seed_suffix = "" if args.seed == 0 else f"_seed{args.seed}"

    from src.model.labels import load_auxiliary_labels
    from src.model.gnn_head import topcow_label_order
    from src.model.segresnet_graph import GraphInjectedSegResNet
    from src.model.train_multitask_gnn import make_gnn_datasets
    from src.model.consistency_loss import laterality_consistency_loss
    from monai.losses import DiceCELoss
    from monai.data import pad_list_data_collate
    # pad_list_data_collate not list_data_collate: see train_baseline.py's comment at the same spot -- a real PersistentDataset cache-write bug (not a worker race, not a transform bug) found during Phase 8 fold-2 prep, 2026-09-17.
    from monai.inferers import sliding_window_inference
    from src.eval.metrics import isles24_metrics

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"whole-brain graph-branch grid size: {wb_size}   lambda_consistency={args.lambda_consistency}")

    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_CKPT.mkdir(parents=True, exist_ok=True)

    labels = load_auxiliary_labels()  # unused by this model, dataset pipeline attaches it regardless
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
    n_side_train = sum(1 for item in train_ds.data if item.get("expected_side_code", -1) >= 0)
    n_side_val = sum(1 for item in val_ds.data if item.get("expected_side_code", -1) >= 0)
    print(f"  laterality-consistency target available: {n_side_train} train / {n_side_val} val subjects")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True,
                               collate_fn=pad_list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             collate_fn=pad_list_data_collate)

    model = GraphInjectedSegResNet(in_channels=6, gnn_stage=args.gnn_stage).to(device)
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
        cons_losses = []
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device)
                image_wb = batch["image_wb"].to(device)
                node_coords = torch.as_tensor(np.asarray(batch["node_coords"]), dtype=torch.float32).to(device)
                node_present = torch.as_tensor(np.asarray(batch["node_present"]), dtype=torch.float32).to(device)
                midline_x = torch.as_tensor(np.asarray(batch["midline_voxel_x"]), dtype=torch.float32).to(device)
                side_code = batch["expected_side_code"].to(device)

                with torch.autocast(device_type="cuda", enabled=args.amp):
                    wb_seg_logits, node_emb = model.forward_wb_seg_and_graph(
                        image_wb, node_coords, node_present, wb_size)
                    cons = laterality_consistency_loss(wb_seg_logits, side_code, midline_x)

                    def predictor(x_patch):
                        return model.forward_seg(x_patch, node_emb, node_present)

                    seg_logits = sliding_window_inference(image, roi_size=tuple(args.patch_size),
                                                           sw_batch_size=1, predictor=predictor, overlap=0.25)

                seg_preds.append(torch.argmax(seg_logits, dim=1)[0].cpu().numpy())
                seg_labels.append(batch["label"].numpy()[0, 0])
                subs.append(batch["subject"][0] if "subject" in batch else "?")
                if bool((side_code >= 0).any().item()):
                    cons_losses.append(float(cons.item()))
        return subs, seg_preds, seg_labels, cons_losses

    best_dice, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_seg, running_cons, n_finite, n_nonfinite = 0.0, 0.0, 0, 0
        for batch in train_loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            image_wb = batch["image_wb"].to(device)
            node_coords = torch.as_tensor(np.asarray(batch["node_coords"]), dtype=torch.float32).to(device)
            node_present = torch.as_tensor(np.asarray(batch["node_present"]), dtype=torch.float32).to(device)
            midline_x = torch.as_tensor(np.asarray(batch["midline_voxel_x"]), dtype=torch.float32).to(device)
            side_code = batch["expected_side_code"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
                wb_seg_logits, node_emb = model.forward_wb_seg_and_graph(
                    image_wb, node_coords, node_present, wb_size)
                seg_logits = model.forward_seg(image, node_emb, node_present)
                seg_loss = seg_loss_fn(seg_logits, label)
                cons_loss = laterality_consistency_loss(wb_seg_logits, side_code, midline_x)
                loss = seg_loss + args.lambda_consistency * cons_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if np.isfinite(float(loss.item())):
                running_seg += float(seg_loss.item())
                running_cons += float(cons_loss.item())
                n_finite += 1
            else:
                n_nonfinite += 1
        scheduler.step()

        dt = time.time() - t0
        mean_seg = running_seg / max(n_finite, 1)
        mean_cons = running_cons / max(n_finite, 1)
        print(f"epoch {epoch+1}/{args.epochs}  seg={mean_seg:.4f}  consistency={mean_cons:.4f}  "
              f"({dt:.1f}s, {n_finite+n_nonfinite} batches"
              f"{f', {n_nonfinite} non-finite skipped' if n_nonfinite else ''})")
        history.append({"epoch": epoch + 1, "seg": mean_seg, "consistency": mean_cons,
                         "seconds": dt, "n_nonfinite": n_nonfinite})

        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            subs, preds, labs, cons_losses = run_validation()
            dices = [isles24_metrics(p, l)["dice"] for p, l in zip(preds, labs)]
            mean_dice = float(np.mean(dices)) if dices else float("nan")
            mean_cons_val = float(np.mean(cons_losses)) if cons_losses else float("nan")
            gates = [round(ca.gate.item(), 4) for ca in model.cross_attn]
            print(f"  val: dice={mean_dice:.4f}  consistency={mean_cons_val:.4f}   gates: {gates}")
            history[-1].update({"val_dice": mean_dice, "val_consistency": mean_cons_val, "gates": gates})
            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(),
                           OUT_CKPT / f"graph_seg_consistency_fold{args.fold}{seed_suffix}_best.pt")
                print("  new best, checkpoint saved")

    print("\nfinal evaluation:")
    subs, preds, labs, cons_losses = run_validation()
    per_subject = []
    for sub, p, l in zip(subs, preds, labs):
        m = isles24_metrics(p, l)
        m["subject"] = sub
        per_subject.append(m)
    agg = {k: float(np.mean([m[k] for m in per_subject]))
           for k in ("dice", "abs_volume_diff_ml", "lesion_f1", "abs_lesion_count_diff")} if per_subject else {}
    print(f"  segmentation: {agg}")
    print(f"  mean laterality-consistency loss (lower = more side-consistent): "
          f"{float(np.mean(cons_losses)) if cons_losses else float('nan'):.4f}")
    print("  B0 reference (plain trunk, same fold protocol): dice=0.247 "
          "(outputs/tables/b0_multitask_fold0.json)")
    print("  B1(seg) reference (graph-injected, no consistency loss): "
          "best_val_dice=0.271, final dice=0.240 (outputs/tables/b1_graph_seg_fold0.json)")

    payload = {
        "fold": args.fold, "n_folds": args.n_folds,
        "config": {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()},
        "history": history, "best_val_dice": best_dice,
        "final_segmentation": {"per_subject": per_subject, "mean": agg},
        "final_consistency_loss": float(np.mean(cons_losses)) if cons_losses else None,
        "b0_reference_dice": 0.247,
        "b1_seg_reference_best_dice": 0.271,
    }
    out_path = OUT_TAB / f"b2_graph_seg_consistency_fold{args.fold}{seed_suffix}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
