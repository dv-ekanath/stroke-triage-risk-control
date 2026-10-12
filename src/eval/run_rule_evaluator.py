"""
KG-XAI Phase 7 (fixed for Phase 8, 2026-09-16) -- runs the Rule Evaluator
(src/eval/rule_evaluator.py) against a trained checkpoint's REAL,
full-resolution predictions on the val split, computes JDCR, and
prints/saves rationales for a manually-reviewable sample.

Fix for CLAUDE.md KNOWN ISSUES #2: the original version scored laterality
on the downscaled whole-brain view (wb_size, e.g. 64x64x48) -- a grid only
B2 was ever trained to segment. B0/B1's high "predicted nothing" rate there
likely reflected that mismatch, not the consistency loss. This version
instead runs the SAME sliding-window inference Dice uses (full resolution)
and checks laterality against `native_midline_voxel_x` (computed on that
same native grid -- see src/model/dataset.py:build_transforms_gnn's
`native_midline` option). The wb view is still used for what it's actually
needed for: computing the graph node embeddings (B1/B2 only).

JDCR is reported both ways -- strict (Phase 8's headline number: predicting
nothing counts against the model) and lenient (the original definition, for
continuity) -- so the effect of the fix itself is visible, not hidden.

Usage
-----
    python -m src.eval.run_rule_evaluator --root D:/ISLES-2024/train --model-type b0
    python -m src.eval.run_rule_evaluator --root D:/ISLES-2024/train --model-type b1
    python -m src.eval.run_rule_evaluator --root D:/ISLES-2024/train --model-type b2
    # Phase 8 (per-fold checkpoints): pass --checkpoint explicitly
    python -m src.eval.run_rule_evaluator --root D:/ISLES-2024/train --model-type b2 \
        --checkpoint outputs/checkpoints/graph_seg_consistency_3fold_fold1_seed0_best.pt --fold 1
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

DEFAULT_CHECKPOINTS = {
    "b0": OUT_CKPT / "segresnet_3fold_fold0_best.pt",
    "b1": OUT_CKPT / "graph_seg_fold0_best.pt",
    "b2": OUT_CKPT / "graph_seg_consistency_fold0_best.pt",
}


def build_model(model_type: str, checkpoint: pathlib.Path, device, edge_index=None):
    if model_type == "b0":
        # Phase 8's real B0: the plain SegResNet train_baseline.py produces
        # (segresnet_3fold_fold{F}[_seed{S}]_best.pt) -- NOT
        # multitask_fold0_best.pt's MultiTaskSegResNet (Phase 3's shelved
        # LVO/collateral heads, a different architecture/state_dict
        # entirely, and superseded, contaminated-split data besides).
        from monai.networks.nets import SegResNet
        model = SegResNet(spatial_dims=3, in_channels=6, out_channels=2,
                           init_filters=16, dropout_prob=0.2).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))

        def predict_native(image, node_emb, node_present, patch_size):
            from monai.inferers import sliding_window_inference
            return sliding_window_inference(image, roi_size=patch_size, sw_batch_size=1,
                                             predictor=model, overlap=0.25)

        compute_node_embeddings = None
    else:
        from src.model.segresnet_graph import GraphInjectedSegResNet
        model = GraphInjectedSegResNet(in_channels=6, edge_index=edge_index).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))

        def compute_node_embeddings(image_wb, node_coords, node_present, wb_size):
            return model.compute_node_embeddings(image_wb, node_coords, node_present, wb_size)

        def predict_native(image, node_emb, node_present, patch_size):
            from monai.inferers import sliding_window_inference
            def predictor(x):
                return model.forward_seg(x, node_emb, node_present)
            return sliding_window_inference(image, roi_size=patch_size, sw_batch_size=1,
                                             predictor=predictor, overlap=0.25)

    model.eval()
    return model, compute_node_embeddings, predict_native


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--audit-json", type=pathlib.Path, default=OUT_TAB / "data_audit.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--patch-size", type=int, nargs=3, default=(128, 128, 64))
    ap.add_argument("--wb-size", type=int, nargs=3, default=(64, 64, 48),
                     help="must match the wb-size the b1/b2 checkpoint was trained with "
                          "(only used to compute graph node embeddings, not for scoring)")
    ap.add_argument("--cache-dir", type=pathlib.Path, default=pathlib.Path("D:/ISLES-2024/cache_1mm_wb64"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-type", choices=["b0", "b1", "b2"], required=True)
    ap.add_argument("--checkpoint", type=pathlib.Path, default=None,
                     help="override the default fold-0 checkpoint (needed for Phase 8's per-fold runs)")
    ap.add_argument("--tag", type=str, default=None,
                     help="output filename suffix, e.g. 'foldN_seedS'; defaults to fold{--fold}")
    ap.add_argument("--n-rationale-samples", type=int, default=10)
    ap.add_argument("--edge-mode", choices=["anatomical", "random"], default="anatomical",
                     help="must match the b1/b2 checkpoint's training --edge-mode -- "
                          "edge_index is a persistent=False buffer, not saved in the checkpoint, "
                          "so this has to be told which graph to reconstruct")
    ap.add_argument("--edge-seed", type=int, default=0,
                     help="must match the checkpoint's training --edge-seed when --edge-mode random")
    args = ap.parse_args()
    wb_size = tuple(args.wb_size)
    checkpoint = args.checkpoint or DEFAULT_CHECKPOINTS[args.model_type]
    tag = args.tag or f"fold{args.fold}"
    edge_index = None
    if args.edge_mode == "random":
        from src.model.gnn_head import random_edge_index
        edge_index = random_edge_index(seed=args.edge_seed)
        print(f"  edge-mode: RANDOM (seed={args.edge_seed}) -- scoring against a non-anatomical graph")

    from src.model.labels import load_auxiliary_labels
    from src.model.gnn_head import topcow_label_order
    from src.model.train_multitask_gnn import make_gnn_datasets
    from src.eval.rule_evaluator import evaluate_laterality, compute_jdcr, generate_rationale
    from src.eval.metrics import expected_calibration_error
    from monai.data import list_data_collate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}   model_type={args.model_type}   checkpoint={checkpoint}")

    labels = load_auxiliary_labels()
    topcow_labels = topcow_label_order()
    _, val_ds, val_ids, _ = make_gnn_datasets(
        args.root, args.audit_json, args.fold, args.n_folds,
        tuple(args.patch_size), wb_size, args.seed, args.cache_dir, labels, topcow_labels,
        native_midline=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=list_data_collate)
    print(f"  val: {len(val_ds)} subjects")

    kg_feats = json.loads((OUT_TAB / "kg_node_features.json").read_text(encoding="utf-8"))
    occ_by_subject = {r["subject"]: r for r in kg_feats["subjects"]}
    side_of_code = {0: "left", 1: "right"}

    model, compute_node_embeddings, predict_native = build_model(args.model_type, checkpoint, device, edge_index)

    rows = []
    with torch.no_grad():
        for batch in val_loader:
            sub = batch["subject"][0]
            image = batch["image"].to(device)
            native_midline_x = float(np.asarray(batch["native_midline_voxel_x"])[0])
            side_code = int(batch["expected_side_code"][0].item())
            expected_side = side_of_code.get(side_code)

            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                node_emb, node_present = None, None
                if compute_node_embeddings is not None:
                    image_wb = batch["image_wb"].to(device)
                    node_coords = torch.as_tensor(np.asarray(batch["node_coords"]), dtype=torch.float32).to(device)
                    node_present = torch.as_tensor(np.asarray(batch["node_present"]), dtype=torch.float32).to(device)
                    node_emb = compute_node_embeddings(image_wb, node_coords, node_present, wb_size)
                seg_logits = predict_native(image, node_emb, node_present, tuple(args.patch_size))
            pred_mask = torch.argmax(seg_logits, dim=1)[0].cpu().numpy().astype(np.uint8)
            # per-subject, not pooled across the whole val set -- keeps memory
            # bounded (full-resolution volumes are a few million voxels each)
            fg_prob = torch.softmax(seg_logits.float(), dim=1)[0, 1].cpu().numpy()
            true_label = batch["label"].numpy()[0, 0]
            ece = expected_calibration_error(fg_prob, true_label, n_bins=10)

            lat = evaluate_laterality(pred_mask, expected_side, native_midline_x)
            occ_row = occ_by_subject.get(sub, {})
            rationale = generate_rationale(sub, occ_row, expected_side, lat)
            rows.append({"subject": sub, "expected_side": expected_side,
                         "laterality": lat, "rationale": rationale,
                         "ece": ece["ece"],
                         "occlusion_status": occ_row.get("occlusion_status"),
                         "confident": occ_row.get("confident")})

    jdcr_strict = compute_jdcr(rows, strict=True)
    jdcr_lenient = compute_jdcr(rows, strict=False)
    print(f"\nJDCR strict (Phase 8 headline, predicting nothing counts against the model): "
          f"{jdcr_strict['jdcr']}  ({jdcr_strict['n_consistent']}/{jdcr_strict['n_evaluated']}, "
          f"coverage={jdcr_strict['coverage']})")
    print(f"JDCR lenient (original definition, for continuity): "
          f"{jdcr_lenient['jdcr']}  ({jdcr_lenient['n_consistent']}/{jdcr_lenient['n_evaluated']}, "
          f"coverage={jdcr_lenient['coverage']})")

    ece_values = [r["ece"] for r in rows if r["ece"] is not None]
    mean_ece = float(np.mean(ece_values)) if ece_values else None
    std_ece = float(np.std(ece_values)) if ece_values else None
    print(f"ECE (segmentation, voxel-level, mean +/- SD across {len(ece_values)} subjects): "
          f"{mean_ece:.4f} +/- {std_ece:.4f}" if ece_values else "ECE: no subjects")
    print("  NOT the base paper's reported ECE (0.130) -- different quantity, "
          "segmentation-only here; see src/eval/metrics.py:expected_calibration_error's docstring")

    # sample for manual review: mix of confident and unlocalized occlusion cases
    confident = [r for r in rows if r["confident"] and r["occlusion_status"] == "ok"]
    unlocalized = [r for r in rows if r["occlusion_status"] == "ok" and not r["confident"]]
    half = args.n_rationale_samples // 2
    sample = confident[:half] + unlocalized[:args.n_rationale_samples - half]
    print(f"\n--- {len(sample)} sample rationales for manual review "
          f"({len(confident[:half])} confident, {len(sample) - len(confident[:half])} unlocalized) ---")
    for r in sample:
        print(f"  {r['rationale']}")

    payload = {"model_type": args.model_type, "checkpoint": str(checkpoint),
               "fold": args.fold, "patch_size": list(args.patch_size), "wb_size": list(wb_size),
               "jdcr_strict": jdcr_strict, "jdcr_lenient": jdcr_lenient,
               "ece_mean": mean_ece, "ece_std": std_ece, "ece_n_subjects": len(ece_values),
               "ece_note": "voxel-level, segmentation-only -- NOT the base paper's 0.130, see docstring",
               "rows": rows, "rationale_sample": [r["subject"] for r in sample]}
    out_path = OUT_TAB / f"jdcr_{args.model_type}_{tag}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
