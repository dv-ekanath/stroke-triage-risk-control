"""
Faithfulness (deletion) test for the channel-occlusion explanations.

Question: does the explanation point at things the model really depends on?
The importance bars in demo/explain.py come from blanking ONE channel at a time,
so "removing the top channel hurts most" would be true by construction. This
study therefore perturbs the input in ways the explanation was NOT built from,
and compares against controls.

Spatial test (native image only; the whole-brain graph view is left intact)
  heat   : blank all 6 channels inside the K voxels with the highest explanation
           heat (max over channels) -- the region the explanation says matters.
  brain  : the same, in a ball of K voxels centred on a random brain voxel
           (weak control: mostly lands far from the lesion).
  lesion : a ball of K voxels centred on a random voxel of the model's own
           predicted lesion (strict control: same location prior as the heat).

Channel test (independent of the single-channel ranking)
  top2    : blank the two channels with the largest importance together.
  bottom2 : blank the two with the smallest.
  random  : blank 3 random channel pairs (averaged).

Metric per patient: lesion-confidence drop = 1 - sum(p_perturbed[pred]) / sum(p_full[pred])
over the model's own predicted lesion `pred = p_full > 0.5`, plus Dice(pred, pred_perturbed).
Patients with an empty prediction are skipped (nothing to explain).

Result file: outputs/tables/xai_faithfulness.json
"""

from __future__ import annotations

import itertools
import json
import pathlib
import time

import numpy as np
import torch

from demo import explain as ex

SEED = 0
N_RANDOM = 3


def _ball(coords: np.ndarray, center: np.ndarray, k: int) -> np.ndarray:
    """Indices (into coords) of the k brain voxels nearest to `center`."""
    d2 = ((coords - center) ** 2).sum(axis=1)
    return np.argpartition(d2, k - 1)[:k]


def _metrics(p_full, pred, p_pert):
    n = float(p_full[pred].sum())
    kept = float(p_pert[pred].sum())
    pp = p_pert > 0.5
    inter = float((pred & pp).sum())
    dice = 2 * inter / max(float(pred.sum()) + float(pp.sum()), 1.0)
    return 1.0 - kept / max(n, 1e-6), dice


def run_one(demo, sub: str, model_id: str, patch, wb, rng) -> dict | None:
    res = demo.explain(sub, model_id)                       # cached heat + importances
    with demo.lock:
        vol = demo.volume(sub)
        image, image_wb = vol["image"].clone(), vol["image_wb"].clone()
    p_full = ex._fg_prob(demo, model_id, image, image_wb, vol, patch, wb)
    pred = p_full > 0.5
    if not pred.any():
        return None

    brain = (image.abs().sum(0) > 0).numpy()
    coords = np.argwhere(brain)
    heat = res["heat"].max(axis=0).astype(np.float32)
    heat_brain = heat[brain]
    k = int(np.clip((heat_brain > 0.05).sum(), 3000, 60000))
    top = np.argpartition(heat_brain, -k)[-k:]              # indices into coords

    def blank_region(idx):
        im = image.clone()
        c = coords[idx]
        im[:, c[:, 0], c[:, 1], c[:, 2]] = 0
        return ex._fg_prob(demo, model_id, im, image_wb, vol, patch, wb)

    out = {"subject": sub, "model": model_id, "k_voxels": k, "pred_voxels": int(pred.sum())}
    out["spatial_heat_drop"], out["spatial_heat_dice"] = _metrics(p_full, pred, blank_region(top))

    pred_idx = np.argwhere(pred)
    for name, centers in (("brain", coords[rng.integers(0, len(coords), N_RANDOM)]),
                          ("lesion", pred_idx[rng.integers(0, len(pred_idx), N_RANDOM)])):
        drops, dices = [], []
        for c in centers:
            d, di = _metrics(p_full, pred, blank_region(_ball(coords, c, k)))
            drops.append(d); dices.append(di)
        out[f"spatial_{name}_drop"], out[f"spatial_{name}_dice"] = float(np.mean(drops)), float(np.mean(dices))

    def blank_channels(chs):
        im, wbv = image.clone(), image_wb.clone()
        for c in chs:
            im[c] = 0; wbv[c] = 0
        return ex._fg_prob(demo, model_id, im, wbv, vol, patch, wb)

    order = np.argsort(-np.nan_to_num(res["drop"], nan=-1.0))   # most -> least important
    out["channels_top2"] = [ex.CHANNELS[i] for i in order[:2]]
    out["chan_top2_drop"], _ = _metrics(p_full, pred, blank_channels(order[:2]))
    out["chan_bottom2_drop"], _ = _metrics(p_full, pred, blank_channels(order[-2:]))
    pairs = list(itertools.combinations(range(len(ex.CHANNELS)), 2))
    sel = rng.choice(len(pairs), N_RANDOM, replace=False)
    out["chan_random_drop"] = float(np.mean([_metrics(p_full, pred, blank_channels(pairs[i]))[0] for i in sel]))
    return out


def summarize(rows: list[dict]) -> dict:
    from scipy.stats import wilcoxon
    s = {"n": len(rows)}
    if len(rows) < 6:
        return s

    def paired(a, b, label):
        x = np.array([r[a] for r in rows]); y = np.array([r[b] for r in rows])
        diff = x - y
        p = float(wilcoxon(x, y, alternative="greater").pvalue) if np.any(diff != 0) else 1.0
        s[label] = {"median_a": round(float(np.median(x)), 4), "median_b": round(float(np.median(y)), 4),
                    "n_a_greater": int((diff > 0).sum()), "n_b_greater": int((diff < 0).sum()),
                    "wilcoxon_p_a_greater": p}

    paired("spatial_heat_drop", "spatial_brain_drop", "heat_vs_random_brain_ball")
    paired("spatial_heat_drop", "spatial_lesion_drop", "heat_vs_random_lesion_ball")
    paired("chan_top2_drop", "chan_bottom2_drop", "top2_vs_bottom2_channels")
    paired("chan_top2_drop", "chan_random_drop", "top2_vs_random_channel_pairs")
    return s


def run_all(demo, model_ids, patch, wb, out_path: pathlib.Path) -> None:
    rng = np.random.default_rng(SEED)
    result = {"seed": SEED, "n_random": N_RANDOM, "models": {}}
    for mid in model_ids:
        rows, skipped, t0 = [], [], time.time()
        for sub in sorted(demo.index):
            try:
                r = run_one(demo, sub, mid, patch, wb, rng)
            except Exception as e:  # keep going; the failure is recorded, not hidden
                skipped.append({"subject": sub, "reason": f"{type(e).__name__}: {e}"})
                continue
            if r is None:
                skipped.append({"subject": sub, "reason": "empty prediction"})
            else:
                rows.append(r)
            print(f"  [faithfulness] {mid} {sub} {'ok' if r else 'skipped'}", flush=True)
        result["models"][mid] = {"summary": summarize(rows), "skipped": skipped, "rows": rows,
                                 "seconds": round(time.time() - t0)}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")   # written after every model
    print("  [faithfulness] finished ->", out_path, flush=True)
