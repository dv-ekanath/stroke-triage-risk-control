"""
Channel-occlusion explanations for the demo.

For one patient and one (non-ensemble) model we blank each of the six input
channels in turn (set to 0, the normalised "no signal" value, in BOTH the
patch view and the whole-brain view the graph branch reads), re-run the same
sliding-window inference the demo uses, and compare the foreground
probability map against the unmodified one.

Two outputs come out of the same six extra passes:

  * a per-channel number: the share of the model's confidence inside its own
    predicted lesion that disappears when that channel is removed
    (``drop``, 0..1), plus the Dice between the full and the occluded
    prediction (``dice_vs_full``);
  * a per-channel spatial map ``heat[c]`` = max(p_full - p_without_c, 0),
    so the page can show WHERE each channel mattered.

This explains the model's behaviour, not the patient's physiology.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch

CHANNELS = ["NCCT", "CTA", "CBF", "CBV", "MTT", "Tmax"]


def _fg_prob(demo, model_id: str, image: torch.Tensor, image_wb: torch.Tensor, vol: dict, patch, wb) -> np.ndarray:
    model, compute_emb, predict_native = demo.models[model_id]
    dev = demo.device
    emb = present = None
    if compute_emb is not None:
        present = vol["node_present"].to(dev)[None]
        emb = compute_emb(image_wb.to(dev)[None], vol["node_coords"].to(dev)[None], present, wb)
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=dev.type == "cuda"):
        logits = predict_native(image.to(dev)[None], emb, present, patch)
    return torch.softmax(logits.float(), dim=1)[0, 1].cpu().numpy()


def compute(demo, sub: str, model_id: str, patch, wb) -> dict:
    """Blocking; ~7 inference passes. Returns arrays + summary (see module docstring)."""
    if demo.model_type[model_id] == "ensemble":
        raise ValueError("explanations are available for single models, not the ensemble")
    with demo.lock:
        vol = demo.volume(sub)
        image, image_wb = vol["image"].clone(), vol["image_wb"].clone()
    p_full = _fg_prob(demo, model_id, image, image_wb, vol, patch, wb)
    pred = p_full > 0.5
    n_pred = int(pred.sum())

    heat = np.zeros((len(CHANNELS),) + p_full.shape, np.float16)
    drop = np.full(len(CHANNELS), np.nan, np.float32)
    dice = np.full(len(CHANNELS), np.nan, np.float32)
    for c in range(len(CHANNELS)):
        im, wbv = image.clone(), image_wb.clone()
        im[c] = 0
        wbv[c] = 0
        p_c = _fg_prob(demo, model_id, im, wbv, vol, patch, wb)
        d = np.clip(p_full - p_c, 0.0, 1.0)
        heat[c] = d.astype(np.float16)
        if n_pred:
            drop[c] = float(d[pred].sum() / max(p_full[pred].sum(), 1e-6))
            pc = p_c > 0.5
            inter = float((pred & pc).sum())
            dice[c] = 2 * inter / max(n_pred + float(pc.sum()), 1.0)
    return {"heat": heat, "drop": drop, "dice_vs_full": dice, "n_pred": n_pred}


def summarize(res: dict) -> dict:
    drop = res["drop"]
    rows = [{"channel": CHANNELS[i],
             "drop": None if np.isnan(drop[i]) else round(float(drop[i]), 4),
             "dice_vs_full": None if np.isnan(res["dice_vs_full"][i]) else round(float(res["dice_vs_full"][i]), 4)}
            for i in range(len(CHANNELS))]
    if res["n_pred"] == 0:
        return {"available": True, "empty_prediction": True, "channels": rows, "top": None,
                "note": "The model predicted no lesion, so there is nothing to attribute."}
    top = max(rows, key=lambda r: r["drop"] or 0.0)
    return {"available": True, "empty_prediction": False, "channels": rows, "top": top["channel"],
            "note": "Each bar is the share of the model's confidence inside its own predicted lesion "
                    "that disappears when that scan channel is blanked."}


def cache_path(cache_dir, sub: str, model_id: str) -> pathlib.Path:
    return pathlib.Path(cache_dir) / "explain" / f"{sub}__{model_id}.npz"


def save(path: pathlib.Path, res: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, heat=res["heat"], drop=res["drop"],
                        dice_vs_full=res["dice_vs_full"], n_pred=np.int64(res["n_pred"]))


def load(path: pathlib.Path) -> dict | None:
    if not path.exists():
        return None
    try:
        z = np.load(path)
        return {"heat": z["heat"], "drop": z["drop"], "dice_vs_full": z["dice_vs_full"], "n_pred": int(z["n_pred"])}
    except Exception:
        return None


def heat_rgb_overlay(base_rgb: np.ndarray, heat2d: np.ndarray, vmax: float = 0.6) -> np.ndarray:
    """Blend a hot (orange -> yellow) heat layer onto an (H, W, 3) float image."""
    h = np.clip(heat2d.astype(np.float32) / vmax, 0, 1)
    colour = np.stack([np.full_like(h, 255.0), 120.0 + 135.0 * h, np.full_like(h, 20.0)], axis=-1)
    a = (h ** 0.7 * 0.85)[..., None]
    return base_rgb * (1 - a) + colour * a
