"""
Evaluation metrics.

Two families:

  ISLES'24 task metrics -- all FOUR, so results are directly comparable to the
  published leaderboard (Kurtlab: Dice 28.50 +/- 21.27, AVD 21.23, lesion-wise
  F1 14.39, ALD 7.18).  Absolute lesion count difference is the one usually
  dropped, and it matters here because a third of the cohort has scattered
  multifocal infarcts.

  Decision metrics -- balanced accuracy and MCC, never plain accuracy.  At
  tau = 70 mL a constant "below cutoff" predictor scores ~0.87, so plain
  accuracy is close to what a real model achieves and reporting it alone is
  misleading.  The majority-class baseline is returned alongside every time.
"""

from __future__ import annotations

import numpy as np

__all__ = ["dice", "absolute_volume_difference", "lesion_count_difference",
           "lesion_wise_f1", "isles24_metrics",
           "balanced_accuracy", "matthews_corrcoef", "decision_metrics",
           "expected_calibration_error"]


def _bin(x):
    return np.asarray(x) > 0


def dice(pred, true) -> float:
    p, t = _bin(pred), _bin(true)
    denom = p.sum() + t.sum()
    if denom == 0:
        return 1.0                       # both empty: perfect agreement
    return float(2.0 * (p & t).sum() / denom)


def absolute_volume_difference(pred, true, vox_ml: float = 1e-3) -> float:
    return float(abs(_bin(pred).sum() - _bin(true).sum()) * vox_ml)


def _components(mask):
    from scipy import ndimage
    return ndimage.label(_bin(mask))


def lesion_count_difference(pred, true) -> int:
    return int(abs(_components(pred)[1] - _components(true)[1]))


def lesion_wise_f1(pred, true, min_overlap: int = 1) -> float:
    """Instance-level F1: a true lesion counts as detected if any predicted
    component overlaps it by at least min_overlap voxels."""
    lp, np_ = _components(pred)
    lt, nt = _components(true)
    if np_ == 0 and nt == 0:
        return 1.0
    if np_ == 0 or nt == 0:
        return 0.0

    tp_t = sum(1 for i in range(1, nt + 1)
               if (lp[lt == i] > 0).sum() >= min_overlap)
    tp_p = sum(1 for j in range(1, np_ + 1)
               if (lt[lp == j] > 0).sum() >= min_overlap)
    prec, rec = tp_p / np_, tp_t / nt
    return float(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))


def isles24_metrics(pred, true, vox_ml: float = 1e-3) -> dict:
    """All four challenge metrics at once."""
    return {
        "dice": dice(pred, true),
        "abs_volume_diff_ml": absolute_volume_difference(pred, true, vox_ml),
        "lesion_f1": lesion_wise_f1(pred, true),
        "abs_lesion_count_diff": lesion_count_difference(pred, true),
    }


def _confusion(y_pred, y_true):
    p, t = np.asarray(y_pred).astype(bool), np.asarray(y_true).astype(bool)
    return (int((p & t).sum()), int((p & ~t).sum()),
            int((~p & t).sum()), int((~p & ~t).sum()))


def balanced_accuracy(y_pred, y_true) -> float:
    tp, fp, fn, tn = _confusion(y_pred, y_true)
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return float(np.nanmean([sens, spec]))


def matthews_corrcoef(y_pred, y_true) -> float:
    tp, fp, fn, tn = _confusion(y_pred, y_true)
    num = tp * tn - fp * fn
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return float(num / den) if den > 0 else 0.0


def expected_calibration_error(probs, labels, n_bins: int = 10) -> dict:
    """Standard equal-width-bin ECE (Guo et al. 2017), scoped here to
    per-VOXEL foreground probability vs. true lesion/not-lesion, over the
    cropped head region every model already restricts its patch/whole-brain
    view to (not the raw scan's background/air, which would trivially
    inflate confidence with near-zero true-positive rate).

    This is NOT the same quantity as the base paper's reported ECE (0.130,
    per-decision calibration of its LVO/collateral/segmentation heads
    together) -- ours is segmentation-only, since the classification heads
    are shelved (see CLAUDE.md, Phase 3-4/5). Report side by side with that
    caveat stated, never as a directly comparable number.

    probs: predicted foreground probability, any shape, flattened here.
    labels: same-shape ground truth (0/1, or any array where >0 is
    foreground -- matches this module's `_bin` convention elsewhere).
    """
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = _bin(labels).ravel().astype(np.float64)
    if p.size == 0:
        return {"ece": None, "n": 0, "n_bins": n_bins, "bins": []}

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        n_in = int(in_bin.sum())
        if n_in == 0:
            bins.append({"range": [round(float(lo), 3), round(float(hi), 3)], "n": 0,
                         "confidence": None, "accuracy": None})
            continue
        conf = float(p[in_bin].mean())
        acc = float(y[in_bin].mean())  # fraction actually foreground in this bin
        ece += (n_in / p.size) * abs(acc - conf)
        bins.append({"range": [round(float(lo), 3), round(float(hi), 3)], "n": n_in,
                     "confidence": round(conf, 4), "accuracy": round(acc, 4)})
    return {"ece": round(float(ece), 4), "n": int(p.size), "n_bins": n_bins, "bins": bins}


def decision_metrics(y_pred_above, y_true_above) -> dict:
    """Threshold-crossing metrics, always with the trivial baseline attached."""
    t = np.asarray(y_true_above).astype(bool)
    prev = float(t.mean())
    return {
        "balanced_accuracy": balanced_accuracy(y_pred_above, t),
        "mcc": matthews_corrcoef(y_pred_above, t),
        "accuracy": float((np.asarray(y_pred_above).astype(bool) == t).mean()),
        "majority_baseline_accuracy": max(prev, 1 - prev),
        "prevalence_above_tau": prev,
        "n": int(t.size),
    }
