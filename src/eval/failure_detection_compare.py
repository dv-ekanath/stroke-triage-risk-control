"""
KG-XAI -- the actual comparison: does the knowledge-grounded triage flag
(laterality + near-zero volume) beat, or just duplicate, standard
statistical uncertainty baselines (ensemble disagreement, MC-dropout) at
catching real segmentation failures?

Reads failure_detection_raw.json (src/eval/failure_detection_baselines.py)
and, if present, mc_dropout_raw.json (src/eval/mc_dropout_baseline.py).
Produces, for each signal:
  - Spearman rho(signal, dice) -- does it predict dice at all?
  - AUROC for detecting SEVERE failure (dice < FAILURE_THRESHOLD, fixed
    before looking at results) -- continuous signals ranked directly;
    the binary knowledge flag reduces to the same Mann-Whitney-based AUC
    used throughout this project.
  - Complementarity: among real failures, what does each method's own
    "flagged" set (knowledge flag as-is; continuous signals thresholded
    at their own top quartile, so flagging RATES are comparable) catch
    that the others miss.

Pre-registered BEFORE running (same discipline as every other check in
this project): FAILURE_THRESHOLD = 0.05 (a severe, unambiguous failure,
not just "below median" which would conflate ordinary task difficulty
with an actual failure worth flagging).

Usage
-----
    python -m src.eval.failure_detection_compare
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

OUT_TAB = pathlib.Path("outputs/tables")
FAILURE_THRESHOLD = 0.05


def auc_continuous(signal: np.ndarray, is_failure: np.ndarray) -> dict:
    """AUROC for a continuous signal ranking failures above non-failures."""
    fail, ok = signal[is_failure], signal[~is_failure]
    if fail.size < 3 or ok.size < 3:
        return {"testable": False, "reason": f"too few (fail={fail.size}, ok={ok.size})"}
    u, p = mannwhitneyu(fail, ok, alternative="greater")
    return {"testable": True, "auc": round(float(u / (fail.size * ok.size)), 4), "p": float(p),
            "n_fail": int(fail.size), "n_ok": int(ok.size)}


def auc_binary(flagged: np.ndarray, is_failure: np.ndarray) -> dict:
    """Same AUC definition but for a binary flag: treat 'flagged' as the
    high end of the ranking (ties within each group don't change the
    Mann-Whitney U). Equivalent to the project's `compare()` pattern."""
    return auc_continuous(flagged.astype(float), is_failure)


def main():
    raw = json.loads((OUT_TAB / "failure_detection_raw.json").read_text())
    dice = np.array([r["dice"] for r in raw])
    is_failure = dice < FAILURE_THRESHOLD
    print(f"ensemble-disagreement dataset: n={len(raw)} (subject,seed) pairs, "
          f"{is_failure.sum()} severe failures (dice<{FAILURE_THRESHOLD})")

    results = {"failure_threshold": FAILURE_THRESHOLD, "n": len(raw), "n_failures": int(is_failure.sum())}

    # --- knowledge flag ---
    kflag = np.array([r["knowledge_flagged"] for r in raw])
    kf_dice_cmp = auc_binary(kflag, is_failure)
    rho_k, p_k = spearmanr(kflag.astype(float), dice)
    print(f"\nknowledge_flagged:  AUC={kf_dice_cmp.get('auc')}  p={kf_dice_cmp.get('p'):.3g}  "
          f"(spearman vs dice: rho={rho_k:.3f} p={p_k:.3g})  n_flagged={int(kflag.sum())}")
    results["knowledge_flag"] = {"auc": kf_dice_cmp, "spearman_rho": round(float(rho_k), 4),
                                  "spearman_p": float(p_k), "n_flagged": int(kflag.sum())}

    # --- ensemble disagreement (continuous) ---
    for sig_name in ("ensemble_disagreement", "volume_cv"):
        sig = np.array([r[sig_name] for r in raw], dtype=float)
        valid = ~np.isnan(sig)
        rho, p = spearmanr(sig[valid], dice[valid])
        auc = auc_continuous(sig[valid], is_failure[valid])
        print(f"{sig_name}:  spearman rho={rho:.4f}  p={p:.3g}  ->  "
              f"{'sanity-check PASS (predicts dice)' if (rho<0 and p<0.05) else 'sanity-check FAIL'}")
        print(f"  AUC (severe-failure detection) = {auc.get('auc')}  p={auc.get('p', float('nan')):.3g}"
              if auc.get("testable") else f"  AUC not testable: {auc.get('reason')}")
        results[sig_name] = {"spearman_rho": round(float(rho), 4), "spearman_p": float(p),
                              "predicts_dice_pass": bool(rho < 0 and p < 0.05), "auc": auc}

    # --- complementarity: top-quartile-disagreement vs knowledge flag ---
    disagreement = np.array([r["ensemble_disagreement"] for r in raw], dtype=float)
    q75 = np.percentile(disagreement, 75)
    disagree_flagged = disagreement >= q75
    print(f"\ncomplementarity (ensemble_disagreement top quartile, cutoff={q75:.4f}):")
    print(f"  knowledge flag rate: {kflag.mean():.1%}   disagreement-top-quartile rate: {disagree_flagged.mean():.1%}")
    both = kflag & disagree_flagged
    k_only = kflag & ~disagree_flagged
    d_only = disagree_flagged & ~kflag
    neither = ~kflag & ~disagree_flagged
    print(f"  both flag:        {both.sum():4d}  (of these, failures: {int((both & is_failure).sum())})")
    print(f"  knowledge only:   {k_only.sum():4d}  (of these, failures: {int((k_only & is_failure).sum())})")
    print(f"  disagreement only:{d_only.sum():4d}  (of these, failures: {int((d_only & is_failure).sum())})")
    print(f"  neither:          {neither.sum():4d}  (of these, failures: {int((neither & is_failure).sum())})")
    # of actual failures, what fraction does each method catch
    n_fail = int(is_failure.sum())
    caught_by_knowledge = int((kflag & is_failure).sum())
    caught_by_disagreement = int((disagree_flagged & is_failure).sum())
    caught_by_either = int(((kflag | disagree_flagged) & is_failure).sum())
    missed_by_both = n_fail - caught_by_either
    print(f"\n  of {n_fail} real failures: knowledge catches {caught_by_knowledge} ({caught_by_knowledge/max(n_fail,1):.1%}), "
          f"disagreement catches {caught_by_disagreement} ({caught_by_disagreement/max(n_fail,1):.1%}), "
          f"either catches {caught_by_either} ({caught_by_either/max(n_fail,1):.1%}), "
          f"neither catches {missed_by_both} ({missed_by_both/max(n_fail,1):.1%})")
    results["complementarity"] = {
        "disagreement_q75_cutoff": round(float(q75), 4),
        "knowledge_flag_rate": round(float(kflag.mean()), 4),
        "disagreement_flag_rate": round(float(disagree_flagged.mean()), 4),
        "both_n": int(both.sum()), "knowledge_only_n": int(k_only.sum()),
        "disagreement_only_n": int(d_only.sum()), "neither_n": int(neither.sum()),
        "n_failures": n_fail, "caught_by_knowledge": caught_by_knowledge,
        "caught_by_disagreement": caught_by_disagreement, "caught_by_either": caught_by_either,
        "missed_by_both": missed_by_both,
    }

    # --- MC-dropout, if available ---
    mc_path = OUT_TAB / "mc_dropout_raw.json"
    if mc_path.exists():
        mc = json.loads(mc_path.read_text())
        mc_dice = np.array([r["dice"] for r in mc])
        mc_fail = mc_dice < FAILURE_THRESHOLD
        print(f"\nMC-dropout dataset (fold0 seed0 only, separate scope): n={len(mc)}  failures={mc_fail.sum()}")
        mc_results = {"n": len(mc), "n_failures": int(mc_fail.sum())}
        mc_kflag = np.array([r["knowledge_flagged"] for r in mc])
        auc_k = auc_binary(mc_kflag, mc_fail)
        print(f"  knowledge_flagged (this subset): AUC={auc_k.get('auc')}")
        mc_results["knowledge_flag_auc"] = auc_k
        for sig_name in ("mc_dropout_disagreement", "mc_dropout_volume_cv"):
            sig = np.array([r[sig_name] for r in mc], dtype=float)
            valid = ~np.isnan(sig)
            rho, p = spearmanr(sig[valid], mc_dice[valid])
            auc = auc_continuous(sig[valid], mc_fail[valid])
            print(f"  {sig_name}: rho={rho:.4f} p={p:.3g}  AUC={auc.get('auc')}")
            mc_results[sig_name] = {"spearman_rho": round(float(rho), 4), "spearman_p": float(p), "auc": auc}
        results["mc_dropout"] = mc_results
    else:
        print("\nMC-dropout data not yet available -- run mc_dropout_baseline.py first")
        results["mc_dropout"] = None

    out = OUT_TAB / "failure_detection_comparison.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
