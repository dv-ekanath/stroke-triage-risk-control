"""
Review 1, item 1.5 -- validate the calibration layer BEFORE touching real data.

Runs four checks.  If any fails the implementation is wrong and no result from
the imaging pipeline can be trusted, so this is the gate that comes first.

  1. PIT uniformity        -- is the conformal predictive distribution calibrated?
  2. Interval coverage     -- does empirical coverage track nominal across alpha?
  3. LTT risk control      -- do realised risks sit at or below their targets,
                              on FRESH draws, at the certified lambda?
  4. Split vs cross        -- how unstable is split conformal at n~24 per fold?
                              (justifies the cross-conformal choice)

Usage
-----
    python3 -m src.experiments.validate_calibration
    python3 -m src.experiments.validate_calibration --trials 500 --tau 70
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from src.conformal.cps import ConformalPredictiveSystem
from src.conformal.decision import evaluate
from src.conformal.ltt import certify
from src.experiments.synthetic import CohortSpec, simulate

OUT_FIG = pathlib.Path("outputs/figures")
OUT_TAB = pathlib.Path("outputs/tables")


def _fit_cps(v, v_hat, sigma, idx_cal, normalise=True):
    cps = ConformalPredictiveSystem(normalise=normalise)
    cps.fit(v[idx_cal], v_hat[idx_cal], sigma[idx_cal] if normalise else None)
    return cps


# ------------------------------------------------------------------ check 1+2

def check_calibration(spec: CohortSpec, trials: int, seed: int):
    """PIT uniformity and interval coverage across alpha."""
    alphas = np.array([0.05, 0.10, 0.20, 0.30])
    pits, cover = [], np.zeros((trials, alphas.size))

    for t in range(trials):
        v, vh, sg = simulate(spec, seed=seed + t)
        n = v.size
        cut = n // 2
        rng = np.random.default_rng(seed + t)
        idx = rng.permutation(n)
        cal, test = idx[:cut], idx[cut:]

        cps = _fit_cps(v, vh, sg, cal)
        pits.append(cps.pit(v[test], vh[test], sg[test]))
        for j, a in enumerate(alphas):
            lo, hi = cps.interval(a, vh[test], sg[test])
            cover[t, j] = np.mean((v[test] >= lo) & (v[test] <= hi))

    pits = np.concatenate(pits)
    rows = [
        {
            "alpha": float(a),
            "nominal": float(1 - a),
            "empirical_mean": float(cover[:, j].mean()),
            "empirical_sd": float(cover[:, j].std()),
            "abs_error_pp": float(abs(cover[:, j].mean() - (1 - a)) * 100),
        }
        for j, a in enumerate(alphas)
    ]
    return pits, rows


# -------------------------------------------------------------------- check 3

def check_risk_control(spec, tau, alpha_missed, alpha_futile, delta, trials, seed):
    """Certify on calibration data, then measure realised risk on fresh draws."""
    realised_m, realised_f, abstain, certified_any = [], [], [], []

    for t in range(trials):
        v, vh, sg = simulate(spec, seed=seed + t)
        n = v.size
        rng = np.random.default_rng(10_000 + seed + t)
        idx = rng.permutation(n)
        cut = n // 2
        cal, test = idx[:cut], idx[cut:]

        # CPS is fit on the calibration half, and LTT certifies on that same
        # half's crossing probabilities.  The test half is never seen.
        cps = _fit_cps(v, vh, sg, cal)
        p_cal = cps.prob_above(tau, vh[cal], sg[cal])
        res = certify(
            p_cal, v[cal], tau,
            alpha_missed=alpha_missed, alpha_futile=alpha_futile,
            delta=delta, method="monotone_fixed_sequence",
        )
        certified_any.append(res.certified_any)
        if not res.certified_any:
            continue

        p_test = cps.prob_above(tau, vh[test], sg[test])
        out = evaluate(p_test, v[test], tau, *res.selected)
        if np.isfinite(out.r_missed):
            realised_m.append(out.r_missed)
        if np.isfinite(out.r_futile):
            realised_f.append(out.r_futile)
        abstain.append(out.abstain_rate)

    def _stat(x, target):
        if not x:
            return {"n": 0}
        a = np.asarray(x)
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "target": float(target),
            "frac_within_target": float(np.mean(a <= target)),
            "p90": float(np.quantile(a, 0.90)),
        }

    return {
        "certified_rate": float(np.mean(certified_any)),
        "R_missed": _stat(realised_m, alpha_missed),
        "R_futile": _stat(realised_f, alpha_futile),
        "abstain_mean": float(np.mean(abstain)) if abstain else float("nan"),
    }


# -------------------------------------------------------------------- check 4

def check_split_vs_cross(spec, trials, seed, alpha=0.10, k_folds=5):
    """Width instability of split conformal at ~n/k per fold vs pooled residuals."""
    w_split, w_cross = [], []
    for t in range(trials):
        v, vh, sg = simulate(spec, seed=seed + t)
        n = v.size
        rng = np.random.default_rng(seed + t)
        folds = np.array_split(rng.permutation(n), k_folds)

        # split conformal: one fold as calibration (~24 cases at n=149)
        cal = folds[0]
        cps = _fit_cps(v, vh, sg, cal)
        lo, hi = cps.interval(alpha, vh, sg)
        w_split.append(float(np.mean(hi - lo)))

        # cross-conformal: pool residuals from all folds
        cps_all = _fit_cps(v, vh, sg, np.arange(n))
        lo, hi = cps_all.interval(alpha, vh, sg)
        w_cross.append(float(np.mean(hi - lo)))

    w_split, w_cross = np.asarray(w_split), np.asarray(w_cross)
    return {
        "alpha": alpha,
        "n_cal_split": int(np.ceil(spec.n / k_folds)),
        "split_width_mean": float(w_split.mean()),
        "split_width_sd": float(w_split.std()),
        "split_width_cv": float(w_split.std() / w_split.mean()),
        "cross_width_mean": float(w_cross.mean()),
        "cross_width_sd": float(w_cross.std()),
        "cross_width_cv": float(w_cross.std() / w_cross.mean()),
    }


# ----------------------------------------------------------------------- plot

def make_figure(pits, cover_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(pits, bins=20, range=(0, 1), density=True,
               color="#4C72B0", edgecolor="white")
    ax[0].axhline(1.0, color="crimson", ls="--", lw=1.5, label="uniform (calibrated)")
    ax[0].set_xlabel("PIT value"); ax[0].set_ylabel("density")
    ax[0].set_title("Conformal predictive distribution\nPIT uniformity")
    ax[0].legend(fontsize=8)

    nom = [r["nominal"] for r in cover_rows]
    emp = [r["empirical_mean"] for r in cover_rows]
    sd = [r["empirical_sd"] for r in cover_rows]
    ax[1].errorbar(nom, emp, yerr=sd, fmt="o-", color="#4C72B0", capsize=4)
    ax[1].plot([0.6, 1.0], [0.6, 1.0], "--", color="crimson", lw=1.5, label="ideal")
    ax[1].set_xlabel("nominal coverage"); ax[1].set_ylabel("empirical coverage")
    ax[1].set_title("Interval coverage\n(mean $\\pm$ SD over trials)")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=149, help="cohort size")
    ap.add_argument("--tau", type=float, default=110.0, help="volume threshold, mL")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--alpha-missed", type=float, default=0.05)
    ap.add_argument("--alpha-futile", type=float, default=0.20)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    spec = CohortSpec(n=args.n)

    print(f"\nSynthetic ISLES'24-like cohort  n={args.n}  tau={args.tau:g} mL"
          f"  trials={args.trials}")
    v, _, _ = simulate(spec, seed=args.seed)
    print(f"  volume  mean={v.mean():.2f} sd={v.std():.2f} "
          f"median={np.median(v):.2f} max={v.max():.1f} mL")
    print(f"  P(V >= tau) = {np.mean(v >= args.tau):.3f}  "
          f"(majority-class baseline accuracy = {max(np.mean(v>=args.tau), 1-np.mean(v>=args.tau)):.3f})")

    print("\n[1/4] PIT uniformity + [2/4] interval coverage")
    pits, cover_rows = check_calibration(spec, args.trials, args.seed)
    print(f"  PIT mean={pits.mean():.4f} (want 0.5)  sd={pits.std():.4f} (want 0.289)")
    for r in cover_rows:
        flag = "OK " if r["abs_error_pp"] < 3.0 else "!! "
        print(f"  {flag}alpha={r['alpha']:.2f}  nominal={r['nominal']:.3f}  "
              f"empirical={r['empirical_mean']:.3f} +/- {r['empirical_sd']:.3f}  "
              f"err={r['abs_error_pp']:.2f} pp")

    print("\n[3/4] LTT risk control on fresh draws")
    risk = check_risk_control(spec, args.tau, args.alpha_missed,
                              args.alpha_futile, args.delta, args.trials, args.seed)
    print(f"  certified in {risk['certified_rate']*100:.1f}% of trials")
    for key in ("R_missed", "R_futile"):
        s = risk[key]
        if s["n"] == 0:
            print(f"  {key}: never observed"); continue
        ok = "OK " if s["frac_within_target"] >= 1 - args.delta else "!! "
        print(f"  {ok}{key}: mean={s['mean']:.4f} target={s['target']:.2f}  "
              f"within target in {s['frac_within_target']*100:.1f}% of trials "
              f"(want >= {(1-args.delta)*100:.0f}%)")
    print(f"  mean abstention at selected lambda: {risk['abstain_mean']:.3f}")

    print("\n[4/4] split conformal vs cross-conformal width stability")
    sc = check_split_vs_cross(spec, args.trials, args.seed)
    print(f"  split  (n_cal={sc['n_cal_split']}): width {sc['split_width_mean']:.1f} "
          f"+/- {sc['split_width_sd']:.1f} mL   CV={sc['split_width_cv']:.3f}")
    print(f"  cross  (n_cal={args.n}): width {sc['cross_width_mean']:.1f} "
          f"+/- {sc['cross_width_sd']:.1f} mL   CV={sc['cross_width_cv']:.3f}")
    print(f"  -> split conformal width is "
          f"{sc['split_width_cv']/sc['cross_width_cv']:.1f}x more variable")

    make_figure(pits, cover_rows, OUT_FIG / "calibration_validation.png")
    payload = {
        "config": vars(args),
        "pit": {"mean": float(pits.mean()), "sd": float(pits.std())},
        "coverage": cover_rows,
        "risk_control": risk,
        "split_vs_cross": sc,
    }
    (OUT_TAB / "calibration_validation.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT_FIG/'calibration_validation.png'}")
    print(f"wrote {OUT_TAB/'calibration_validation.json'}")


if __name__ == "__main__":
    main()
