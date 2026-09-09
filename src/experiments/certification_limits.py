"""
Review 1, item 1.5b / plan item 3.2 -- what can n=149 actually certify?

Two sweeps, both on synthetic cohorts matched to the ISLES'24 volume
distribution, so both are available BEFORE the imaging data is touched.

  A. tau sweep at n=149  -- for each candidate threshold, how many cases sit
     above it, and can either risk be certified?
  B. n sweep at fixed tau -- how many patients would be needed?

The binding constraint is not n.  It is the size of the CONDITIONING GROUP.
R_futile = P(say eligible | V >= tau) can only be certified from cases with
V >= tau, and in a cohort with mean 33 mL almost nobody sits above 110 mL.
This is the proposal's "boundary band may be nearly empty" risk, quantified.

Usage
-----
    python3 -m src.experiments.certification_limits
    python3 -m src.experiments.certification_limits --trials 200
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from src.conformal.cps import ConformalPredictiveSystem
from src.conformal.decision import evaluate
from src.conformal.ltt import certify, min_n_to_certify
from src.experiments.synthetic import CohortSpec, simulate

OUT_FIG = pathlib.Path("outputs/figures")
OUT_TAB = pathlib.Path("outputs/tables")

# Candidate thresholds spanning the clinically discussed range:
#   70  mL  DAWN / DEFUSE-3 legacy exclusion
#   110 mL  therapeutic ceiling, Korean nationwide registry (PMID 42403349)
TAUS = [10, 20, 30, 40, 50, 70, 90, 110, 130, 150]
NS = [50, 100, 149, 250, 500, 1000, 2000]


def _one_trial(spec, tau, alpha_m, alpha_f, delta, seed):
    """Certify on the full cohort; measure realised risk on a large fresh draw."""
    v, vh, sg = simulate(spec, seed=seed)

    cps = ConformalPredictiveSystem(normalise=True).fit(v, vh, sg)
    p_cal = cps.prob_above(tau, vh, sg)
    res = certify(p_cal, v, tau, alpha_missed=alpha_m, alpha_futile=alpha_f,
                  delta=delta, method="monotone_fixed_sequence")

    # fresh, large evaluation cohort -- approximates the true risk
    big = CohortSpec(n=20_000, rel_err=spec.rel_err, abs_err_ml=spec.abs_err_ml,
                     sigma_noise=spec.sigma_noise)
    v2, vh2, sg2 = simulate(big, seed=seed + 777_000)
    out = None
    if res.certified_any:
        p2 = cps.prob_above(tau, vh2, sg2)
        out = evaluate(p2, v2, tau, *res.selected)
    return res, out


def sweep(spec_fn, values, key, tau_fixed, n_fixed, alpha_m, alpha_f, delta, trials):
    rows = []
    for val in values:
        spec = spec_fn(val)
        tau = val if key == "tau" else tau_fixed
        cert, held_m, held_f, abst = [], [], [], []
        n_large_obs, n_small_obs = [], []

        for t in range(trials):
            res, out = _one_trial(spec, tau, alpha_m, alpha_f, delta, seed=t)
            cert.append(res.certified_any)
            n_large_obs.append(res.n_large)
            n_small_obs.append(res.n_small)
            if out is not None:
                if np.isfinite(out.r_missed):
                    held_m.append(out.r_missed <= alpha_m)
                if np.isfinite(out.r_futile):
                    held_f.append(out.r_futile <= alpha_f)
                abst.append(out.abstain_rate)

        rows.append({
            key: float(val),
            "n": int(spec.n),
            "tau": float(tau),
            "mean_n_above_tau": float(np.mean(n_large_obs)),
            "mean_n_below_tau": float(np.mean(n_small_obs)),
            "certified_rate": float(np.mean(cert)),
            "risk_held_missed": float(np.mean(held_m)) if held_m else float("nan"),
            "risk_held_futile": float(np.mean(held_f)) if held_f else float("nan"),
            "mean_abstain": float(np.mean(abst)) if abst else float("nan"),
        })
    return rows


def _print_table(rows, key, title):
    print(f"\n{title}")
    print(f"  {key:>6}  {'n':>5}  {'n>=tau':>7}  {'cert%':>6}  "
          f"{'held_m':>7}  {'held_f':>7}  {'abstain':>8}")
    print("  " + "-" * 60)
    for r in rows:
        hm = "  --  " if np.isnan(r["risk_held_missed"]) else f"{r['risk_held_missed']:.3f} "
        hf = "  --  " if np.isnan(r["risk_held_futile"]) else f"{r['risk_held_futile']:.3f} "
        ab = "   --   " if np.isnan(r["mean_abstain"]) else f"{r['mean_abstain']:.3f}   "
        print(f"  {r[key]:>6.0f}  {r['n']:>5d}  {r['mean_n_above_tau']:>7.1f}  "
              f"{r['certified_rate']*100:>5.1f}%  {hm:>7}  {hf:>7}  {ab:>8}")



def make_figure(tau_rows, n_rows, tau_fixed, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    taus = [r["tau"] for r in tau_rows]
    rate = [r["certified_rate"] for r in tau_rows]
    ax[0].plot(taus, rate, "o-", color="#4C72B0", lw=2)
    ax[0].fill_between(taus, 0, rate, alpha=0.15, color="#4C72B0")
    ax[0].axvline(70, color="crimson", ls="--", lw=1.5, label="$\\tau$=70 (legacy)")
    ax[0].axvline(110, color="darkorange", ls="--", lw=1.5,
                  label="$\\tau$=110 (ceiling)")
    ax[0].set_xlabel("$\\tau$ (mL)")
    ax[0].set_ylabel("fraction of trials certified")
    ax[0].set_title("Feasible threshold window at $n$=149")
    ax[0].set_ylim(-0.03, 1.03); ax[0].legend(fontsize=8)

    ns = [r["n"] for r in n_rows]
    ax[1].plot(ns, [r["certified_rate"] for r in n_rows], "o-",
               color="#55A868", lw=2)
    ax[1].axvline(149, color="crimson", ls="--", lw=1.5, label="ISLES'24 ($n$=149)")
    ax[1].axhline(0.9, color="grey", ls=":", lw=1)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("cohort size $n$ (log scale)")
    ax[1].set_ylabel("fraction of trials certified")
    ax[1].set_title(f"Sample size needed at $\\tau$={tau_fixed:g} mL")
    ax[1].set_ylim(-0.03, 1.03); ax[1].legend(fontsize=8)

    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--alpha-missed", type=float, default=0.05)
    ap.add_argument("--alpha-futile", type=float, default=0.20)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--tau", type=float, default=70.0, help="tau for the n sweep")
    args = ap.parse_args()

    OUT_TAB.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("CERTIFICATION LIMITS")
    print("=" * 68)
    print(f"alpha_missed={args.alpha_missed}  alpha_futile={args.alpha_futile}  "
          f"delta={args.delta}  trials={args.trials}")

    print("\nFloor: smallest conditioning-group size that can EVER certify")
    print("(assumes a perfect rule with zero observed errors)")
    for a in (0.01, 0.05, 0.10, 0.20, 0.30):
        print(f"  risk <= {a:.2f}  needs n >= {min_n_to_certify(a, args.delta):>4d} "
              f"cases in that group")

    tau_rows = sweep(lambda t: CohortSpec(n=149), TAUS, "tau",
                     None, 149, args.alpha_missed, args.alpha_futile,
                     args.delta, args.trials)
    _print_table(tau_rows, "tau", "A. tau sweep at n = 149 (ISLES'24 public set)")

    n_rows = sweep(lambda n: CohortSpec(n=int(n)), NS, "n",
                   args.tau, None, args.alpha_missed, args.alpha_futile,
                   args.delta, args.trials)
    _print_table(n_rows, "n", f"B. n sweep at tau = {args.tau:g} mL")

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    make_figure(tau_rows, n_rows, args.tau, OUT_FIG / "certification_limits.png")
    print(f"\nwrote {OUT_FIG/'certification_limits.png'}")

    payload = {"config": vars(args), "tau_sweep": tau_rows, "n_sweep": n_rows,
               "min_group_n": {str(a): min_n_to_certify(a, args.delta)
                               for a in (0.01, 0.05, 0.10, 0.20, 0.30)}}
    (OUT_TAB / "certification_limits.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT_TAB/'certification_limits.json'}")


if __name__ == "__main__":
    main()
