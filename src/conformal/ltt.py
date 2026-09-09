"""
Learn-then-Test risk control for the triage decision (proposal v4, N1 step 2).

Angelopoulos, Bates, Candes, Jordan, Lei, "Learn then Test: Calibrating
Predictive Algorithms to Achieve Risk Control", Ann. Appl. Stat. 19(2), 2025.
arXiv:2110.01052

The idea: treat "is rule lam safe?" as a hypothesis test.  For each lam on a
grid, test  H0(lam): R(lam) > alpha  using a concentration p-value.  Reject H0
with family-wise error rate control across the grid, and every surviving lam
carries a finite-sample guarantee  P(R(lam) <= alpha) >= 1 - delta.

Two departures from the vanilla recipe, both forced by this application:

1. TWO risks, controlled simultaneously.  A lam is certified only if both
   H0_missed and H0_futile are rejected.  Under the union-intersection
   principle the valid p-value for the intersection null is max(p1, p2), so no
   extra multiplicity correction is needed for having two risks.

2. The risks are CONDITIONAL, so the effective n is the size of the
   conditioning group, not the calibration set.  With a right-skewed cohort
   (mean 33 mL) and tau = 110 mL the "large" group is small, and using n
   everywhere would manufacture significance that is not there.

On multiplicity across the grid: Bonferroni over a dense 2-D grid at n=149 will
certify nothing, so `split_fixed_sequence` is the default.  It spends part of
the calibration data learning a promising ORDER over lam, then walks that order
on the held-out part testing at level delta with no correction at all (valid
because the sequence is fixed before seeing the test half).  This is the
multi-dimensional-lambda variant recommended in the LTT paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import binom

from .decision import evaluate

__all__ = ["hb_p_value", "min_n_to_certify", "LTTResult", "certify",
           "make_lambda_grid"]

_E = float(np.e)


def _h1(a: float, b: float) -> float:
    """KL divergence between Bernoulli(a) and Bernoulli(b)."""
    out = 0.0
    if a > 0:
        out += a * np.log(a / b)
    if a < 1:
        out += (1 - a) * np.log((1 - a) / (1 - b))
    return out


def hb_p_value(r_hat: float, n: int, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for H0: R > alpha, from n iid bounded losses.

    Returns 1.0 (cannot reject) when n == 0 or the risk is not observed, which
    is the honest answer for an empty conditioning group.
    """
    if n <= 0 or not np.isfinite(r_hat):
        return 1.0
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0,1), got {alpha}")
    r_hat = float(np.clip(r_hat, 0.0, 1.0))
    if r_hat >= alpha:
        return 1.0                      # observed risk already exceeds the target

    p_hoeffding = float(np.exp(-n * _h1(r_hat, alpha)))
    p_bentkus = float(_E * binom.cdf(np.ceil(n * r_hat), n, alpha))
    return float(min(1.0, p_hoeffding, p_bentkus))


def min_n_to_certify(alpha: float, delta: float, max_n: int = 100_000) -> int:
    """Smallest conditioning-group size that can EVER certify risk <= alpha at
    level delta, assuming the most favourable case of zero observed errors.

    This is the hard floor the data audit checks against.  R_futile can only be
    certified from cases with V >= tau, so if the cohort has fewer than this
    many of them, no model and no amount of tuning yields a guarantee at that
    threshold -- the threshold has to move, or the claim has to change.
    """
    for n in range(1, max_n + 1):
        if hb_p_value(0.0, n, alpha) <= delta:
            return n
    return -1


def make_lambda_grid(n_lo: int = 41, n_hi: int = 41) -> np.ndarray:
    """Candidate (lam_lo, lam_hi) pairs with lam_lo <= lam_hi."""
    lo = np.linspace(0.0, 1.0, n_lo)
    hi = np.linspace(0.0, 1.0, n_hi)
    return np.array([(a, b) for a in lo for b in hi if a <= b], dtype=float)


@dataclass
class LTTResult:
    """Outcome of a certification run."""
    certified: np.ndarray                 # (k, 2) certified lam pairs
    selected: tuple[float, float] | None  # abstention-minimising certified lam
    alpha_missed: float
    alpha_futile: float
    delta: float
    method: str
    n_small: int
    n_large: int
    n_tested: int
    diagnostics: dict = field(default_factory=dict)

    @property
    def certified_any(self) -> bool:
        return self.certified.shape[0] > 0

    def summary(self) -> str:
        head = (
            f"LTT [{self.method}] alpha_missed={self.alpha_missed:.2f} "
            f"alpha_futile={self.alpha_futile:.2f} delta={self.delta:.2f} | "
            f"n_small={self.n_small} n_large={self.n_large} "
            f"tested={self.n_tested}"
        )
        if not self.certified_any:
            return head + "\n  NOTHING CERTIFIED at this sample size."
        lo, hi = self.selected
        d = self.diagnostics
        return (
            head
            + f"\n  certified {self.certified.shape[0]} lambda(s)"
            + f"\n  selected  lam_lo={lo:.3f} lam_hi={hi:.3f}"
            + f"  abstain={d.get('abstain_rate', float('nan')):.3f}"
            + f"  R_missed={d.get('r_missed', float('nan')):.3f}"
            + f"  R_futile={d.get('r_futile', float('nan')):.3f}"
        )


def _risks_and_pvals(p, y, tau, lam, alpha_missed, alpha_futile):
    """Empirical risks and the intersection p-value for one lam."""
    out = evaluate(p, y, tau, lam[0], lam[1])
    p_m = hb_p_value(out.r_missed, out.n_small, alpha_missed)
    p_f = hb_p_value(out.r_futile, out.n_large, alpha_futile)
    return out, max(p_m, p_f)          # union-intersection: max is valid


def certify(
    p_cal,
    y_cal,
    tau: float,
    alpha_missed: float = 0.05,
    alpha_futile: float = 0.20,
    delta: float = 0.10,
    grid: np.ndarray | None = None,
    method: str = "monotone_fixed_sequence",
    n_levels: int = 201,
    seed: int = 0,
) -> LTTResult:
    """Certify triage rules with FWER control.

    The default method exploits a structural fact about this decision rule that
    makes the general 2-D machinery unnecessary:

        R_missed = P(p > lam_hi | V <  tau)   depends ONLY on lam_hi
        R_futile = P(p < lam_lo | V >= tau)   depends ONLY on lam_lo

    The two risks decouple, so the certified region is a product set rather than
    an arbitrary subset of the grid, and each risk is monotone in its own
    parameter.  That reduces certification to two one-dimensional fixed-sequence
    walks:

      * lam_hi starts at 1 (nothing called ineligible, R_missed = 0) and walks
        down until the risk test fails;
      * lam_lo starts at 0 (nothing called eligible, R_futile = 0) and walks up
        until the risk test fails.

    Each walk is a fixed sequence decided before looking at the data, so no
    multiplicity correction is needed within a walk; the two walks split delta
    between them.  Nothing is held out for learning an ordering, so all n
    calibration cases contribute -- which matters at n = 149.

    Parameters
    ----------
    p_cal, y_cal
        Calibrated crossing probabilities and true volumes on calibration data.
    tau
        Volume threshold in mL.
    alpha_missed, alpha_futile
        Risk targets.  alpha_missed should be the tighter of the two.
    delta
        FWER level: with probability >= 1-delta every certified lam satisfies
        both risk constraints.
    method
        'monotone_fixed_sequence' (default) or 'bonferroni' (dense 2-D grid,
        kept for the ablation showing it certifies nothing at n = 149).
    """
    p_cal = np.asarray(p_cal, dtype=float).ravel()
    y_cal = np.asarray(y_cal, dtype=float).ravel()
    if p_cal.shape != y_cal.shape:
        raise ValueError(f"p_cal {p_cal.shape} and y_cal {y_cal.shape} differ")

    small = y_cal < tau
    large = ~small
    n_small, n_large = int(small.sum()), int(large.sum())
    common = dict(
        alpha_missed=alpha_missed, alpha_futile=alpha_futile, delta=delta,
        method=method, n_small=n_small, n_large=n_large,
    )

    if method == "bonferroni":
        if grid is None:
            grid = make_lambda_grid()
        thresh = delta / grid.shape[0]
        certified, rows = [], []
        for lam in grid:
            out, pv = _risks_and_pvals(p_cal, y_cal, tau, lam, alpha_missed, alpha_futile)
            if pv <= thresh:
                certified.append(lam)
                rows.append(out)
        n_tested = grid.shape[0]

    elif method == "monotone_fixed_sequence":
        levels = np.linspace(0.0, 1.0, n_levels)
        half = delta / 2.0                      # one half per walk
        p_small, p_large = p_cal[small], p_cal[large]
        n_tested = 0

        # walk lam_hi downward from 1: R_missed = P(p > lam_hi | small)
        hi_ok = []
        for lam_hi in levels[::-1]:
            n_tested += 1
            r = float(np.mean(p_small > lam_hi)) if n_small else float("nan")
            if hb_p_value(r, n_small, alpha_missed) > half:
                break
            hi_ok.append(lam_hi)

        # walk lam_lo upward from 0: R_futile = P(p < lam_lo | large)
        lo_ok = []
        for lam_lo in levels:
            n_tested += 1
            r = float(np.mean(p_large < lam_lo)) if n_large else float("nan")
            if hb_p_value(r, n_large, alpha_futile) > half:
                break
            lo_ok.append(lam_lo)

        # product set, restricted to lam_lo <= lam_hi
        certified = [(a, b) for a in lo_ok for b in hi_ok if a <= b]
        rows = None

    else:
        raise ValueError(f"unknown method {method!r}")

    if not certified:
        return LTTResult(
            certified=np.empty((0, 2)), selected=None, n_tested=n_tested, **common
        )

    certified_arr = np.asarray(certified, dtype=float)
    if rows is None:
        # abstention shrinks as lam_lo grows and lam_hi shrinks; evaluate only
        # the corner rather than every pair
        best_lo = float(certified_arr[:, 0].max())
        cand = certified_arr[certified_arr[:, 0] == best_lo]
        best_hi = float(cand[:, 1].min())
        chosen = evaluate(p_cal, y_cal, tau, best_lo, best_hi)
    else:
        best = int(np.argmin([r.abstain_rate for r in rows]))
        chosen = rows[best]

    return LTTResult(
        certified=certified_arr,
        selected=(chosen.lam_lo, chosen.lam_hi),
        n_tested=n_tested,
        diagnostics={
            "abstain_rate": chosen.abstain_rate,
            "r_missed": chosen.r_missed,
            "r_futile": chosen.r_futile,
            "acc_on_decided": chosen.accuracy_on_decided,
        },
        **common,
    )
