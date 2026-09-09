"""
The triage decision rule and its two clinical risks.

Decision rule, parameterised by lam = (lam_lo, lam_hi) on the calibrated
crossing probability p = P(V > tau):

      p <  lam_lo             -> ELIGIBLE    (model asserts V <  tau)
      p >  lam_hi             -> INELIGIBLE  (model asserts V >= tau)
      lam_lo <= p <= lam_hi   -> ABSTAIN     (refer to clinician)

The two risks are deliberately NOT symmetric:

  R_missed = P(say INELIGIBLE | true V <  tau)
      Denying thrombectomy to a patient who would have benefited.

  R_futile = P(say ELIGIBLE   | true V >= tau)
      Futile reperfusion: procedural and haemorrhage risk for no tissue benefit.

The 2026 AHA/ASA guideline makes EVT in large core Class 1A in both the 0-6 h
and 6-24 h windows, so denying treatment is now the worse error and
alpha_missed << alpha_futile.  The asymmetry is set by the guideline, not by
taste -- which is the point worth making in the write-up.

Note both risks are CONDITIONAL.  Their effective sample size is the number of
calibration cases in the conditioning group (V < tau or V >= tau), not n.  At
n=149 with a right-skewed cohort those groups are very unequal, and treating
them as n is the single easiest way to certify something that is not true.
ltt.py handles this explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ELIGIBLE, ABSTAIN, INELIGIBLE = 0, 1, 2
_LABELS = {ELIGIBLE: "ELIGIBLE", ABSTAIN: "ABSTAIN", INELIGIBLE: "INELIGIBLE"}

__all__ = [
    "ELIGIBLE", "ABSTAIN", "INELIGIBLE", "label",
    "decide", "DecisionOutcome", "evaluate",
]


def label(code: int) -> str:
    return _LABELS[int(code)]


def decide(p, lam_lo: float, lam_hi: float) -> np.ndarray:
    """Map calibrated crossing probabilities to {ELIGIBLE, ABSTAIN, INELIGIBLE}."""
    if not (0.0 <= lam_lo <= lam_hi <= 1.0):
        raise ValueError(f"need 0 <= lam_lo <= lam_hi <= 1, got ({lam_lo}, {lam_hi})")
    p = np.asarray(p, dtype=float).ravel()
    out = np.full(p.shape, ABSTAIN, dtype=int)
    out[p < lam_lo] = ELIGIBLE
    out[p > lam_hi] = INELIGIBLE
    return out


@dataclass(frozen=True)
class DecisionOutcome:
    """Empirical risks and rates for one lam on one dataset."""
    lam_lo: float
    lam_hi: float
    r_missed: float          # P(INELIGIBLE | V <  tau)
    r_futile: float          # P(ELIGIBLE   | V >= tau)
    n_small: int             # cases with V <  tau   (denominator of r_missed)
    n_large: int             # cases with V >= tau   (denominator of r_futile)
    abstain_rate: float
    decided_rate: float
    accuracy_on_decided: float

    def as_row(self) -> dict:
        return {
            "lam_lo": self.lam_lo, "lam_hi": self.lam_hi,
            "R_missed": self.r_missed, "R_futile": self.r_futile,
            "n_small": self.n_small, "n_large": self.n_large,
            "abstain_rate": self.abstain_rate,
            "acc_on_decided": self.accuracy_on_decided,
        }


def evaluate(p, y_true, tau: float, lam_lo: float, lam_hi: float) -> DecisionOutcome:
    """Empirical risks of rule lam on cases with calibrated probabilities p."""
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y_true, dtype=float).ravel()
    if p.shape != y.shape:
        raise ValueError(f"p {p.shape} and y_true {y.shape} differ")

    d = decide(p, lam_lo, lam_hi)
    small = y < tau                      # truly eligible on the volume criterion
    large = ~small
    n_small, n_large = int(small.sum()), int(large.sum())

    # nan, not 0, when a conditioning group is empty: an unobserved risk is
    # unknown, and reporting it as zero would certify a rule on no evidence.
    r_missed = float(np.mean(d[small] == INELIGIBLE)) if n_small else float("nan")
    r_futile = float(np.mean(d[large] == ELIGIBLE)) if n_large else float("nan")

    decided = d != ABSTAIN
    n_dec = int(decided.sum())
    if n_dec:
        correct = ((d == ELIGIBLE) & small) | ((d == INELIGIBLE) & large)
        acc = float(correct[decided].sum() / n_dec)
    else:
        acc = float("nan")

    return DecisionOutcome(
        lam_lo=float(lam_lo), lam_hi=float(lam_hi),
        r_missed=r_missed, r_futile=r_futile,
        n_small=n_small, n_large=n_large,
        abstain_rate=float(np.mean(~decided)),
        decided_rate=float(np.mean(decided)),
        accuracy_on_decided=acc,
    )
