"""
Conformal predictive systems (CPS) for infarct volume.

Emits a *calibrated CDF* per patient rather than an interval at one fixed alpha.

Why a CDF and not an interval (proposal v4, N1 step 1):
  - P(V > tau) is then calibrated for EVERY tau simultaneously, so the threshold
    sweep is a by-product instead of a separate experiment;
  - the triage decision becomes a threshold on a calibrated probability, which is
    what the Learn-then-Test layer in ltt.py certifies.

References
  Vovk et al., "Nonparametric predictive distributions based on conformal
      prediction" (2017/2019)  -- the split CPS below
  Vovk & Manokhin, "Cross-conformal predictive distributions" (2018)
      -- the cross-conformal aggregation used at n=149
"""

from __future__ import annotations

import numpy as np

__all__ = ["ConformalPredictiveSystem", "CrossConformalPredictiveSystem"]


def _as_1d(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


class ConformalPredictiveSystem:
    """Split conformal predictive system with optional difficulty normalisation.

    Calibration residuals are  C_i = (y_i - yhat_i) / sigma_i.  The predictive
    distribution at a test point is the empirical CDF of  yhat + sigma * C,
    which is what makes the resulting probabilities calibrated.

    Parameters
    ----------
    normalise
        If True, divide residuals by a per-case difficulty estimate (here the
        H3 quantile head's spread).  This is what makes interval width adapt to
        the case rather than being constant, without costing validity.
    """

    def __init__(self, normalise: bool = True, eps: float = 1e-6):
        self.normalise = normalise
        self.eps = float(eps)
        self.residuals_: np.ndarray | None = None

    # ------------------------------------------------------------------ fit

    def fit(self, y_cal, yhat_cal, sigma_cal=None) -> "ConformalPredictiveSystem":
        y = _as_1d(y_cal, "y_cal")
        yhat = _as_1d(yhat_cal, "yhat_cal")
        if y.shape != yhat.shape:
            raise ValueError(f"y_cal {y.shape} and yhat_cal {yhat.shape} differ")

        sigma = self._check_sigma(sigma_cal, y.shape[0])
        self.residuals_ = np.sort((y - yhat) / sigma)
        return self

    def _check_sigma(self, sigma, n: int) -> np.ndarray:
        if not self.normalise or sigma is None:
            return np.ones(n)
        s = _as_1d(sigma, "sigma")
        if s.shape[0] != n:
            raise ValueError(f"sigma has {s.shape[0]} entries, expected {n}")
        if np.any(s < 0):
            raise ValueError("sigma has negative entries")
        return s + self.eps

    def _require_fit(self) -> np.ndarray:
        if self.residuals_ is None:
            raise RuntimeError("call fit() before using the predictive system")
        return self.residuals_

    # -------------------------------------------------------------- predict

    def cdf(self, v, yhat_test, sigma_test=None) -> np.ndarray:
        """P(V <= v) for each test case.  Shape (n_test,) or (n_test, n_v)."""
        res = self._require_fit()
        yhat = _as_1d(yhat_test, "yhat_test")
        sigma = self._check_sigma(sigma_test, yhat.shape[0])

        v_arr = np.asarray(v, dtype=float)
        scalar_v = v_arr.ndim == 0
        v_arr = np.atleast_1d(v_arr)

        # standardised query points, one row per test case
        c = (v_arr[None, :] - yhat[:, None]) / sigma[:, None]

        n = res.shape[0]
        # (n_cal + 1) denominator is the conformal correction: with n calibration
        # residuals the CDF can only be resolved to 1/(n+1), and pretending
        # otherwise is exactly the over-confidence this project is about.
        below = np.searchsorted(res, c, side="left")
        p = below / (n + 1.0)

        return p[:, 0] if scalar_v else p

    def prob_above(self, tau: float, yhat_test, sigma_test=None) -> np.ndarray:
        """P(V > tau) -- the calibrated crossing probability the triage rule uses."""
        return 1.0 - self.cdf(float(tau), yhat_test, sigma_test)

    def quantile(self, q, yhat_test, sigma_test=None) -> np.ndarray:
        """Inverse CDF.  Used only to report intervals for comparability."""
        res = self._require_fit()
        yhat = _as_1d(yhat_test, "yhat_test")
        sigma = self._check_sigma(sigma_test, yhat.shape[0])

        q_arr = np.atleast_1d(np.asarray(q, dtype=float))
        if np.any((q_arr < 0) | (q_arr > 1)):
            raise ValueError("quantile levels must lie in [0, 1]")

        n = res.shape[0]
        # order statistic index under the (n+1) convention, clipped to the sample
        idx = np.clip(np.ceil(q_arr * (n + 1)).astype(int) - 1, 0, n - 1)
        out = yhat[:, None] + sigma[:, None] * res[idx][None, :]
        return out[:, 0] if np.isscalar(q) or np.asarray(q).ndim == 0 else out

    def interval(self, alpha: float, yhat_test, sigma_test=None):
        """Two-sided (1-alpha) predictive interval, for the baseline comparison."""
        lo = self.quantile(alpha / 2.0, yhat_test, sigma_test)
        hi = self.quantile(1.0 - alpha / 2.0, yhat_test, sigma_test)
        return lo, hi

    def pit(self, y_test, yhat_test, sigma_test=None) -> np.ndarray:
        """Probability integral transform.  Uniform on [0,1] iff calibrated."""
        y = _as_1d(y_test, "y_test")
        yhat = _as_1d(yhat_test, "yhat_test")
        sigma = self._check_sigma(sigma_test, yhat.shape[0])
        res = self._require_fit()
        c = (y - yhat) / sigma
        return np.searchsorted(res, c, side="left") / (res.shape[0] + 1.0)


class CrossConformalPredictiveSystem(ConformalPredictiveSystem):
    """Cross-conformal CPS: pool out-of-fold residuals from every CV fold.

    This is the construction the proposal needs at n=149.  Split conformal on a
    single 80/20 split leaves ~24 calibration points per fold, where the
    alpha=0.05 quantile is literally the maximum observed residual and interval
    width is set by one worst case (proposal v3 section 1.5).  Pooling
    out-of-fold residuals uses all n cases instead.

    Validity is approximate rather than exact -- the pooled residuals are not
    fully exchangeable.  That is the standard cross-conformal trade and it must
    be stated in the write-up, not hidden.  validate_calibration.py measures the
    actual coverage error empirically so the size of the trade is reported.
    """

    def fit_from_folds(self, y, yhat_oof, sigma_oof=None, fold_id=None):
        """Fit from out-of-fold predictions.

        Parameters
        ----------
        y, yhat_oof, sigma_oof
            Full-length arrays where entry i was predicted by the model that did
            NOT see case i in training.
        fold_id
            Optional fold assignment, retained for per-fold diagnostics.
        """
        self.fold_id_ = None if fold_id is None else np.asarray(fold_id).ravel()
        return self.fit(y, yhat_oof, sigma_oof)
