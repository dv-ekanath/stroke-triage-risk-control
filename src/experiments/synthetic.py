"""
Synthetic ISLES'24-like cohort.

Moments are matched to the real dataset descriptor so that conclusions about
sample size transfer:  scan-level final infarct volume 33.16 +/- 48.67 mL,
range <0.01 to 318.25 mL, heavily right-skewed (proposal v3 section 2.1).
A lognormal reproduces those two moments and the skew.

This exists so the whole calibration layer can be validated -- and the
certification-limits table produced -- with no imaging data and no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CohortSpec", "simulate"]

# ISLES'24 public training set, scan level
MEAN_ML, SD_ML = 33.16, 48.67
MAX_ML = 318.25


@dataclass(frozen=True)
class CohortSpec:
    """Parameters of the simulated cohort and of the volume predictor."""
    n: int = 149
    mean_ml: float = MEAN_ML
    sd_ml: float = SD_ML
    # Predictor error, as a fraction of true volume plus a floor in mL.
    # Defaults are deliberately optimistic-but-plausible for a model at the
    # ISLES'24 leaderboard level; sweep them to test robustness.
    rel_err: float = 0.35
    abs_err_ml: float = 6.0
    bias_ml: float = 0.0
    # Multiplicative noise on the difficulty estimate the H3 head would emit,
    # i.e. how badly the model knows its own error.
    sigma_noise: float = 0.25

    def lognormal_params(self) -> tuple[float, float]:
        m, s = self.mean_ml, self.sd_ml
        sigma2 = np.log1p((s * s) / (m * m))
        mu = np.log(m) - 0.5 * sigma2
        return mu, float(np.sqrt(sigma2))


def simulate(spec: CohortSpec | None = None, seed: int = 0):
    """Draw (V, V_hat, sigma_hat) for one synthetic cohort.

    Returns
    -------
    v : true final infarct volume, mL
    v_hat : model point prediction, mL (clipped at 0)
    sigma_hat : model difficulty estimate, mL -- stands in for the spread of
        the H3 quantile head, and is deliberately an imperfect estimate of the
        true error scale.
    """
    spec = spec or CohortSpec()
    rng = np.random.default_rng(seed)

    mu, sigma = spec.lognormal_params()
    v = rng.lognormal(mu, sigma, spec.n)
    v = np.clip(v, 0.0, MAX_ML)

    # Heteroscedastic error: big lesions are harder in absolute terms.
    scale = spec.rel_err * v + spec.abs_err_ml
    v_hat = np.clip(v + spec.bias_ml + rng.normal(0.0, scale), 0.0, None)

    # The model's own estimate of that scale, itself noisy.
    sigma_hat = scale * np.exp(rng.normal(0.0, spec.sigma_noise, spec.n))
    return v, v_hat, sigma_hat
