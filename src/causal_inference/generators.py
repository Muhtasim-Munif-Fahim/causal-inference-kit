"""Synthetic observational data with known ground-truth effects."""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _draw_covariates(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    """Draw a covariate block: one binary column followed by continuous ones."""
    if k == 0:
        return np.empty((n, 0))
    cols = [rng.binomial(1, 0.5, size=n).astype(float)]
    if k > 1:
        cols.append(rng.normal(size=(n, k - 1)))
    return np.column_stack(cols)


def simulate_observational_data(
    n: int = 2000,
    d_x: int = 3,
    d_w: int = 2,
    ate: float = 2.0,
    confounding: float = 1.0,
    selection_bias: float = 0.0,
    effect_heterogeneity: float = 0.5,
    seed: int = 42,
):
    """Simulate an observational study with known potential outcomes.

    The process:

    1. Draw observed confounders ``X`` (one binary column, the rest
       continuous) and extra covariates ``W`` that predict the outcome but
       not treatment assignment.
    2. Draw a hidden confounder ``U`` that enters both the treatment log-odds
       and the outcome.
    3. Assign treatment from a logistic propensity score built on ``X`` and
       ``U``.
    4. Draw potential outcomes ``Y(0)`` and ``Y(1)``, where the unit-level
       treatment effect can vary linearly with the first confounder.
    5. Return the observed outcome ``Y = Y(T)`` and drop ``U``.

    Parameters
    ----------
    n : int
        Number of units.
    d_x : int
        Number of observed confounders.
    d_w : int
        Number of outcome-only covariates.
    ate : float
        Center of the unit-level treatment effect.
    confounding : float
        Scale of the ``X`` coefficients in the treatment log-odds. Larger
        values create stronger confounding and more imbalance between groups.
    selection_bias : float
        Scale of the hidden confounder ``U`` in the treatment log-odds. When
        non-zero the data violates ignorability: treatment and outcome share
        an unobserved cause, so no estimator adjusting only for ``X`` and
        ``W`` recovers the true ATE.
    effect_heterogeneity : float
        Scale of the ``X``-dependent part of the unit treatment effect.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n, d_x)
        Observed confounders.
    W : ndarray of shape (n, d_w)
        Outcome-only covariates.
    treatment : ndarray of shape (n,)
        Binary treatment indicator.
    outcome : ndarray of shape (n,)
        Observed outcome.
    true_ate : float
        Population average treatment effect, ``mean(unit_effect)``.
    """
    rng = np.random.default_rng(seed)

    X = _draw_covariates(rng, n, d_x)
    W = _draw_covariates(rng, n, d_w)
    U = rng.normal(size=n)

    beta_x = rng.normal(size=d_x)
    beta_x /= np.linalg.norm(beta_x)
    gamma_x = rng.normal(size=d_x)
    gamma_x /= np.linalg.norm(gamma_x)
    delta_w = rng.normal(size=d_w)
    delta_w /= np.linalg.norm(delta_w)

    logit_p = 0.1 + confounding * (X @ beta_x) + selection_bias * U
    propensity = _sigmoid(logit_p)
    treatment = rng.binomial(1, propensity).astype(float)

    noise_0 = rng.normal(size=n)
    noise_1 = rng.normal(size=n)

    baseline = X @ gamma_x + W @ delta_w + 0.5 * U
    modifier = X[:, 0] if d_x > 0 else np.zeros(n)
    unit_effect = ate + effect_heterogeneity * modifier
    y0 = baseline + noise_0
    y1 = baseline + unit_effect + noise_1

    outcome = np.where(treatment == 1, y1, y0)
    true_ate = float(np.mean(unit_effect))
    return X, W, treatment, outcome, true_ate
