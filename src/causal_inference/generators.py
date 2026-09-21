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


def simulate_did_data(
    n: int = 1000,
    ate: float = 1.5,
    time_trend: float = 1.0,
    group_effect: float = 0.5,
    n_pre: int = 1,
    n_post: int = 1,
    seed: int = 0,
):
    """Simulate a panel with parallel trends for difference-in-differences.

    Each unit is observed for ``n_pre`` pre-treatment periods and ``n_post``
    post-treatment periods. Treatment applies to the treated group in every
    post period (canonical simultaneous adoption). Trends are parallel by
    construction: both groups share the same time effect, so two-period DiD
    and two-way fixed effects recover ``ate`` in expectation.

    With the default ``n_pre=1, n_post=1`` the design is the textbook
    two-period, two-group panel. Periods are ``0, ..., n_pre + n_post - 1``
    and treatment starts at time ``n_pre``.

    Parameters
    ----------
    n : int
        Number of units, each observed in every period.
    ate : float
        True effect of the treatment on the treated group.
    time_trend : float
        Common linear time effect ``time_trend * t`` shared by both groups.
    group_effect : float
        Fixed outcome difference between groups.
    n_pre : int
        Number of pre-treatment periods.
    n_post : int
        Number of post-treatment periods.
    seed : int

    Returns
    -------
    unit : ndarray of shape (n * n_times,)
        Unit identifier ``0, ..., n - 1``.
    group : ndarray of shape (n * n_times,)
        Group indicator, 1 = treated (ever-treated).
    period : ndarray of shape (n * n_times,)
        Period index.
    outcome : ndarray of shape (n * n_times,)
    true_did : float
        The constant ATT ``ate``.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if n_pre < 1:
        raise ValueError("n_pre must be at least 1")
    if n_post < 1:
        raise ValueError("n_post must be at least 1")

    rng = np.random.default_rng(seed)
    group = rng.binomial(1, 0.5, size=n).astype(float)
    unit_fe = rng.normal(size=n)
    n_times = n_pre + n_post
    noise = rng.normal(size=(n, n_times))
    times = np.arange(n_times, dtype=float)
    post = (times >= n_pre).astype(float)
    outcome = (
        group_effect * group[:, None]
        + time_trend * times[None, :]
        + ate * group[:, None] * post[None, :]
        + unit_fe[:, None]
        + noise
    )
    unit = np.repeat(np.arange(n), n_times)
    group_all = np.repeat(group, n_times)
    period_all = np.tile(times, n)
    return unit, group_all, period_all, outcome.ravel(), float(ate)


def simulate_synthetic_control_data(
    n_donors: int = 8,
    n_pre: int = 12,
    n_post: int = 8,
    ate: float = 5.0,
    n_factors: int = 2,
    noise: float = 0.1,
    seed: int = 0,
):
    """Simulate a balanced panel for Abadie-style synthetic control.

    One treated unit and ``n_donors`` untreated donors are observed for
    ``n_pre`` pre-treatment and ``n_post`` post-treatment periods. Outcomes
    follow an interactive factor model

    ``Y_it = alpha_i + lambda_i' F_t + eps_it``,

    where the treated unit's intercept and factor loadings are a convex
    combination of the donors'. A constant ``ate`` is then added to the
    treated unit in every post-treatment period. With small noise a
    synthetic control that matches the pre-treatment path therefore
    recovers ``ate`` as the average post-treatment gap.

    Parameters
    ----------
    n_donors : int
        Number of untreated donor units.
    n_pre : int
        Number of pre-treatment periods (times ``0, ..., n_pre - 1``).
    n_post : int
        Number of post-treatment periods.
    ate : float
        Constant post-treatment effect added to the treated unit.
    n_factors : int
        Dimension of the latent factors ``F_t``.
    noise : float
        Standard deviation of the idiosyncratic outcome shock.
    seed : int

    Returns
    -------
    unit : ndarray of shape (n_units * n_times,)
        Unit id. The treated unit is ``0``; donors are ``1, ..., n_donors``.
    time : ndarray of shape (n_units * n_times,)
        Period index ``0, ..., n_pre + n_post - 1``.
    outcome : ndarray of shape (n_units * n_times,)
    treated_unit : int
        Always ``0``.
    treatment_time : int
        First post-treatment period, equal to ``n_pre``.
    true_effect : float
        The constant post-treatment effect ``ate``.
    """
    if n_donors < 1:
        raise ValueError("n_donors must be at least 1")
    if n_pre < 1:
        raise ValueError("n_pre must be at least 1")
    if n_post < 1:
        raise ValueError("n_post must be at least 1")
    if n_factors < 1:
        raise ValueError("n_factors must be at least 1")
    if noise < 0:
        raise ValueError("noise must be non-negative")

    rng = np.random.default_rng(seed)
    n_units = n_donors + 1
    n_times = n_pre + n_post

    k = min(3, n_donors)
    raw = np.linspace(0.5, 0.1, k)
    true_w = np.zeros(n_donors)
    true_w[:k] = raw / raw.sum()

    factors = np.zeros((n_factors, n_times))
    factors[:, 0] = rng.normal(size=n_factors)
    for t in range(1, n_times):
        factors[:, t] = 0.85 * factors[:, t - 1] + rng.normal(size=n_factors)

    lambda_donors = rng.normal(size=(n_donors, n_factors))
    alpha_donors = rng.normal(size=n_donors)
    lambda_treated = true_w @ lambda_donors
    alpha_treated = float(true_w @ alpha_donors)

    panel = np.empty((n_units, n_times))
    panel[0] = alpha_treated + lambda_treated @ factors + noise * rng.normal(size=n_times)
    for j in range(n_donors):
        panel[j + 1] = (
            alpha_donors[j] + lambda_donors[j] @ factors + noise * rng.normal(size=n_times)
        )
    panel[0, n_pre:] += ate

    unit = np.repeat(np.arange(n_units), n_times)
    time = np.tile(np.arange(n_times), n_units)
    outcome = panel.reshape(-1)
    return unit, time, outcome, 0, int(n_pre), float(ate)


def simulate_rd_data(
    n: int = 2000,
    cutoff: float = 0.0,
    ate: float = 2.0,
    noise: float = 0.5,
    slope: float = 1.0,
    slope_jump: float = 0.0,
    curvature: float = 0.0,
    spread: float = 1.0,
    seed: int = 0,
):
    """Simulate a sharp regression-discontinuity design.

    The running variable is drawn uniformly on
    ``[cutoff - spread, cutoff + spread]``. Treatment is assigned
    deterministically as ``T = 1{R >= cutoff}``. Potential outcomes
    share a polynomial baseline that is continuous at the cutoff, so
    the only jump in ``E[Y | R]`` at the threshold is ``ate``:

    ``Y = slope * (R - c) + slope_jump * T * (R - c)
        + curvature * (R - c)^2 + ate * T + ε``.

    Local linear RD therefore recovers ``ate`` in expectation when
    ``curvature`` is zero (linear sides) and approximately when
    ``curvature`` is small relative to the bandwidth. A non-zero
    ``slope`` makes the global treated-minus-control difference a
    biased estimator of the cutoff effect, because treated units have
    systematically larger running-variable values.

    Parameters
    ----------
    n : int
        Number of units.
    cutoff : float
        Treatment threshold.
    ate : float
        Jump in the outcome at the cutoff.
    noise : float
        Standard deviation of the idiosyncratic outcome shock.
    slope : float
        Common linear slope in the running variable.
    slope_jump : float
        Extra slope on the treated side (the intercept jump is still
        ``ate``).
    curvature : float
        Shared quadratic term, continuous at the cutoff.
    spread : float
        Half-width of the uniform support of the running variable.
    seed : int

    Returns
    -------
    running : ndarray of shape (n,)
        Running (forcing) variable.
    treatment : ndarray of shape (n,)
        Sharp treatment indicator ``1{running >= cutoff}``.
    outcome : ndarray of shape (n,)
    cutoff : float
    true_effect : float
        The cutoff jump ``ate``.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if spread <= 0:
        raise ValueError("spread must be positive")
    if noise < 0:
        raise ValueError("noise must be non-negative")

    rng = np.random.default_rng(seed)
    running = rng.uniform(cutoff - spread, cutoff + spread, size=n)
    treatment = (running >= cutoff).astype(float)
    centered = running - cutoff
    outcome = (
        slope * centered
        + slope_jump * treatment * centered
        + curvature * centered ** 2
        + ate * treatment
        + noise * rng.normal(size=n)
    )
    return running, treatment, outcome, float(cutoff), float(ate)
