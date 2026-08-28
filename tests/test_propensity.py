"""Tests for propensity score estimation and overlap diagnostics."""

import numpy as np
import pytest

from causal_inference.generators import simulate_observational_data
from causal_inference.propensity import (
    logistic_regression,
    propensity_histogram,
    propensity_scores,
    standardized_mean_differences,
    weighted_standardized_mean_differences,
)


def _logistic_sample(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    beta = np.array([0.8, -0.5, 1.2])
    intercept = 0.3
    p = 1.0 / (1.0 + np.exp(-(intercept + X @ beta)))
    y = rng.binomial(1, p)
    return X, y, beta, intercept


def test_logistic_recovers_coefficients_on_large_sample():
    X, y, beta, intercept = _logistic_sample()
    fit = logistic_regression(X, y, l2=0.0)
    assert fit.converged
    np.testing.assert_allclose(fit.coef, beta, atol=0.08)
    np.testing.assert_allclose(fit.intercept, intercept, atol=0.08)


def test_predictions_in_unit_interval():
    X, y, _, _ = _logistic_sample(n=2000)
    p, fit = propensity_scores(X, y)
    assert np.all(p > 0.0)
    assert np.all(p < 1.0)


def test_returns_converged_flag():
    X, y, _, _ = _logistic_sample()
    fit = logistic_regression(X, y)
    assert fit.converged
    assert 0 < fit.n_iter <= 2000


def test_perfect_separation_stays_finite():
    rng = np.random.default_rng(1)
    X = rng.choice([-1.0, 1.0], size=(500, 2))
    y = (X[:, 0] > 0).astype(float)
    fit = logistic_regression(X, y, l2=1e-4)
    assert np.all(np.isfinite(fit.coef))
    assert np.isfinite(fit.intercept)
    p, _ = propensity_scores(X, y)
    assert np.all(np.isfinite(p))
    assert np.all((p > 0.0) & (p < 1.0))


def test_perfect_separation_gives_extreme_predictions():
    X = np.array([-1.0] * 400 + [1.0] * 400).reshape(-1, 1)
    y = (X[:, 0] > 0).astype(float)
    p, _ = propensity_scores(X, y, l2=1e-5)
    assert p[y == 1].mean() > 0.999
    assert p[y == 0].mean() < 0.001


def test_scaling_invariance_of_predictions():
    X, y, _, _ = _logistic_sample(n=5000)
    p1, _ = propensity_scores(X, y)
    scaled = X * 10.0 + 2.0
    p2, _ = propensity_scores(scaled, y)
    np.testing.assert_allclose(p1, p2, atol=1e-8)


def test_l2_penalty_shrinks_coefficients():
    X, y, _, _ = _logistic_sample(n=5000)
    small = logistic_regression(X, y, l2=1e-6)
    large = logistic_regression(X, y, l2=10.0)
    assert np.linalg.norm(large.coef) < np.linalg.norm(small.coef)


def test_constant_covariate_column():
    rng = np.random.default_rng(3)
    X = np.column_stack([np.full(400, 3.0), rng.normal(size=400)])
    y = rng.binomial(1, 0.5, size=400)
    fit = logistic_regression(X, y)
    assert np.all(np.isfinite(fit.coef))
    assert fit.coef[0] == pytest.approx(0.0)


def test_single_covariate():
    X, y, _, _ = _logistic_sample(n=3000)
    p, fit = propensity_scores(X[:, 0], y)
    assert p.shape == (3000,)
    assert fit.coef.shape == (1,)


def test_single_sample():
    X = np.array([[1.0, 2.0]])
    y = np.array([1.0])
    fit = logistic_regression(X, y, l2=0.01)
    assert np.all(np.isfinite(fit.coef))
    p, _ = propensity_scores(X, y)
    assert p[0] > 0.9


def test_all_treated_no_crash():
    X = np.arange(200.0).reshape(100, 2)
    y = np.ones(100)
    p, _ = propensity_scores(X, y)
    assert np.all(np.isfinite(p))
    assert p.mean() > 0.9


def test_invalid_target_values_raise():
    X = np.zeros((10, 2))
    with pytest.raises(ValueError):
        logistic_regression(X, np.array([0, 1, 2] * 3 + [0]))


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        logistic_regression(np.zeros((10, 2)), np.zeros(9))


def test_empty_input_raises():
    with pytest.raises(ValueError):
        logistic_regression(np.zeros((0, 2)), np.zeros(0))


def test_propensity_scores_match_manual_formula():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 2))
    y = rng.binomial(1, 0.4, size=200)
    p, fit = propensity_scores(X, y)
    manual = 1.0 / (1.0 + np.exp(-(fit.intercept + X @ fit.coef)))
    np.testing.assert_allclose(p, manual, atol=1e-12)


def test_propensity_scores_discriminate_on_simulated_data():
    X, _, treatment, _, _ = simulate_observational_data(
        confounding=2.0, n=3000, seed=9
    )
    p, _ = propensity_scores(X, treatment)
    assert p[treatment == 1].mean() > p[treatment == 0].mean()


def test_smd_balanced_data_near_zero():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(5000, 3))
    treatment = rng.binomial(1, 0.5, size=5000)
    smd = standardized_mean_differences(X, treatment)
    assert np.abs(smd).max() < 0.1


def test_smd_imbalanced_data_nonzero():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(5000, 1))
    treatment = rng.binomial(1, 1.0 / (1.0 + np.exp(-X[:, 0])), size=5000)
    smd = standardized_mean_differences(X, treatment)
    assert abs(smd[0]) > 0.3


def test_smd_zero_variance_column():
    X = np.column_stack([np.ones(100), np.random.default_rng(7).normal(size=100)])
    treatment = np.array([1] * 50 + [0] * 50)
    smd = standardized_mean_differences(X, treatment)
    assert smd[0] == 0.0
    assert np.isfinite(smd[1])


def test_smd_requires_both_groups():
    X = np.zeros((10, 2))
    with pytest.raises(ValueError):
        standardized_mean_differences(X, np.ones(10))


def test_weighted_smd_constant_weights_match_unweighted():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(3000, 3))
    treatment = rng.binomial(1, 0.5, size=3000)
    weights = np.full(3000, 2.0)
    wsmd = weighted_standardized_mean_differences(X, treatment, weights)
    smd = standardized_mean_differences(X, treatment)
    np.testing.assert_allclose(wsmd, smd, atol=1e-10)


def test_weighted_smd_reduces_imbalance_with_ipw_weights():
    smd_maxes, wsmd_maxes = [], []
    for seed in range(8):
        X, _, treatment, _, _ = simulate_observational_data(
            confounding=1.5, n=15000, seed=seed
        )
        p, _ = propensity_scores(X, treatment)
        p_treat = treatment.mean()
        weights = (
            treatment * (p_treat / p)
            + (1.0 - treatment) * ((1.0 - p_treat) / (1.0 - p))
        )
        smd_maxes.append(np.abs(standardized_mean_differences(X, treatment)).max())
        wsmd_maxes.append(
            np.abs(weighted_standardized_mean_differences(X, treatment, weights)).max()
        )
    assert np.mean(smd_maxes) > 0.5
    assert np.mean(wsmd_maxes) < 0.1
    assert np.mean(wsmd_maxes) < 0.5 * np.mean(smd_maxes)


def test_weighted_smd_rejects_bad_weights():
    X = np.zeros((10, 2))
    treatment = np.array([1, 0] * 5)
    with pytest.raises(ValueError):
        weighted_standardized_mean_differences(X, treatment, -np.ones(10))
    with pytest.raises(ValueError):
        weighted_standardized_mean_differences(X, treatment, np.ones(5))


def test_histogram_counts_sum_to_group_sizes():
    rng = np.random.default_rng(10)
    p = rng.uniform(size=1000)
    treatment = rng.binomial(1, 0.5, size=1000)
    hist = propensity_histogram(p, treatment, bins=10)
    assert hist.treated.sum() == (treatment == 1).sum()
    assert hist.control.sum() == (treatment == 0).sum()
    assert hist.edges.shape == (11,)
    assert hist.edges[0] == pytest.approx(0.0)
    assert hist.edges[-1] == pytest.approx(1.0)


def test_histogram_places_all_scores_in_one_bin():
    p = np.full(200, 0.7)
    treatment = np.array([1] * 100 + [0] * 100)
    hist = propensity_histogram(p, treatment, bins=10)
    assert hist.treated.sum() == 100
    assert hist.control.sum() == 100
    assert (hist.treated > 0).sum() == 1
    assert (hist.control > 0).sum() == 1


def test_histogram_drops_nan():
    p = np.array([0.1, 0.9, np.nan, 0.5])
    treatment = np.array([1, 1, 1, 0])
    hist = propensity_histogram(p, treatment, bins=4)
    assert hist.treated.sum() == 2
    assert hist.control.sum() == 1


def test_histogram_rejects_bad_bins():
    with pytest.raises(ValueError):
        propensity_histogram(np.array([0.5, 0.5]), np.array([1, 0]), bins=0)
