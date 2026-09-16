"""Tests for treatment-effect estimators."""

import numpy as np
import pytest

from causal_inference.estimators import (
    aipw_ate,
    difference_in_differences,
    difference_in_means,
    ipw_ate,
    ipw_att,
    ipw_weights,
    outcome_regression,
    propensity_matching,
)
from causal_inference.generators import simulate_did_data, simulate_observational_data
from causal_inference.propensity import propensity_scores


def _confounded(n=20000, confounding=1.5, seed=0):
    X, W, treatment, outcome, true_ate = simulate_observational_data(
        n=n, confounding=confounding, seed=seed
    )
    return X, W, treatment, outcome, true_ate


def test_ipw_ate_recovers_ate_without_selection_bias():
    X, _, treatment, outcome, true_ate = _confounded()
    estimate = ipw_ate(X, treatment, outcome)
    assert abs(estimate - true_ate) < 0.3


def test_ipw_ate_improves_on_naive_difference():
    X, _, treatment, outcome, true_ate = _confounded()
    ipw = ipw_ate(X, treatment, outcome)
    naive = difference_in_means(treatment, outcome)
    assert abs(ipw - true_ate) < abs(naive - true_ate)


def test_ipw_att_recovers_true_att():
    X, _, treatment, outcome, _ = simulate_observational_data(
        n=20000, confounding=1.5, effect_heterogeneity=3.0, seed=3
    )
    true_att = 2.0 + 3.0 * X[treatment == 1, 0].mean()
    estimate = ipw_att(X, treatment, outcome)
    assert abs(estimate - true_att) < 0.3


def test_ipw_ate_stabilized_and_raw_close():
    X, _, treatment, outcome, true_ate = _confounded()
    stabilized = ipw_ate(X, treatment, outcome, stabilized=True)
    raw = ipw_ate(X, treatment, outcome, stabilized=False)
    assert abs(stabilized - raw) < 0.2
    assert abs(stabilized - true_ate) < 0.3
    assert abs(raw - true_ate) < 0.3


def test_ipw_ate_normalized_and_hajek_close():
    X, _, treatment, outcome, true_ate = _confounded()
    normalized = ipw_ate(X, treatment, outcome, normalized=True)
    unnormalized = ipw_ate(X, treatment, outcome, normalized=False, stabilized=False)
    assert abs(normalized - unnormalized) < 0.2
    assert abs(normalized - true_ate) < 0.3


def test_ipw_ate_rejects_stabilized_unnormalized():
    X, _, treatment, outcome, _ = _confounded(n=500)
    with pytest.raises(ValueError):
        ipw_ate(X, treatment, outcome, stabilized=True, normalized=False)


def test_ipw_uses_explicit_propensity():
    X, _, treatment, outcome, true_ate = _confounded(n=5000)
    p, _ = propensity_scores(X, treatment)
    with_p = ipw_ate(X, treatment, outcome, propensity=p)
    without_p = ipw_ate(X, treatment, outcome)
    assert with_p == pytest.approx(without_p, abs=1e-8)


def test_ipw_rejects_out_of_range_propensity():
    X, _, treatment, outcome, _ = _confounded(n=500)
    p = np.full(500, 0.0)
    with pytest.raises(ValueError):
        ipw_ate(X, treatment, outcome, propensity=p)
    with pytest.raises(ValueError):
        ipw_ate(X, treatment, outcome, propensity=np.full(499, 0.5))


def test_ipw_rejects_bad_inputs():
    X, _, treatment, outcome, _ = _confounded(n=100)
    with pytest.raises(ValueError):
        ipw_ate(X, treatment[:50], outcome)
    with pytest.raises(ValueError):
        ipw_ate(X, treatment, outcome[:50])
    with pytest.raises(ValueError):
        ipw_ate(X, treatment + 2, outcome)


def test_ipw_finite_under_extreme_propensity():
    X = np.array([[-1.0], [1.0], [-1.0], [1.0]])
    treatment = np.array([1.0, 1.0, 0.0, 0.0])
    outcome = np.array([3.0, 4.0, 1.0, 2.0])
    estimate = ipw_ate(X, treatment, outcome)
    assert np.isfinite(estimate)


def test_ipw_att_requires_both_groups():
    X = np.zeros((10, 1))
    with pytest.raises(ValueError):
        ipw_att(X, np.ones(10), np.zeros(10))


def test_ipw_weights_shape_and_positivity():
    X, _, treatment, _, _ = _confounded(n=2000)
    weights = ipw_weights(X, treatment)
    assert weights.shape == (2000,)
    assert np.all(weights > 0)


def test_ipw_weights_matches_manual_formula():
    X, _, treatment, _, _ = _confounded(n=2000)
    p, _ = propensity_scores(X, treatment)
    p_treat = treatment.mean()
    manual = np.where(
        treatment == 1,
        p_treat / p,
        (1.0 - p_treat) / (1.0 - p),
    )
    np.testing.assert_allclose(ipw_weights(X, treatment), manual, atol=1e-12)
    raw = np.where(treatment == 1, 1.0 / p, 1.0 / (1.0 - p))
    np.testing.assert_allclose(
        ipw_weights(X, treatment, stabilized=False), raw, atol=1e-12
    )


def test_ipw_weights_reject_bad_inputs():
    X, _, treatment, _, _ = _confounded(n=500)
    with pytest.raises(ValueError):
        ipw_weights(X, treatment[:250])
    with pytest.raises(ValueError):
        ipw_weights(X, treatment, propensity=np.full(500, 0.0))
    with pytest.raises(ValueError):
        ipw_weights(X, np.ones(500))


def test_aipw_matches_hand_calculation():
    X = np.array([[0.0], [0.0], [1.0], [1.0], [0.0]])
    treatment = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    outcome = np.array([2.0, 4.0, 6.0, 1.0, 1.0])
    p = np.full(5, 0.5)
    assert aipw_ate(X, treatment, outcome, propensity=p) == pytest.approx(3.2)


def test_aipw_ate_recovers_ate_without_selection_bias():
    X, W, treatment, outcome, true_ate = _confounded()
    estimate = aipw_ate(X, treatment, outcome, W=W)
    assert abs(estimate - true_ate) < 0.3


def test_aipw_ate_improves_on_naive_difference():
    X, W, treatment, outcome, true_ate = _confounded()
    aipw = aipw_ate(X, treatment, outcome, W=W)
    naive = difference_in_means(treatment, outcome)
    assert abs(aipw - true_ate) < abs(naive - true_ate)


def test_aipw_doubly_robust_to_wrong_propensity():
    """Correct outcome regression, constant (wrong) propensity."""
    X, W, treatment, outcome, true_ate = _confounded()
    p_wrong = np.full(treatment.shape[0], 0.5)
    estimate = aipw_ate(X, treatment, outcome, propensity=p_wrong, W=W)
    assert abs(estimate - true_ate) < 0.3


def test_aipw_doubly_robust_to_wrong_outcome():
    """Correct propensity, intercept-only (wrong) outcome regression."""
    X, _, treatment, outcome, true_ate = _confounded()
    p, _ = propensity_scores(X, treatment)
    X_intercept_only = np.zeros((X.shape[0], 1))
    estimate = aipw_ate(X_intercept_only, treatment, outcome, propensity=p)
    assert abs(estimate - true_ate) < 0.3


def test_aipw_normalized_and_unnormalized_close():
    X, W, treatment, outcome, true_ate = _confounded()
    plain = aipw_ate(X, treatment, outcome, W=W, normalized=False)
    hajek = aipw_ate(X, treatment, outcome, W=W, normalized=True)
    assert abs(plain - hajek) < 0.2
    assert abs(plain - true_ate) < 0.3
    assert abs(hajek - true_ate) < 0.3


def test_aipw_uses_explicit_propensity():
    X, W, treatment, outcome, _ = _confounded(n=5000)
    p, _ = propensity_scores(X, treatment)
    with_p = aipw_ate(X, treatment, outcome, propensity=p, W=W)
    without_p = aipw_ate(X, treatment, outcome, W=W)
    assert with_p == pytest.approx(without_p, abs=1e-8)


def test_aipw_without_w_still_recovers_ate():
    X, _, treatment, outcome, true_ate = _confounded()
    estimate = aipw_ate(X, treatment, outcome)
    assert abs(estimate - true_ate) < 0.3


def test_aipw_rejects_out_of_range_propensity():
    X, W, treatment, outcome, _ = _confounded(n=500)
    with pytest.raises(ValueError):
        aipw_ate(X, treatment, outcome, propensity=np.full(500, 0.0), W=W)
    with pytest.raises(ValueError):
        aipw_ate(X, treatment, outcome, propensity=np.full(499, 0.5), W=W)


def test_aipw_rejects_bad_inputs():
    X, W, treatment, outcome, _ = _confounded(n=100)
    with pytest.raises(ValueError):
        aipw_ate(X, treatment[:50], outcome, W=W)
    with pytest.raises(ValueError):
        aipw_ate(X, treatment, outcome[:50], W=W)
    with pytest.raises(ValueError):
        aipw_ate(X, treatment, outcome, W=W[:50])
    with pytest.raises(ValueError):
        aipw_ate(X, np.ones(100), outcome, W=W)


def test_aipw_requires_both_groups():
    X = np.zeros((10, 1))
    with pytest.raises(ValueError):
        aipw_ate(X, np.ones(10), np.zeros(10))


def test_outcome_regression_shapes_and_in_sample_fit():
    X, W, treatment, outcome, _ = _confounded(n=2000)
    mu1, mu0 = outcome_regression(X, treatment, outcome, W=W)
    assert mu1.shape == (2000,)
    assert mu0.shape == (2000,)
    assert np.all(np.isfinite(mu1))
    assert np.all(np.isfinite(mu0))
    treated = treatment == 1
    assert abs((outcome[treated] - mu1[treated]).mean()) < 1e-8
    assert abs((outcome[~treated] - mu0[~treated]).mean()) < 1e-8


def test_outcome_regression_requires_both_groups():
    X = np.zeros((10, 1))
    with pytest.raises(ValueError):
        outcome_regression(X, np.ones(10), np.zeros(10))


def test_difference_in_means_basic():
    treatment = np.array([1.0, 1.0, 0.0, 0.0])
    outcome = np.array([3.0, 5.0, 1.0, 2.0])
    assert difference_in_means(treatment, outcome) == pytest.approx(2.5)


def test_difference_in_means_requires_both_groups():
    with pytest.raises(ValueError):
        difference_in_means(np.ones(5), np.zeros(5))


def test_matching_att_recovers_att():
    X, _, treatment, outcome, _ = simulate_observational_data(
        n=12000, confounding=1.0, effect_heterogeneity=3.0, seed=7
    )
    true_att = 2.0 + 3.0 * X[treatment == 1, 0].mean()
    estimate = propensity_matching(X, treatment, outcome, with_replacement=True)
    assert abs(estimate - true_att) < 0.4


def test_matching_att_improves_on_naive():
    X, _, treatment, outcome, true_ate = simulate_observational_data(
        n=12000, confounding=1.5, seed=11
    )
    matched = propensity_matching(X, treatment, outcome, with_replacement=True)
    naive = difference_in_means(treatment, outcome)
    assert abs(matched - true_ate) < abs(naive - true_ate)


def test_matching_with_and_without_replacement_recover_att_with_caliper():
    X, _, treatment, outcome, _ = simulate_observational_data(
        n=12000, confounding=1.0, effect_heterogeneity=3.0, seed=13
    )
    true_att = 2.0 + 3.0 * X[treatment == 1, 0].mean()
    no_replace = propensity_matching(X, treatment, outcome, caliper=0.2)
    with_replace = propensity_matching(X, treatment, outcome, caliper=0.2, with_replacement=True)
    assert abs(no_replace - true_att) < 0.4
    assert abs(with_replace - true_att) < 0.4


def test_matching_caliper_drops_far_matches():
    X, _, treatment, outcome, _ = simulate_observational_data(
        n=6000, confounding=2.0, effect_heterogeneity=3.0, seed=17
    )
    estimate = propensity_matching(
        X, treatment, outcome, caliper=0.1, with_replacement=True
    )
    assert np.isfinite(estimate)


def test_matching_caliper_too_small_raises():
    X, _, treatment, outcome, _ = simulate_observational_data(
        n=2000, confounding=2.0, seed=19
    )
    with pytest.raises(ValueError):
        propensity_matching(X, treatment, outcome, caliper=1e-9)


def test_matching_requires_both_groups():
    X = np.zeros((10, 1))
    with pytest.raises(ValueError):
        propensity_matching(X, np.ones(10), np.zeros(10))


def test_matching_single_treated_unit():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    treatment = np.array([1.0, 0.0, 0.0, 0.0])
    outcome = np.array([5.0, 1.0, 2.0, 3.0])
    estimate = propensity_matching(X, treatment, outcome)
    assert np.isfinite(estimate)


def test_matching_multiple_neighbors():
    X, _, treatment, outcome, _ = simulate_observational_data(
        n=12000, confounding=1.0, effect_heterogeneity=3.0, seed=23
    )
    true_att = 2.0 + 3.0 * X[treatment == 1, 0].mean()
    estimate = propensity_matching(
        X, treatment, outcome, with_replacement=True, n_neighbors=5
    )
    assert abs(estimate - true_att) < 0.5


def test_matching_ate_with_replacement_recovers_ate():
    X, _, treatment, outcome, true_ate = simulate_observational_data(
        n=12000, confounding=1.0, seed=29
    )
    estimate = propensity_matching(
        X, treatment, outcome, estimand="ate", with_replacement=True
    )
    assert abs(estimate - true_ate) < 0.5


def test_matching_ate_requires_with_replacement():
    X, _, treatment, outcome, _ = simulate_observational_data(n=500, seed=31)
    with pytest.raises(ValueError):
        propensity_matching(X, treatment, outcome, estimand="ate")


def test_matching_rejects_bad_n_neighbors():
    X, _, treatment, outcome, _ = simulate_observational_data(n=200, seed=37)
    with pytest.raises(ValueError):
        propensity_matching(X, treatment, outcome, n_neighbors=0)


def test_matching_uses_explicit_propensity():
    X, _, treatment, outcome, _ = simulate_observational_data(n=1000, seed=41)
    p, _ = propensity_scores(X, treatment)
    with_p = propensity_matching(X, treatment, outcome, propensity=p)
    without_p = propensity_matching(X, treatment, outcome)
    assert with_p == pytest.approx(without_p, abs=1e-8)


def test_did_recovers_effect_with_parallel_trends():
    group, period, outcome, true_did = simulate_did_data(n=2000, seed=2)
    estimate = difference_in_differences(group, period, outcome)
    assert abs(estimate - true_did) < 0.2


def test_did_zero_effect():
    group, period, outcome, _ = simulate_did_data(n=2000, ate=0.0, seed=4)
    estimate = difference_in_differences(group, period, outcome)
    assert abs(estimate) < 0.2


def test_did_negative_effect():
    group, period, outcome, true_did = simulate_did_data(n=2000, ate=-1.5, seed=6)
    estimate = difference_in_differences(group, period, outcome)
    assert abs(estimate - true_did) < 0.2


def test_did_removes_time_trend_bias():
    group, period, outcome, true_did = simulate_did_data(
        n=2000, ate=1.0, time_trend=5.0, seed=8
    )
    naive_pre_post = outcome[period == 1].mean() - outcome[period == 0].mean()
    estimate = difference_in_differences(group, period, outcome)
    assert abs(naive_pre_post - true_did) > 1.0
    assert abs(estimate - true_did) < 0.2


def test_did_empty_cell_raises():
    group = np.array([1.0, 1.0, 1.0, 1.0])
    period = np.array([0.0, 1.0, 0.0, 1.0])
    outcome = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        difference_in_differences(group, period, outcome)


def test_did_rejects_invalid_values():
    group = np.array([1.0, 2.0, 0.0, 0.0])
    period = np.array([0.0, 1.0, 0.0, 1.0])
    outcome = np.arange(4.0)
    with pytest.raises(ValueError):
        difference_in_differences(group, period, outcome)


def test_did_length_mismatch_raises():
    with pytest.raises(ValueError):
        difference_in_differences(np.zeros(3), np.zeros(4), np.zeros(3))
