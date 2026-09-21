"""Tests for treatment-effect estimators."""

import numpy as np
import pytest

from causal_inference.estimators import (
    DifferenceInDifferencesResult,
    aipw_ate,
    difference_in_differences,
    difference_in_means,
    ipw_ate,
    ipw_att,
    ipw_weights,
    outcome_regression,
    propensity_matching,
    regression_discontinuity,
    synthetic_control,
)
from causal_inference.generators import (
    simulate_did_data,
    simulate_observational_data,
    simulate_rd_data,
    simulate_synthetic_control_data,
)
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
    unit, group, period, outcome, true_did = simulate_did_data(n=2000, seed=2)
    result = difference_in_differences(group, period, outcome, unit=unit)
    assert abs(result.estimate - true_did) < 0.2
    assert result.se > 0
    assert result.method == "2x2"
    assert result.se_type == "cluster"
    assert result.n_clusters == 2000


def test_did_zero_effect():
    unit, group, period, outcome, _ = simulate_did_data(n=2000, ate=0.0, seed=4)
    result = difference_in_differences(group, period, outcome, unit=unit)
    assert abs(result.estimate) < 0.2


def test_did_negative_effect():
    unit, group, period, outcome, true_did = simulate_did_data(n=2000, ate=-1.5, seed=6)
    result = difference_in_differences(group, period, outcome, unit=unit)
    assert abs(result.estimate - true_did) < 0.2


def test_did_removes_time_trend_bias():
    unit, group, period, outcome, true_did = simulate_did_data(
        n=2000, ate=1.0, time_trend=5.0, seed=8
    )
    naive_pre_post = outcome[period == 1].mean() - outcome[period == 0].mean()
    result = difference_in_differences(group, period, outcome, unit=unit)
    assert abs(naive_pre_post - true_did) > 1.0
    assert abs(result.estimate - true_did) < 0.2


def test_did_repeated_cross_section_robust_se():
    _, group, period, outcome, true_did = simulate_did_data(n=2000, seed=2)
    result = difference_in_differences(group, period, outcome)
    assert abs(result.estimate - true_did) < 0.2
    assert result.se > 0
    assert result.se_type == "hc1"
    assert result.n_units is None
    assert result.n_clusters is None


def test_did_clustered_matches_four_cell_point_estimate():
    unit, group, period, outcome, _ = simulate_did_data(n=800, seed=10)
    clustered = difference_in_differences(group, period, outcome, unit=unit)
    robust = difference_in_differences(group, period, outcome)
    assert clustered.estimate == pytest.approx(robust.estimate, abs=1e-10)
    assert clustered.se_type == "cluster"
    assert robust.se_type == "hc1"


def test_did_hand_calculation():
    group = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    period = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    outcome = np.array([1.0, 4.0, 2.0, 6.0, 0.0, 1.0, 1.0, 3.0])
    unit = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    result = difference_in_differences(group, period, outcome, unit=unit)
    assert result.estimate == pytest.approx(2.0)
    assert result.se == pytest.approx(np.sqrt(0.5))
    assert result.treated_mean_pre == pytest.approx(1.5)
    assert result.treated_mean_post == pytest.approx(5.0)
    assert result.control_mean_pre == pytest.approx(0.5)
    assert result.control_mean_post == pytest.approx(2.0)
    assert isinstance(result, DifferenceInDifferencesResult)


def test_did_hand_calculation_without_unit():
    group = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    period = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    outcome = np.array([1.0, 4.0, 2.0, 6.0, 0.0, 1.0, 1.0, 3.0])
    result = difference_in_differences(group, period, outcome)
    assert result.estimate == pytest.approx(2.0)
    # four independent cells: vars 0.5, 2.0, 0.5, 2.0 with n=2
    assert result.se == pytest.approx(np.sqrt(2.5))


def test_did_twfe_recovers_effect_on_multi_period_panel():
    unit, group, period, outcome, true_did = simulate_did_data(
        n=400, n_pre=3, n_post=3, ate=2.0, time_trend=1.0, seed=11
    )
    result = difference_in_differences(
        group, period, outcome, unit=unit, treatment_time=3
    )
    assert result.method == "twfe"
    assert result.se_type == "cluster"
    assert result.n_times == 6
    assert result.n_units == 400
    assert abs(result.estimate - true_did) < 0.25
    assert result.se > 0
    assert abs(result.estimate - true_did) < 4.0 * result.se


def test_did_twfe_matches_2x2_when_two_periods():
    unit, group, period, outcome, _ = simulate_did_data(n=500, seed=13)
    two_by_two = difference_in_differences(group, period, outcome, unit=unit)
    twfe = difference_in_differences(
        group, period, outcome, unit=unit, cluster=False
    )
    assert two_by_two.estimate == pytest.approx(twfe.estimate, abs=1e-8)


def test_did_twfe_explicit_treatment_indicator():
    unit, group, period, outcome, true_did = simulate_did_data(
        n=300, n_pre=2, n_post=2, ate=1.5, seed=17
    )
    treatment = ((group == 1) & (period >= 2)).astype(float)
    result = difference_in_differences(
        group, period, outcome, unit=unit, treatment=treatment
    )
    assert result.method == "twfe"
    assert abs(result.estimate - true_did) < 0.3


def test_did_twfe_zero_and_negative_effects():
    for ate, seed in ((0.0, 19), (-2.0, 23)):
        unit, group, period, outcome, true_did = simulate_did_data(
            n=350, n_pre=3, n_post=2, ate=ate, seed=seed
        )
        result = difference_in_differences(
            group, period, outcome, unit=unit, treatment_time=3
        )
        assert abs(result.estimate - true_did) < 0.3


def test_did_cluster_without_unit_raises():
    _, group, period, outcome, _ = simulate_did_data(n=50, seed=1)
    with pytest.raises(ValueError, match="clustered"):
        difference_in_differences(group, period, outcome, cluster=True)


def test_did_multi_period_without_unit_raises():
    _, group, period, outcome, _ = simulate_did_data(
        n=40, n_pre=2, n_post=2, seed=1
    )
    with pytest.raises(ValueError, match="unit identifiers"):
        difference_in_differences(group, period, outcome)


def test_did_multi_period_requires_treatment_time():
    unit, group, period, outcome, _ = simulate_did_data(
        n=40, n_pre=2, n_post=2, seed=1
    )
    with pytest.raises(ValueError, match="treatment_time"):
        difference_in_differences(group, period, outcome, unit=unit)


def test_did_collinear_treatment_raises():
    unit = np.array([0, 0, 0, 1, 1, 1])
    group = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    period = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    outcome = np.arange(6.0)
    with pytest.raises(ValueError, match="collinear"):
        difference_in_differences(
            group, period, outcome, unit=unit, treatment_time=1
        )


def test_did_group_must_be_constant_within_unit():
    unit = np.array([0, 0, 1, 1])
    group = np.array([1.0, 1.0, 1.0, 0.0])
    period = np.array([0.0, 1.0, 0.0, 1.0])
    outcome = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="constant within unit"):
        difference_in_differences(group, period, outcome, unit=unit)


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
    with pytest.raises(ValueError):
        difference_in_differences(
            np.zeros(4), np.zeros(4), np.zeros(4), unit=np.zeros(3)
        )


def _sc_long(treated, donors, times=None):
    """Stack a treated path and donor matrix into long unit/time/outcome."""
    donors = np.asarray(donors, dtype=float)
    treated = np.asarray(treated, dtype=float).ravel()
    n_times = treated.size
    if times is None:
        times = np.arange(n_times)
    unit = np.concatenate(
        [np.repeat(0, n_times), np.repeat(np.arange(1, donors.shape[0] + 1), n_times)]
    )
    time = np.tile(times, donors.shape[0] + 1)
    outcome = np.concatenate([treated, donors.reshape(-1)])
    return unit, time, outcome


def test_synthetic_control_hand_calculation():
    treated = np.array([2.0, 4.0, 10.0])
    donors = np.array([[2.0, 4.0, 5.0], [0.0, 0.0, 0.0]])
    unit, time, outcome = _sc_long(treated, donors)
    result = synthetic_control(unit, time, outcome, treated_unit=0, treatment_time=2)
    assert result.estimate == pytest.approx(5.0)
    np.testing.assert_allclose(result.weights, [1.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(result.gap, [0.0, 0.0, 5.0], atol=1e-8)
    assert result.pre_rmspe == pytest.approx(0.0, abs=1e-8)
    assert result.placebo_p_value is None


def test_synthetic_control_convex_combination():
    treated = np.array([3.0, 3.0, 10.0])
    donors = np.array([[6.0, 6.0, 8.0], [0.0, 0.0, 4.0]])
    unit, time, outcome = _sc_long(treated, donors)
    result = synthetic_control(unit, time, outcome, treated_unit=0, treatment_time=2)
    np.testing.assert_allclose(result.weights, [0.5, 0.5], atol=1e-6)
    assert result.estimate == pytest.approx(4.0)
    np.testing.assert_allclose(result.synthetic_outcome, [3.0, 3.0, 6.0], atol=1e-6)


def test_synthetic_control_weights_nonnegative_and_sum_to_one():
    treated = np.array([4.0, 4.0, 10.0])
    donors = np.array([[3.0, 3.0, 7.0], [2.0, 2.0, 1.0]])
    unit, time, outcome = _sc_long(treated, donors)
    result = synthetic_control(unit, time, outcome, treated_unit=0, treatment_time=2)
    assert np.all(result.weights >= -1e-12)
    assert result.weights.sum() == pytest.approx(1.0)
    assert result.weights[0] > result.weights[1]


def test_synthetic_control_recovers_effect_on_factor_panel():
    unit, time, outcome, treated, t0, true_effect = simulate_synthetic_control_data(
        n_donors=6, n_pre=15, n_post=8, ate=5.0, noise=0.05, seed=11
    )
    result = synthetic_control(unit, time, outcome, treated, t0)
    assert abs(result.estimate - true_effect) < 0.5
    assert result.pre_rmspe < 0.5
    assert np.all(result.weights >= -1e-12)
    assert result.weights.sum() == pytest.approx(1.0)


def test_synthetic_control_zero_effect():
    unit, time, outcome, treated, t0, _ = simulate_synthetic_control_data(
        n_donors=6, n_pre=15, n_post=8, ate=0.0, noise=0.05, seed=13
    )
    result = synthetic_control(unit, time, outcome, treated, t0)
    assert abs(result.estimate) < 0.5


def test_synthetic_control_negative_effect():
    unit, time, outcome, treated, t0, true_effect = simulate_synthetic_control_data(
        n_donors=6, n_pre=15, n_post=8, ate=-3.0, noise=0.05, seed=17
    )
    result = synthetic_control(unit, time, outcome, treated, t0)
    assert abs(result.estimate - true_effect) < 0.5


def test_synthetic_control_noise_free_recovers_exact_effect():
    unit, time, outcome, treated, t0, true_effect = simulate_synthetic_control_data(
        n_donors=5, n_pre=12, n_post=6, ate=4.0, noise=0.0, seed=19
    )
    result = synthetic_control(unit, time, outcome, treated, t0)
    assert result.estimate == pytest.approx(true_effect, abs=0.05)
    assert result.pre_rmspe == pytest.approx(0.0, abs=0.05)


def test_synthetic_control_placebo_ranks_treated_when_effect_is_large():
    unit, time, outcome, treated, t0, _ = simulate_synthetic_control_data(
        n_donors=6, n_pre=12, n_post=8, ate=8.0, noise=0.0, seed=23
    )
    result = synthetic_control(unit, time, outcome, treated, t0, placebo=True)
    assert result.placebo_p_value == pytest.approx(1.0 / 7.0)
    assert result.placebo_estimates.shape == (6,)
    assert result.placebo_ratios.shape == (6,)
    treated_ratio = result.post_rmspe / result.pre_rmspe if result.pre_rmspe > 1e-15 else np.inf
    assert np.all(result.placebo_ratios <= treated_ratio + 1e-12)


def test_synthetic_control_placebo_requires_two_donors():
    treated = np.array([1.0, 2.0, 4.0])
    donors = np.array([[1.0, 2.0, 2.0]])
    unit, time, outcome = _sc_long(treated, donors)
    with pytest.raises(ValueError):
        synthetic_control(unit, time, outcome, treated_unit=0, treatment_time=2, placebo=True)


def test_synthetic_control_rejects_missing_treated_unit():
    unit, time, outcome, _, t0, _ = simulate_synthetic_control_data(n_donors=3, seed=3)
    with pytest.raises(ValueError):
        synthetic_control(unit, time, outcome, treated_unit=99, treatment_time=t0)


def test_synthetic_control_rejects_no_pre_period():
    unit, time, outcome, treated, _, _ = simulate_synthetic_control_data(n_donors=3, seed=5)
    with pytest.raises(ValueError):
        synthetic_control(unit, time, outcome, treated, treatment_time=-1)


def test_synthetic_control_rejects_no_post_period():
    unit, time, outcome, treated, _, _ = simulate_synthetic_control_data(
        n_donors=3, n_pre=4, n_post=4, seed=7
    )
    with pytest.raises(ValueError):
        synthetic_control(unit, time, outcome, treated, treatment_time=100)


def test_synthetic_control_rejects_unbalanced_panel():
    unit = np.array([0, 0, 1])
    time = np.array([0, 1, 0])
    outcome = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        synthetic_control(unit, time, outcome, treated_unit=0, treatment_time=1)


def test_synthetic_control_rejects_duplicate_observations():
    unit = np.array([0, 0, 1, 1, 0])
    time = np.array([0, 1, 0, 1, 0])
    outcome = np.arange(5.0)
    with pytest.raises(ValueError):
        synthetic_control(unit, time, outcome, treated_unit=0, treatment_time=1)


def test_synthetic_control_rejects_length_mismatch():
    with pytest.raises(ValueError):
        synthetic_control(np.zeros(3), np.zeros(4), np.zeros(3), 0, 1)


def test_synthetic_control_donor_ids_exclude_treated():
    unit, time, outcome, treated, t0, _ = simulate_synthetic_control_data(
        n_donors=4, seed=29
    )
    result = synthetic_control(unit, time, outcome, treated, t0)
    assert treated not in set(result.donor_ids.tolist())
    assert result.donor_ids.tolist() == [1, 2, 3, 4]
    assert result.weights.shape == (4,)


def test_rd_hand_calculation():
    running = np.array([-2.0, -1.0, 1.0, 2.0])
    outcome = np.array([0.0, 1.0, 5.0, 6.0])
    result = regression_discontinuity(
        running, outcome, cutoff=0.0, bandwidth=10.0, kernel="uniform"
    )
    assert result.estimate == pytest.approx(2.0)
    assert result.intercept_left == pytest.approx(2.0)
    assert result.intercept_right == pytest.approx(4.0)
    assert result.slope_left == pytest.approx(1.0)
    assert result.slope_right == pytest.approx(1.0)
    assert result.n_left == 2
    assert result.n_right == 2


def test_rd_kernel_does_not_change_exact_linear_fit():
    running = np.array([-2.0, -1.0, 1.0, 2.0])
    outcome = np.array([0.0, 1.0, 5.0, 6.0])
    for kernel in ("triangular", "uniform", "epanechnikov"):
        result = regression_discontinuity(
            running, outcome, cutoff=0.0, bandwidth=3.0, kernel=kernel
        )
        assert result.estimate == pytest.approx(2.0)
        assert result.kernel == kernel


def test_rd_recovers_effect_on_linear_dgp():
    running, _, outcome, cutoff, true_effect = simulate_rd_data(
        n=6000, ate=2.0, slope=1.0, noise=0.4, seed=11
    )
    result = regression_discontinuity(
        running, outcome, cutoff=cutoff, bandwidth=0.5, kernel="triangular"
    )
    assert abs(result.estimate - true_effect) < 0.2
    assert result.n_left >= 2
    assert result.n_right >= 2
    assert result.se > 0


def test_rd_zero_effect():
    running, _, outcome, cutoff, _ = simulate_rd_data(
        n=5000, ate=0.0, slope=1.5, noise=0.4, seed=13
    )
    result = regression_discontinuity(running, outcome, cutoff=cutoff, bandwidth=0.5)
    assert abs(result.estimate) < 0.2


def test_rd_negative_effect():
    running, _, outcome, cutoff, true_effect = simulate_rd_data(
        n=5000, ate=-1.5, slope=1.0, noise=0.4, seed=17
    )
    result = regression_discontinuity(running, outcome, cutoff=cutoff, bandwidth=0.5)
    assert abs(result.estimate - true_effect) < 0.2


def test_rd_improves_on_naive_when_slope_biases_means():
    running, treatment, outcome, cutoff, true_effect = simulate_rd_data(
        n=8000, ate=2.0, slope=2.0, noise=0.3, seed=19
    )
    naive = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
    result = regression_discontinuity(running, outcome, cutoff=cutoff, bandwidth=0.4)
    assert abs(naive - true_effect) > 0.8
    assert abs(result.estimate - true_effect) < 0.2


def test_rd_nonzero_cutoff():
    running, _, outcome, cutoff, true_effect = simulate_rd_data(
        n=5000, cutoff=3.0, ate=1.5, slope=1.0, noise=0.3, seed=23
    )
    result = regression_discontinuity(running, outcome, cutoff=cutoff, bandwidth=0.5)
    assert abs(result.estimate - true_effect) < 0.2
    assert result.cutoff == pytest.approx(3.0)


def test_rd_slope_jump_still_recovers_intercept():
    running, _, outcome, cutoff, true_effect = simulate_rd_data(
        n=6000, ate=2.0, slope=1.0, slope_jump=1.5, noise=0.3, seed=29
    )
    result = regression_discontinuity(running, outcome, cutoff=cutoff, bandwidth=0.5)
    assert abs(result.estimate - true_effect) < 0.25


def test_rd_automatic_bandwidth_recovers_effect():
    running, _, outcome, cutoff, true_effect = simulate_rd_data(
        n=6000, ate=2.0, slope=1.0, noise=0.4, seed=31
    )
    result = regression_discontinuity(running, outcome, cutoff=cutoff)
    assert result.bandwidth > 0
    assert np.isfinite(result.bandwidth)
    assert abs(result.estimate - true_effect) < 0.25


def test_rd_treated_below_cutoff():
    running = np.array([-2.0, -1.0, 1.0, 2.0])
    outcome = np.array([4.0, 3.0, 1.0, 2.0])
    result = regression_discontinuity(
        running,
        outcome,
        cutoff=0.0,
        bandwidth=10.0,
        kernel="uniform",
        treated_above=False,
    )
    assert result.estimate == pytest.approx(2.0)
    assert result.treated_above is False


def test_rd_validates_sharp_treatment():
    running, treatment, outcome, cutoff, true_effect = simulate_rd_data(
        n=800, ate=2.0, seed=37
    )
    result = regression_discontinuity(
        running, outcome, cutoff=cutoff, bandwidth=0.6, treatment=treatment
    )
    assert abs(result.estimate - true_effect) < 0.4
    flipped = 1.0 - treatment
    with pytest.raises(ValueError, match="sharp"):
        regression_discontinuity(
            running, outcome, cutoff=cutoff, bandwidth=0.6, treatment=flipped
        )


def test_rd_rejects_bad_bandwidth():
    running, _, outcome, cutoff, _ = simulate_rd_data(n=200, seed=41)
    with pytest.raises(ValueError):
        regression_discontinuity(running, outcome, cutoff=cutoff, bandwidth=0.0)
    with pytest.raises(ValueError):
        regression_discontinuity(running, outcome, cutoff=cutoff, bandwidth=-1.0)


def test_rd_rejects_unknown_kernel():
    running, _, outcome, cutoff, _ = simulate_rd_data(n=200, seed=43)
    with pytest.raises(ValueError):
        regression_discontinuity(running, outcome, cutoff=cutoff, bandwidth=0.5, kernel="cosine")


def test_rd_rejects_empty_side():
    running = np.array([-0.2, -0.1, 0.1, 0.2])
    outcome = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        regression_discontinuity(running, outcome, cutoff=0.0, bandwidth=0.05)


def test_rd_rejects_length_mismatch():
    with pytest.raises(ValueError):
        regression_discontinuity(np.zeros(3), np.zeros(4), cutoff=0.0, bandwidth=1.0)


def test_rd_rejects_nonfinite_running():
    running = np.array([-1.0, np.nan, 1.0, 2.0])
    outcome = np.arange(4.0)
    with pytest.raises(ValueError):
        regression_discontinuity(running, outcome, cutoff=0.0, bandwidth=2.0)


def test_rd_rectangular_alias_is_uniform():
    running = np.array([-2.0, -1.0, 1.0, 2.0])
    outcome = np.array([0.0, 1.0, 5.0, 6.0])
    result = regression_discontinuity(
        running, outcome, cutoff=0.0, bandwidth=10.0, kernel="rectangular"
    )
    assert result.kernel == "uniform"
    assert result.estimate == pytest.approx(2.0)
