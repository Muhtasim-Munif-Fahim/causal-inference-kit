"""Tests for the synthetic data generator."""

import numpy as np
import pytest

from causal_inference.generators import (
    simulate_did_data,
    simulate_iv_data,
    simulate_observational_data,
    simulate_rd_data,
    simulate_synthetic_control_data,
)


def test_returns_expected_shapes():
    X, W, treatment, outcome, true_ate = simulate_observational_data(
        n=500, d_x=3, d_w=2, seed=1
    )
    assert X.shape == (500, 3)
    assert W.shape == (500, 2)
    assert treatment.shape == (500,)
    assert outcome.shape == (500,)
    assert np.isfinite(true_ate)


def test_seed_reproducibility():
    a = simulate_observational_data(seed=7)
    b = simulate_observational_data(seed=7)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)


def test_different_seeds_differ():
    a = simulate_observational_data(seed=1)
    b = simulate_observational_data(seed=2)
    assert not np.array_equal(a[0], b[0])
    assert not np.array_equal(a[3], b[3])


def test_treatment_has_both_groups():
    _, _, treatment, _, _ = simulate_observational_data(n=2000, seed=3)
    assert set(np.unique(treatment)) == {0.0, 1.0}
    share = treatment.mean()
    assert 0.2 < share < 0.8


def test_first_covariate_of_each_block_is_binary():
    X, W, _, _, _ = simulate_observational_data(d_x=4, d_w=3, seed=5)
    assert set(np.unique(X[:, 0])) <= {0.0, 1.0}
    assert set(np.unique(W[:, 0])) <= {0.0, 1.0}


def test_true_ate_equals_requested_ate_without_heterogeneity():
    for seed in range(5):
        _, _, _, _, true_ate = simulate_observational_data(
            ate=2.5, effect_heterogeneity=0.0, seed=seed
        )
        assert true_ate == pytest.approx(2.5)


def test_true_ate_uses_first_confounder_when_heterogeneous():
    X, _, _, _, true_ate = simulate_observational_data(
        ate=1.0, effect_heterogeneity=4.0, seed=11
    )
    expected = 1.0 + 4.0 * X[:, 0].mean()
    assert true_ate == pytest.approx(expected)


def test_true_att_differs_from_ate_under_selection_on_effect_modifier():
    X, _, treatment, _, true_ate = simulate_observational_data(
        ate=1.0, effect_heterogeneity=5.0, confounding=2.0, n=5000, seed=13
    )
    true_att = 1.0 + 5.0 * X[treatment == 1, 0].mean()
    assert abs(true_att - true_ate) > 0.1


def test_confounding_creates_imbalance():
    X, _, treatment, _, _ = simulate_observational_data(
        confounding=3.0, n=5000, seed=17
    )
    d = X[treatment == 1].mean(axis=0) - X[treatment == 0].mean(axis=0)
    assert np.abs(d).max() > 0.15


def test_no_confounding_gives_balanced_groups():
    X, _, treatment, _, _ = simulate_observational_data(
        confounding=0.0, n=5000, seed=19
    )
    d = X[treatment == 1].mean(axis=0) - X[treatment == 0].mean(axis=0)
    assert np.abs(d).max() < 0.15


def test_outcome_only_covariates_balanced_across_groups():
    _, W, treatment, _, _ = simulate_observational_data(
        confounding=3.0, n=5000, seed=23
    )
    d = W[treatment == 1].mean(axis=0) - W[treatment == 0].mean(axis=0)
    assert np.abs(d).max() < 0.15


def test_selection_bias_inflates_naive_estimator_error():
    naive_errors = {0.0: [], 3.0: []}
    for seed in range(8):
        for sb in (0.0, 3.0):
            _, _, treatment, outcome, true_ate = simulate_observational_data(
                n=8000,
                confounding=0.0,
                selection_bias=sb,
                effect_heterogeneity=0.0,
                seed=seed,
            )
            naive = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
            naive_errors[sb].append(abs(naive - true_ate))
    assert np.mean(naive_errors[3.0]) > np.mean(naive_errors[0.0])


def test_zero_ate_dataset():
    _, _, treatment, outcome, true_ate = simulate_observational_data(
        ate=0.0, effect_heterogeneity=0.0, seed=29
    )
    assert true_ate == pytest.approx(0.0)
    assert np.isfinite(outcome).all()


def test_small_n():
    X, W, treatment, outcome, true_ate = simulate_observational_data(n=10, seed=31)
    assert X.shape[0] == 10
    assert X.shape[1] == 3
    assert treatment.sum() >= 0
    assert np.isfinite(true_ate)


def test_zero_covariate_dimensions():
    X, W, treatment, outcome, true_ate = simulate_observational_data(
        d_x=0, d_w=0, n=100, seed=37
    )
    assert X.shape == (100, 0)
    assert W.shape == (100, 0)
    assert np.isfinite(outcome).all()
    assert np.isfinite(true_ate)


def test_zero_covariate_dimensions_balanced_treatment():
    _, _, treatment, _, _ = simulate_observational_data(
        d_x=0, d_w=0, n=5000, seed=41
    )
    assert 0.4 < treatment.mean() < 0.6


def test_outcome_is_continuous_and_finite():
    for seed in range(4):
        _, _, _, outcome, _ = simulate_observational_data(seed=seed)
        assert np.isfinite(outcome).all()
        assert len(np.unique(outcome)) > 10


def test_did_shapes():
    unit, group, period, outcome, true_did = simulate_did_data(n=300, seed=1)
    assert unit.shape == (600,)
    assert group.shape == (600,)
    assert period.shape == (600,)
    assert outcome.shape == (600,)
    assert true_did == pytest.approx(1.5)
    assert set(np.unique(period)) == {0.0, 1.0}
    assert set(np.unique(unit)) == set(range(300))


def test_did_seed_reproducibility():
    a = simulate_did_data(n=200, seed=5)
    b = simulate_did_data(n=200, seed=5)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)


def test_did_has_both_groups_in_both_periods():
    _, group, period, _, _ = simulate_did_data(n=500, seed=9)
    for g in (0.0, 1.0):
        for p in (0.0, 1.0):
            assert ((group == g) & (period == p)).sum() > 100


def test_did_time_trend_shifts_both_groups():
    _, group, period, outcome, _ = simulate_did_data(
        n=2000, ate=0.0, time_trend=2.0, seed=11
    )
    for g in (0.0, 1.0):
        pre = outcome[(group == g) & (period == 0)].mean()
        post = outcome[(group == g) & (period == 1)].mean()
        assert post - pre > 1.5


def test_did_multi_period_shapes_and_adoption():
    unit, group, period, outcome, true_did = simulate_did_data(
        n=80, n_pre=3, n_post=2, ate=2.5, seed=13
    )
    assert unit.shape == (80 * 5,)
    assert set(np.unique(period)) == {0.0, 1.0, 2.0, 3.0, 4.0}
    assert true_did == pytest.approx(2.5)
    treated_post = (group == 1) & (period >= 3)
    treated_pre = (group == 1) & (period < 3)
    control_post = (group == 0) & (period >= 3)
    control_pre = (group == 0) & (period < 3)
    assert treated_post.sum() > 0
    assert treated_pre.sum() > 0
    assert control_post.sum() > 0
    assert control_pre.sum() > 0
    assert np.isfinite(outcome).all()


def test_did_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        simulate_did_data(n=0)
    with pytest.raises(ValueError):
        simulate_did_data(n_pre=0)
    with pytest.raises(ValueError):
        simulate_did_data(n_post=0)


def test_sc_shapes_and_ids():
    unit, time, outcome, treated_unit, treatment_time, true_effect = (
        simulate_synthetic_control_data(n_donors=5, n_pre=6, n_post=4, ate=3.5, seed=1)
    )
    n_obs = 6 * 10
    assert unit.shape == (n_obs,)
    assert time.shape == (n_obs,)
    assert outcome.shape == (n_obs,)
    assert treated_unit == 0
    assert treatment_time == 6
    assert true_effect == pytest.approx(3.5)
    assert set(np.unique(unit)) == set(range(6))
    assert set(np.unique(time)) == set(range(10))


def test_sc_seed_reproducibility():
    a = simulate_synthetic_control_data(n_donors=4, seed=5)
    b = simulate_synthetic_control_data(n_donors=4, seed=5)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)


def test_sc_balanced_panel():
    unit, time, outcome, _, _, _ = simulate_synthetic_control_data(
        n_donors=4, n_pre=5, n_post=3, seed=9
    )
    for u in np.unique(unit):
        assert set(time[unit == u]) == set(range(8))
    assert np.isfinite(outcome).all()


def test_sc_post_shift_on_treated_unit():
    unit, time, outcome, treated, t0, ate = simulate_synthetic_control_data(
        n_donors=6, n_pre=10, n_post=6, ate=7.0, noise=0.0, seed=11
    )
    treated_pre = outcome[(unit == treated) & (time < t0)].mean()
    treated_post = outcome[(unit == treated) & (time >= t0)].mean()
    donor_pre = outcome[(unit != treated) & (time < t0)].mean()
    donor_post = outcome[(unit != treated) & (time >= t0)].mean()
    assert (treated_post - treated_pre) - (donor_post - donor_pre) == pytest.approx(ate, abs=0.5)


def test_sc_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        simulate_synthetic_control_data(n_donors=0)
    with pytest.raises(ValueError):
        simulate_synthetic_control_data(n_pre=0)
    with pytest.raises(ValueError):
        simulate_synthetic_control_data(n_post=0)


def test_rd_shapes_and_assignment():
    running, treatment, outcome, cutoff, true_effect = simulate_rd_data(
        n=500, cutoff=0.5, ate=3.0, seed=1
    )
    assert running.shape == (500,)
    assert treatment.shape == (500,)
    assert outcome.shape == (500,)
    assert cutoff == pytest.approx(0.5)
    assert true_effect == pytest.approx(3.0)
    assert set(np.unique(treatment)) == {0.0, 1.0}
    np.testing.assert_array_equal(treatment, (running >= cutoff).astype(float))
    assert np.isfinite(outcome).all()


def test_rd_seed_reproducibility():
    a = simulate_rd_data(n=400, seed=5)
    b = simulate_rd_data(n=400, seed=5)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)


def test_rd_both_sides_nonempty():
    running, treatment, _, cutoff, _ = simulate_rd_data(n=2000, seed=9)
    assert (running < cutoff).sum() > 200
    assert (running >= cutoff).sum() > 200
    assert 0.3 < treatment.mean() < 0.7


def test_rd_noise_free_matches_polynomial():
    running, treatment, outcome, cutoff, ate = simulate_rd_data(
        n=300,
        cutoff=0.0,
        ate=2.0,
        noise=0.0,
        slope=1.5,
        slope_jump=0.5,
        curvature=0.25,
        seed=11,
    )
    z = running - cutoff
    expected = (
        1.5 * z + 0.5 * treatment * z + 0.25 * z ** 2 + ate * treatment
    )
    np.testing.assert_allclose(outcome, expected)


def test_rd_rejects_bad_arguments():
    with pytest.raises(ValueError):
        simulate_rd_data(n=0)
    with pytest.raises(ValueError):
        simulate_rd_data(spread=0.0)
    with pytest.raises(ValueError):
        simulate_rd_data(noise=-0.1)


def test_iv_shapes_and_one_sided_noncompliance():
    instrument, treatment, outcome, covariates, true_effect = simulate_iv_data(
        n=800, late=2.5, instrument_prob=0.4, compliance=0.5, n_covariates=3, seed=1
    )
    assert instrument.shape == (800,)
    assert treatment.shape == (800,)
    assert outcome.shape == (800,)
    assert covariates.shape == (800, 3)
    assert true_effect == pytest.approx(2.5)
    assert set(np.unique(instrument)) == {0.0, 1.0}
    assert set(np.unique(treatment)) <= {0.0, 1.0}
    assert np.all(treatment[instrument == 0] == 0)
    assert np.isfinite(outcome).all()
    assert treatment[instrument == 1].mean() == pytest.approx(0.5, abs=0.08)


def test_iv_seed_reproducibility():
    a = simulate_iv_data(n=400, n_covariates=2, seed=5)
    b = simulate_iv_data(n=400, n_covariates=2, seed=5)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)


def test_iv_different_seeds_differ():
    a = simulate_iv_data(n=400, seed=1)
    b = simulate_iv_data(n=400, seed=2)
    assert not np.array_equal(a[2], b[2])


def test_iv_noise_free_outcome_is_late_times_treatment():
    instrument, treatment, outcome, covariates, late = simulate_iv_data(
        n=300, late=4.0, confounding=0.0, noise=0.0, seed=11
    )
    assert covariates.shape == (300, 0)
    np.testing.assert_allclose(outcome, late * treatment)
    assert np.any(instrument == 0)
    assert np.any(instrument == 1)


def test_iv_covariates_shift_outcome():
    _, _, outcome, covariates, _ = simulate_iv_data(
        n=200,
        late=0.0,
        confounding=0.0,
        noise=0.0,
        n_covariates=2,
        covariate_effect=2.0,
        seed=13,
    )
    np.testing.assert_allclose(outcome, 2.0 * covariates.sum(axis=1))


def test_iv_full_compliance_sets_treatment_equal_to_instrument():
    instrument, treatment, _, _, late = simulate_iv_data(
        n=100, late=1.0, compliance=1.0, seed=15
    )
    np.testing.assert_array_equal(treatment, instrument)
    assert late == pytest.approx(1.0)


def test_iv_rejects_bad_arguments():
    with pytest.raises(ValueError):
        simulate_iv_data(n=1)
    with pytest.raises(ValueError):
        simulate_iv_data(instrument_prob=0.0)
    with pytest.raises(ValueError):
        simulate_iv_data(instrument_prob=1.0)
    with pytest.raises(ValueError):
        simulate_iv_data(compliance=0.0)
    with pytest.raises(ValueError):
        simulate_iv_data(compliance=1.1)
    with pytest.raises(ValueError):
        simulate_iv_data(noise=-0.1)
    with pytest.raises(ValueError):
        simulate_iv_data(n_covariates=-1)
