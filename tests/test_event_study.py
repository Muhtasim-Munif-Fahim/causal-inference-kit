"""Tests for event-study / dynamic difference-in-differences."""

from __future__ import annotations

import numpy as np
import pytest

from causal_inference import (
    EventStudyResult,
    difference_in_differences,
    event_study_did,
    simulate_did_data,
)


def test_event_study_exported() -> None:
    assert callable(event_study_did)
    assert EventStudyResult is not None


def test_event_study_recovers_flat_post_effect() -> None:
    unit, group, period, outcome, true_did = simulate_did_data(
        n=800, ate=2.0, n_pre=3, n_post=3, seed=3
    )
    result = event_study_did(
        unit, group, period, outcome, treatment_time=3, reference=-1
    )
    assert isinstance(result, EventStudyResult)
    assert result.reference == -1
    assert result.treatment_time == 3.0
    assert result.n_times == 6
    # Pre-treatment leads (except reference) near zero; post lags near ATT.
    for k, coef in zip(result.relative_times, result.coefficients):
        if k < 0:
            assert abs(coef) < 0.35, f"lead k={k} coef={coef}"
        else:
            assert abs(coef - true_did) < 0.35, f"lag k={k} coef={coef}"


def test_event_study_two_period_matches_did_point_estimate() -> None:
    unit, group, period, outcome, true_did = simulate_did_data(
        n=600, ate=1.5, n_pre=1, n_post=1, seed=5
    )
    # treatment_time defaults to the post period (1). Relative times: -1 (ref), 0.
    did = difference_in_differences(group, period, outcome, unit=unit)
    es = event_study_did(unit, group, period, outcome, reference=-1)
    assert list(es.relative_times) == [0]
    assert es.coefficients[0] == pytest.approx(did.estimate, abs=1e-8)
    assert abs(es.coefficients[0] - true_did) < 0.25


def test_event_study_as_dict_and_ses_positive() -> None:
    unit, group, period, outcome, _ = simulate_did_data(
        n=400, ate=1.0, n_pre=2, n_post=2, seed=1
    )
    result = event_study_did(
        unit, group, period, outcome, treatment_time=2, cluster=True
    )
    mapping = result.as_dict()
    assert set(mapping) == set(int(k) for k in result.relative_times)
    assert result.se_type == "cluster"
    assert np.all(result.ses >= 0.0)
    assert result.n == outcome.shape[0]
    assert result.n_units == len(np.unique(unit))


def test_event_study_hc1_option() -> None:
    unit, group, period, outcome, _ = simulate_did_data(
        n=300, ate=0.5, n_pre=2, n_post=2, seed=8
    )
    clustered = event_study_did(
        unit, group, period, outcome, treatment_time=2, cluster=True
    )
    robust = event_study_did(
        unit, group, period, outcome, treatment_time=2, cluster=False
    )
    np.testing.assert_allclose(clustered.coefficients, robust.coefficients)
    assert robust.se_type == "hc1"
    assert clustered.se_type == "cluster"


def test_event_study_requires_treatment_time_for_multi_period() -> None:
    unit, group, period, outcome, _ = simulate_did_data(
        n=100, n_pre=2, n_post=2, seed=0
    )
    with pytest.raises(ValueError, match="treatment_time is required"):
        event_study_did(unit, group, period, outcome)


def test_event_study_rejects_bad_reference() -> None:
    unit, group, period, outcome, _ = simulate_did_data(
        n=100, n_pre=2, n_post=2, seed=0
    )
    with pytest.raises(ValueError, match="reference relative time"):
        event_study_did(
            unit, group, period, outcome, treatment_time=2, reference=99
        )


def test_event_study_zero_effect_pre_and_post_near_zero() -> None:
    unit, group, period, outcome, _ = simulate_did_data(
        n=700, ate=0.0, n_pre=3, n_post=3, seed=11
    )
    result = event_study_did(
        unit, group, period, outcome, treatment_time=3, reference=-1
    )
    assert np.max(np.abs(result.coefficients)) < 0.3
