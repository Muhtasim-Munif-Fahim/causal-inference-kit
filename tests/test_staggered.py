"""Tests for staggered DiD group-time ATT."""

from __future__ import annotations

import numpy as np
import pytest

from causal_inference.staggered import (
    simulate_staggered_did_data,
    staggered_did,
)


def test_staggered_recovers_constant_att():
    unit, time, y, cohort, true = simulate_staggered_did_data(
        n_units=200, n_times=6, cohorts=(3, 4), att=2.0, seed=1
    )
    res = staggered_did(unit, time, y, cohort, control_group="notyet")
    assert abs(res.overall_att_weighted - true) < 0.35
    assert res.n_cohorts == 2
    assert len(res.group_time) >= 2


def test_staggered_never_treated_control():
    unit, time, y, cohort, true = simulate_staggered_did_data(
        n_units=150, n_times=5, cohorts=(3,), att=1.5, seed=2
    )
    res = staggered_did(unit, time, y, cohort, control_group="never")
    assert abs(res.overall_att_weighted - true) < 0.4


def test_staggered_bootstrap_se():
    unit, time, y, cohort, _ = simulate_staggered_did_data(
        n_units=80, n_times=5, cohorts=(3, 4), att=1.0, seed=3
    )
    res = staggered_did(
        unit, time, y, cohort, control_group="notyet", bootstrap=30, seed=0
    )
    assert res.overall_se is not None
    assert res.overall_se > 0


def test_staggered_rejects_bad_control():
    unit, time, y, cohort, _ = simulate_staggered_did_data(n_units=40, seed=4)
    with pytest.raises(ValueError, match="control_group"):
        staggered_did(unit, time, y, cohort, control_group="always")
