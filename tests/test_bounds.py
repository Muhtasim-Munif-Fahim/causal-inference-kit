"""Tests for Manski bounds, Lee bounds and Imbens-Manski intervals."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from causal_inference import (
    LeeBoundsResult,
    ManskiBoundsResult,
    imbens_manski_interval,
    lee_bounds,
    manski_bounds,
    trimmed_mean,
)
from causal_inference.cli import main


def _selection_data(seed=0, n=4000, effect=1.0, p_to=0.2):
    """Randomized trial with monotone selection.

    Units are always observed (60%), observed only when treated (p_to),
    or never observed. "Observed only if treated" units have higher Y(0).
    """
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, n)
    u = rng.random(n)
    always = u < 0.6
    treated_only = (u >= 0.6) & (u < 0.6 + p_to)
    y0 = rng.normal(size=n) + 1.5 * treated_only
    y1 = y0 + effect
    s = always | (treated_only & (t == 1))
    y = np.where(t == 1, y1, y0)
    y = np.where(s, y, np.nan)
    return t, s.astype(int), y


class TestTrimmedMean:
    def test_integer_trim(self):
        y = np.arange(1.0, 11.0)
        assert trimmed_mean(y, 0.2, "upper") == pytest.approx(np.mean(np.arange(1.0, 9.0)))
        assert trimmed_mean(y, 0.2, "lower") == pytest.approx(np.mean(np.arange(3.0, 11.0)))

    def test_fractional_boundary_weight(self):
        y = np.array([4.0, 1.0, 3.0, 2.0])
        # Keep 2.5 units of mass from the bottom: 1 + 2 + 0.5 * 3.
        assert trimmed_mean(y, 0.375, "upper") == pytest.approx((1 + 2 + 1.5) / 2.5)
        assert trimmed_mean(y, 0.0, "lower") == pytest.approx(2.5)

    def test_matches_quantile_definition_with_repeats(self):
        # Repeating each value m times makes the trim integral; it must agree.
        rng = np.random.default_rng(0)
        y = rng.normal(size=7)
        p = 3 / 7 * 0.5
        rep = np.repeat(y, 14)
        k = int(round((1 - p) * rep.size))
        assert trimmed_mean(y, p, "upper") == pytest.approx(np.sort(rep)[:k].mean())
        assert trimmed_mean(y, p, "lower") == pytest.approx(np.sort(rep)[-k:].mean())

    @pytest.mark.parametrize("args", [([], 0.1, "upper"), ([1.0], 1.0, "upper"), ([1.0], 0.1, "middle"), ([np.nan], 0.1, "upper")])
    def test_validation(self, args):
        with pytest.raises(ValueError):
            trimmed_mean(*args)


class TestImbensManski:
    def test_point_identified_reduces_to_two_sided(self):
        lo, hi, c = imbens_manski_interval(1.0, 1.0, 0.5, 0.5, 0.05)
        assert c == pytest.approx(1.959964, abs=1e-5)
        assert (lo, hi) == pytest.approx((1.0 - 0.5 * c, 1.0 + 0.5 * c))

    def test_wide_bounds_use_one_sided(self):
        _, _, c = imbens_manski_interval(0.0, 50.0, 1.0, 1.0, 0.05)
        assert c == pytest.approx(1.644854, abs=1e-5)

    def test_critical_value_solves_equation(self):
        lo, hi, se_l, se_u = 0.0, 0.8, 0.3, 0.5
        _, _, c = imbens_manski_interval(lo, hi, se_l, se_u, 0.1)
        phi = lambda z: 0.5 * math.erfc(-z / math.sqrt(2))
        assert phi(c + (hi - lo) / 0.5) - phi(-c) == pytest.approx(0.9, abs=1e-9)
        assert 1.2815 < c < 1.6449

    def test_zero_se(self):
        assert imbens_manski_interval(0.0, 1.0, 0.0, 0.0) == (0.0, 1.0, 0.0)

    @pytest.mark.parametrize("args", [(1.0, 0.0, 1.0, 1.0), (0.0, 1.0, -1.0, 1.0), (0.0, 1.0, 1.0, 1.0, 1.5)])
    def test_validation(self, args):
        with pytest.raises(ValueError):
            imbens_manski_interval(*args)


class TestManski:
    def _binary(self, seed=0, n=3000):
        rng = np.random.default_rng(seed)
        t = rng.integers(0, 2, n)
        y = (rng.random(n) < 0.3 + 0.2 * t).astype(float)
        return t, y

    def test_closed_form_no_assumptions(self):
        t, y = self._binary()
        res = manski_bounds(t, y, 0.0, 1.0)
        p1 = t.mean()
        m1, m0 = y[t == 1].mean(), y[t == 0].mean()
        assert isinstance(res, ManskiBoundsResult)
        assert res.lower == pytest.approx(p1 * m1 - (1 - p1) * m0 - p1)
        assert res.upper == pytest.approx(p1 * m1 + (1 - p1) - (1 - p1) * m0)
        assert res.width == pytest.approx(1.0)
        assert res.lower <= 0 <= res.upper

    def test_brute_force_over_unobserved_outcomes(self):
        # On a tiny binary example, enumerate every fill-in of the missing
        # potential outcomes: the extremes of the ATE are the Manski bounds.
        t = np.array([1, 1, 0, 0, 0])
        y = np.array([1.0, 0.0, 1.0, 0.0, 0.0])
        ates = []
        for fill in itertools.product([0.0, 1.0], repeat=t.size):
            y1 = np.where(t == 1, y, fill)
            y0 = np.where(t == 0, y, fill)
            ates.append(np.mean(y1 - y0))
        res = manski_bounds(t, y, 0.0, 1.0)
        assert res.lower == pytest.approx(min(ates))
        assert res.upper == pytest.approx(max(ates))

    def test_shape_restrictions_tighten(self):
        t, y = self._binary()
        base = manski_bounds(t, y, 0, 1)
        mtr = manski_bounds(t, y, 0, 1, ["mtr"])
        mts = manski_bounds(t, y, 0, 1, ["mts"])
        both = manski_bounds(t, y, 0, 1, ["mts", "mtr"])
        assert mtr.lower == 0.0 and mtr.upper == pytest.approx(base.upper)
        assert mts.upper == pytest.approx(base.naive_difference) and mts.lower == pytest.approx(base.lower)
        assert (both.lower, both.upper) == pytest.approx((0.0, base.naive_difference))
        assert both.assumptions == ("mtr", "mts") and not both.refuted

    def test_refuted_when_naive_negative(self):
        t, y = self._binary()
        res = manski_bounds(t, 1.0 - y, 0, 1, ["mtr", "mts"])
        assert res.refuted
        assert "refuted" in res.summary()

    def test_default_range_and_bootstrap(self):
        t, y = self._binary()
        res = manski_bounds(t, 3 * y + 2, n_boot=100, random_state=0)
        assert (res.y_min, res.y_max) == (2.0, 5.0)
        assert res.width == pytest.approx(3.0)
        assert 0 < res.se_lower < 0.2 and 0 < res.se_upper < 0.2
        assert res.ci_lower < res.lower and res.ci_upper > res.upper
        again = manski_bounds(t, 3 * y + 2, n_boot=100, random_state=0)
        assert again.se_lower == res.se_lower

    def test_validation(self):
        t, y = self._binary(n=100)
        with pytest.raises(ValueError):
            manski_bounds(t, y, 0.5, 1.0)
        with pytest.raises(ValueError):
            manski_bounds(t, y, 1.0, 1.0)
        with pytest.raises(ValueError):
            manski_bounds(t, y, 0, 1, ["miv"])
        with pytest.raises(ValueError):
            manski_bounds(np.ones(100), y, 0, 1)
        with pytest.raises(ValueError):
            manski_bounds(t * 2, y, 0, 1)
        with pytest.raises(ValueError):
            manski_bounds(t, y, 0, 1, n_boot=5)


class TestLee:
    def test_bounds_contain_always_observed_effect(self):
        t, s, y = _selection_data()
        res = lee_bounds(t, s, y)
        assert isinstance(res, LeeBoundsResult)
        assert res.trimmed_arm == "treated"
        assert res.lower < 1.0 < res.upper
        expected_p = (res.p_observed_treated - res.p_observed_control) / res.p_observed_treated
        assert res.trimming_proportion == pytest.approx(expected_p)

    def test_matches_manual_trimming(self):
        t, s, y = _selection_data(seed=3, n=1000)
        res = lee_bounds(t, s, y)
        y1 = y[(t == 1) & (s == 1)]
        y0 = y[(t == 0) & (s == 1)]
        p = res.trimming_proportion
        assert res.lower == pytest.approx(trimmed_mean(y1, p, "upper") - y0.mean())
        assert res.upper == pytest.approx(trimmed_mean(y1, p, "lower") - y0.mean())

    def test_no_differential_attrition_point_identifies(self):
        rng = np.random.default_rng(1)
        n = 2000
        t = np.repeat([0, 1], n // 2)
        s = np.tile([1, 1, 1, 0], n // 4)
        y = rng.normal(size=n) + 0.5 * t
        res = lee_bounds(t, s, y)
        assert res.trimming_proportion == 0.0
        diff = y[(t == 1) & (s == 1)].mean() - y[(t == 0) & (s == 1)].mean()
        assert res.lower == pytest.approx(diff) and res.upper == pytest.approx(diff)

    def test_trims_control_when_it_is_observed_more(self):
        t, s, y = _selection_data(seed=2)
        flipped = 1 - t
        res = lee_bounds(flipped, s, -np.nan_to_num(y))
        assert res.trimmed_arm == "control"
        assert res.lower < 1.0 < res.upper

    def test_bounds_shrink_as_selection_gap_closes(self):
        widths = [lee_bounds(*_selection_data(seed=4, p_to=p)).width for p in (0.3, 0.15, 0.05)]
        assert widths[0] > widths[1] > widths[2]

    def test_bootstrap_interval(self):
        t, s, y = _selection_data(seed=5, n=2000)
        res = lee_bounds(t, s, y, n_boot=100, random_state=7)
        assert res.se_lower > 0 and res.se_upper > 0
        assert res.ci_lower < res.lower and res.ci_upper > res.upper
        assert 1.6448 < res.critical_value < 1.96
        assert "Imbens-Manski" in res.summary()

    def test_coverage_of_true_effect(self):
        hits = 0
        for seed in range(30):
            t, s, y = _selection_data(seed=100 + seed, n=600)
            res = lee_bounds(t, s, y, n_boot=60, random_state=seed)
            hits += res.ci_lower <= 1.0 <= res.ci_upper
        assert hits >= 27

    def test_validation(self):
        t, s, y = _selection_data(n=200)
        with pytest.raises(ValueError):
            lee_bounds(t, s, y[:-1])
        bad = y.copy()
        bad[np.flatnonzero(s)[0]] = np.nan
        with pytest.raises(ValueError):
            lee_bounds(t, s, bad)
        with pytest.raises(ValueError):
            lee_bounds(t, np.zeros_like(s), y)
        with pytest.raises(ValueError):
            lee_bounds(t, s, y, n_boot=3)


class TestCLI:
    def test_manski_cli(self, tmp_path, capsys):
        rng = np.random.default_rng(0)
        t = rng.integers(0, 2, 400)
        y = (rng.random(400) < 0.4 + 0.2 * t).astype(float)
        path = tmp_path / "d.csv"
        pd.DataFrame({"treatment": t, "outcome": y}).to_csv(path, index=False)
        assert main(["bounds", "--data", str(path), "--assume", "mtr", "--boot", "50"]) == 0
        out = capsys.readouterr().out
        assert "Manski ATE bounds (assumptions: mtr)" in out and "ci_upper" in out

    def test_lee_cli(self, tmp_path, capsys):
        t, s, y = _selection_data(n=800)
        path = tmp_path / "d.csv"
        pd.DataFrame({"treatment": t, "outcome": y, "observed": s}).to_csv(path, index=False)
        assert main(["bounds", "--data", str(path), "--method", "lee", "--boot", "0"]) == 0
        assert "Lee bounds" in capsys.readouterr().out

    def test_lee_cli_missing_column(self, tmp_path):
        path = tmp_path / "d.csv"
        pd.DataFrame({"treatment": [0, 1], "outcome": [1.0, 2.0]}).to_csv(path, index=False)
        with pytest.raises(SystemExit):
            main(["bounds", "--data", str(path), "--method", "lee"])
