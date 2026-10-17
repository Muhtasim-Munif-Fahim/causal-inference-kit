"""Tests for the Goodman-Bacon decomposition of staggered TWFE DiD."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_inference import bacon_decomposition, difference_in_differences
from causal_inference.cli import main
from causal_inference.staggered import simulate_staggered_did_data


def _panel(cohorts_by_unit, n_times=7, effect=lambda t, g: 1.0, seed=0):
    rng = np.random.default_rng(seed)
    unit, time, y, cohort = [], [], [], []
    for i, g in enumerate(cohorts_by_unit):
        alpha = rng.normal()
        for t in range(1, n_times + 1):
            d = g > 0 and t >= g
            unit.append(i)
            time.append(t)
            cohort.append(g)
            y.append(alpha + 0.4 * t + (effect(t, g) if d else 0.0) + rng.normal(0, 0.3))
    return np.array(unit), np.array(time), np.array(y), np.array(cohort)


def test_weights_sum_to_one_and_reproduce_twfe():
    u, t, y, c, _ = simulate_staggered_did_data(n_units=97, n_times=7, cohorts=(2, 4, 5), seed=3)
    res = bacon_decomposition(u, t, y, c)
    assert res.total_weight == pytest.approx(1.0, abs=1e-12)
    assert res.weighted_estimate == pytest.approx(res.twfe_estimate, abs=1e-10)
    assert all(cmp.weight >= 0 for cmp in res.comparisons)
    # 3 timing groups + never treated: 3 clean + 3 pairs x 2 timing comparisons
    assert len(res.comparisons) == 9
    assert set(res.by_type) == {"treated_vs_never", "earlier_vs_later", "later_vs_earlier"}


def test_twfe_matches_package_twfe_estimator():
    u, t, y, c = _panel([0, 3, 5, 3, 5, 0, 4, 4] * 5, seed=1)
    res = bacon_decomposition(u, t, y, c)
    d = ((c > 0) & (t >= c)).astype(float)
    twfe = difference_in_differences((c > 0).astype(int), t, y, unit=u, treatment=d)
    assert res.twfe_estimate == pytest.approx(twfe.estimate, abs=1e-8)


def test_always_treated_and_late_cohort_handling():
    # cohort 1 = always treated; cohort 9 (> last period 7) = never treated
    u, t, y, c = _panel([0, 1, 3, 4, 6, 9] * 10, effect=lambda t, g: 1 + 0.5 * (t - g), seed=2)
    res = bacon_decomposition(u, t, y, c)
    kinds = {cmp.kind for cmp in res.comparisons}
    assert "later_vs_always" in kinds
    assert res.total_weight == pytest.approx(1.0, abs=1e-12)
    assert res.weighted_estimate == pytest.approx(res.twfe_estimate, abs=1e-10)
    assert res.forbidden_weight == pytest.approx(
        res.by_type["later_vs_earlier"]["weight"] + res.by_type["later_vs_always"]["weight"]
    )


def test_two_by_two_estimates_by_hand():
    # one early (3) and one late (5) group, noise-free with constant effect 2
    u, t, y, c = _panel([3, 3, 5, 5], n_times=6, effect=lambda t, g: 2.0, seed=0)
    res = bacon_decomposition(u, t, y, c)
    Y = pd.DataFrame({"u": u, "t": t, "y": y, "c": c}).groupby(["c", "t"])["y"].mean().unstack()
    early, late = Y.loc[3].to_numpy(), Y.loc[5].to_numpy()
    by_kind = {cmp.kind: cmp for cmp in res.comparisons}
    exp_early = (early[2:4].mean() - early[:2].mean()) - (late[2:4].mean() - late[:2].mean())
    exp_late = (late[4:].mean() - late[2:4].mean()) - (early[4:].mean() - early[2:4].mean())
    assert by_kind["earlier_vs_later"].estimate == pytest.approx(exp_early)
    assert by_kind["later_vs_earlier"].estimate == pytest.approx(exp_late)
    # equal-size groups: Goodman-Bacon weights depend only on timing shares
    w = {k: v.weight for k, v in by_kind.items()}
    assert w["earlier_vs_later"] + w["later_vs_earlier"] == pytest.approx(1.0)


def test_dynamic_effects_bias_later_vs_earlier():
    # effects that grow over time make already-treated controls bad controls
    u, t, y, c = _panel([0, 2, 5] * 40, n_times=8, effect=lambda t, g: 0.8 * (t - g + 1), seed=4)
    res = bacon_decomposition(u, t, y, c)
    clean = res.by_type["treated_vs_never"]["estimate"]
    forbidden = res.by_type["later_vs_earlier"]["estimate"]
    assert forbidden < clean - 1.0


def test_to_frame():
    u, t, y, c, _ = simulate_staggered_did_data(n_units=40, n_times=5, cohorts=(3, 4), seed=0)
    frame = bacon_decomposition(u, t, y, c).to_frame()
    assert list(frame.columns) == ["type", "treated", "control", "estimate", "weight"]
    assert frame["weight"].sum() == pytest.approx(1.0)


def test_validation():
    u, t, y, c = _panel([0, 3, 4] * 3, n_times=5)
    with pytest.raises(ValueError, match="balanced"):
        bacon_decomposition(u[:-1], t[:-1], y[:-1], c[:-1])
    c_bad = c.copy()
    c_bad[0] = 4
    with pytest.raises(ValueError, match="multiple cohort"):
        bacon_decomposition(u, t, y, c_bad)
    with pytest.raises(ValueError, match="no variation"):
        bacon_decomposition(u, t, y, np.zeros_like(c))
    with pytest.raises(ValueError, match="same length"):
        bacon_decomposition(u, t, y[:-1], c)


def test_cli_simulated_and_csv(tmp_path, capsys):
    assert main(["bacon", "--n", "60", "--n-times", "5", "--cohorts", "3,4"]) == 0
    out = capsys.readouterr().out
    assert "Goodman-Bacon decomposition" in out and "later_vs_earlier" in out
    u, t, y, c = _panel([0, 3, 5] * 5)
    path = tmp_path / "panel.csv"
    pd.DataFrame({"unit": u, "time": t, "outcome": y, "cohort": c}).to_csv(path, index=False)
    assert main(["bacon", "--data", str(path)]) == 0
    assert "TWFE=" in capsys.readouterr().out
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"unit": u, "time": t}).to_csv(bad, index=False)
    assert main(["bacon", "--data", str(bad)]) == 2
    assert "missing columns" in capsys.readouterr().err
