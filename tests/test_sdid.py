"""Tests for synthetic difference-in-differences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_inference import simulate_sdid_panel, synthetic_did
from causal_inference.cli import main

DATA = Path(__file__).parent / "data" / "california_prop99.csv"


def _prop99():
    # Abadie, Diamond & Hainmueller (2010) cigarette sales panel as shipped
    # with the reference ``synthdid`` R package (BSD-3).
    df = pd.read_csv(DATA, sep=";")
    return df["State"].to_numpy(), df["Year"].to_numpy(), df["PacksPerCapita"].to_numpy(dtype=float)


def test_prop99_matches_reference_implementation():
    unit, time, outcome = _prop99()
    res = synthetic_did(unit, time, outcome, ["California"], 1989, se_method="none")
    # Arkhangelsky et al. (2021), Table 1 / synthdid: SDID -15.6, DiD -27.3.
    assert res.estimate == pytest.approx(-15.604, abs=0.01)
    assert res.did_estimate == pytest.approx(-27.349, abs=0.01)
    assert res.sc_estimate == pytest.approx(-19.6, abs=0.2)
    # Time weights concentrate on 1986-1988 as reported in the paper.
    weights = dict(zip(res.pre_times.tolist(), res.time_weights.tolist()))
    assert weights[1986] == pytest.approx(0.366, abs=0.005)
    assert weights[1987] == pytest.approx(0.206, abs=0.005)
    assert weights[1988] == pytest.approx(0.427, abs=0.005)
    assert sum(w for y, w in weights.items() if y < 1986) < 1e-3
    assert res.unit_weights.sum() == pytest.approx(1.0)
    assert np.all(res.unit_weights >= 0)
    top = res.control_ids[np.argmax(res.unit_weights)]
    assert top == "Nevada"


def test_prop99_placebo_standard_error_is_reasonable():
    unit, time, outcome = _prop99()
    res = synthetic_did(unit, time, outcome, ["California"], 1989, se_method="placebo", n_placebo=60, seed=1)
    assert res.se_method == "placebo"
    # synthdid reports a placebo SE of about 9-10 packs for this example.
    assert 5.0 < res.std_err < 15.0
    assert res.placebo_estimates.shape == (60,)
    assert res.ci_lower < res.estimate < res.ci_upper


def test_did_comparison_is_two_by_two_of_group_period_means():
    unit, time, outcome, treated, t0, _ = simulate_sdid_panel(n_control=10, n_treated=3, n_pre=6, n_post=3, seed=2)
    res = synthetic_did(unit, time, outcome, treated, t0, se_method="none")
    panel = outcome.reshape(13, 9)
    co, tr = panel[:10], panel[10:]
    manual = (tr[:, t0:].mean() - tr[:, :t0].mean()) - (co[:, t0:].mean() - co[:, :t0].mean())
    assert res.did_estimate == pytest.approx(manual)


def test_exact_recovery_without_noise_under_parallel_trends():
    rng = np.random.default_rng(0)
    n_co, n_tr, n_pre, n_post = 8, 2, 6, 3
    alpha = rng.normal(size=n_co + n_tr)
    beta = rng.normal(size=n_pre + n_post).cumsum()
    panel = alpha[:, None] + beta[None, :]
    panel[n_co:, n_pre:] += 1.5
    unit = np.repeat(np.arange(n_co + n_tr), n_pre + n_post)
    time = np.tile(np.arange(n_pre + n_post), n_co + n_tr)
    res = synthetic_did(unit, time, panel.ravel(), list(range(n_co, n_co + n_tr)), n_pre, se_method="jackknife")
    assert res.estimate == pytest.approx(1.5, abs=1e-8)
    assert res.did_estimate == pytest.approx(1.5, abs=1e-8)
    assert res.std_err == pytest.approx(0.0, abs=1e-6)


def test_sdid_beats_did_when_parallel_trends_fail():
    errs = {"sdid": [], "did": [], "sc": []}
    covered = 0
    for seed in range(8):
        unit, time, outcome, treated, t0, att = simulate_sdid_panel(seed=seed)
        res = synthetic_did(unit, time, outcome, treated, t0)
        errs["sdid"].append(res.estimate - att)
        errs["did"].append(res.did_estimate - att)
        errs["sc"].append(res.sc_estimate - att)
        covered += res.ci_lower <= att <= res.ci_upper
        assert res.se_method == "jackknife"
    rmse = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in errs.items()}
    assert rmse["sdid"] < 0.5
    assert rmse["sdid"] < rmse["did"] / 4
    assert rmse["sdid"] <= rmse["sc"] + 0.05
    assert covered >= 6


def test_larger_zeta_spreads_unit_weights():
    unit, time, outcome, treated, t0, _ = simulate_sdid_panel(seed=3)
    tight = synthetic_did(unit, time, outcome, treated, t0, se_method="none", zeta=1e-3)
    loose = synthetic_did(unit, time, outcome, treated, t0, se_method="none", zeta=1e3)
    assert loose.effective_controls > tight.effective_controls
    np.testing.assert_allclose(loose.unit_weights, np.full(loose.unit_weights.size, 1 / loose.unit_weights.size), atol=1e-3)


def test_string_ids_and_unsorted_input():
    unit, time, outcome, treated, t0, _ = simulate_sdid_panel(n_control=12, n_treated=2, seed=4)
    names = np.array([f"u{u:02d}" for u in unit])
    perm = np.random.default_rng(0).permutation(unit.size)
    a = synthetic_did(unit, time, outcome, treated, t0, se_method="none")
    b = synthetic_did(names[perm], time[perm], outcome[perm], [f"u{u:02d}" for u in treated], t0, se_method="none")
    assert a.estimate == pytest.approx(b.estimate)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"treated_units": []}, "at least one treated"),
        ({"treated_units": [999]}, "appear in the panel"),
        ({"treatment_time": 1}, "two pre-treatment"),
        ({"treatment_time": 100}, "post-treatment"),
        ({"se_method": "bootstrap"}, "se_method"),
        ({"zeta": -1.0}, "zeta"),
    ],
)
def test_validation(kwargs, message):
    unit, time, outcome, treated, t0, _ = simulate_sdid_panel(n_control=6, n_treated=1, n_pre=5, n_post=2, seed=0)
    args = {"treated_units": treated, "treatment_time": t0}
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        synthetic_did(unit, time, outcome, args.pop("treated_units"), args.pop("treatment_time"), **args)


def test_jackknife_needs_two_treated_units():
    unit, time, outcome, treated, t0, _ = simulate_sdid_panel(n_control=6, n_treated=1, n_pre=5, n_post=2, seed=0)
    with pytest.raises(ValueError, match="two treated"):
        synthetic_did(unit, time, outcome, treated, t0, se_method="jackknife")


def test_cli_simulated_and_csv(tmp_path, capsys):
    assert main(["sdid", "--n-control", "15", "--n-treated", "3", "--se", "jackknife"]) == 0
    out = capsys.readouterr().out
    assert "SDID estimate=" in out and "DiD estimate=" in out and "true ATT=" in out
    unit, time, outcome = _prop99()
    df = pd.DataFrame({"unit": unit, "time": time, "outcome": outcome, "treated": (unit == "California").astype(int)})
    path = tmp_path / "p99.csv"
    df.to_csv(path, index=False)
    assert main(["sdid", "--data", str(path), "--treatment-time", "1989", "--se", "none"]) == 0
    out = capsys.readouterr().out
    assert "SDID estimate=-15.60" in out
    assert "1988=0.427" in out
    assert main(["sdid", "--data", str(path)]) == 2
