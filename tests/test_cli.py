"""Tests for the command-line interface and the demo script."""

import os
import sys

import pandas as pd
import pytest

from causal_inference.cli import main

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def _simulate(tmp_path, seed=1, n=200):
    out = tmp_path / "data.csv"
    main(["simulate", "--n", str(n), "--seed", str(seed), "--out", str(out)])
    return out


def test_simulate_writes_csv(tmp_path):
    out = _simulate(tmp_path)
    df = pd.read_csv(out)
    assert {"treatment", "outcome", "X0", "X1", "X2", "W0", "W1"} <= set(df.columns)
    assert len(df) == 200


def test_simulate_prints_true_ate(capsys, tmp_path):
    out = _simulate(tmp_path)
    captured = capsys.readouterr()
    assert "true ATE" in captured.out
    assert "200 rows" in captured.out


def test_simulate_seed_reproducibility(tmp_path):
    first = _simulate(tmp_path, seed=3)
    second = tmp_path / "other.csv"
    main(["simulate", "--n", "200", "--seed", "3", "--out", str(second)])
    a = pd.read_csv(first)
    b = pd.read_csv(second)
    pd.testing.assert_frame_equal(a, b)


def test_simulate_selection_bias_flag(tmp_path):
    out = _simulate(tmp_path)
    df = pd.read_csv(out)
    assert (df["treatment"].isin([0.0, 1.0])).all()


def test_estimate_prints_estimators(capsys, tmp_path):
    out = _simulate(tmp_path)
    main(["estimate", "--data", str(out)])
    captured = capsys.readouterr()
    for name in ("naive", "ipw", "ipw_att", "matching"):
        assert name in captured.out


def test_estimate_with_true_ate_shows_bias(capsys, tmp_path):
    out = _simulate(tmp_path)
    main(["estimate", "--data", str(out), "--true-ate", "2.0"])
    captured = capsys.readouterr()
    assert "bias" in captured.out


def test_report_writes_markdown(tmp_path):
    out = _simulate(tmp_path)
    report = tmp_path / "report.md"
    main(["report", "--data", str(out), "--true-ate", "2.0", "--bootstrap", "20", "--out", str(report)])
    text = report.read_text(encoding="utf-8")
    assert "# Causal inference report" in text
    assert "## Covariate balance" in text
    assert "## Bias and RMSE vs ground truth" in text
    assert "### Assumption caveats" in text


def test_report_missing_columns_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(bad, index=False)
    with pytest.raises(SystemExit):
        main(["estimate", "--data", str(bad)])


def test_unknown_subcommand_exits(capsys, tmp_path):
    with pytest.raises(SystemExit):
        main(["bogus"])


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "causal-inference-kit" in capsys.readouterr().out


def test_report_requires_true_ate(tmp_path):
    out = _simulate(tmp_path)
    with pytest.raises(SystemExit):
        main(["report", "--data", str(out)])


def test_demo_writes_report(tmp_path):
    import run_demo

    out_dir = tmp_path / "demo_output"
    code = run_demo.main(["--n", "400", "--bootstrap", "20", "--out-dir", str(out_dir)])
    assert code == 0
    report = out_dir / "demo_report.md"
    assert report.exists()
    assert "## Difference-in-differences" in report.read_text(encoding="utf-8")


def test_demo_prints_estimates(capsys, tmp_path):
    import run_demo

    run_demo.main(["--n", "300", "--bootstrap", "10", "--out-dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert "ipw (stabilized)" in captured.out
    assert "bootstrap evaluation" in captured.out
