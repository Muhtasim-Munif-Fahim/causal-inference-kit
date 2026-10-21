"""Tests for double / debiased machine learning (dml_plr, dml_irm)."""

import numpy as np
import pytest

from causal_inference import dml_irm, dml_plr
from causal_inference.cli import main
from causal_inference.dml import PolynomialRidge


class OLS:
    def fit(self, X, y):
        Z = np.column_stack([np.ones(len(X)), X])
        self.beta_ = np.linalg.lstsq(Z, y, rcond=None)[0]
        return self

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), X]) @ self.beta_


class LogitMLE:
    """Unpenalised logistic regression by Newton-Raphson."""

    def fit(self, X, y):
        Z = np.column_stack([np.ones(len(X)), X])
        b = np.zeros(Z.shape[1])
        for _ in range(100):
            p = 1 / (1 + np.exp(-Z @ b))
            step = np.linalg.solve(Z.T @ (Z * (p * (1 - p))[:, None]), Z.T @ (y - p))
            b += step
            if np.max(np.abs(step)) < 1e-13:
                break
        self.beta_ = b
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-np.column_stack([np.ones(len(X)), X]) @ self.beta_))
        return np.column_stack([1 - p, p])


def _ref_data():
    r = np.random.default_rng(2024)
    n = 200
    X = r.normal(size=(n, 3))
    m = 1 / (1 + np.exp(-(X[:, 0] - 0.5 * X[:, 1])))
    D = (r.random(n) < m).astype(float)
    Y = 1.5 * D + X[:, 0] + np.sin(X[:, 1]) + r.normal(size=n)
    Dc = X[:, 0] + r.normal(size=n)
    Yc = 0.7 * Dc + X[:, 1] ** 2 + r.normal(size=n)
    idx = np.arange(n)
    folds = [(idx[idx % 4 != k], idx[idx % 4 == k]) for k in range(4)]
    return X, D, Y, Dc, Yc, folds


# Computed offline with DoubleML 0.11.4 (LinearRegression / unpenalised LogisticRegression,
# the same 4 folds, trimming_threshold=0.01): (coef, se)
REF_PLR = (0.6854896555245094, 0.13426021356947673)
REF_ATE = (1.8466212890573968, 0.20358723811799787)
REF_ATTE = (2.0439070381451945, 0.31881257945164504)


class TestMatchesDoubleML:
    def test_plr(self):
        X, _, _, Dc, Yc, folds = _ref_data()
        res = dml_plr(X, Dc, Yc, ml_l=OLS(), ml_m=OLS(), folds=folds)
        assert (res.estimate, res.se) == pytest.approx(REF_PLR, rel=1e-10)

    @pytest.mark.parametrize("target,ref", [("ate", REF_ATE), ("att", REF_ATTE)])
    def test_irm(self, target, ref):
        X, D, Y, _, _, folds = _ref_data()
        res = dml_irm(X, D, Y, ml_g=OLS(), ml_m=LogitMLE(), target=target, folds=folds, trimming=0.01)
        assert (res.estimate, res.se) == pytest.approx(ref, rel=1e-6)


def test_plr_is_residual_on_residual_regression():
    X, _, _, Dc, Yc, folds = _ref_data()
    res = dml_plr(X, Dc, Yc, ml_l=OLS(), ml_m=OLS(), folds=folds)
    u = np.empty_like(Yc)
    v = np.empty_like(Dc)
    for tr, te in folds:
        u[te] = Yc[te] - OLS().fit(X[tr], Yc[tr]).predict(X[te])
        v[te] = Dc[te] - OLS().fit(X[tr], Dc[tr]).predict(X[te])
    theta = (u @ v) / (v @ v)
    assert res.estimate == pytest.approx(theta)
    psi = (u - theta * v) * v
    assert res.se == pytest.approx(np.sqrt(np.mean(psi**2) / np.mean(v**2) ** 2 / len(u)))
    assert res.ci_lower < res.estimate < res.ci_upper
    assert np.mean(res.scores) == pytest.approx(0.0, abs=1e-12)


def _nonlinear_plr(seed, n=600, theta=0.5):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 3))
    D = np.tanh(X[:, 0]) + 0.5 * X[:, 1] ** 2 + r.normal(size=n)
    Y = theta * D + np.cos(X[:, 0]) + X[:, 1] ** 2 + r.normal(size=n)
    return X, D, Y


def test_plr_coverage_with_default_learners_and_naive_bias():
    hits = 0
    for seed in range(30):
        X, D, Y = _nonlinear_plr(seed)
        res = dml_plr(X, D, Y, random_state=seed)
        hits += res.ci_lower <= 0.5 <= res.ci_upper
    assert hits >= 24  # nominal 95% of 30
    X, D, Y = _nonlinear_plr(0, n=3000)
    naive = np.linalg.lstsq(np.column_stack([np.ones(len(D)), D]), Y, rcond=None)[0][1]
    assert abs(naive - 0.5) > 0.3  # confounding through X1^2
    assert abs(dml_plr(X, D, Y, random_state=0).estimate - 0.5) < 0.08


def _irm_data(seed, n=3000):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 3))
    p = 1 / (1 + np.exp(-(0.8 * X[:, 0] - 0.5 * X[:, 1])))
    D = (r.random(n) < p).astype(float)
    tau = 1.0 + X[:, 0]  # heterogeneous effect; treated have larger X0 -> ATT > ATE
    Y = X[:, 0] + X[:, 1] ** 2 + tau * D + r.normal(size=n)
    return X, D, Y, tau


def test_irm_ate_and_att_recover_truth():
    X, D, Y, tau = _irm_data(1)
    ate = dml_irm(X, D, Y, random_state=0)
    att = dml_irm(X, D, Y, target="att", random_state=0)
    assert abs(ate.estimate - tau.mean()) < 3 * ate.se
    assert abs(att.estimate - tau[D == 1].mean()) < 3 * att.se
    assert att.estimate > ate.estimate
    assert set(ate.nuisance_rmse) == {"ml_g0", "ml_g1", "ml_m"}
    assert "DML-IRM ATE" in ate.summary()


def test_repeated_splitting_aggregation():
    X, D, Y = _nonlinear_plr(5, n=300)
    res = dml_plr(X, D, Y, n_rep=7, random_state=1)
    th, se = res.estimates_by_rep, res.se_by_rep
    assert res.n_rep == 7 and len(np.unique(th)) == 7
    assert res.estimate == pytest.approx(np.median(th))
    assert res.se == pytest.approx(np.sqrt(np.median(se**2 + (th - np.median(th)) ** 2)))
    ub = dml_plr(X, D, Y, n_rep=7, random_state=1, se_aggregation="median_upper_bound")
    assert ub.se == pytest.approx((np.median(th + 1.96 * se) - np.median(th)) / 1.96)
    # explicit repeated folds give the same answer as the drawn ones
    n = len(Y)
    idx = np.arange(n)
    reps = [[(idx[(idx + s) % 3 != k], idx[(idx + s) % 3 == k]) for k in range(3)] for s in range(2)]
    assert dml_plr(X, D, Y, folds=reps).n_rep == 2


def test_polynomial_ridge_fits_quadratic():
    r = np.random.default_rng(0)
    X = r.normal(size=(400, 2))
    y = 1 + X[:, 0] - 2 * X[:, 0] * X[:, 1] + 0.5 * X[:, 1] ** 2
    pred = PolynomialRidge(alpha=1e-8).fit(X, y).predict(X)
    assert np.allclose(pred, y, atol=1e-5)


def test_validation():
    X, D, Y, _, _, folds = _ref_data()
    with pytest.raises(ValueError, match="binary"):
        dml_irm(X, D + 0.5, Y)
    with pytest.raises(ValueError, match="target"):
        dml_irm(X, D, Y, target="atc")
    with pytest.raises(ValueError, match="partition"):
        dml_plr(X, D, Y, folds=folds[:3])
    with pytest.raises(ValueError, match="n_folds"):
        dml_plr(X, D, Y, n_folds=1)
    with pytest.raises(ValueError, match="se_aggregation"):
        dml_plr(X, D, Y, se_aggregation="mean")
    with pytest.raises(ValueError, match="same number"):
        dml_plr(X, D[:-1], Y)


def test_cli_dml(tmp_path, capsys):
    out = tmp_path / "obs.csv"
    main(["simulate", "--n", "600", "--seed", "1", "--out", str(out)])
    capsys.readouterr()
    assert main(["dml", "--data", str(out)]) == 0
    text = capsys.readouterr().out
    assert "DML-IRM ATE" in text and "rmse_ml_m" in text
    assert main(["dml", "--data", str(out), "--model", "plr", "--reps", "2"]) == 0
    assert "DML-PLR theta" in capsys.readouterr().out
