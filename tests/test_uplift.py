import numpy as np
import pytest

from causal_inference import (
    aipw_ate,
    aipw_scores,
    qini_curve,
    rate,
    simulate_observational_data,
)


def _rct(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    tau = 2.0 * X[:, 0]
    t = rng.binomial(1, 0.5, size=n).astype(float)
    y = X[:, 1] + tau * t + rng.normal(size=n)
    return X, t, y, tau


def test_qini_hand_example():
    score = [3.0, 2.0, 1.0, 0.0]
    t = [1, 0, 1, 0]
    y = [1.0, 0.0, 0.0, 0.0]
    res = qini_curve(score, t, y, n_points=4)
    # Q(k) = Y_t(k) - Y_c(k) N_t/N_c = 1 for every k >= 1; scaled by 1/n.
    np.testing.assert_allclose(res.qini, [0, 0.25, 0.25, 0.25, 0.25])
    np.testing.assert_allclose(res.random, np.linspace(0, 0.25, 5))
    # qini - random = [0, .1875, .125, .0625, 0]; trapezoid with step 0.25.
    assert res.qini_coefficient == pytest.approx(0.09375)
    assert res.qini_coefficient > 0


def test_oracle_beats_random_and_reverse_on_qini():
    X, t, y, tau = _rct()
    rng = np.random.default_rng(1)
    oracle = qini_curve(tau, t, y)
    noise = qini_curve(rng.normal(size=tau.size), t, y)
    worst = qini_curve(-tau, t, y)
    assert oracle.qini_coefficient > 0.2
    assert abs(noise.qini_coefficient) < 0.1
    assert worst.qini_coefficient < -0.2
    assert oracle.auuc > noise.auuc > worst.auuc
    # The curve ends at the overall difference in means times the treated share.
    diff = y[t == 1].mean() - y[t == 0].mean()
    assert oracle.qini[-1] == pytest.approx(diff * t.mean(), rel=1e-9)


def test_aipw_scores_average_to_aipw_ate():
    X, W, t, y, _ = simulate_observational_data(n=1500, seed=3)
    gamma = aipw_scores(X, t, y, W=W)
    assert gamma.mean() == pytest.approx(aipw_ate(X, t, y, W=W))


def test_rate_closed_form_on_given_scores():
    gamma = np.array([4.0, 1.0, 3.0, 0.0])
    score = np.array([0.9, 0.1, 0.5, 0.2])  # order: 4, 3, 0, 1
    sorted_gamma = np.array([4.0, 3.0, 0.0, 1.0])
    toc = np.cumsum(sorted_gamma) / np.arange(1, 5) - gamma.mean()
    autoc = rate(score, dr_scores=gamma, n_bootstrap=0)
    qini = rate(score, dr_scores=gamma, target="qini", n_bootstrap=0)
    assert autoc.estimate == pytest.approx(toc.mean())
    assert qini.estimate == pytest.approx(np.mean(np.arange(1, 5) / 4 * toc))
    assert qini.target == "QINI"
    assert np.isnan(autoc.std_err)


def test_rate_detects_real_heterogeneity_under_confounding():
    X, W, t, y, _ = simulate_observational_data(n=4000, effect_heterogeneity=1.5, confounding=1.0, seed=7)
    tau = 1.5 * X[:, 0]  # true CATE ordering (X[:, 0] is the effect modifier)
    good = rate(tau, t, y, X, W=W, n_bootstrap=200, seed=0)
    rng = np.random.default_rng(0)
    null = rate(rng.normal(size=t.size), t, y, X, W=W, n_bootstrap=200, seed=0)
    assert good.estimate > 0 and good.ci_low > 0 and good.z > 3
    assert null.ci_low < 0 < null.ci_high
    assert good.toc[0] > good.toc[-1]
    assert good.toc[-1] == pytest.approx(0.0, abs=1e-9)  # TOC(1) = 0 by definition


def test_rate_bootstrap_is_seeded_and_ci_contains_estimate():
    X, t, y, tau = _rct(n=1000)
    a = rate(tau, t, y, X, propensity=np.full(t.size, 0.5), n_bootstrap=100, seed=5)
    b = rate(tau, t, y, X, propensity=np.full(t.size, 0.5), n_bootstrap=100, seed=5)
    assert a.std_err == b.std_err > 0
    assert a.ci_low < a.estimate < a.ci_high


def test_validation():
    with pytest.raises(ValueError, match="binary"):
        qini_curve([1, 2], [0, 2], [1, 1])
    with pytest.raises(ValueError, match="both treatment groups"):
        qini_curve([1, 2], [1, 1], [1, 1])
    with pytest.raises(ValueError, match="one entry per unit"):
        qini_curve([1, 2, 3], [0, 1], [1, 1])
    with pytest.raises(ValueError, match="target"):
        rate([1, 2], dr_scores=[1, 2], target="auc")
    with pytest.raises(ValueError, match="dr_scores, or"):
        rate([1, 2])
    with pytest.raises(ValueError, match="n_points"):
        qini_curve([1, 2], [0, 1], [1, 1], n_points=1)
