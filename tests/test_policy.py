"""Tests for doubly robust policy evaluation and policy trees."""

from __future__ import annotations


import numpy as np
import pytest

from causal_inference import (
    PolicyTree,
    aipw_scores,
    dr_policy_scores,
    policy_tree,
    policy_value,
)
from causal_inference.cli import main


def _rct(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    T = rng.binomial(1, 0.5, n).astype(float)
    tau = np.where(X[:, 0] > 0.3, 1.5, -1.0)
    Y = X[:, 1] + T * tau + rng.normal(size=n)
    return X, T, Y, tau


def test_scores_match_aipw_and_known_nuisances():
    X, T, Y, _ = _rct(800)
    p = np.full(800, 0.5)
    G = dr_policy_scores(X, T, Y, propensity=p)
    assert G.shape == (800, 2)
    np.testing.assert_allclose(G[:, 1] - G[:, 0], aipw_scores(X, T, Y, propensity=p))
    # with zero outcome models the rewards are pure IPW
    G0 = dr_policy_scores(X, T, Y, propensity=p, mu0=np.zeros(800), mu1=np.zeros(800))
    np.testing.assert_allclose(G0[:, 1], T * Y / 0.5)
    np.testing.assert_allclose(G0[:, 0], (1 - T) * Y / 0.5)


def test_policy_value_baselines_and_gains():
    X, T, Y, tau = _rct()
    G = dr_policy_scores(X, T, Y, propensity=np.full(len(T), 0.5))
    none = policy_value(np.zeros(len(T), dtype=int), G)
    allt = policy_value(np.ones(len(T), dtype=int), G)
    assert none.value == pytest.approx(G[:, 0].mean())
    assert allt.value == pytest.approx(G[:, 1].mean())
    assert none.gain_vs_control == 0.0 and allt.gain_vs_treat_all == 0.0
    oracle = policy_value((tau > 0).astype(int), G)
    true_gain = np.mean(np.maximum(tau, 0.0))
    assert abs(oracle.gain_vs_control - true_gain) < 4 * oracle.gain_vs_control_se + 0.02
    assert oracle.ci_low < oracle.value < oracle.ci_high
    assert oracle.treated_fraction == pytest.approx(np.mean(tau > 0))


def test_tree_recovers_threshold_rule_and_generalises():
    X, T, Y, tau = _rct(4000, seed=1)
    tree = policy_tree(X, T, Y, propensity=np.full(4000, 0.5), max_depth=1)
    assert tree.tree_.feature == 0
    assert abs(tree.tree_.threshold - 0.3) < 0.15
    assert tree.tree_.left.action == 0 and tree.tree_.right.action == 1
    Xt, Tt, Yt, taut = _rct(4000, seed=2)
    agree = np.mean(tree.predict(Xt) == (taut > 0))
    assert agree > 0.93
    assert "x0 <=" in tree.rules()


def test_stump_search_matches_brute_force():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 2))
    G = rng.normal(size=(60, 3))
    tree = PolicyTree(max_depth=1, min_leaf_size=5, n_thresholds=8).fit(X, G)
    best = G.sum(0).max()
    for j in range(2):
        for t in tree.thresholds_[j]:
            m = X[:, j] <= t
            if m.sum() < 5 or (~m).sum() < 5:
                continue
            best = max(best, G[m].sum(0).max() + G[~m].sum(0).max())
    assert tree.reward_ == pytest.approx(best)
    np.testing.assert_allclose(G[np.arange(60), tree.predict(X)].sum(), tree.reward_)


def test_depth_two_finds_interaction_and_is_monotone_in_depth():
    rng = np.random.default_rng(4)
    n = 3000
    X = rng.uniform(-1, 1, size=(n, 2))
    gain = np.where((X[:, 0] > 0) & (X[:, 1] > 0), 2.0, -1.0)
    G = np.column_stack([np.zeros(n), gain + 0.5 * rng.normal(size=n)])
    rewards = [PolicyTree(max_depth=d, min_leaf_size=20).fit(X, G).reward_ for d in range(3)]
    assert rewards[0] <= rewards[1] + 1e-9 <= rewards[2] + 2e-9
    t2 = PolicyTree(max_depth=2, min_leaf_size=20).fit(X, G)
    acc = np.mean(t2.predict(X) == (gain > 0))
    assert acc > 0.97 and t2.n_leaves_ >= 3


def test_treatment_cost_shrinks_treated_set():
    X, T, Y, _ = _rct(3000, seed=5)
    G = dr_policy_scores(X, T, Y, propensity=np.full(3000, 0.5))
    free = PolicyTree(max_depth=1).fit(X, G).predict(X).mean()
    pricey = PolicyTree(max_depth=1, treatment_cost=5.0).fit(X, G).predict(X).mean()
    assert free > 0.2 and pricey == 0.0
    with pytest.raises(ValueError):
        PolicyTree(treatment_cost=[0.0, 1.0, 2.0]).fit(X, G)


def test_validation():
    X, T, Y, _ = _rct(200)
    G = dr_policy_scores(X, T, Y)
    for bad in ({"max_depth": 4}, {"max_depth": -1}, {"min_leaf_size": 0}, {"n_thresholds": 0}):
        with pytest.raises(ValueError):
            PolicyTree(**bad)
    with pytest.raises(ValueError):
        PolicyTree().predict(X)
    with pytest.raises(ValueError):
        PolicyTree().fit(X[:10], G)
    with pytest.raises(ValueError):
        PolicyTree().fit(X, G[:, :1])
    with pytest.raises(ValueError):
        policy_value(np.full(200, 2), G)
    with pytest.raises(ValueError):
        dr_policy_scores(X, T, Y, mu0=np.zeros(200))
    with pytest.raises(ValueError):
        dr_policy_scores(X, np.ones(200), Y)
    fitted = PolicyTree(max_depth=0).fit(X, G)
    assert fitted.n_leaves_ == 1
    with pytest.raises(ValueError):
        fitted.predict(np.zeros((3, 5)))


def test_cli_policy(tmp_path, capsys):
    out = tmp_path / "obs.csv"
    main(["simulate", "--n", "1200", "--seed", "1", "--out", str(out)])
    capsys.readouterr()
    assert main(["policy", "--data", str(out), "--depth", "1"]) == 0
    text = capsys.readouterr().out
    assert "Policy tree" in text and "gain_vs_control" in text and "held-out" in text
    assert main(["policy", "--data", str(out), "--test-frac", "0", "--cost", "100"]) == 0
    text = capsys.readouterr().out
    assert "in-sample" in text and "-> control" in text
    with pytest.raises(SystemExit):
        main(["policy", "--data", str(out), "--depth", "5"])
