"""Policy learning: doubly robust policy evaluation and exhaustive policy trees.

A CATE estimate answers "who benefits?"; a *policy* answers "whom should we
treat?". Following Athey & Wager (2021, *Econometrica*) and Zhou, Athey &
Wager (2023, *Operations Research*), each unit gets a doubly robust (AIPW)
reward for each action ``a``:

``Gamma_i(a) = mu_a(X_i) + 1{T_i = a} (Y_i - mu_a(X_i)) / P(T_i = a | X_i)``.

The value of a policy ``pi`` is estimated by ``mean_i Gamma_i(pi(X_i))``. That
estimate is unbiased if either the outcome models or the propensity score are
correct, and the CLT gives it a standard error.

* :func:`dr_policy_scores` builds the ``(n, 2)`` reward matrix
  ``[Gamma(0), Gamma(1)]`` from OLS outcome models and a logistic propensity,
  or from nuisances you pass in (cross-fitted ones are better).
* :func:`policy_value` evaluates any action vector. It reports the value, the
  gain over treat-none and over treat-all, and their standard errors.
* :class:`PolicyTree` runs an exhaustive search over depth-``L`` trees of
  axis-aligned splits (``L <= 3``) and returns the tree with the highest
  total reward. Depth-1 subtrees are solved exactly with sorted cumulative
  sums, so depth 2 costs ``O(d q * d n log n)``. Split thresholds are taken
  from ``n_thresholds`` quantiles of each feature. An optional per-action
  ``treatment_cost`` is subtracted from the rewards.

Evaluate a learned policy on held-out data. In-sample policy values are
optimistic because the tree was chosen to maximise them.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import List, Optional, Sequence

import numpy as np

from .estimators import _coerce_arrays, _validate_propensity, outcome_regression
from .propensity import propensity_scores

__all__ = [
    "PolicyNode",
    "PolicyTree",
    "PolicyValueResult",
    "dr_policy_scores",
    "policy_tree",
    "policy_value",
]


def dr_policy_scores(
    X,
    treatment,
    outcome,
    propensity=None,
    W=None,
    mu0=None,
    mu1=None,
    clip: float = 1e-3,
) -> np.ndarray:
    """Doubly robust rewards ``[Gamma(0), Gamma(1)]`` with shape ``(n, 2)``.

    ``mu0`` / ``mu1`` default to OLS outcome regressions on ``X`` (and
    ``W``), and ``propensity`` defaults to a logistic fit on ``X``. The
    propensity is clipped to ``[clip, 1 - clip]``. Each row satisfies
    ``Gamma(1) - Gamma(0) = aipw_scores``.
    """
    X, treatment, outcome = _coerce_arrays(X, treatment, outcome)
    n = treatment.shape[0]
    if not ((treatment == 1).any() and (treatment == 0).any()):
        raise ValueError("both treatment groups must be present")
    if not 0.0 <= clip < 0.5:
        raise ValueError("clip must be in [0, 0.5)")
    if propensity is None:
        p, _ = propensity_scores(X, treatment)
    else:
        p = _validate_propensity(propensity, n)
    p = np.clip(np.asarray(p, dtype=float), clip, 1.0 - clip)
    if (mu0 is None) != (mu1 is None):
        raise ValueError("pass both mu0 and mu1, or neither")
    if mu0 is None:
        mu1, mu0 = outcome_regression(X, treatment, outcome, W=W)
    mu0 = np.asarray(mu0, dtype=float).reshape(-1)
    mu1 = np.asarray(mu1, dtype=float).reshape(-1)
    if mu0.shape[0] != n or mu1.shape[0] != n:
        raise ValueError("mu0 and mu1 must have one entry per unit")
    g1 = mu1 + treatment * (outcome - mu1) / p
    g0 = mu0 + (1.0 - treatment) * (outcome - mu0) / (1.0 - p)
    return np.column_stack([g0, g1])


def _check_scores(scores) -> np.ndarray:
    G = np.asarray(scores, dtype=float)
    if G.ndim != 2 or G.shape[1] < 2 or G.shape[0] == 0:
        raise ValueError("scores must be an (n, K) reward matrix with K >= 2")
    if not np.all(np.isfinite(G)):
        raise ValueError("scores must be finite")
    return G


@dataclass
class PolicyValueResult:
    """Doubly robust value of a policy, with CLT standard errors."""

    value: float
    se: float
    ci_low: float
    ci_high: float
    gain_vs_control: float
    gain_vs_control_se: float
    gain_vs_treat_all: float
    gain_vs_treat_all_se: float
    treated_fraction: float
    n: int


def _mean_se(v: np.ndarray):
    n = v.shape[0]
    se = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return float(v.mean()), se


def policy_value(actions, scores, alpha: float = 0.05) -> PolicyValueResult:
    """Evaluate integer ``actions`` (one per row) against a reward matrix.

    "Control" means action 0 for everyone, and "treat all" means the last
    action for everyone (action 1 in the binary case).
    """
    G = _check_scores(scores)
    a = np.asarray(actions).reshape(-1)
    if a.shape[0] != G.shape[0]:
        raise ValueError("actions must have one entry per row of scores")
    if not np.all(np.equal(np.mod(a, 1), 0)) or a.min() < 0 or a.max() >= G.shape[1]:
        raise ValueError(f"actions must be integers in [0, {G.shape[1] - 1}]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    a = a.astype(int)
    rows = np.arange(G.shape[0])
    chosen = G[rows, a]
    value, se = _mean_se(chosen)
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    gc, gc_se = _mean_se(chosen - G[:, 0])
    ga, ga_se = _mean_se(chosen - G[:, -1])
    return PolicyValueResult(
        value=value,
        se=se,
        ci_low=value - z * se,
        ci_high=value + z * se,
        gain_vs_control=gc,
        gain_vs_control_se=gc_se,
        gain_vs_treat_all=ga,
        gain_vs_treat_all_se=ga_se,
        treated_fraction=float(np.mean(a != 0)),
        n=int(G.shape[0]),
    )


@dataclass
class PolicyNode:
    """A node of a fitted policy tree (a leaf when ``feature`` is ``None``)."""

    n: int
    reward: float
    action: int
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["PolicyNode"] = None
    right: Optional["PolicyNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


_TOL = 1e-10


def _leaf(G: np.ndarray) -> PolicyNode:
    totals = G.sum(axis=0)
    a = int(np.argmax(totals))
    return PolicyNode(n=int(G.shape[0]), reward=float(totals[a]), action=a)


class PolicyTree:
    """Exhaustive-search policy tree (Athey & Wager 2021; Zhou et al. 2023).

    Parameters
    ----------
    max_depth : int
        Tree depth, from 0 (a single best action) to 3.
    min_leaf_size : int
        Minimum number of units in each leaf.
    n_thresholds : int
        Number of candidate split points per feature, taken from interior
        quantiles of the training data.
    treatment_cost : float or sequence
        Cost per unit subtracted from each action's reward. A scalar applies
        to every action except 0.
    """

    def __init__(
        self,
        max_depth: int = 2,
        min_leaf_size: int = 10,
        n_thresholds: int = 20,
        treatment_cost=0.0,
    ) -> None:
        if not isinstance(max_depth, (int, np.integer)) or not 0 <= max_depth <= 3:
            raise ValueError("max_depth must be an integer in [0, 3]")
        if min_leaf_size < 1:
            raise ValueError("min_leaf_size must be >= 1")
        if n_thresholds < 1:
            raise ValueError("n_thresholds must be >= 1")
        self.max_depth = int(max_depth)
        self.min_leaf_size = int(min_leaf_size)
        self.n_thresholds = int(n_thresholds)
        self.treatment_cost = treatment_cost

    # -- fitting --------------------------------------------------------------
    def _costs(self, k: int) -> np.ndarray:
        c = np.asarray(self.treatment_cost, dtype=float)
        if c.ndim == 0:
            out = np.full(k, float(c))
            out[0] = 0.0
            return out
        if c.shape != (k,):
            raise ValueError(f"treatment_cost must be a scalar or have length {k}")
        return c

    def fit(self, X, scores) -> "PolicyTree":
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        G = _check_scores(scores)
        if X.shape[0] != G.shape[0]:
            raise ValueError("X and scores must have the same number of rows")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must be finite")
        G = G - self._costs(G.shape[1])[None, :]
        qs = np.linspace(0.0, 1.0, self.n_thresholds + 2)[1:-1]
        self.thresholds_: List[np.ndarray] = [
            np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])
        ]
        self.n_features_in_ = X.shape[1]
        self.n_actions_ = G.shape[1]
        self.tree_ = self._search(X, G, self.max_depth)
        self.reward_ = self.tree_.reward
        self.mean_reward_ = self.reward_ / X.shape[0]
        return self

    def _search(self, X: np.ndarray, G: np.ndarray, depth: int) -> PolicyNode:
        best = _leaf(G)
        n = X.shape[0]
        if depth == 0 or n < 2 * self.min_leaf_size:
            return best
        if depth == 1:
            return self._best_stump(X, G, best)
        for j in range(X.shape[1]):
            for t in self.thresholds_[j]:
                mask = X[:, j] <= t
                nl = int(mask.sum())
                if nl < self.min_leaf_size or n - nl < self.min_leaf_size:
                    continue
                left = self._search(X[mask], G[mask], depth - 1)
                right = self._search(X[~mask], G[~mask], depth - 1)
                r = left.reward + right.reward
                if r > best.reward + _TOL * max(1.0, abs(best.reward)):
                    best = PolicyNode(n=n, reward=r, action=-1, feature=j, threshold=float(t), left=left, right=right)
        return best

    def _best_stump(self, X: np.ndarray, G: np.ndarray, best: PolicyNode) -> PolicyNode:
        n = X.shape[0]
        total = G.sum(axis=0)
        best_split = None
        best_r = best.reward
        for j in range(X.shape[1]):
            order = np.argsort(X[:, j], kind="mergesort")
            xs = X[order, j]
            cs = np.cumsum(G[order], axis=0)
            ks = np.searchsorted(xs, self.thresholds_[j], side="right")
            ok = (ks >= self.min_leaf_size) & (n - ks >= self.min_leaf_size)
            if not ok.any():
                continue
            ks_ok = ks[ok]
            left = cs[ks_ok - 1]
            right = total[None, :] - left
            r = left.max(axis=1) + right.max(axis=1)
            i = int(np.argmax(r))
            if r[i] > best_r + _TOL * max(1.0, abs(best_r)):
                best_r = float(r[i])
                best_split = (j, float(self.thresholds_[j][ok][i]))
        if best_split is None:
            return best
        j, t = best_split
        mask = X[:, j] <= t
        return PolicyNode(
            n=n, reward=best_r, action=-1, feature=j, threshold=t,
            left=_leaf(G[mask]), right=_leaf(G[~mask]),
        )

    # -- inference ------------------------------------------------------------
    def _check_fitted(self) -> None:
        if not hasattr(self, "tree_"):
            raise ValueError("PolicyTree is not fitted; call fit first")

    def predict(self, X) -> np.ndarray:
        """Recommended action for each row of ``X``."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {X.shape[1]}")
        out = np.empty(X.shape[0], dtype=int)
        for i, x in enumerate(X):
            node = self.tree_
            while not node.is_leaf:
                node = node.left if x[node.feature] <= node.threshold else node.right
            out[i] = node.action
        return out

    @property
    def n_leaves_(self) -> int:
        self._check_fitted()

        def count(node):
            return 1 if node.is_leaf else count(node.left) + count(node.right)

        return count(self.tree_)

    def rules(self, feature_names: Optional[Sequence[str]] = None, action_names: Optional[Sequence[str]] = None) -> str:
        """Human-readable if/else listing of the tree."""
        self._check_fitted()
        names = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(self.n_features_in_)]
        acts = list(action_names) if action_names is not None else [f"action {a}" for a in range(self.n_actions_)]
        lines: List[str] = []

        def walk(node, indent):
            pad = "  " * indent
            if node.is_leaf:
                lines.append(f"{pad}-> {acts[node.action]}  (n={node.n})")
                return
            lines.append(f"{pad}if {names[node.feature]} <= {node.threshold:.4g}:")
            walk(node.left, indent + 1)
            lines.append(f"{pad}else:  # {names[node.feature]} > {node.threshold:.4g}")
            walk(node.right, indent + 1)

        walk(self.tree_, 0)
        return "\n".join(lines)


def policy_tree(
    X,
    treatment,
    outcome,
    propensity=None,
    W=None,
    max_depth: int = 2,
    min_leaf_size: int = 10,
    n_thresholds: int = 20,
    treatment_cost=0.0,
) -> PolicyTree:
    """Fit a :class:`PolicyTree` on the doubly robust rewards of a binary treatment."""
    scores = dr_policy_scores(X, treatment, outcome, propensity=propensity, W=W)
    return PolicyTree(
        max_depth=max_depth,
        min_leaf_size=min_leaf_size,
        n_thresholds=n_thresholds,
        treatment_cost=treatment_cost,
    ).fit(X, scores)
