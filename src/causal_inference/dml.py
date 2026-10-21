"""Double / debiased machine learning (Chernozhukov et al., 2018).

Plugging flexible ML predictions straight into a treatment-effect formula
causes two problems: regularisation bias and overfitting bias. DML removes
both by combining

* a **Neyman-orthogonal score**, whose expectation is insensitive to small
  errors in the nuisance functions, and
* **cross-fitting**: the nuisances for fold ``k`` are fitted on the other
  folds, so every prediction is out of sample.

The result is a root-n consistent, asymptotically normal estimate even when
the nuisances converge slowly (at ``o(n^{-1/4})``).

Two models are provided.

``dml_plr``: partially linear regression ``Y = theta D + g(X) + e``, with
``D = m(X) + v``. ``D`` may be continuous. With ``l(X) = E[Y|X]``, the
"partialling-out" score gives

    theta = sum (D - m)(Y - l) / sum (D - m)^2

(Robinson-style residual-on-residual regression).

``dml_irm``: interactive regression model for binary ``D`` with
``g_d(X) = E[Y | D=d, X]`` and propensity ``m(X)``. The doubly robust (AIPW)
score estimates the ATE, or the ATT with ``target="att"``.

Standard errors come from the score's empirical variance. With ``n_rep > 1``
the sample splitting is repeated and the estimates are aggregated by the
median, with ``se^2 = median(se_s^2 + (theta_s - theta)^2)`` as in the
paper (``se_aggregation="median_upper_bound"`` reproduces DoubleML >= 0.10,
which takes the median of the upper 95% bounds instead).

Learners are any objects with ``fit(X, y)`` / ``predict(X)`` (classifiers:
``predict_proba``), e.g. scikit-learn estimators. The NumPy defaults are a
ridge regression on degree-2 polynomial features and the kit's L2 logistic
regression.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from math import erf, sqrt

import numpy as np

from .propensity import logistic_regression

__all__ = [
    "DMLResult",
    "PolynomialRidge",
    "LogisticClassifier",
    "dml_plr",
    "dml_irm",
]


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


class PolynomialRidge:
    """Ridge regression on standardised degree-``degree`` polynomial features."""

    def __init__(self, degree: int = 2, alpha: float = 1.0):
        self.degree = degree
        self.alpha = alpha

    def _features(self, X):
        Z = (X - self.mu_) / self.sd_
        cols = [Z]
        if self.degree >= 2:
            d = Z.shape[1]
            iu = np.triu_indices(d)
            cols.append((Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]])
        return np.hstack(cols)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mu_ = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd_ = np.where(sd > 0, sd, 1.0)
        F = self._features(X)
        self.fmu_ = F.mean(axis=0)
        Fc = F - self.fmu_
        self.ymu_ = y.mean()
        A = Fc.T @ Fc + self.alpha * np.eye(F.shape[1])
        self.coef_ = np.linalg.solve(A, Fc.T @ (y - self.ymu_))
        return self

    def predict(self, X):
        F = self._features(np.asarray(X, dtype=float))
        return self.ymu_ + (F - self.fmu_) @ self.coef_


class LogisticClassifier:
    """Thin wrapper over :func:`logistic_regression` with ``predict_proba``."""

    def __init__(self, l2: float = 1e-3):
        self.l2 = l2

    def fit(self, X, y):
        fit = logistic_regression(np.asarray(X, dtype=float), np.asarray(y, dtype=float), l2=self.l2)
        self.coef_, self.intercept_ = fit.coef, fit.intercept
        return self

    def predict_proba(self, X):
        z = np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
        return np.column_stack([1.0 - p, p])


@dataclass
class DMLResult:
    """Cross-fitted DML estimate with normal-theory inference."""

    model: str
    target: str
    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    z: float
    p_value: float
    n: int
    n_folds: int
    n_rep: int
    estimates_by_rep: np.ndarray
    se_by_rep: np.ndarray
    scores: np.ndarray = field(repr=False)
    nuisance_rmse: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"DML-{self.model.upper()} {self.target}: {self.estimate:.4f} (SE {self.se:.4f}, "
            f"95% CI [{self.ci_lower:.4f}, {self.ci_upper:.4f}], p={self.p_value:.3g}; "
            f"{self.n_folds}-fold cross-fitting x {self.n_rep})"
        )


def _clone(est):
    if est is None:
        return None
    try:  # scikit-learn estimators
        from sklearn.base import clone  # type: ignore

        return clone(est)
    except Exception:
        return copy.deepcopy(est)


def _prepare(X, treatment, outcome):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    d = np.asarray(treatment, dtype=float).ravel()
    y = np.asarray(outcome, dtype=float).ravel()
    if not (X.shape[0] == d.size == y.size):
        raise ValueError("X, treatment and outcome must have the same number of rows")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(d)) and np.all(np.isfinite(y))):
        raise ValueError("inputs must be finite")
    return X, d, y


def _make_folds(n, n_folds, rng, strata=None):
    if n_folds < 2 or n_folds > n:
        raise ValueError("n_folds must be between 2 and n")
    fold_id = np.empty(n, dtype=int)
    if strata is None:
        perm = rng.permutation(n)
        for k, part in enumerate(np.array_split(perm, n_folds)):
            fold_id[part] = k
    else:  # stratify on binary treatment so every fold has both arms
        for s in np.unique(strata):
            idx = rng.permutation(np.flatnonzero(strata == s))
            for k, part in enumerate(np.array_split(idx, n_folds)):
                fold_id[part] = k
    return [(np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)) for k in range(n_folds)]


def _splits_from(folds, n):
    """Accept one split ``[(tr, te), ...]`` or repetitions ``[[(tr, te), ...], ...]``."""
    first = folds[0]
    single = len(first) == 2 and np.ndim(first[0]) == 1 and not isinstance(first[0][0], (tuple, list))
    if single:
        try:
            np.asarray(first[0], dtype=int)
        except (TypeError, ValueError):
            single = False
    reps = [folds] if single else list(folds)
    return [_check_folds(rep, n) for rep in reps]


def _check_folds(folds, n):
    seen = np.zeros(n, dtype=int)
    out = []
    for tr, te in folds:
        tr, te = np.asarray(tr, dtype=int), np.asarray(te, dtype=int)
        seen[te] += 1
        if np.intersect1d(tr, te).size:
            raise ValueError("train and test indices overlap in a fold")
        out.append((tr, te))
    if not np.all(seen == 1):
        raise ValueError("test folds must partition the sample")
    return out


def _aggregate(model, target, thetas, ses, psis, n, n_folds, rmse, level=0.95, se_aggregation="median_variance"):
    thetas = np.asarray(thetas)
    ses = np.asarray(ses)
    theta = float(np.median(thetas))
    if se_aggregation == "median_variance":  # Chernozhukov et al. (2018), Definition 3.3
        se = float(np.sqrt(np.median(ses**2 + (thetas - theta) ** 2)))
    else:  # DoubleML >= 0.10: median of the upper 95% bounds, mapped back to an SE
        se = float((np.median(thetas + 1.96 * ses) - theta) / 1.96)
    zcrit = 1.959963984540054 if level == 0.95 else _z_quantile(0.5 + level / 2)
    z = theta / se if se > 0 else np.inf
    p = 2.0 * (1.0 - _norm_cdf(abs(z))) if np.isfinite(z) else 0.0
    return DMLResult(
        model=model,
        target=target,
        estimate=theta,
        se=se,
        ci_lower=theta - zcrit * se,
        ci_upper=theta + zcrit * se,
        z=float(z),
        p_value=float(p),
        n=n,
        n_folds=n_folds,
        n_rep=len(thetas),
        estimates_by_rep=thetas,
        se_by_rep=ses,
        scores=np.asarray(psis[0]),
        nuisance_rmse=rmse,
    )


def _check_agg(se_aggregation):
    if se_aggregation not in ("median_variance", "median_upper_bound"):
        raise ValueError("se_aggregation must be 'median_variance' or 'median_upper_bound'")


def _z_quantile(p: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _predict_proba1(est, X):
    if hasattr(est, "predict_proba"):
        return np.asarray(est.predict_proba(X))[:, 1]
    return np.asarray(est.predict(X), dtype=float)


def dml_plr(
    X,
    treatment,
    outcome,
    ml_l=None,
    ml_m=None,
    n_folds: int = 5,
    n_rep: int = 1,
    random_state=None,
    folds=None,
    level: float = 0.95,
    se_aggregation: str = "median_variance",
) -> DMLResult:
    """DML for the partially linear model, using the partialling-out score.

    ``ml_l`` learns ``E[Y|X]`` and ``ml_m`` learns ``E[D|X]``. If ``D`` is
    binary and ``ml_m`` has ``predict_proba``, the probability is used. Pass
    ``folds`` (a list of ``(train_idx, test_idx)``, or a list of such lists
    for repeated splitting) to fix the sample splitting.
    """
    X, d, y = _prepare(X, treatment, outcome)
    _check_agg(se_aggregation)
    n = y.size
    ml_l = PolynomialRidge() if ml_l is None else ml_l
    ml_m = PolynomialRidge() if ml_m is None else ml_m
    rng = np.random.default_rng(random_state)
    splits = _splits_from(folds, n) if folds is not None else [_make_folds(n, int(n_folds), rng) for _ in range(int(n_rep))]
    thetas, ses, psis, rl, rm = [], [], [], [], []
    for sp in splits:
        l_hat = np.empty(n)
        m_hat = np.empty(n)
        for tr, te in sp:
            l_hat[te] = np.asarray(_clone(ml_l).fit(X[tr], y[tr]).predict(X[te]), dtype=float)
            m_hat[te] = _predict_proba1(_clone(ml_m).fit(X[tr], d[tr]), X[te])
        u, v = y - l_hat, d - m_hat
        denom = float(np.mean(v * v))
        if denom <= 0:
            raise ValueError("treatment residuals are identically zero; theta is not identified")
        theta = float(np.mean(v * u) / denom)
        psi = (u - theta * v) * v
        se = float(np.sqrt(np.mean(psi**2) / denom**2 / n))
        thetas.append(theta)
        ses.append(se)
        psis.append(psi)
        rl.append(float(np.sqrt(np.mean(u**2))))
        rm.append(float(np.sqrt(np.mean(v**2))))
    rmse = {"ml_l": float(np.median(rl)), "ml_m": float(np.median(rm))}
    return _aggregate("plr", "theta", thetas, ses, psis, n, len(splits[0]), rmse, level, se_aggregation)


def dml_irm(
    X,
    treatment,
    outcome,
    ml_g=None,
    ml_m=None,
    target: str = "ate",
    n_folds: int = 5,
    n_rep: int = 1,
    trimming: float = 0.01,
    random_state=None,
    folds=None,
    level: float = 0.95,
    se_aggregation: str = "median_variance",
) -> DMLResult:
    """DML for the interactive regression model (binary treatment), AIPW score.

    ``ml_g`` learns ``E[Y | X]`` separately in each arm and ``ml_m`` the
    propensity score, which is clipped to ``[trimming, 1 - trimming]``.
    ``target`` is ``"ate"`` or ``"att"``. Folds are stratified by treatment.
    """
    X, d, y = _prepare(X, treatment, outcome)
    _check_agg(se_aggregation)
    if not np.all(np.isin(d, (0.0, 1.0))):
        raise ValueError("dml_irm requires a binary 0/1 treatment")
    if target not in ("ate", "att"):
        raise ValueError("target must be 'ate' or 'att'")
    if not (0.0 <= trimming < 0.5):
        raise ValueError("trimming must be in [0, 0.5)")
    n = y.size
    ml_g = PolynomialRidge() if ml_g is None else ml_g
    ml_m = LogisticClassifier() if ml_m is None else ml_m
    rng = np.random.default_rng(random_state)
    splits = (
        _splits_from(folds, n)
        if folds is not None
        else [_make_folds(n, int(n_folds), rng, strata=d) for _ in range(int(n_rep))]
    )
    thetas, ses, psis, rg0, rg1, rm = [], [], [], [], [], []
    for sp in splits:
        g0 = np.empty(n)
        g1 = np.empty(n)
        m = np.empty(n)
        for tr, te in sp:
            t1 = tr[d[tr] == 1]
            t0 = tr[d[tr] == 0]
            if t1.size == 0 or t0.size == 0:
                raise ValueError("a training fold lacks treated or control units")
            g1[te] = np.asarray(_clone(ml_g).fit(X[t1], y[t1]).predict(X[te]), dtype=float)
            g0[te] = np.asarray(_clone(ml_g).fit(X[t0], y[t0]).predict(X[te]), dtype=float)
            m[te] = _predict_proba1(_clone(ml_m).fit(X[tr], d[tr]), X[te])
        m = np.clip(m, trimming, 1.0 - trimming)
        if target == "ate":
            psi_b = g1 - g0 + d * (y - g1) / m - (1 - d) * (y - g0) / (1 - m)
            psi_a = -np.ones(n)
        else:
            p = d.mean()
            psi_b = d * (y - g0) / p - m * (1 - d) * (y - g0) / (p * (1 - m))
            psi_a = -d / p
        theta = float(-np.mean(psi_b) / np.mean(psi_a))
        psi = psi_a * theta + psi_b
        J = float(np.mean(psi_a))
        se = float(np.sqrt(np.mean(psi**2) / J**2 / n))
        thetas.append(theta)
        ses.append(se)
        psis.append(psi)
        rg0.append(float(np.sqrt(np.mean((y - g0)[d == 0] ** 2))))
        rg1.append(float(np.sqrt(np.mean((y - g1)[d == 1] ** 2))))
        rm.append(float(np.sqrt(np.mean((d - m) ** 2))))
    rmse = {"ml_g0": float(np.median(rg0)), "ml_g1": float(np.median(rg1)), "ml_m": float(np.median(rm))}
    return _aggregate("irm", target.upper(), thetas, ses, psis, n, len(splits[0]), rmse, level, se_aggregation)
