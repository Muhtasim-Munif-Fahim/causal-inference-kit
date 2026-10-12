"""Evaluating CATE / uplift models: Qini curves and the rank-weighted ATE.

A CATE model is most often used to *rank* units ("treat the top 20%"), so it
should be judged on how well that ranking works rather than on its point
estimates. Two standard tools:

* :func:`qini_curve` builds Radcliffe's (2007) Qini curve from observed
  outcomes. Units are sorted by predicted uplift, and for the top ``k``
  ``Q(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k)`` (incremental gain, scaled by
  ``1 / n``). The Qini coefficient is the area between that curve and the
  random-targeting diagonal, and the uplift curve
  ``(mean_t(k) - mean_c(k)) * k / n`` is returned alongside. Both assume
  treatment was randomized; for observational data use :func:`rate`.
* :func:`rate` gives the *rank-weighted average treatment effect* of
  Yadlowsky, Fleming, Shah, Brunskill & Wager (2021, JASA 2024). The Targeting
  Operator Characteristic ``TOC(q) = E[tau | top q by score] - ATE`` is
  estimated with doubly robust AIPW scores ``Gamma_i``, so it is valid under
  confounding. Integrating TOC gives ``AUTOC`` (uniform weights) or ``QINI``
  (weights ``q``). A RATE significantly above zero means the score really
  ranks units by their benefit. The standard error comes from a bootstrap
  over units.

The scores should be computed on data the CATE model was not trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from .estimators import _coerce_arrays, _validate_propensity, outcome_regression
from .propensity import propensity_scores

__all__ = [
    "QiniCurveResult",
    "RATEResult",
    "aipw_scores",
    "qini_curve",
    "rate",
]


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral (``np.trapezoid`` only exists in NumPy >= 2)."""
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2.0)


def _ranking(score, n):
    s = np.asarray(score, dtype=float).ravel()
    if s.shape[0] != n:
        raise ValueError("score must have one entry per unit")
    if not np.all(np.isfinite(s)):
        raise ValueError("score must be finite")
    # Highest predicted uplift first; ties keep input order (stable).
    return np.argsort(-s, kind="stable")


@dataclass
class QiniCurveResult:
    """Qini and uplift curves on a grid of targeted fractions."""

    fractions: np.ndarray
    qini: np.ndarray
    random: np.ndarray
    uplift: np.ndarray
    qini_coefficient: float
    auuc: float
    n: int


def qini_curve(score, treatment, outcome, n_points: int = 100) -> QiniCurveResult:
    """Qini and uplift curves for a randomized experiment, with their areas.

    ``fractions`` runs from 0 to 1 in ``n_points`` steps. Curve values are
    divided by ``n``, so ``qini[-1]`` is the overall difference in means times
    the treated share. ``qini_coefficient`` is the trapezoidal area between
    the Qini curve and the random line ``fractions * qini[-1]``, and ``auuc``
    is the area under the uplift curve. Larger values mean better targeting.
    """
    t = np.asarray(treatment, dtype=float).ravel()
    y = np.asarray(outcome, dtype=float).ravel()
    n = t.shape[0]
    if y.shape[0] != n or n == 0:
        raise ValueError("treatment and outcome must be non-empty and the same length")
    if not np.all(np.isin(t, (0.0, 1.0))):
        raise ValueError("treatment must be binary 0/1")
    if not (t.any() and (1 - t).any()):
        raise ValueError("both treatment groups must be present")
    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError("n_points must be an integer >= 2")
    order = _ranking(score, n)
    t_s, y_s = t[order], y[order]
    n_t = np.concatenate([[0.0], np.cumsum(t_s)])
    n_c = np.concatenate([[0.0], np.cumsum(1.0 - t_s)])
    y_t = np.concatenate([[0.0], np.cumsum(y_s * t_s)])
    y_c = np.concatenate([[0.0], np.cumsum(y_s * (1.0 - t_s))])

    fractions = np.linspace(0.0, 1.0, int(n_points) + 1)
    k = np.rint(fractions * n).astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(n_c[k] > 0, n_t[k] / n_c[k], 0.0)
        qini = (y_t[k] - y_c[k] * ratio) / n
        mean_t = np.where(n_t[k] > 0, y_t[k] / np.maximum(n_t[k], 1), 0.0)
        mean_c = np.where(n_c[k] > 0, y_c[k] / np.maximum(n_c[k], 1), 0.0)
    uplift = np.where((n_t[k] > 0) & (n_c[k] > 0), (mean_t - mean_c) * k / n, 0.0)
    random_line = fractions * qini[-1]
    return QiniCurveResult(
        fractions=fractions,
        qini=qini,
        random=random_line,
        uplift=uplift,
        qini_coefficient=_trapezoid(qini - random_line, fractions),
        auuc=_trapezoid(uplift, fractions),
        n=int(n),
    )


def aipw_scores(X, treatment, outcome, propensity=None, W=None) -> np.ndarray:
    """Per-unit doubly robust (AIPW) scores, whose mean is the AIPW ATE.

    ``Gamma_i = m1 - m0 + T (Y - m1) / e - (1 - T)(Y - m0) / (1 - e)``, using
    OLS outcome models and a logistic propensity score unless ``propensity``
    is supplied.
    """
    X, treatment, outcome = _coerce_arrays(X, treatment, outcome)
    if propensity is None:
        p, _ = propensity_scores(X, treatment)
    else:
        p = _validate_propensity(propensity, treatment.shape[0])
    if not ((treatment == 1).any() and (treatment == 0).any()):
        raise ValueError("both treatment groups must be present")
    mu1, mu0 = outcome_regression(X, treatment, outcome, W=W)
    return (
        mu1
        - mu0
        + treatment * (outcome - mu1) / p
        - (1.0 - treatment) * (outcome - mu0) / (1.0 - p)
    )


@dataclass
class RATEResult:
    """Rank-weighted average treatment effect with a bootstrap standard error."""

    target: str
    estimate: float
    std_err: float
    ci_low: float
    ci_high: float
    toc_fractions: np.ndarray
    toc: np.ndarray
    n: int

    @property
    def z(self) -> float:
        return self.estimate / self.std_err if self.std_err > 0 else float("inf")


def _rate_from_sorted(gamma_sorted: np.ndarray, target: str) -> float:
    n = gamma_sorted.shape[0]
    ks = np.arange(1, n + 1)
    toc = np.cumsum(gamma_sorted) / ks - gamma_sorted.mean()
    if target == "AUTOC":
        return float(toc.mean())
    return float(np.mean(ks / n * toc))


def rate(
    score,
    treatment=None,
    outcome=None,
    X=None,
    *,
    propensity=None,
    W=None,
    dr_scores=None,
    target: str = "AUTOC",
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    seed: int = 0,
    n_points: int = 100,
) -> RATEResult:
    """Rank-weighted ATE (AUTOC or QINI) of a prioritization ``score``.

    Pass ``dr_scores`` (for example from :func:`aipw_scores` on held-out
    data), or pass ``treatment``, ``outcome`` and ``X`` and the AIPW scores
    are computed here. ``TOC(q)`` is the mean ``Gamma`` among the top
    ``ceil(q n)`` units minus the overall mean. ``AUTOC`` averages it over
    ``q``, and ``QINI`` averages ``q * TOC(q)``. ``toc`` is reported on an
    ``n_points`` grid.
    """
    target = str(target).upper()
    if target not in ("AUTOC", "QINI"):
        raise ValueError("target must be 'AUTOC' or 'QINI'")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 0:
        raise ValueError("n_bootstrap must be a non-negative integer")
    if dr_scores is None:
        if treatment is None or outcome is None or X is None:
            raise ValueError("pass dr_scores, or treatment, outcome and X")
        gamma = aipw_scores(X, treatment, outcome, propensity=propensity, W=W)
    else:
        gamma = np.asarray(dr_scores, dtype=float).ravel()
        if not np.all(np.isfinite(gamma)):
            raise ValueError("dr_scores must be finite")
    n = gamma.shape[0]
    if n < 2:
        raise ValueError("need at least two units")
    order = _ranking(score, n)
    g_sorted = gamma[order]
    estimate = _rate_from_sorted(g_sorted, target)

    fractions = np.linspace(0.0, 1.0, int(n_points) + 1)[1:]
    k = np.maximum(1, np.ceil(fractions * n).astype(int))
    toc = np.cumsum(g_sorted)[k - 1] / k - g_sorted.mean()

    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(n)
        boots = np.empty(int(n_bootstrap))
        for b in range(int(n_bootstrap)):
            idx = rng.integers(0, n, size=n)
            # Keep the original ranking of the resampled units.
            idx = idx[np.argsort(ranks[idx], kind="stable")]
            boots[b] = _rate_from_sorted(gamma[idx], target)
        std_err = float(boots.std(ddof=1)) if n_bootstrap > 1 else 0.0
    else:
        std_err = float("nan")
    # Normal-approximation interval around the point estimate.
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    return RATEResult(
        target=target,
        estimate=estimate,
        std_err=std_err,
        ci_low=estimate - z * std_err,
        ci_high=estimate + z * std_err,
        toc_fractions=fractions,
        toc=toc,
        n=int(n),
    )
