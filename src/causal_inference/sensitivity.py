"""Sensitivity analysis for unmeasured confounding.

Two complementary tools:

* **E-values** (VanderWeele & Ding, *Annals of Internal Medicine*, 2017).
  The E-value is the minimum strength of association, on the risk-ratio
  scale, that an unmeasured confounder would need with *both* the treatment
  and the outcome (conditional on the measured covariates) to fully explain
  away an observed association.  For a risk ratio ``RR >= 1``

      E = RR + sqrt(RR * (RR - 1))

  (protective effects use ``1 / RR``).  The E-value for the confidence
  limit closest to the null says how strong confounding must be to move
  the interval to include 1.  Odds ratios, hazard ratios and standardized
  mean differences are first converted to approximate risk ratios.

* **Rosenbaum bounds** for matched-pair designs (Rosenbaum, *Observational
  Studies*, 2002).  Within each pair, hidden bias may change the odds of
  treatment by at most a factor ``Gamma``.  For the Wilcoxon signed-rank
  statistic this bounds the probability that a pair's difference is
  positive between ``1 / (1 + Gamma)`` and ``Gamma / (1 + Gamma)``, which
  yields bounds on the one-sided p-value.  ``Gamma = 1`` is the usual
  randomization test; the *sensitivity value* is the smallest ``Gamma`` at
  which the upper-bound p-value crosses ``alpha``.

Only ``numpy`` is required; the normal tail uses :func:`math.erfc`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "EValueResult",
    "RosenbaumBoundsResult",
    "bias_factor",
    "e_value",
    "e_value_from_ate",
    "rosenbaum_bounds",
    "rosenbaum_sensitivity_value",
]

_MEASURES = ("RR", "OR", "HR", "SMD")


@dataclass
class EValueResult:
    """E-values for a point estimate and its confidence interval.

    Attributes
    ----------
    measure : str
        Input effect measure (``"RR"``, ``"OR"``, ``"HR"`` or ``"SMD"``).
    estimate, lower, upper : float or None
        The input estimate and confidence limits on their original scale.
    rr, rr_lower, rr_upper : float or None
        The same quantities on the (approximate) risk-ratio scale.
    e_value : float
        E-value for the point estimate (``1.0`` when ``rr == 1``).
    e_value_ci : float or None
        E-value for the confidence limit closest to the null; ``1.0`` when
        the interval already contains the null.  ``None`` without limits.
    """

    measure: str
    estimate: float
    lower: Optional[float]
    upper: Optional[float]
    rr: float
    rr_lower: Optional[float]
    rr_upper: Optional[float]
    e_value: float
    e_value_ci: Optional[float]


def _rr_e_value(rr: float) -> float:
    if rr < 1.0:
        rr = 1.0 / rr
    return float(rr + math.sqrt(rr * (rr - 1.0)))


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _to_rr(value: float, measure: str, rare: bool) -> float:
    if measure == "RR":
        return _positive(value, "estimate")
    if measure == "OR":
        value = _positive(value, "estimate")
        return value if rare else math.sqrt(value)
    if measure == "HR":
        value = _positive(value, "estimate")
        if rare:
            return value
        root = math.sqrt(value)
        return (1.0 - 0.5**root) / (1.0 - 0.5 ** (1.0 / root))
    # SMD (Cohen's d) -> RR ~= exp(0.91 d)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("estimate must be finite")
    return math.exp(0.91 * value)


def e_value(
    estimate: float,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    *,
    measure: str = "RR",
    rare: bool = False,
    se: Optional[float] = None,
) -> EValueResult:
    """E-value for a risk ratio, odds ratio, hazard ratio or standardized difference.

    Parameters
    ----------
    estimate :
        Point estimate on the scale given by ``measure``.
    lower, upper :
        Optional confidence limits on the same scale.
    measure :
        ``"RR"`` (risk ratio), ``"OR"`` (odds ratio), ``"HR"`` (hazard ratio)
        or ``"SMD"`` (standardized mean difference, Cohen's d).
    rare :
        For OR / HR: treat the outcome as rare (<15%), so the ratio already
        approximates a risk ratio.  Otherwise ``RR ~= sqrt(OR)`` and the
        VanderWeele HR conversion are used.
    se :
        For ``"SMD"`` only: standard error of ``d``.  When given (and no
        limits are), the interval ``exp(0.91 d -/+ 1.78 se)`` is used, as in
        VanderWeele & Ding (2017).
    """
    measure = str(measure).upper()
    if measure not in _MEASURES:
        raise ValueError(f"measure must be one of {_MEASURES}")
    if (lower is None) != (upper is None):
        raise ValueError("provide both lower and upper, or neither")
    if se is not None and measure != "SMD":
        raise ValueError("se is only used with measure='SMD'")

    rr = _to_rr(estimate, measure, rare)
    rr_lower = rr_upper = None
    if lower is not None:
        if float(lower) > float(estimate) or float(upper) < float(estimate):
            raise ValueError("confidence limits must satisfy lower <= estimate <= upper")
        rr_lower = _to_rr(lower, measure, rare)
        rr_upper = _to_rr(upper, measure, rare)
    elif se is not None:
        se = float(se)
        if not math.isfinite(se) or se < 0.0:
            raise ValueError("se must be a non-negative finite number")
        rr_lower = math.exp(0.91 * float(estimate) - 1.78 * se)
        rr_upper = math.exp(0.91 * float(estimate) + 1.78 * se)

    point = 1.0 if rr == 1.0 else _rr_e_value(rr)
    ci_value: Optional[float] = None
    if rr_lower is not None and rr_upper is not None:
        if rr_lower <= 1.0 <= rr_upper:
            ci_value = 1.0
        elif rr > 1.0:
            ci_value = _rr_e_value(rr_lower)
        else:
            ci_value = _rr_e_value(rr_upper)
    return EValueResult(
        measure=measure,
        estimate=float(estimate),
        lower=None if lower is None else float(lower),
        upper=None if upper is None else float(upper),
        rr=float(rr),
        rr_lower=None if rr_lower is None else float(rr_lower),
        rr_upper=None if rr_upper is None else float(rr_upper),
        e_value=float(point),
        e_value_ci=None if ci_value is None else float(ci_value),
    )


def e_value_from_ate(
    ate: float,
    outcome_sd: float,
    se: Optional[float] = None,
) -> EValueResult:
    """E-value for a continuous-outcome effect (e.g. from ``aipw_ate``).

    The ATE is standardized as ``d = ate / outcome_sd`` and converted with
    ``RR ~= exp(0.91 d)``.  ``se`` is the standard error of the ATE on the
    original scale; it is standardized the same way.
    """
    sd = _positive(outcome_sd, "outcome_sd")
    d = float(ate) / sd
    d_se = None if se is None else float(se) / sd
    return e_value(d, measure="SMD", se=d_se)


def bias_factor(rr_eu: float, rr_ud: float) -> float:
    """Ding & VanderWeele (2016) joint bounding factor.

    ``rr_eu`` is the treatment-confounder risk ratio and ``rr_ud`` the
    confounder-outcome risk ratio.  An observed RR divided by this factor is
    the most the confounder could shift it toward the null; ``bias_factor(E,
    E)`` equals the observed RR exactly when ``E`` is its E-value.
    """
    a = _positive(rr_eu, "rr_eu")
    b = _positive(rr_ud, "rr_ud")
    if a < 1.0 or b < 1.0:
        raise ValueError("rr_eu and rr_ud must be >= 1")
    return float(a * b / (a + b - 1.0))


# ---------------------------------------------------------------- Rosenbaum
def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    n = values.shape[0]
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


@dataclass
class RosenbaumBoundsResult:
    """Wilcoxon signed-rank Rosenbaum bounds over a grid of ``Gamma``.

    Attributes
    ----------
    gammas : list of float
        Sensitivity parameters evaluated (``Gamma >= 1``).
    p_upper, p_lower : list of float
        Upper / lower bounds on the one-sided p-value for a positive effect.
    statistic : float
        Observed signed-rank statistic ``T+`` (sum of ranks of positive
        differences, average ranks for ties).
    n_pairs : int
        Number of pairs with a non-zero difference.
    n_zero : int
        Number of zero differences dropped.
    """

    gammas: List[float]
    p_upper: List[float]
    p_lower: List[float]
    statistic: float
    n_pairs: int
    n_zero: int
    alternative: str = "greater"
    rows: List[dict] = field(default_factory=list)


def _signed_rank_parts(differences, alternative: str):
    d = np.asarray(differences, dtype=np.float64).ravel()
    if d.size == 0 or not np.all(np.isfinite(d)):
        raise ValueError("differences must be a non-empty array of finite values")
    if alternative not in ("greater", "less"):
        raise ValueError("alternative must be 'greater' or 'less'")
    if alternative == "less":
        d = -d
    nonzero = d[d != 0.0]
    n_zero = int(d.size - nonzero.size)
    if nonzero.size == 0:
        raise ValueError("all paired differences are zero")
    ranks = _average_ranks(np.abs(nonzero))
    t_plus = float(ranks[nonzero > 0].sum())
    return ranks, t_plus, int(nonzero.size), n_zero


def _bounds_for_gamma(ranks: np.ndarray, t_plus: float, gamma: float, continuity: bool):
    sum_r = float(ranks.sum())
    sum_r2 = float(np.sum(ranks**2))
    cc = 0.5 if continuity else 0.0
    out = []
    for p in (gamma / (1.0 + gamma), 1.0 / (1.0 + gamma)):
        mean = p * sum_r
        var = p * (1.0 - p) * sum_r2
        z = (t_plus - mean - cc) / math.sqrt(var)
        out.append(_norm_sf(z))
    return out[0], out[1]


def rosenbaum_bounds(
    differences: Sequence[float],
    gammas: Sequence[float] = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0),
    *,
    alternative: str = "greater",
    continuity: bool = True,
) -> RosenbaumBoundsResult:
    """Rosenbaum sensitivity bounds for the Wilcoxon signed-rank test.

    Parameters
    ----------
    differences :
        Treated-minus-control outcome difference for each matched pair.
    gammas :
        Values of the sensitivity parameter ``Gamma >= 1``.
    alternative :
        ``"greater"`` tests a positive effect; ``"less"`` a negative one.
    continuity :
        Apply a 0.5 continuity correction to the normal approximation.
    """
    gammas = [float(g) for g in gammas]
    if not gammas or any(not math.isfinite(g) or g < 1.0 for g in gammas):
        raise ValueError("gammas must be a non-empty list of values >= 1")
    ranks, t_plus, n_pairs, n_zero = _signed_rank_parts(differences, alternative)
    p_upper: List[float] = []
    p_lower: List[float] = []
    rows: List[dict] = []
    for g in gammas:
        hi, lo = _bounds_for_gamma(ranks, t_plus, g, continuity)
        p_upper.append(hi)
        p_lower.append(lo)
        rows.append({"gamma": g, "p_lower": lo, "p_upper": hi})
    return RosenbaumBoundsResult(
        gammas=gammas,
        p_upper=p_upper,
        p_lower=p_lower,
        statistic=t_plus,
        n_pairs=n_pairs,
        n_zero=n_zero,
        alternative=alternative,
        rows=rows,
    )


def rosenbaum_sensitivity_value(
    differences: Sequence[float],
    alpha: float = 0.05,
    *,
    alternative: str = "greater",
    continuity: bool = True,
    gamma_max: float = 100.0,
    tol: float = 1e-6,
) -> float:
    """Smallest ``Gamma`` at which the upper-bound p-value reaches ``alpha``.

    Returns ``1.0`` when the result is not significant even without hidden
    bias, and ``inf`` if it stays significant up to ``gamma_max``.
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not float(gamma_max) > 1.0:
        raise ValueError("gamma_max must be > 1")
    ranks, t_plus, _, _ = _signed_rank_parts(differences, alternative)

    def upper(g: float) -> float:
        return _bounds_for_gamma(ranks, t_plus, g, continuity)[0]

    if upper(1.0) >= alpha:
        return 1.0
    if upper(float(gamma_max)) < alpha:
        return math.inf
    lo, hi = 1.0, float(gamma_max)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if upper(mid) < alpha:
            lo = mid
        else:
            hi = mid
    return float(hi)
