"""Partial identification: Manski bounds, Lee bounds and Imbens-Manski intervals.

Point estimates need strong assumptions, such as unconfoundedness or random
attrition. Bounds report the set of effects consistent with the data under
weaker assumptions instead.

* **Manski (1990) worst-case bounds** need only a bounded outcome
  ``y_min <= Y <= y_max``. The unobserved potential outcome of each arm is
  replaced by the outcome bounds, so

      E[Y(1)] in [P1 E[Y|T=1] + P0 y_min,  P1 E[Y|T=1] + P0 y_max]

  and similarly for ``E[Y(0)]``. The no-assumption ATE interval always has
  width ``y_max - y_min`` and contains zero. Two shape restrictions tighten
  it (Manski 1997; Manski & Pepper 2000):

  - *monotone treatment response* (``"mtr"``, ``Y(1) >= Y(0)``) raises the
    lower bound to 0;
  - *monotone treatment selection* (``"mts"``,
    ``E[Y(t)|T=1] >= E[Y(t)|T=0]``) caps the upper bound at the naive
    difference in means.

* **Lee (2009) bounds** handle sample selection or attrition after random
  assignment. Outcomes are observed only when ``S = 1``. Under monotonicity
  (treatment can only move selection in one direction), the arm with the
  higher observation rate contains a share ``p = |p1 - p0| / max(p1, p0)``
  of "marginal" units. Trimming that share from the top, or from the bottom,
  of its outcome distribution gives sharp bounds on the effect for units
  that would be observed under either assignment. Trimming is exact
  (fractional at the boundary), so ties and small samples are handled
  without rounding.

* **Imbens & Manski (2004) confidence intervals** cover the *true effect*,
  not the whole identified set, with probability ``1 - alpha``. The interval
  is ``[lo - C se_lo, hi + C se_hi]``, where ``C`` solves

      Phi(C + width / max(se_lo, se_hi)) - Phi(-C) = 1 - alpha,

  so ``C`` moves from ``z_{1-alpha/2}`` (point identification) towards
  ``z_{1-alpha}`` (wide bounds).

Standard errors of the bounds come from a nonparametric bootstrap that
resamples within arms. Only ``numpy`` is required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

__all__ = [
    "ManskiBoundsResult",
    "LeeBoundsResult",
    "manski_bounds",
    "lee_bounds",
    "trimmed_mean",
    "imbens_manski_interval",
]


def _norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _norm_ppf(p: float) -> float:
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _check_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return alpha


def imbens_manski_interval(
    lower: float, upper: float, se_lower: float, se_upper: float, alpha: float = 0.05
) -> Tuple[float, float, float]:
    """Imbens-Manski (2004) confidence interval for a partially identified parameter.

    Returns ``(ci_lower, ci_upper, critical_value)``.
    """
    alpha = _check_alpha(alpha)
    lower, upper = float(lower), float(upper)
    if not (math.isfinite(lower) and math.isfinite(upper)) or upper < lower:
        raise ValueError("bounds must be finite with lower <= upper")
    if se_lower < 0 or se_upper < 0 or not (math.isfinite(se_lower) and math.isfinite(se_upper)):
        raise ValueError("standard errors must be finite and non-negative")
    sigma = max(se_lower, se_upper)
    if sigma == 0.0:
        return lower, upper, 0.0
    delta = (upper - lower) / sigma
    target = 1.0 - alpha
    lo, hi = 0.0, 40.0
    for _ in range(200):
        c = 0.5 * (lo + hi)
        if _norm_cdf(c + delta) - _norm_cdf(-c) < target:
            lo = c
        else:
            hi = c
    c = 0.5 * (lo + hi)
    return lower - c * se_lower, upper + c * se_upper, c


def trimmed_mean(values, proportion: float, side: str) -> float:
    """Mean after exactly trimming ``proportion`` of the mass from one side.

    ``side="upper"`` drops the largest values; ``side="lower"`` drops the
    smallest. The boundary observation receives fractional weight.
    """
    y = np.sort(np.asarray(values, dtype=np.float64).ravel())
    if y.size == 0 or not np.all(np.isfinite(y)):
        raise ValueError("values must be a non-empty array of finite numbers")
    if not 0.0 <= proportion < 1.0:
        raise ValueError("proportion must lie in [0, 1)")
    if side not in ("upper", "lower"):
        raise ValueError("side must be 'upper' or 'lower'")
    if side == "lower":
        y = y[::-1]
    keep = (1.0 - proportion) * y.size
    k = int(math.floor(keep + 1e-12))
    frac = keep - k
    total = float(y[:k].sum())
    if k < y.size and frac > 1e-12:
        total += frac * float(y[k])
    return total / keep


def _binary(values, name: str) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64).ravel()
    if a.size == 0 or not np.all(np.isin(a, (0.0, 1.0))):
        raise ValueError(f"{name} must be a non-empty binary 0/1 array")
    return a.astype(bool)


def _bootstrap_se(stat, groups, n_boot: int, seed) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = [g[rng.integers(0, g.size, g.size)] for g in groups]
        try:
            draws.append(stat(np.concatenate(idx)))
        except ValueError:
            continue
    if len(draws) < max(10, n_boot // 2):
        raise ValueError("too many degenerate bootstrap resamples; increase the sample size")
    d = np.asarray(draws)
    return float(d[:, 0].std(ddof=1)), float(d[:, 1].std(ddof=1))


@dataclass
class ManskiBoundsResult:
    """Bounds on the ATE for a bounded outcome."""

    lower: float
    upper: float
    assumptions: Tuple[str, ...]
    naive_difference: float
    p_treated: float
    y_min: float
    y_max: float
    refuted: bool
    se_lower: float = float("nan")
    se_upper: float = float("nan")
    ci_lower: float = float("nan")
    ci_upper: float = float("nan")
    n: int = 0

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def summary(self) -> str:
        tag = "+".join(self.assumptions) or "none"
        out = (
            f"Manski ATE bounds (assumptions: {tag}): [{self.lower:.4f}, {self.upper:.4f}], "
            f"width {self.width:.4f}; naive difference {self.naive_difference:.4f}"
        )
        if math.isfinite(self.ci_lower):
            out += f"; Imbens-Manski CI [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        if self.refuted:
            out += " (assumptions refuted by the data)"
        return out


def _manski_point(t, y, y_min, y_max, assumptions):
    if t.all() or not t.any():
        raise ValueError("both treatment arms must be present")
    p1 = float(t.mean())
    p0 = 1.0 - p1
    m1 = float(y[t].mean())
    m0 = float(y[~t].mean())
    naive = m1 - m0
    lower = (p1 * m1 + p0 * y_min) - (p0 * m0 + p1 * y_max)
    upper = (p1 * m1 + p0 * y_max) - (p0 * m0 + p1 * y_min)
    if "mtr" in assumptions:
        lower = max(lower, 0.0)
    if "mts" in assumptions:
        upper = min(upper, naive)
    return lower, upper, naive, p1


def manski_bounds(
    treatment,
    outcome,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    assumptions: Iterable[str] = (),
    *,
    n_boot: int = 0,
    alpha: float = 0.05,
    random_state=None,
) -> ManskiBoundsResult:
    """Worst-case (Manski) ATE bounds, optionally tightened by MTR and/or MTS.

    ``y_min`` / ``y_max`` default to the observed outcome range. Set
    ``n_boot > 0`` to add bootstrap standard errors and an Imbens-Manski
    confidence interval.
    """
    t = _binary(treatment, "treatment")
    y = np.asarray(outcome, dtype=np.float64).ravel()
    if y.shape != t.shape or not np.all(np.isfinite(y)):
        raise ValueError("outcome must be finite and match treatment in length")
    assumptions = tuple(sorted(set(str(a).lower() for a in assumptions)))
    if any(a not in ("mtr", "mts") for a in assumptions):
        raise ValueError("assumptions must be drawn from {'mtr', 'mts'}")
    y_min = float(np.min(y)) if y_min is None else float(y_min)
    y_max = float(np.max(y)) if y_max is None else float(y_max)
    if not y_min < y_max:
        raise ValueError("y_min must be smaller than y_max")
    if y.min() < y_min - 1e-12 or y.max() > y_max + 1e-12:
        raise ValueError("outcomes fall outside [y_min, y_max]")
    lower, upper, naive, p1 = _manski_point(t, y, y_min, y_max, assumptions)
    refuted = lower > upper
    res = ManskiBoundsResult(lower, upper, assumptions, naive, p1, y_min, y_max, refuted, n=int(y.size))
    if n_boot and not refuted:
        if n_boot < 20:
            raise ValueError("n_boot must be 0 or at least 20")
        data = np.column_stack([t, y])

        def stat(idx):
            lo, hi, _, _ = _manski_point(data[idx, 0].astype(bool), data[idx, 1], y_min, y_max, assumptions)
            return lo, hi

        groups = [np.flatnonzero(t), np.flatnonzero(~t)]
        res.se_lower, res.se_upper = _bootstrap_se(stat, groups, n_boot, random_state)
        res.ci_lower, res.ci_upper, _ = imbens_manski_interval(
            lower, upper, res.se_lower, res.se_upper, alpha
        )
    return res


@dataclass
class LeeBoundsResult:
    """Lee (2009) sharp bounds under monotone sample selection."""

    lower: float
    upper: float
    trimming_proportion: float
    trimmed_arm: str
    p_observed_treated: float
    p_observed_control: float
    n_observed_treated: int
    n_observed_control: int
    se_lower: float = float("nan")
    se_upper: float = float("nan")
    ci_lower: float = float("nan")
    ci_upper: float = float("nan")
    critical_value: float = float("nan")

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def summary(self) -> str:
        out = (
            f"Lee bounds: [{self.lower:.4f}, {self.upper:.4f}] after trimming "
            f"{self.trimming_proportion:.2%} of the {self.trimmed_arm} arm "
            f"(observed {self.p_observed_treated:.3f} treated vs {self.p_observed_control:.3f} control)"
        )
        if math.isfinite(self.ci_lower):
            out += f"; Imbens-Manski CI [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        return out


def _lee_point(t, s, y):
    if not (t & s).any() or not (~t & s).any():
        raise ValueError("each arm needs at least one observed outcome")
    p1 = float(s[t].mean())
    p0 = float(s[~t].mean())
    y1 = y[t & s]
    y0 = y[~t & s]
    if p1 >= p0:
        p = (p1 - p0) / p1
        lower = trimmed_mean(y1, p, "upper") - float(y0.mean())
        upper = trimmed_mean(y1, p, "lower") - float(y0.mean())
        arm = "treated"
    else:
        p = (p0 - p1) / p0
        lower = float(y1.mean()) - trimmed_mean(y0, p, "lower")
        upper = float(y1.mean()) - trimmed_mean(y0, p, "upper")
        arm = "control"
    return lower, upper, p, arm, p1, p0


def lee_bounds(
    treatment,
    observed,
    outcome,
    *,
    n_boot: int = 0,
    alpha: float = 0.05,
    random_state=None,
) -> LeeBoundsResult:
    """Lee (2009) trimming bounds on the effect for always-observed units.

    Parameters
    ----------
    treatment : array of 0/1
        Randomized assignment.
    observed : array of 0/1
        Selection indicator, 1 when the outcome is observed.
    outcome : array
        Outcomes. Values where ``observed == 0`` are ignored and may be NaN.
    n_boot :
        Bootstrap replications for standard errors and the Imbens-Manski
        interval (0 to skip).
    """
    t = _binary(treatment, "treatment")
    s = _binary(observed, "observed")
    y = np.asarray(outcome, dtype=np.float64).ravel()
    if not (t.shape == s.shape == y.shape):
        raise ValueError("treatment, observed and outcome must have the same length")
    if not np.all(np.isfinite(y[s])):
        raise ValueError("observed outcomes must be finite")
    if t.all() or not t.any():
        raise ValueError("both treatment arms must be present")
    y = np.where(s, y, 0.0)
    lower, upper, p, arm, p1, p0 = _lee_point(t, s, y)
    res = LeeBoundsResult(
        lower, upper, p, arm, p1, p0, int((t & s).sum()), int((~t & s).sum())
    )
    if n_boot:
        if n_boot < 20:
            raise ValueError("n_boot must be 0 or at least 20")

        def stat(idx):
            lo, hi, *_ = _lee_point(t[idx], s[idx], y[idx])
            return lo, hi

        groups = [np.flatnonzero(t), np.flatnonzero(~t)]
        res.se_lower, res.se_upper = _bootstrap_se(stat, groups, n_boot, random_state)
        res.ci_lower, res.ci_upper, res.critical_value = imbens_manski_interval(
            lower, upper, res.se_lower, res.se_upper, alpha
        )
    return res
