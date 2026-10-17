"""Goodman-Bacon decomposition of the staggered two-way fixed-effects DiD.

Goodman-Bacon (2021), "Difference-in-differences with variation in
treatment timing", *Journal of Econometrics* 225(2).

With staggered adoption the TWFE coefficient ``tau`` from
``Y_it = a_i + b_t + tau D_it + e_it`` on a balanced panel is *exactly*
a weighted average of every 2x2 DiD that can be formed from the timing
groups:

- **treated vs never treated** (``k`` vs ``U``),
- **earlier vs later treated** (``k`` treated, ``l`` not yet treated as the
  control, using periods before ``l`` adopts),
- **later vs earlier treated** (``l`` treated, the already-treated ``k`` as
  the control, using periods after ``k`` adopted) -- the comparison that
  biases TWFE when effects change over time,
- **later vs always treated** (units treated in every period act as an
  already-treated control).

Weights are non-negative, sum to one, and grow with group size and the
variance of the treatment indicator in each 2x2 subsample (timing groups
treated in the middle of the panel get the most weight). Inspecting how
much weight lands on "later vs earlier" comparisons, and whether their
estimates differ from the clean ones, is the standard diagnostic before
trusting a staggered TWFE estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "BaconComparison",
    "BaconDecompositionResult",
    "bacon_decomposition",
]

COMPARISON_TYPES = (
    "treated_vs_never",
    "earlier_vs_later",
    "later_vs_earlier",
    "later_vs_always",
)


@dataclass
class BaconComparison:
    """One 2x2 DiD in the decomposition."""

    kind: str
    treated: object
    control: object
    estimate: float
    weight: float


@dataclass
class BaconDecompositionResult:
    """TWFE estimate with its exact 2x2 decomposition."""

    twfe_estimate: float
    comparisons: List[BaconComparison]
    by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    n_units: int = 0
    n_periods: int = 0

    @property
    def weighted_estimate(self) -> float:
        """``sum(weight * estimate)``; equals :attr:`twfe_estimate` up to rounding."""
        return float(sum(c.weight * c.estimate for c in self.comparisons))

    @property
    def total_weight(self) -> float:
        return float(sum(c.weight for c in self.comparisons))

    @property
    def forbidden_weight(self) -> float:
        """Weight on comparisons that use already-treated units as controls."""
        return float(
            sum(c.weight for c in self.comparisons if c.kind in ("later_vs_earlier", "later_vs_always"))
        )

    def to_frame(self):
        """Comparisons as a pandas DataFrame (requires pandas)."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "type": c.kind,
                    "treated": c.treated,
                    "control": c.control,
                    "estimate": c.estimate,
                    "weight": c.weight,
                }
                for c in self.comparisons
            ]
        )


def _never_treated(value, last_time) -> bool:
    if value is None:
        return True
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"cohort values must be numeric, got {value!r}") from None
    return (not np.isfinite(v)) or v <= 0 or v > last_time


def bacon_decomposition(unit, time, outcome, cohort) -> BaconDecompositionResult:
    """Decompose the staggered TWFE DiD into weighted 2x2 comparisons.

    Parameters
    ----------
    unit, time, outcome : array-like of shape (n,)
        Long-format balanced panel (every unit observed once in every
        period).
    cohort : array-like of shape (n,)
        First treated period for each row's unit (constant within unit),
        using the same convention as :func:`staggered_did`: ``0``,
        negative, ``NaN`` or a period after the panel ends mean never
        treated. A cohort at or before the first period means always
        treated. Treatment is absorbing: ``D_it = 1{time >= cohort}``.

    Returns
    -------
    BaconDecompositionResult
        ``twfe_estimate`` (the TWFE coefficient), every 2x2
        ``comparisons`` with estimate and weight, and ``by_type`` totals
        (``weight`` and weight-averaged ``estimate`` per comparison type).
    """
    unit = np.asarray(unit)
    time = np.asarray(time)
    outcome = np.asarray(outcome, dtype=float)
    cohort = np.asarray(cohort, dtype=object)
    n = unit.shape[0]
    if not (time.shape[0] == outcome.shape[0] == cohort.shape[0] == n):
        raise ValueError("unit, time, outcome and cohort must have the same length")
    if n == 0:
        raise ValueError("empty panel")
    if not np.all(np.isfinite(outcome)):
        raise ValueError("outcome contains non-finite values")

    units, u_idx = np.unique(unit, return_inverse=True)
    times, t_idx = np.unique(time, return_inverse=True)
    N, T = units.shape[0], times.shape[0]
    if T < 2:
        raise ValueError("need at least two periods")
    if n != N * T:
        raise ValueError("panel must be balanced: each unit observed once per period")
    Y = np.full((N, T), np.nan)
    Y[u_idx, t_idx] = outcome
    seen = np.zeros((N, T), dtype=int)
    np.add.at(seen, (u_idx, t_idx), 1)
    if np.any(seen != 1):
        raise ValueError("panel must be balanced: each unit observed once per period")

    last_time = float(times[-1])
    unit_group: List[object] = [None] * N
    assigned = [False] * N
    for i in range(n):
        u = u_idx[i]
        g = None if _never_treated(cohort[i], last_time) else float(cohort[i])
        if assigned[u] and unit_group[u] != g:
            raise ValueError(f"unit {units[u]!r} has multiple cohort values")
        unit_group[u] = g
        assigned[u] = True

    # Treatment start index per unit (T = never treated, 0 = always treated).
    start = np.array(
        [T if g is None else int(np.searchsorted(times.astype(float), g, side="left")) for g in unit_group]
    )
    D = (np.arange(T)[None, :] >= start[:, None]).astype(float)

    def _two_way_demean(M):
        return M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + M.mean()

    Dd = _two_way_demean(D)
    var_d = float(np.mean(Dd**2))
    if var_d <= 1e-15:
        raise ValueError("treatment has no variation after removing unit and time effects")
    twfe = float(np.sum(Dd * _two_way_demean(Y)) / np.sum(Dd**2))

    # Timing groups keyed by start index.
    starts = sorted(set(int(s) for s in start))
    share = {s: float(np.mean(start == s)) for s in starts}
    dbar = {s: (T - s) / T for s in starts}
    path = {s: Y[start == s].mean(axis=0) for s in starts}
    label = {s: (None if s == T else times[s]) for s in starts}
    never = T if T in starts else None
    always = 0 if 0 in starts else None
    timed = [s for s in starts if 0 < s < T]

    comps: List[BaconComparison] = []

    def _dd(treated, control, pre, post):
        return float(
            (path[treated][post].mean() - path[treated][pre].mean())
            - (path[control][post].mean() - path[control][pre].mean())
        )

    for k in timed:
        if never is not None:
            nk, nu = share[k], share[never]
            n_ku = nk / (nk + nu)
            w = (nk + nu) ** 2 * n_ku * (1 - n_ku) * dbar[k] * (1 - dbar[k]) / var_d
            est = _dd(k, never, slice(0, k), slice(k, T))
            comps.append(BaconComparison("treated_vs_never", label[k], None, est, float(w)))
    for a, k in enumerate(timed):
        for l in timed[a + 1:]:
            nk, nl = share[k], share[l]
            n_kl = nk / (nk + nl)
            dk, dl = dbar[k], dbar[l]
            # k (earlier) treated, l (later) not yet treated; periods [0, l)
            w_k = (
                ((nk + nl) * (1 - dl)) ** 2 * n_kl * (1 - n_kl)
                * ((dk - dl) / (1 - dl)) * ((1 - dk) / (1 - dl)) / var_d
            )
            est_k = _dd(k, l, slice(0, k), slice(k, l))
            comps.append(BaconComparison("earlier_vs_later", label[k], label[l], est_k, float(w_k)))
            # l (later) treated, k already treated; periods [k, T)
            w_l = (
                ((nk + nl) * dk) ** 2 * n_kl * (1 - n_kl)
                * (dl / dk) * ((dk - dl) / dk) / var_d
            )
            est_l = _dd(l, k, slice(k, l), slice(l, T))
            comps.append(BaconComparison("later_vs_earlier", label[l], label[k], est_l, float(w_l)))
    if always is not None:
        for l in timed:
            na, nl = share[always], share[l]
            n_al = na / (na + nl)
            dl = dbar[l]
            # always-treated (dbar = 1) as the already-treated control
            w = (na + nl) ** 2 * n_al * (1 - n_al) * dl * (1 - dl) / var_d
            est = _dd(l, always, slice(0, l), slice(l, T))
            comps.append(BaconComparison("later_vs_always", label[l], label[always], est, float(w)))

    by_type: Dict[str, Dict[str, float]] = {}
    for kind in COMPARISON_TYPES:
        rows = [c for c in comps if c.kind == kind]
        if not rows:
            continue
        wsum = float(sum(c.weight for c in rows))
        avg = float(sum(c.weight * c.estimate for c in rows) / wsum) if wsum > 0 else float("nan")
        by_type[kind] = {"weight": wsum, "estimate": avg, "n_comparisons": float(len(rows))}

    return BaconDecompositionResult(
        twfe_estimate=twfe,
        comparisons=comps,
        by_type=by_type,
        n_units=N,
        n_periods=T,
    )
