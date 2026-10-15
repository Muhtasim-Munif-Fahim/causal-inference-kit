"""Staggered adoption difference-in-differences (group-time ATT).

Implements a Callaway & Sant'Anna (2021)-style *group-time* ATT for
panels with staggered treatment adoption. Each cohort ``g`` (first
treatment period) is compared, at calendar time ``t``, against units
that are still untreated at ``t`` (never-treated, or not-yet-treated).

For each pair ``(g, t)`` with ``t >= g``:

    ATT(g, t) = (Ybar_{G=g,t} - Ybar_{G=g, g-1})
              - (Ybar_{C(t),t} - Ybar_{C(t), g-1})

where ``C(t)`` is the control group at ``t``. Aggregates report the
simple average of all group-time ATTs and a cohort-size-weighted
overall ATT. Optional bootstrap standard errors resample units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "GroupTimeATT",
    "StaggeredDiDResult",
    "staggered_did",
    "simulate_staggered_did_data",
]


@dataclass
class GroupTimeATT:
    """One group-time average treatment effect on the treated."""

    cohort: object
    time: object
    att: float
    n_treated: int
    n_control: int


@dataclass
class StaggeredDiDResult:
    """Staggered DiD fit with group-time and aggregated ATTs."""

    group_time: List[GroupTimeATT]
    overall_att: float
    overall_att_weighted: float
    overall_se: Optional[float]
    n_units: int
    n_cohorts: int
    control_group: str

    @property
    def estimate(self) -> float:
        return self.overall_att_weighted


def _panel_means(
    unit: np.ndarray,
    time: np.ndarray,
    outcome: np.ndarray,
    cohort: np.ndarray,
) -> Tuple[Dict[Tuple[object, object], float], Dict[object, object], Dict[object, int]]:
    """Return mean outcome per (cohort_label, time), unit->cohort, cohort sizes.

    Never-treated units are labelled with cohort ``None``.
    """
    unit_ids = np.unique(unit)
    unit_cohort: Dict[object, object] = {}
    for u in unit_ids:
        mask = unit == u
        vals = np.unique(cohort[mask])
        if vals.size != 1:
            raise ValueError(f"unit {u!r} has multiple cohort values {vals}")
        g = vals[0]
        # Sentinel: cohort <= 0 or nan means never treated
        if g is None or (isinstance(g, float) and not np.isfinite(g)) or g <= 0:
            unit_cohort[u] = None
        else:
            unit_cohort[u] = g

    means: Dict[Tuple[object, object], List[float]] = {}
    for i in range(unit.shape[0]):
        g = unit_cohort[unit[i]]
        key = (g, time[i])
        means.setdefault(key, []).append(float(outcome[i]))
    mean_map = {k: float(np.mean(v)) for k, v in means.items()}

    sizes: Dict[object, int] = {}
    for g in unit_cohort.values():
        sizes[g] = sizes.get(g, 0) + 1
    return mean_map, unit_cohort, sizes


def _control_cohorts(
    unit_cohort: Dict[object, object],
    t,
    control_group: str,
) -> List[object]:
    if control_group == "never":
        return [None] if None in set(unit_cohort.values()) else []
    if control_group == "notyet":
        # Cohorts with g > t, plus never-treated
        out = []
        seen = set()
        for g in unit_cohort.values():
            if g in seen:
                continue
            seen.add(g)
            if g is None or g > t:
                out.append(g)
        return out
    raise ValueError("control_group must be 'never' or 'notyet'")


def _att_gt(
    mean_map,
    sizes,
    unit_cohort,
    g,
    t,
    control_group: str,
) -> Optional[GroupTimeATT]:
    pre = None
    # pre-period = period immediately before g among observed times for cohort g
    times_for_g = sorted({tt for (gg, tt) in mean_map if gg == g})
    earlier = [tt for tt in times_for_g if tt < g]
    if not earlier:
        return None
    pre = earlier[-1]
    if (g, t) not in mean_map or (g, pre) not in mean_map:
        return None
    ctrl = _control_cohorts(unit_cohort, t, control_group)
    # Aggregate control means across control cohort labels
    y_c_t, y_c_pre, n_c = [], [], 0
    for c in ctrl:
        if (c, t) in mean_map and (c, pre) in mean_map:
            y_c_t.append(mean_map[(c, t)])
            y_c_pre.append(mean_map[(c, pre)])
            n_c += sizes.get(c, 0)
    if n_c == 0 or not y_c_t:
        return None
    treated_delta = mean_map[(g, t)] - mean_map[(g, pre)]
    control_delta = float(np.mean(y_c_t)) - float(np.mean(y_c_pre))
    att = treated_delta - control_delta
    return GroupTimeATT(
        cohort=g,
        time=t,
        att=float(att),
        n_treated=int(sizes.get(g, 0)),
        n_control=int(n_c),
    )


def staggered_did(
    unit,
    time,
    outcome,
    cohort,
    *,
    control_group: str = "notyet",
    bootstrap: int = 0,
    seed: Optional[int] = None,
) -> StaggeredDiDResult:
    """Estimate group-time and aggregated ATTs under staggered adoption.

    Parameters
    ----------
    unit, time, outcome :
        Long panel columns.
    cohort :
        First treatment period for each observation (constant within unit).
        Use ``0``, ``-1``, or ``nan`` for never-treated units.
    control_group : {'never', 'notyet'}
        Comparison group for each ``(g, t)``.
    bootstrap :
        Number of unit-level bootstrap draws for the SE of the weighted
        overall ATT. ``0`` skips the SE.
    seed :
        Bootstrap RNG seed.
    """
    unit = np.asarray(unit).ravel()
    time = np.asarray(time).ravel()
    outcome = np.asarray(outcome, dtype=float).ravel()
    cohort = np.asarray(cohort).ravel()
    n = unit.shape[0]
    if not (time.shape[0] == outcome.shape[0] == cohort.shape[0] == n):
        raise ValueError("unit, time, outcome, cohort must have the same length")
    if n == 0:
        raise ValueError("at least one observation is required")
    if control_group not in {"never", "notyet"}:
        raise ValueError("control_group must be 'never' or 'notyet'")

    mean_map, unit_cohort, sizes = _panel_means(unit, time, outcome, cohort)
    cohorts = sorted({g for g in unit_cohort.values() if g is not None})
    times = sorted(set(time.tolist()))
    group_time: List[GroupTimeATT] = []
    for g in cohorts:
        for t in times:
            if t < g:
                continue
            row = _att_gt(mean_map, sizes, unit_cohort, g, t, control_group)
            if row is not None:
                group_time.append(row)

    if not group_time:
        raise ValueError("no identifiable group-time ATTs; check cohort/time coding")

    overall = float(np.mean([r.att for r in group_time]))
    weights = np.asarray([r.n_treated for r in group_time], dtype=float)
    overall_w = float(np.average([r.att for r in group_time], weights=weights))

    se = None
    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        unit_ids = np.unique(unit)
        boots = []
        for _ in range(int(bootstrap)):
            draw = rng.choice(unit_ids, size=unit_ids.size, replace=True)
            # rebuild panel for drawn units (with replacement → duplicate rows)
            masks = []
            for u in draw:
                masks.append(unit == u)
            # Concatenate selected rows
            idx = np.concatenate([np.flatnonzero(m) for m in masks])
            try:
                res = staggered_did(
                    unit[idx],
                    time[idx],
                    outcome[idx],
                    cohort[idx],
                    control_group=control_group,
                    bootstrap=0,
                )
                boots.append(res.overall_att_weighted)
            except ValueError:
                continue
        if len(boots) >= 2:
            se = float(np.std(boots, ddof=1))

    return StaggeredDiDResult(
        group_time=group_time,
        overall_att=overall,
        overall_att_weighted=overall_w,
        overall_se=se,
        n_units=int(len(unit_cohort)),
        n_cohorts=int(len(cohorts)),
        control_group=control_group,
    )


def simulate_staggered_did_data(
    n_units: int = 120,
    n_times: int = 6,
    cohorts: Sequence[int] = (3, 4, 5),
    att: float = 2.0,
    seed: int = 0,
):
    """Simulate a staggered-adoption panel with constant post-treatment ATT.

    Never-treated share is about one quarter of units. Returns
    ``unit, time, outcome, cohort, true_att``.
    """
    rng = np.random.default_rng(seed)
    cohorts = list(cohorts)
    n_treated = int(0.75 * n_units)
    n_never = n_units - n_treated
    unit_list = []
    time_list = []
    y_list = []
    c_list = []
    # Assign cohorts
    treated_units = list(range(n_treated))
    never_units = list(range(n_treated, n_units))
    assign = {}
    for i, u in enumerate(treated_units):
        assign[u] = cohorts[i % len(cohorts)]
    for u in never_units:
        assign[u] = 0
    for u in range(n_units):
        alpha = rng.normal(0, 1)
        g = assign[u]
        for t in range(1, n_times + 1):
            tau = rng.normal(0, 0.3)
            treat = 1.0 if (g > 0 and t >= g) else 0.0
            y = alpha + 0.5 * t + att * treat + tau + rng.normal(0, 0.5)
            unit_list.append(u)
            time_list.append(t)
            y_list.append(y)
            c_list.append(g)
    return (
        np.asarray(unit_list),
        np.asarray(time_list),
        np.asarray(y_list),
        np.asarray(c_list, dtype=float),
        float(att),
    )
