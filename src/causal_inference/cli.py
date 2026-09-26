"""Command-line interface for the causal-inference toolkit."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import __version__
from .estimators import (
    aipw_ate,
    difference_in_means,
    difference_in_differences,
    event_study_did,
    ipw_ate,
    ipw_att,
    ipw_weights,
    propensity_matching,
)
from .evaluate import evaluate, standard_estimators
from .generators import simulate_did_data, simulate_observational_data
from .propensity import propensity_scores
from .report import render_report


def _save_dataset(path, X, W, treatment, outcome) -> None:
    columns = {"treatment": treatment, "outcome": outcome}
    for j in range(X.shape[1]):
        columns[f"X{j}"] = X[:, j]
    for j in range(W.shape[1]):
        columns[f"W{j}"] = W[:, j]
    pd.DataFrame(columns).to_csv(path, index=False)


def _load_dataset(path):
    df = pd.read_csv(path)
    if "treatment" not in df.columns or "outcome" not in df.columns:
        raise SystemExit("data file must contain 'treatment' and 'outcome' columns")
    x_cols = [c for c in df.columns if c.startswith("X")]
    w_cols = [c for c in df.columns if c.startswith("W")]
    X = df[x_cols].to_numpy(dtype=float)
    W = df[w_cols].to_numpy(dtype=float)
    treatment = df["treatment"].to_numpy(dtype=float)
    outcome = df["outcome"].to_numpy(dtype=float)
    return X, W, treatment, outcome


def _cmd_simulate(args) -> int:
    X, W, treatment, outcome, true_ate = simulate_observational_data(
        n=args.n,
        d_x=args.d_x,
        d_w=args.d_w,
        ate=args.ate,
        confounding=args.confounding,
        selection_bias=args.selection_bias,
        effect_heterogeneity=args.heterogeneity,
        seed=args.seed,
    )
    _save_dataset(args.out, X, W, treatment, outcome)
    print(
        f"wrote {len(treatment)} rows to {args.out} "
        f"(true ATE {true_ate:.3f}, treatment share {treatment.mean():.1%})"
    )
    return 0


def _cmd_estimate(args) -> int:
    X, W, treatment, outcome = _load_dataset(args.data)
    p, _ = propensity_scores(X, treatment)
    estimators = {
        "naive": difference_in_means(treatment, outcome),
        "ipw": ipw_ate(X, treatment, outcome, propensity=p),
        "aipw": aipw_ate(X, treatment, outcome, propensity=p, W=W),
        "ipw_att": ipw_att(X, treatment, outcome, propensity=p),
        "matching": propensity_matching(
            X, treatment, outcome, propensity=p, with_replacement=True
        ),
    }
    show_bias = args.true_ate is not None
    if show_bias:
        header = f"{'estimator':<12}{'estimate':>10}{'bias':>10}"
        rule = "-" * 32
    else:
        header = f"{'estimator':<12}{'estimate':>10}"
        rule = "-" * 22
    print(header)
    print(rule)
    for name, estimate in estimators.items():
        if show_bias:
            print(f"{name:<12}{estimate:>10.3f}{estimate - args.true_ate:>10.3f}")
        else:
            print(f"{name:<12}{estimate:>10.3f}")
    return 0


def _cmd_report(args) -> int:
    X, W, treatment, outcome = _load_dataset(args.data)
    results = evaluate(
        standard_estimators(),
        X,
        W,
        treatment,
        outcome,
        args.true_ate,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    weights = ipw_weights(X, treatment)
    markdown = render_report(X, W, treatment, outcome, results, args.true_ate, weights=weights)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(markdown)
    print(f"wrote {args.out}")
    return 0


def _add_common_data_args(parser) -> None:
    parser.add_argument("--data", default="data.csv", help="path to the dataset CSV")



def _cmd_event_study(args) -> int:
    unit, group, period, outcome, true_did = simulate_did_data(
        n=args.n,
        ate=args.ate,
        time_trend=args.time_trend,
        n_pre=args.n_pre,
        n_post=args.n_post,
        seed=args.seed,
    )
    treatment_time = float(args.n_pre)
    result = event_study_did(
        unit,
        group,
        period,
        outcome,
        treatment_time=treatment_time,
        reference=args.reference,
        cluster=not args.hc1,
    )
    print(
        f"event-study DiD  n={result.n} units={result.n_units} "
        f"times={result.n_times} T0={result.treatment_time:g} "
        f"reference={result.reference} se={result.se_type}"
    )
    print(f"true ATT (flat): {true_did:.4f}")
    print(f"{'k':>6}{'estimate':>12}{'se':>12}")
    for k, coef, se in zip(result.relative_times, result.coefficients, result.ses):
        print(f"{int(k):>6}{float(coef):>12.4f}{float(se):>12.4f}")
    print(f"{result.reference:>6}{0.0:>12.4f}{'(ref)':>12}")
    return 0



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causal-inference",
        description="Estimate treatment effects from observational data.",
    )
    parser.add_argument(
        "--version", action="version", version=f"causal-inference-kit {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="write a synthetic dataset to CSV")
    simulate.add_argument("--n", type=int, default=2000, help="number of units")
    simulate.add_argument("--d-x", type=int, default=3, help="confounder columns")
    simulate.add_argument("--d-w", type=int, default=2, help="outcome-only covariate columns")
    simulate.add_argument("--ate", type=float, default=2.0, help="average treatment effect")
    simulate.add_argument("--confounding", type=float, default=1.0, help="confounding strength")
    simulate.add_argument("--selection-bias", type=float, default=0.0, help="unobserved confounding")
    simulate.add_argument("--heterogeneity", type=float, default=0.5, help="effect heterogeneity")
    simulate.add_argument("--seed", type=int, default=42, help="random seed")
    simulate.add_argument("--out", default="data.csv", help="output CSV path")
    simulate.set_defaults(func=_cmd_simulate)

    estimate = subparsers.add_parser("estimate", help="print point estimates from a dataset")
    _add_common_data_args(estimate)
    estimate.add_argument("--true-ate", type=float, default=None, help="ground-truth ATE for a bias column")
    estimate.set_defaults(func=_cmd_estimate)

    report = subparsers.add_parser("report", help="write a markdown evaluation report")
    _add_common_data_args(report)
    report.add_argument("--true-ate", type=float, required=True, help="ground-truth ATE")
    report.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples")
    report.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    report.add_argument("--out", default="report.md", help="output markdown path")
    report.set_defaults(func=_cmd_report)

    event = subparsers.add_parser(
        "event-study",
        help="fit an event-study / dynamic DiD on a synthetic panel",
    )
    event.add_argument("--n", type=int, default=500, help="number of units")
    event.add_argument("--n-pre", type=int, default=3, help="pre-treatment periods")
    event.add_argument("--n-post", type=int, default=3, help="post-treatment periods")
    event.add_argument("--ate", type=float, default=1.5, help="true flat ATT")
    event.add_argument("--time-trend", type=float, default=1.0, help="common linear time effect")
    event.add_argument("--reference", type=int, default=-1, help="omitted relative-time period")
    event.add_argument("--seed", type=int, default=0, help="simulation seed")
    event.add_argument(
        "--hc1",
        action="store_true",
        help="use HC1 robust SE instead of clustering by unit",
    )
    event.set_defaults(func=_cmd_event_study)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
