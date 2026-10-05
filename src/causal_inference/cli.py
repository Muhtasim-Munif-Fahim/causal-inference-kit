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
    linear_mediation,
    t_learner,
    s_learner,
    x_learner,
    r_learner,
    dr_learner,
    ipw_ate,
    ipw_att,
    ipw_weights,
    matching_weights_ate,
    overlap_ate,
    propensity_matching,
)
from .evaluate import evaluate, standard_estimators
from .generators import simulate_did_data, simulate_mediation_data, simulate_observational_data
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
        "overlap": overlap_ate(X, treatment, outcome, propensity=p),
        "matching_weights": matching_weights_ate(X, treatment, outcome, propensity=p),
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




def _cmd_mediation(args) -> int:
    treatment, mediator, outcome, covariates, truth = simulate_mediation_data(
        n=args.n,
        direct_effect=args.direct,
        a_path=args.a_path,
        b_path=args.b_path,
        confounding=args.confounding,
        n_covariates=args.n_covariates,
        seed=args.seed,
    )
    cov = covariates if args.n_covariates > 0 else None
    result = linear_mediation(treatment, mediator, outcome, covariates=cov)
    print(
        f"linear mediation  n={result.n} covariates={result.n_covariates}"
    )
    print(f"{'effect':<12}{'estimate':>12}{'se':>12}{'truth':>12}")
    rows = [
        ("total", result.total_effect, result.se_total, truth["total"]),
        ("direct", result.direct_effect, result.se_direct, truth["direct"]),
        ("indirect", result.indirect_effect, result.se_indirect, truth["indirect"]),
        ("a_path", result.a_path, result.se_a, truth["a_path"]),
        ("b_path", result.b_path, result.se_b, truth["b_path"]),
    ]
    for name, est, se, true in rows:
        print(f"{name:<12}{est:>12.4f}{se:>12.4f}{true:>12.4f}")
    return 0




def _cmd_t_learner(args) -> int:
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
    result = t_learner(X, treatment, outcome, W=W if args.d_w > 0 else None)
    print(
        f"T-learner CATE  n={result.n} treated={result.n_treated} "
        f"control={result.n_control}"
    )
    print(f"{'metric':<16}{'value':>12}")
    print("-" * 28)
    print(f"{'mean_cate':<16}{result.mean_cate:>12.4f}")
    print(f"{'true_ate':<16}{true_ate:>12.4f}")
    print(f"{'bias':<16}{result.mean_cate - true_ate:>12.4f}")
    print(f"{'cate_std':<16}{float(result.cate.std()):>12.4f}")
    print(f"{'cate_min':<16}{float(result.cate.min()):>12.4f}")
    print(f"{'cate_max':<16}{float(result.cate.max()):>12.4f}")
    return 0




def _cmd_s_learner(args) -> int:
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
    result = s_learner(X, treatment, outcome, W=W if args.d_w > 0 else None)
    print(
        f"S-learner CATE  n={result.n} treated={result.n_treated} "
        f"control={result.n_control}"
    )
    print(f"{'metric':<16}{'value':>12}")
    print("-" * 28)
    print(f"{'mean_cate':<16}{result.mean_cate:>12.4f}")
    print(f"{'true_ate':<16}{true_ate:>12.4f}")
    print(f"{'bias':<16}{result.mean_cate - true_ate:>12.4f}")
    print(f"{'cate_std':<16}{float(result.cate.std()):>12.4f}")
    print(f"{'cate_min':<16}{float(result.cate.min()):>12.4f}")
    print(f"{'cate_max':<16}{float(result.cate.max()):>12.4f}")
    return 0



def _cmd_x_learner(args) -> int:
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
    result = x_learner(X, treatment, outcome, W=W if args.d_w > 0 else None)
    print(
        f"X-learner CATE  n={result.n} treated={result.n_treated} "
        f"control={result.n_control}"
    )
    print(f"{'metric':<16}{'value':>12}")
    print("-" * 28)
    print(f"{'mean_cate':<16}{result.mean_cate:>12.4f}")
    print(f"{'true_ate':<16}{true_ate:>12.4f}")
    print(f"{'bias':<16}{result.mean_cate - true_ate:>12.4f}")
    print(f"{'cate_std':<16}{float(result.cate.std()):>12.4f}")
    print(f"{'cate_min':<16}{float(result.cate.min()):>12.4f}")
    print(f"{'cate_max':<16}{float(result.cate.max()):>12.4f}")
    return 0




def _cmd_r_learner(args) -> int:
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
    result = r_learner(X, treatment, outcome, W=W if args.d_w > 0 else None)
    print(
        f"R-learner CATE  n={result.n} treated={result.n_treated} "
        f"control={result.n_control}"
    )
    print(f"{'metric':<16}{'value':>12}")
    print("-" * 28)
    print(f"{'mean_cate':<16}{result.mean_cate:>12.4f}")
    print(f"{'true_ate':<16}{true_ate:>12.4f}")
    print(f"{'bias':<16}{result.mean_cate - true_ate:>12.4f}")
    print(f"{'cate_std':<16}{float(result.cate.std()):>12.4f}")
    print(f"{'cate_min':<16}{float(result.cate.min()):>12.4f}")
    print(f"{'cate_max':<16}{float(result.cate.max()):>12.4f}")
    return 0



def _cmd_dr_learner(args) -> int:
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
    result = dr_learner(X, treatment, outcome, W=W if args.d_w > 0 else None)
    print(
        f"DR-learner CATE  n={result.n} treated={result.n_treated} "
        f"control={result.n_control}"
    )
    print(f"{'metric':<16}{'value':>12}")
    print("-" * 28)
    print(f"{'mean_cate':<16}{result.mean_cate:>12.4f}")
    print(f"{'true_ate':<16}{true_ate:>12.4f}")
    print(f"{'bias':<16}{result.mean_cate - true_ate:>12.4f}")
    print(f"{'cate_std':<16}{float(result.cate.std()):>12.4f}")
    print(f"{'cate_min':<16}{float(result.cate.min()):>12.4f}")
    print(f"{'cate_max':<16}{float(result.cate.max()):>12.4f}")
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

    med = subparsers.add_parser(
        "mediation",
        help="fit linear product-of-coefficients mediation on a synthetic SEM",
    )
    med.add_argument("--n", type=int, default=2000, help="number of units")
    med.add_argument("--direct", type=float, default=1.0, help="true controlled direct effect")
    med.add_argument("--a-path", type=float, default=1.5, help="true treatment→mediator coefficient")
    med.add_argument("--b-path", type=float, default=0.8, help="true mediator→outcome coefficient")
    med.add_argument("--confounding", type=float, default=0.0, help="shared M/Y confounder strength")
    med.add_argument("--n-covariates", type=int, default=0, help="exogenous covariate columns")
    med.add_argument("--seed", type=int, default=0, help="simulation seed")
    med.set_defaults(func=_cmd_mediation)

    tl = subparsers.add_parser(
        "t-learner",
        help="fit a T-learner CATE on synthetic observational data",
    )
    tl.add_argument("--n", type=int, default=2000, help="number of units")
    tl.add_argument("--d-x", type=int, default=3, help="confounder columns")
    tl.add_argument("--d-w", type=int, default=2, help="outcome-only covariate columns")
    tl.add_argument("--ate", type=float, default=2.0, help="true average treatment effect")
    tl.add_argument("--confounding", type=float, default=1.0, help="confounding strength")
    tl.add_argument("--selection-bias", type=float, default=0.0, help="unobserved confounding")
    tl.add_argument("--heterogeneity", type=float, default=0.5, help="effect heterogeneity")
    tl.add_argument("--seed", type=int, default=0, help="simulation seed")
    tl.set_defaults(func=_cmd_t_learner)

    sl = subparsers.add_parser(
        "s-learner",
        help="fit an S-learner CATE on synthetic observational data",
    )
    sl.add_argument("--n", type=int, default=2000, help="number of units")
    sl.add_argument("--d-x", type=int, default=3, help="confounder columns")
    sl.add_argument("--d-w", type=int, default=2, help="outcome-only covariate columns")
    sl.add_argument("--ate", type=float, default=2.0, help="true average treatment effect")
    sl.add_argument("--confounding", type=float, default=1.0, help="confounding strength")
    sl.add_argument("--selection-bias", type=float, default=0.0, help="unobserved confounding")
    sl.add_argument("--heterogeneity", type=float, default=0.5, help="effect heterogeneity")
    sl.add_argument("--seed", type=int, default=0, help="simulation seed")
    sl.set_defaults(func=_cmd_s_learner)

    xl = subparsers.add_parser(
        "x-learner",
        help="fit an X-learner CATE on synthetic observational data",
    )
    xl.add_argument("--n", type=int, default=2000, help="number of units")
    xl.add_argument("--d-x", type=int, default=3, help="confounder columns")
    xl.add_argument("--d-w", type=int, default=2, help="outcome-only covariate columns")
    xl.add_argument("--ate", type=float, default=2.0, help="true average treatment effect")
    xl.add_argument("--confounding", type=float, default=1.0, help="confounding strength")
    xl.add_argument("--selection-bias", type=float, default=0.0, help="unobserved confounding")
    xl.add_argument("--heterogeneity", type=float, default=0.5, help="effect heterogeneity")
    xl.add_argument("--seed", type=int, default=0, help="simulation seed")
    xl.set_defaults(func=_cmd_x_learner)

    rl = subparsers.add_parser(
        "r-learner",
        help="fit an R-learner CATE on synthetic observational data",
    )
    rl.add_argument("--n", type=int, default=2000, help="number of units")
    rl.add_argument("--d-x", type=int, default=3, help="confounder columns")
    rl.add_argument("--d-w", type=int, default=2, help="outcome-only covariate columns")
    rl.add_argument("--ate", type=float, default=2.0, help="true average treatment effect")
    rl.add_argument("--confounding", type=float, default=1.0, help="confounding strength")
    rl.add_argument("--selection-bias", type=float, default=0.0, help="unobserved confounding")
    rl.add_argument("--heterogeneity", type=float, default=0.5, help="effect heterogeneity")
    rl.add_argument("--seed", type=int, default=0, help="simulation seed")
    rl.set_defaults(func=_cmd_r_learner)

    drl = subparsers.add_parser(
        "dr-learner",
        help="fit a DR-learner CATE on synthetic observational data",
    )
    drl.add_argument("--n", type=int, default=2000, help="number of units")
    drl.add_argument("--d-x", type=int, default=3, help="confounder columns")
    drl.add_argument("--d-w", type=int, default=2, help="outcome-only covariate columns")
    drl.add_argument("--ate", type=float, default=2.0, help="true average treatment effect")
    drl.add_argument("--confounding", type=float, default=1.0, help="confounding strength")
    drl.add_argument("--selection-bias", type=float, default=0.0, help="unobserved confounding")
    drl.add_argument("--heterogeneity", type=float, default=0.5, help="effect heterogeneity")
    drl.add_argument("--seed", type=int, default=0, help="simulation seed")
    drl.set_defaults(func=_cmd_dr_learner)


    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
