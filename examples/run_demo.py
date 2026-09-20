"""End-to-end demo: simulate, estimate, evaluate and write a report."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causal_inference.estimators import (
    aipw_ate,
    difference_in_differences,
    difference_in_means,
    ipw_ate,
    ipw_att,
    ipw_weights,
    propensity_matching,
    regression_discontinuity,
    synthetic_control,
)
from causal_inference.evaluate import evaluate, standard_estimators
from causal_inference.generators import (
    simulate_did_data,
    simulate_observational_data,
    simulate_rd_data,
    simulate_synthetic_control_data,
)
from causal_inference.propensity import propensity_scores
from causal_inference.report import render_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8000, help="sample size")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "output"),
        help="directory for the generated report",
    )
    args = parser.parse_args(argv)

    X, W, treatment, outcome, true_ate = simulate_observational_data(
        n=args.n, confounding=1.5, selection_bias=0.0, seed=args.seed
    )

    print("=" * 60)
    print("Causal inference demo")
    print("=" * 60)
    print(f"synthetic sample: n={len(treatment)}, treated {treatment.mean():.1%}")
    print(f"ground-truth ATE: {true_ate:.3f}\n")

    p, _ = propensity_scores(X, treatment)
    estimates = [
        ("naive difference", difference_in_means(treatment, outcome)),
        ("ipw (stabilized)", ipw_ate(X, treatment, outcome, propensity=p)),
        ("aipw (doubly robust)", aipw_ate(X, treatment, outcome, propensity=p, W=W)),
        ("ipw (ATT)", ipw_att(X, treatment, outcome, propensity=p)),
        (
            "matching (ATT)",
            propensity_matching(X, treatment, outcome, propensity=p, with_replacement=True),
        ),
    ]
    print(f"{'estimator':<24}{'estimate':>10}{'bias':>10}")
    print("-" * 44)
    for name, estimate in estimates:
        print(f"{name:<24}{estimate:>10.3f}{estimate - true_ate:>10.3f}")

    print(f"\nbootstrap evaluation ({args.bootstrap} resamples, fixed first stage):")
    results = evaluate(
        standard_estimators(),
        X,
        W,
        treatment,
        outcome,
        true_ate,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    print(f"{'estimator':<12}{'estimate':>10}{'se':>8}{'bias':>8}{'rmse':>8}")
    print("-" * 46)
    for result in results:
        print(
            f"{result.name:<12}{result.estimate:>10.3f}{result.se:>8.3f}"
            f"{result.bias:>8.3f}{result.rmse:>8.3f}"
        )

    group, period, did_outcome, true_did = simulate_did_data(n=args.n, seed=args.seed)
    did = difference_in_differences(group, period, did_outcome)
    print(f"\ndifference-in-differences on a simulated panel: {did:.3f} (true {true_did:.3f})")

    sc_unit, sc_time, sc_outcome, sc_treated, sc_t0, true_sc = simulate_synthetic_control_data(
        n_donors=8, n_pre=12, n_post=8, ate=5.0, seed=args.seed
    )
    sc = synthetic_control(
        sc_unit, sc_time, sc_outcome, sc_treated, sc_t0, placebo=True
    )
    print(
        f"synthetic control on a simulated donor panel: {sc.estimate:.3f} "
        f"(true {true_sc:.3f}, placebo p {sc.placebo_p_value:.3f})"
    )

    rd_running, _, rd_outcome, rd_cutoff, true_rd = simulate_rd_data(
        n=args.n, cutoff=0.0, ate=2.0, slope=1.0, seed=args.seed
    )
    rd = regression_discontinuity(rd_running, rd_outcome, cutoff=rd_cutoff)
    print(
        f"regression discontinuity on a simulated running-variable sample: "
        f"{rd.estimate:.3f} (true {true_rd:.3f}, h={rd.bandwidth:.3f}, "
        f"n_left={rd.n_left}, n_right={rd.n_right})"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "demo_report.md")
    extra_section = (
        "## Difference-in-differences\n\n"
        f"On a simulated two-period two-group panel the DiD estimate is "
        f"**{did:.3f}** against a true effect of **{true_did:.3f}**.\n\n"
        "## Synthetic control\n\n"
        f"On a simulated donor panel the synthetic control estimate is "
        f"**{sc.estimate:.3f}** against a true effect of **{true_sc:.3f}**. "
        f"Pre-treatment RMSPE is {sc.pre_rmspe:.3f}; the in-space placebo "
        f"p-value is {sc.placebo_p_value:.3f}.\n\n"
        "## Regression discontinuity\n\n"
        f"On a simulated sharp RD sample the local-linear estimate at the "
        f"cutoff is **{rd.estimate:.3f}** against a true effect of "
        f"**{true_rd:.3f}**. The Imbens–Kalyanaraman bandwidth is "
        f"{rd.bandwidth:.3f} ({rd.n_left} left / {rd.n_right} right)."
    )
    markdown = render_report(
        X,
        W,
        treatment,
        outcome,
        results,
        true_ate,
        weights=ipw_weights(X, treatment),
        extra_markdown=extra_section,
    )
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(markdown)
    print(f"\nreport written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
