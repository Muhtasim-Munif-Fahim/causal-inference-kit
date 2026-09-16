"""causal_inference: a small toolkit for treatment-effect estimation."""

from .evaluate import EvaluationResult, evaluate, standard_estimators
from .estimators import (
    aipw_ate,
    difference_in_differences,
    difference_in_means,
    ipw_ate,
    ipw_att,
    ipw_weights,
    outcome_regression,
    propensity_matching,
)
from .generators import simulate_did_data, simulate_observational_data
from .propensity import (
    LogisticFit,
    PropensityHistogram,
    logistic_regression,
    propensity_histogram,
    propensity_scores,
    standardized_mean_differences,
    weighted_standardized_mean_differences,
)
from .report import render_report

__version__ = "0.1.0"

__all__ = [
    "EvaluationResult",
    "LogisticFit",
    "PropensityHistogram",
    "aipw_ate",
    "difference_in_differences",
    "difference_in_means",
    "evaluate",
    "ipw_ate",
    "ipw_att",
    "ipw_weights",
    "logistic_regression",
    "outcome_regression",
    "propensity_histogram",
    "propensity_matching",
    "propensity_scores",
    "render_report",
    "simulate_did_data",
    "simulate_observational_data",
    "standard_estimators",
    "standardized_mean_differences",
    "weighted_standardized_mean_differences",
]
