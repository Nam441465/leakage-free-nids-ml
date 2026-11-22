"""Evaluation metrics and statistical testing modules."""

from src.evaluation.metrics import (
    evaluate_binary_metrics,
    evaluate_multiclass_metrics,
    compute_recall_at_fixed_fpr,
)
from src.evaluation.stats import (
    compute_mean_std,
    compute_paired_difference_ci,
    paired_t_test_with_holm,
)

__all__ = [
    "evaluate_binary_metrics",
    "evaluate_multiclass_metrics",
    "compute_recall_at_fixed_fpr",
    "compute_mean_std",
    "compute_paired_difference_ci",
    "paired_t_test_with_holm",
]
