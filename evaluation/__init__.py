"""
evaluation/
===========
Phase 4: Validation and Performance Evaluation

Modules:
    metrics    — Accuracy, Precision, Recall, F1, DER, RTF, t-test, Pearson r
    validator  — Ground truth CSV loader, per-call/aggregate validation pipeline
"""

from evaluation.metrics import (
    compute_classification_metrics,
    build_confusion_matrix,
    compute_der,
    compute_paired_ttest,
    compute_pearson_correlation,
    compute_rtf,
)
from evaluation.validator import (
    load_ground_truth,
    run_validation,
    build_qa_comparison_table,
)

__all__ = [
    "compute_classification_metrics",
    "build_confusion_matrix",
    "compute_der",
    "compute_paired_ttest",
    "compute_pearson_correlation",
    "compute_rtf",
    "load_ground_truth",
    "run_validation",
    "build_qa_comparison_table",
]
