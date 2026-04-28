"""
methods/
========
Phase 2: Speaker Role Detection

Modules:
    method1_lexical  — Rule-based Keyword Density classifier (baseline)
    method2_acoustic — Deep Neural Network on MFCC + d-vector features
    method3_hybrid   — Confidence-Weighted Ensemble Fusion (proposed solution)
"""

from methods.method1_lexical import (
    classify_transcript_lexical,
    classify_segment_lexical,
    compute_lexical_density,
)
from methods.method2_acoustic import (
    classify_transcript_acoustic,
    classify_segment_acoustic,
    build_feature_vector,
    train_acoustic_model,
)
from methods.method3_hybrid import (
    classify_transcript_hybrid,
    fuse_predictions,
    compute_confidence_statistics,
)

__all__ = [
    "classify_transcript_lexical",
    "classify_segment_lexical",
    "compute_lexical_density",
    "classify_transcript_acoustic",
    "classify_segment_acoustic",
    "build_feature_vector",
    "train_acoustic_model",
    "classify_transcript_hybrid",
    "fuse_predictions",
    "compute_confidence_statistics",
]
