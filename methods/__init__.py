"""
methods/
========
Phase 2: Speaker Role Detection

Modules:
    method1_lexical  — Rule-based Keyword Density classifier (baseline)
    method2_acoustic — Deep Neural Network on MFCC + d-vector features
    # method4_hybrid removed
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
# Method 4 hybrid removed — Method 3 LLM is the proposed system

__all__ = [
    "classify_transcript_lexical",
    "classify_segment_lexical",
    "compute_lexical_density",
    "classify_transcript_acoustic",
    "classify_segment_acoustic",
    "build_feature_vector",
    "train_acoustic_model",
    "compute_confidence_statistics",
]
