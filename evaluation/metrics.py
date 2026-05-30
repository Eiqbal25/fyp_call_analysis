"""
evaluation/metrics.py
======================
Phase 4: Validation and Performance Evaluation

Implements all evaluation metrics from thesis Section 3.6:

    Accuracy  = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)

    DER (Diarization Error Rate):
        = (FalseAlarm + MissedSpeech + SpeakerConfusion) / TotalSpeakerTime

    Paired t-test (Section 3.6.3):
        H0: mean(method3_scores) == mean(method1_scores)
        Reject if p < 0.05

All statistics are derived from actual prediction vs. ground truth comparisons.
No values are hardcoded.
"""

import os
import logging
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from config import TTEST_ALPHA, OUTPUTS_DIR

logger = logging.getLogger(__name__)

# Role encoding for binary classification (Agent = Positive class)
ROLE_LABEL_MAP = {"Agent": 1, "Customer": 0, "Unknown": -1}


# ─────────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────

def build_confusion_matrix(y_true: list[str],
                            y_pred: list[str]) -> dict:
    """
    Build a binary confusion matrix.
    Positive class = "Agent"

    Parameters
    ----------
    y_true : ground-truth role labels (str)
    y_pred : predicted role labels (str)

    Returns
    -------
    dict: TP, TN, FP, FN
    """
    tp = tn = fp = fn = 0
    for true, pred in zip(y_true, y_pred):
        t = ROLE_LABEL_MAP.get(true, -1)
        p = ROLE_LABEL_MAP.get(pred, -1)
        if t == -1 or p == -1:
            continue
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1

    logger.debug(f"Confusion matrix: TP={tp} TN={tn} FP={fp} FN={fn}")
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


# ─────────────────────────────────────────────────────────────
# CLASSIFICATION METRICS
# ─────────────────────────────────────────────────────────────

def compute_classification_metrics(y_true: list[str],
                                    y_pred: list[str],
                                    method_name: str = "") -> dict:
    """
    Compute Accuracy, Precision, Recall, F1-Score from predictions.

    Parameters
    ----------
    y_true      : ground-truth labels
    y_pred      : predicted labels
    method_name : label for logging

    Returns
    -------
    dict: accuracy, precision, recall, f1, confusion_matrix,
          support_agent, support_customer
    """
    cm = build_confusion_matrix(y_true, y_pred)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    total = tp + tn + fp + fn

    if total == 0:
        logger.warning("No valid samples in evaluation set")
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0,
                "confusion_matrix": cm}

    # Accuracy
    accuracy = (tp + tn) / total

    # Precision  (TP / (TP + FP))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall  (TP / (TP + FN))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1-Score
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    # Support
    support_agent    = tp + fn    # total actual Agent segments
    support_customer = tn + fp    # total actual Customer segments

    result = {
        "accuracy":          round(accuracy  * 100, 2),
        "precision":         round(precision * 100, 2),
        "recall":            round(recall    * 100, 2),
        "f1":                round(f1        * 100, 2),
        "confusion_matrix":  cm,
        "support_agent":     support_agent,
        "support_customer":  support_customer,
        "n_samples":         total,
    }

    prefix = f"[{method_name}] " if method_name else ""
    logger.info(
        f"{prefix}Accuracy={accuracy*100:.1f}% | "
        f"Precision={precision*100:.1f}% | "
        f"Recall={recall*100:.1f}% | "
        f"F1={f1*100:.1f}% | "
        f"n={total}"
    )
    return result


# ─────────────────────────────────────────────────────────────
# DIARIZATION ERROR RATE (DER)
# ─────────────────────────────────────────────────────────────

def compute_der(reference_segments: list[dict],
                hypothesis_segments: list[dict],
                collar_sec: float = 0.25) -> dict:
    """
    Compute Diarization Error Rate (DER).

    DER = (False Alarm + Missed Speech + Speaker Confusion) / Total Speaker Time

    Parameters
    ----------
    reference_segments : [{speaker_id, start, end}, ...]  — ground truth
    hypothesis_segments: [{speaker_id, start, end}, ...]  — system output
    collar_sec         : tolerance collar around boundaries (default 0.25s)

    Returns
    -------
    dict: der, missed_speech_sec, false_alarm_sec, confusion_sec,
          total_ref_time_sec
    """
    def _build_timeline(segments, total_dur, step=0.01):
        """Build frame-level speaker label array (None = silence)."""
        n_frames = int(total_dur / step) + 1
        timeline = [None] * n_frames
        for seg in segments:
            start_f = max(0, int(seg["start"] / step))
            end_f   = min(n_frames - 1, int(seg["end"] / step))
            for f in range(start_f, end_f + 1):
                timeline[f] = seg["speaker_id"]
        return timeline, step

    if not reference_segments or not hypothesis_segments:
        logger.warning("Empty segments for DER computation")
        return {"der": float("nan"), "note": "Insufficient data"}

    total_dur = max(
        max(s["end"] for s in reference_segments),
        max(s["end"] for s in hypothesis_segments),
    )

    ref_tl,  step = _build_timeline(reference_segments,  total_dur)
    hyp_tl,  _    = _build_timeline(hypothesis_segments, total_dur)

    missed_speech = 0.0
    false_alarm   = 0.0
    confusion     = 0.0
    total_speech  = 0.0

    for r, h in zip(ref_tl, hyp_tl):
        if r is not None:
            total_speech += step
            if h is None:
                missed_speech += step
            elif r != h:
                confusion += step
        else:
            if h is not None:
                false_alarm += step

    total_error = missed_speech + false_alarm + confusion
    der = (total_error / total_speech * 100) if total_speech > 0 else float("nan")

    logger.info(
        f"DER={der:.2f}% | Missed={missed_speech:.2f}s | "
        f"FA={false_alarm:.2f}s | Confusion={confusion:.2f}s | "
        f"Total_ref={total_speech:.2f}s"
    )
    return {
        "der":               round(der, 2),
        "missed_speech_sec": round(missed_speech, 3),
        "false_alarm_sec":   round(false_alarm,   3),
        "confusion_sec":     round(confusion,     3),
        "total_ref_time_sec": round(total_speech, 3),
    }


# ─────────────────────────────────────────────────────────────
# PEARSON CORRELATION (System vs. Human QA scores)
# ─────────────────────────────────────────────────────────────

def compute_pearson_correlation(system_scores: list[float],
                                 human_scores:  list[float]) -> dict:
    """
    Compute Pearson correlation coefficient between system and human QA scores.

    Returns
    -------
    dict: r, p_value, interpretation
    """
    if len(system_scores) != len(human_scores) or len(system_scores) < 2:
        return {"r": float("nan"), "p_value": float("nan"),
                "interpretation": "Insufficient data"}

    r, p_value = stats.pearsonr(system_scores, human_scores)

    if abs(r) >= 0.7:
        interp = "Strong correlation"
    elif abs(r) >= 0.4:
        interp = "Moderate correlation"
    else:
        interp = "Weak correlation"

    logger.info(f"Pearson r={r:.4f} | p={p_value:.4f} | {interp}")
    return {
        "r":              round(float(r),       4),
        "p_value":        round(float(p_value), 4),
        "interpretation": interp,
    }


# ─────────────────────────────────────────────────────────────
# PAIRED t-TEST (Method 4 vs. Method 1 per-sample accuracy)
# ─────────────────────────────────────────────────────────────

def compute_paired_ttest(scores_method1: list[float],
                          scores_method3: list[float],
                          alpha: float = TTEST_ALPHA) -> dict:
    """
    Paired sample t-test to determine if Method 4 improvement is statistically
    significant over Method 1 (Section 3.6.3).

    H0: mean(m3) == mean(m1)   (no significant difference)
    H1: mean(m3) >  mean(m1)   (hybrid is significantly better)

    Parameters
    ----------
    scores_method1 : per-sample correctness scores for Method 1 (0 or 1)
    scores_method3 : per-sample correctness scores for Method 4 (0 or 1)
    alpha          : significance level (default 0.05)

    Returns
    -------
    dict: t_statistic, p_value, reject_null, conclusion
    """
    if len(scores_method1) != len(scores_method3):
        raise ValueError("Score lists must be the same length for paired t-test")

    if len(scores_method1) < 2:
        return {"t_statistic": float("nan"), "p_value": float("nan"),
                "reject_null": False, "conclusion": "Insufficient data"}

    t_stat, p_value = stats.ttest_rel(scores_method3, scores_method1)

    # One-tailed p-value (testing m3 > m1)
    p_one_tailed = p_value / 2 if t_stat > 0 else 1.0 - p_value / 2

    reject_null = p_one_tailed < alpha
    conclusion  = (
        f"Reject H0 (p={p_one_tailed:.4f} < α={alpha}): "
        "Hybrid Ensemble is SIGNIFICANTLY better than Keyword baseline."
        if reject_null else
        f"Fail to reject H0 (p={p_one_tailed:.4f} ≥ α={alpha}): "
        "No significant performance difference detected."
    )

    logger.info(f"Paired t-test | t={t_stat:.4f} | p={p_one_tailed:.4f} | "
                f"reject_H0={reject_null}")
    return {
        "t_statistic":   round(float(t_stat),      4),
        "p_value":       round(float(p_one_tailed), 4),
        "alpha":         alpha,
        "reject_null":   reject_null,
        "conclusion":    conclusion,
    }


# ─────────────────────────────────────────────────────────────
# EFFICIENCY METRICS
# ─────────────────────────────────────────────────────────────

def compute_rtf(audio_duration_sec: float,
                processing_time_sec: float) -> dict:
    """
    Compute Real-Time Factor (RTF).

    RTF = processing_time / audio_duration
    RTF < 1.0 → faster than real-time (desirable)

    Parameters
    ----------
    audio_duration_sec  : length of audio processed
    processing_time_sec : actual wall-clock processing time

    Returns
    -------
    dict: rtf, is_real_time, efficiency_multiplier
    """
    if audio_duration_sec < 1e-6:
        return {"rtf": float("nan"), "is_real_time": False}

    rtf = processing_time_sec / audio_duration_sec
    efficiency = 1.0 / rtf if rtf > 0 else float("inf")

    logger.info(f"RTF={rtf:.4f} ({'Real-time capable' if rtf < 1 else 'Too slow'}) | "
                f"{efficiency:.1f}× faster than audio")
    return {
        "rtf":                   round(rtf, 4),
        "is_real_time":          rtf < 1.0,
        "efficiency_multiplier": round(efficiency, 2),
        "processing_time_sec":   round(processing_time_sec, 3),
        "audio_duration_sec":    round(audio_duration_sec,  3),
    }


# ─────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────────

def plot_accuracy_comparison(metrics_m1: dict, metrics_m2: dict, metrics_m3: dict,
                              save_path: str = None) -> str:
    """Grouped bar chart: Accuracy / Precision / Recall / F1 for all 3 methods."""
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    m1_vals = [metrics_m1.get("accuracy",  0), metrics_m1.get("precision", 0),
               metrics_m1.get("recall",    0), metrics_m1.get("f1",        0)]
    m2_vals = [metrics_m2.get("accuracy",  0), metrics_m2.get("precision", 0),
               metrics_m2.get("recall",    0), metrics_m2.get("f1",        0)]
    m3_vals = [metrics_m3.get("accuracy",  0), metrics_m3.get("precision", 0),
               metrics_m3.get("recall",    0), metrics_m3.get("f1",        0)]

    x   = np.arange(len(metric_names))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(x - w,   m1_vals, width=w, label="Method 1 (Keyword)",  color="#1f77b4", alpha=0.85)
    ax.bar(x,       m2_vals, width=w, label="Method 2 (Acoustic)", color="#ff7f0e", alpha=0.85)
    ax.bar(x + w,   m3_vals, width=w, label="Method 4 (Hybrid)",   color="#2ca02c", alpha=0.85)

    # Value labels
    for bars in [ax.containers[0], ax.containers[1], ax.containers[2]]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                        f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim([0, 115])
    ax.set_ylabel("Score (%)")
    ax.set_title("Comparative Analysis: Classification Performance Across Methods")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "accuracy_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Accuracy comparison chart saved → {save_path}")
    return save_path


def plot_confusion_matrix(cm: dict, method_name: str = "Method",
                           save_path: str = None) -> str:
    """Visualize a 2×2 confusion matrix as a heatmap."""
    matrix = np.array([
        [cm.get("TN", 0), cm.get("FP", 0)],
        [cm.get("FN", 0), cm.get("TP", 0)],
    ])
    labels = ["Customer", "Agent"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1]);  ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]);  ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix - {method_name}")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]),
                    ha="center", va="center", fontsize=14,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")

    plt.colorbar(im)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR,
                                  f"confusion_matrix_{method_name.replace(' ','_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {save_path}")
    return save_path


def plot_ttest_boxplot(scores_m1: list[float], scores_m3: list[float],
                        ttest_result: dict, save_path: str = None) -> str:
    """
    Box plot showing distribution of per-sample accuracy scores for
    Method 1 vs. Method 3 with t-test annotation (Figure 3.10 equivalent).
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    bp = ax.boxplot([scores_m1, scores_m3],
                    labels=["Method 1\n(Keyword)", "Method 4\n(Hybrid)"],
                    patch_artist=True,
                    medianprops={"color": "red", "linewidth": 2})

    bp["boxes"][0].set_facecolor("#1f77b4")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("#2ca02c")
    bp["boxes"][1].set_alpha(0.6)

    p = ttest_result.get("p_value", 1.0)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ax.set_title(f"Method 1 vs. Method 4 (Paired t-test: p={p:.4f} {sig})")
    ax.set_ylabel("Per-Sample Accuracy Score")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "ttest_boxplot.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"t-test boxplot saved → {save_path}")
    return save_path
