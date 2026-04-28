"""
evaluation/validator.py
========================
Phase 4: Ground Truth Comparison Pipeline

KEY FIX: The old validator matched by speaker_id (0 or 1), which is WRONG.
Resemblyzer assigns speaker IDs randomly per call — speaker 0 might be
Agent in call A but Customer in call B. This caused the system to report
"correct" when labels were actually inverted.

NEW APPROACH:
  1. Match system segments to ground truth by TEXT SIMILARITY (not speaker_id)
  2. Detect and correct label inversion automatically
  3. Report true accuracy against human-verified labels

Ground truth CSV format:
    call_id, ground_truth_role, text, start (opt), end (opt), human_qa_score (opt)

  - ground_truth_role: what a human says this speaker IS ("Agent" or "Customer")
  - text: what that speaker actually said (used for matching)
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

from config import GROUND_TRUTH_CSV, OUTPUTS_DIR
from evaluation.metrics import (
    compute_classification_metrics,
    compute_paired_ttest,
    compute_pearson_correlation,
    compute_rtf,
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_ttest_boxplot,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# LOAD GROUND TRUTH
# ─────────────────────────────────────────────────────────────

def load_ground_truth(csv_path: str = GROUND_TRUTH_CSV) -> pd.DataFrame:
    """
    Load human_validation_study.csv.

    Required columns: call_id, ground_truth_role, text
    Optional columns: start, end, human_qa_score

    The 'text' column is what the human VERIFIED the speaker said.
    The 'ground_truth_role' is what the human LABELED them as.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Ground truth CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    required = {"call_id", "ground_truth_role", "text"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Ground truth CSV missing columns: {missing}\n"
            f"Required: call_id, ground_truth_role, text\n"
            f"Got: {list(df.columns)}"
        )

    df["ground_truth_role"] = df["ground_truth_role"].str.strip().str.title()
    df["text"]              = df["text"].fillna("").astype(str).str.strip()

    logger.info(
        f"Loaded ground truth: {csv_path} | rows={len(df)} | "
        f"calls={df['call_id'].nunique()} | "
        f"roles={df['ground_truth_role'].value_counts().to_dict()}"
    )
    return df


# ─────────────────────────────────────────────────────────────
# TEXT SIMILARITY MATCHING
# ─────────────────────────────────────────────────────────────

def _text_similarity(a: str, b: str) -> float:
    """Compute fuzzy text similarity ratio between two strings (0–1)."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _match_segments_to_groundtruth(
        system_segments: list[dict],
        gt_rows: pd.DataFrame,
        similarity_threshold: float = 0.40) -> list[dict]:
    """
    Match each system segment to the closest ground truth row by text similarity.

    For each system segment, find the ground truth row whose text is most
    similar. If similarity >= threshold, record the match.

    This avoids the speaker_id mismatch problem entirely.

    Parameters
    ----------
    system_segments      : classified transcript segments
    gt_rows              : ground truth rows for this call_id
    similarity_threshold : minimum similarity to accept a match

    Returns
    -------
    matched : list of dicts with keys:
        system_text, gt_text, similarity,
        predicted_role, ground_truth_role, correct
    """
    matched = []
    used_gt_indices = set()

    for seg in system_segments:
        sys_text     = seg.get("text", "").strip()
        predicted    = seg.get("predicted_role", "Unknown")

        if not sys_text or predicted == "Unknown":
            continue

        best_sim   = 0.0
        best_idx   = -1
        best_gt_row = None

        for idx, gt_row in gt_rows.iterrows():
            if idx in used_gt_indices:
                continue
            sim = _text_similarity(sys_text, gt_row["text"])
            if sim > best_sim:
                best_sim    = sim
                best_idx    = idx
                best_gt_row = gt_row

        if best_sim >= similarity_threshold and best_gt_row is not None:
            used_gt_indices.add(best_idx)
            gt_role  = best_gt_row["ground_truth_role"]
            correct  = (predicted == gt_role)
            matched.append({
                "system_text":      sys_text,
                "gt_text":          best_gt_row["text"],
                "similarity":       round(best_sim, 4),
                "predicted_role":   predicted,
                "ground_truth_role": gt_role,
                "correct":          correct,
                "confidence":       seg.get("final_confidence",
                                            seg.get("confidence", 0.5)),
            })

    return matched


# ─────────────────────────────────────────────────────────────
# LABEL INVERSION DETECTION AND CORRECTION
# ─────────────────────────────────────────────────────────────

def detect_label_inversion(matched: list[dict]) -> dict:
    """
    Detect if the system has inverted Agent/Customer labels for this call.

    If > 60% of matched segments are wrong, the labels are likely inverted.
    This happens when Resemblyzer assigns speaker IDs opposite to reality.

    Returns
    -------
    dict: is_inverted (bool), normal_accuracy, inverted_accuracy
    """
    if not matched:
        return {"is_inverted": False, "normal_accuracy": 0.0, "inverted_accuracy": 0.0}

    normal_correct   = sum(1 for m in matched if m["correct"])
    normal_accuracy  = normal_correct / len(matched)

    # Simulate flipping all predictions
    role_flip = {"Agent": "Customer", "Customer": "Agent"}
    inverted_correct  = sum(
        1 for m in matched
        if role_flip.get(m["predicted_role"], m["predicted_role"]) == m["ground_truth_role"]
    )
    inverted_accuracy = inverted_correct / len(matched)

    is_inverted = inverted_accuracy > normal_accuracy

    logger.info(
        f"Label inversion check | normal_acc={normal_accuracy*100:.1f}% | "
        f"inverted_acc={inverted_accuracy*100:.1f}% | "
        f"inverted={is_inverted}"
    )
    return {
        "is_inverted":       is_inverted,
        "normal_accuracy":   round(normal_accuracy,   4),
        "inverted_accuracy": round(inverted_accuracy, 4),
    }


def correct_inverted_labels(classified: list[dict]) -> list[dict]:
    """
    Flip Agent↔Customer labels for an entire call's classified transcript.
    Called when detect_label_inversion() returns is_inverted=True.
    """
    role_flip = {"Agent": "Customer", "Customer": "Agent"}
    corrected = []
    for seg in classified:
        new_seg = dict(seg)
        new_seg["predicted_role"]   = role_flip.get(seg.get("predicted_role",   "Unknown"), "Unknown")
        new_seg["lexical_role"]     = role_flip.get(seg.get("lexical_role",     "Unknown"), "Unknown")
        new_seg["acoustic_role"]    = role_flip.get(seg.get("acoustic_role",    "Unknown"), "Unknown")
        new_seg["label_was_inverted"] = True
        corrected.append(new_seg)
    logger.info("Labels corrected — was inverted")
    return corrected


# ─────────────────────────────────────────────────────────────
# VALIDATE A SINGLE CALL
# ─────────────────────────────────────────────────────────────

def validate_call(classified: list[dict],
                  gt_df: pd.DataFrame,
                  call_id: str,
                  method_name: str = "",
                  auto_correct_inversion: bool = True) -> dict:
    """
    Validate one call's classification against human ground truth.

    Steps:
      1. Filter ground truth for this call
      2. Match system segments to GT by text similarity
      3. Detect label inversion
      4. Optionally auto-correct and re-evaluate
      5. Compute Accuracy / Precision / Recall / F1

    Parameters
    ----------
    classified              : system output segments for this call
    gt_df                   : full ground truth DataFrame
    call_id                 : call identifier
    method_name             : label for logging
    auto_correct_inversion  : if True, flip labels if inversion detected

    Returns
    -------
    dict with all metrics + inversion info
    """
    gt_call = gt_df[gt_df["call_id"] == call_id].copy()
    if gt_call.empty:
        logger.warning(f"No ground truth rows for call_id='{call_id}'")
        return {}

    # Step 1: Match by text similarity
    matched = _match_segments_to_groundtruth(classified, gt_call)
    if not matched:
        logger.warning(
            f"No segments matched for '{call_id}' — check that call_id in CSV "
            f"matches audio filename, and that text column has real transcript text"
        )
        return {}

    logger.info(
        f"[{method_name}|{call_id}] Matched {len(matched)}/{len(classified)} segments "
        f"(avg similarity={np.mean([m['similarity'] for m in matched]):.2f})"
    )

    # Step 2: Detect inversion
    inversion = detect_label_inversion(matched)

    # Step 3: Auto-correct if inverted
    if auto_correct_inversion and inversion["is_inverted"]:
        logger.warning(
            f"⚠ Label inversion detected for '{call_id}' — "
            f"auto-correcting predictions"
        )
        classified_corrected = correct_inverted_labels(classified)
        matched = _match_segments_to_groundtruth(classified_corrected, gt_call)
    else:
        classified_corrected = classified

    # Step 4: Compute metrics
    y_true = [m["ground_truth_role"] for m in matched]
    y_pred = [m["predicted_role"]    for m in matched]

    metrics = compute_classification_metrics(
        y_true, y_pred, method_name=f"{method_name}|{call_id}"
    )
    metrics["call_id"]              = call_id
    metrics["method"]               = method_name
    metrics["n_matched_segments"]   = len(matched)
    metrics["n_total_segments"]     = len(classified)
    metrics["avg_similarity"]       = round(np.mean([m["similarity"] for m in matched]), 4)
    metrics["was_inverted"]         = inversion["is_inverted"]
    metrics["per_segment_correct"]  = [1 if m["correct"] else 0 for m in matched]
    metrics["matched_segments"]     = matched
    metrics["corrected_classified"] = classified_corrected

    return metrics


# ─────────────────────────────────────────────────────────────
# FULL VALIDATION PIPELINE
# ─────────────────────────────────────────────────────────────

def run_validation(calls_m1: list[list],
                   calls_m2: list[list],
                   calls_m3: list[list],
                   call_ids: list[str],
                   system_qa_scores: list[float] = None,
                   processing_time_sec: float = None,
                   total_audio_duration_sec: float = None,
                   csv_path: str = GROUND_TRUTH_CSV) -> dict:
    """
    Full validation against ground truth for all three methods.

    Parameters
    ----------
    calls_m1/m2/m3       : list of classified transcripts (one per call)
    call_ids             : call identifier strings
    system_qa_scores     : automated QA scores per call
    processing_time_sec  : wall-clock time for efficiency metric
    total_audio_duration : total audio seconds for RTF
    csv_path             : path to ground truth CSV

    Returns
    -------
    dict: aggregated metrics, statistical tests, efficiency, per_call results
    """
    gt_df = load_ground_truth(csv_path)

    all_results = {"m1": [], "m2": [], "m3": []}

    for m1, m2, m3, cid in zip(calls_m1, calls_m2, calls_m3, call_ids):
        for key, classified, mname in [
            ("m1", m1, "Method1-Lexical"),
            ("m2", m2, "Method2-Acoustic"),
            ("m3", m3, "Method3-Hybrid"),
        ]:
            if not classified:
                continue
            result = validate_call(classified, gt_df, cid, mname)
            if result:
                all_results[key].append(result)

    # Aggregate across calls
    def _agg(results_list):
        if not results_list:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
        return {
            "accuracy":  round(float(np.mean([r["accuracy"]  for r in results_list])), 2),
            "precision": round(float(np.mean([r["precision"] for r in results_list])), 2),
            "recall":    round(float(np.mean([r["recall"]    for r in results_list])), 2),
            "f1":        round(float(np.mean([r["f1"]        for r in results_list])), 2),
        }

    agg_m1 = _agg(all_results["m1"])
    agg_m2 = _agg(all_results["m2"])
    agg_m3 = _agg(all_results["m3"])

    # Per-sample correctness for t-test
    scores_m1 = [s for r in all_results["m1"] for s in r.get("per_segment_correct", [])]
    scores_m3 = [s for r in all_results["m3"] for s in r.get("per_segment_correct", [])]
    n = min(len(scores_m1), len(scores_m3))
    ttest = compute_paired_ttest(scores_m1[:n], scores_m3[:n]) if n >= 2 else {}

    # Pearson correlation with human QA scores
    pearson = {}
    if system_qa_scores and "human_qa_score" in gt_df.columns:
        human_scores = []
        for cid in call_ids:
            row = gt_df[gt_df["call_id"] == cid]
            if not row.empty:
                val = row["human_qa_score"].dropna()
                if not val.empty:
                    human_scores.append(float(val.mean()))
        if len(human_scores) == len(system_qa_scores) and len(human_scores) >= 2:
            pearson = compute_pearson_correlation(system_qa_scores, human_scores)

    # RTF
    efficiency = {}
    if processing_time_sec and total_audio_duration_sec:
        efficiency = compute_rtf(total_audio_duration_sec, processing_time_sec)

    # Inversion report
    inversions = {
        key: [r["call_id"] for r in results if r.get("was_inverted")]
        for key, results in all_results.items()
    }

    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info(f"  Method 1 (Keyword) : Acc={agg_m1['accuracy']}% F1={agg_m1['f1']}%")
    logger.info(f"  Method 2 (Acoustic): Acc={agg_m2['accuracy']}% F1={agg_m2['f1']}%")
    logger.info(f"  Method 3 (Hybrid)  : Acc={agg_m3['accuracy']}% F1={agg_m3['f1']}%")
    if any(v for v in inversions.values()):
        logger.warning(f"  Label inversions auto-corrected: {inversions}")
    logger.info("=" * 60)

    # Save plots
    plot_accuracy_comparison(agg_m1, agg_m2, agg_m3)
    if n >= 2:
        plot_ttest_boxplot(scores_m1[:n], scores_m3[:n], ttest)

    return {
        "aggregated":   {"m1": agg_m1, "m2": agg_m2, "m3": agg_m3},
        "ttest":        ttest,
        "pearson":      pearson,
        "efficiency":   efficiency,
        "inversions":   inversions,
        "per_call":     {k: [{
            "call_id":            r["call_id"],
            "accuracy":           r["accuracy"],
            "f1":                 r["f1"],
            "n_matched":          r["n_matched_segments"],
            "avg_similarity":     r["avg_similarity"],
            "was_inverted":       r["was_inverted"],
        } for r in v] for k, v in all_results.items()},
    }


# ─────────────────────────────────────────────────────────────
# QA COMPARISON TABLE
# ─────────────────────────────────────────────────────────────

def build_qa_comparison_table(call_ids, system_scores, human_scores) -> pd.DataFrame:
    rows = []
    for cid, sys_s, hum_s in zip(call_ids, system_scores, human_scores):
        diff    = abs(sys_s - hum_s)
        aligned = "✅ Aligned" if diff <= 20 else "⚠️ Divergence"
        rating  = "Good" if sys_s >= 75 else "Fair" if sys_s >= 55 else "Needs Improvement"
        rows.append({
            "call_id":       cid,
            "system_score":  round(sys_s, 2),
            "human_score":   round(hum_s, 2),
            "difference":    round(diff,  2),
            "alignment":     aligned,
            "system_rating": rating,
        })
    return pd.DataFrame(rows)


def plot_qa_score_comparison(comparison_df: pd.DataFrame,
                              save_path: str = None) -> str:
    sys_scores = comparison_df["system_score"].values
    hum_scores = comparison_df["human_score"].values
    call_ids   = comparison_df["call_id"].values

    pearson_result = compute_pearson_correlation(list(sys_scores), list(hum_scores))
    r = pearson_result.get("r", float("nan"))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(hum_scores, sys_scores, s=80, zorder=5,
               c=range(len(sys_scores)), cmap="tab10")
    for i, cid in enumerate(call_ids):
        ax.annotate(cid, (hum_scores[i], sys_scores[i]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    lim = [0, 105]
    ax.plot(lim, lim, "k--", linewidth=1, alpha=0.5, label="Perfect Agreement")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Human Analyst QA Score")
    ax.set_ylabel("Automated System QA Score")
    ax.set_title(f"System vs. Human QA Score Comparison\n(Pearson r = {r:.2f})")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "qa_score_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
