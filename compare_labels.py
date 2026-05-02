"""
compare_labels.py
=================
Compares the system's final Agent/Customer labels against your human-verified
transcripts in the human_transcripts/ folder.

This is ONLY for label accuracy checking — not for QA scores.

Usage:
    python compare_labels.py
    python compare_labels.py --call_id food_malay
    python compare_labels.py --show_diff     # print every wrong segment

What it produces:
    outputs/label_comparison_report.txt   — full text report
    outputs/label_comparison_{call}.csv   — per-segment diff table per call
    outputs/label_accuracy_summary.png    — bar chart per call

UPGRADE: Matching uses timestamp overlap + text similarity combined.
Old version used text-only which caused wrong GT rows to be matched
when two consecutive system segments were both within the same GT row's
time range.
"""

import os
import sys
import json
import glob
import argparse
import logging
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("compare_labels")

from config import (
    HUMAN_TRANSCRIPTS_DIR,
    OUTPUTS_DIR,
    LABEL_MATCH_SIMILARITY_THRESHOLD,
)


# ─────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare system labels vs human transcript"
    )
    parser.add_argument("--call_id",   default=None,
                        help="Compare only this call")
    parser.add_argument("--results",
                        default=os.path.join(OUTPUTS_DIR, "pipeline_results.json"),
                        help="Path to pipeline_results.json")
    parser.add_argument("--show_diff", action="store_true",
                        help="Print every wrong segment to console")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# LOAD HUMAN TRANSCRIPTS
# ─────────────────────────────────────────────────────────────

def load_human_transcript(call_id: str) -> pd.DataFrame:
    """
    Load human_transcripts/{call_id}.csv

    Required columns: segment_id, role, text
    Optional columns: start, end
    """
    csv_path = os.path.join(HUMAN_TRANSCRIPTS_DIR, f"{call_id}.csv")
    if not os.path.isfile(csv_path):
        return None

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()

    required = {"segment_id", "role", "text"}
    missing  = required - set(df.columns)
    if missing:
        logger.error(
            f"Human transcript {csv_path} missing columns: {missing}\n"
            f"Required: segment_id, role, text\n"
            f"Got: {list(df.columns)}"
        )
        return None

    df["role"] = df["role"].str.strip().str.title()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df.sort_values("segment_id").reset_index(drop=True)
    return df


def list_available_human_transcripts() -> list[str]:
    """Return list of call_ids that have human transcript CSVs."""
    files = glob.glob(os.path.join(HUMAN_TRANSCRIPTS_DIR, "*.csv"))
    # Exclude the example file
    return [
        os.path.splitext(os.path.basename(f))[0]
        for f in sorted(files)
        if "example" not in os.path.basename(f).lower()
    ]


# ─────────────────────────────────────────────────────────────
# SIMILARITY HELPERS
# ─────────────────────────────────────────────────────────────

def _text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _timestamp_overlap_score(sys_start: float, sys_end: float,
                               gt_start: float,  gt_end: float) -> float:
    """
    Returns fraction of the system segment that overlaps the GT row's time range.
    Range 0.0 – 1.0.  1.0 = system segment is completely inside GT row range.
    """
    seg_dur = max(sys_end - sys_start, 0.01)
    overlap_start = max(sys_start, gt_start)
    overlap_end   = min(sys_end,   gt_end)
    overlap       = max(0.0, overlap_end - overlap_start)
    return min(1.0, overlap / seg_dur)


# ─────────────────────────────────────────────────────────────
# SEGMENT MATCHING — UPGRADED: timestamp + text combined
# ─────────────────────────────────────────────────────────────

def match_and_compare(system_segments: list[dict],
                      human_df: pd.DataFrame,
                      call_id: str) -> pd.DataFrame:
    """
    Match each system segment to the best GT row using a combined score:

        combined = (text_similarity × 0.6) + (timestamp_overlap × 0.4)

    KEY FIX — Many-to-one matching:
    Human annotators write coarse segments (e.g. one row = 8 seconds).
    Whisper splits the same audio into many fine segments (3-4 per human row).
    The old code used used_human_idx to prevent one human row matching twice,
    which caused all but the first system segment in a time range to be silently
    dropped — inflating accuracy by only counting easy matches.

    Fix: each human row CAN match multiple system segments.
    Unmatched system segments (below threshold) are recorded as UNMATCHED
    and counted as wrong in the accuracy, giving a true result.
    """
    rows = []

    # Pre-compute GT start/end (fill missing with estimated values)
    gt_starts = []
    gt_ends   = []
    for idx, hrow in human_df.iterrows():
        s = float(hrow["start"]) if "start" in hrow.index and pd.notna(hrow.get("start")) else float(idx) * 5.0
        e = float(hrow["end"])   if "end"   in hrow.index and pd.notna(hrow.get("end"))   else s + 5.0
        gt_starts.append(s)
        gt_ends.append(e)

    for seg in system_segments:
        sys_text  = seg.get("text", "").strip()
        sys_role  = seg.get("predicted_role", "Unknown")
        sys_start = float(seg.get("start", 0))
        sys_end   = float(seg.get("end", sys_start + 1.0))

        if not sys_text or sys_role == "Unknown":
            continue

        best_combined = 0.0
        best_idx      = -1
        best_row      = None
        best_text_sim = 0.0

        # No used_human_idx — each human row can match multiple system segments
        for idx, hrow in human_df.iterrows():
            gt_s = gt_starts[idx]
            gt_e = gt_ends[idx]

            text_sim  = _text_similarity(sys_text, str(hrow["text"]))
            time_sim  = _timestamp_overlap_score(sys_start, sys_end, gt_s, gt_e)
            combined  = (text_sim * 0.6) + (time_sim * 0.4)

            if combined > best_combined:
                best_combined  = combined
                best_idx       = idx
                best_row       = hrow
                best_text_sim  = text_sim

        if best_combined >= LABEL_MATCH_SIMILARITY_THRESHOLD and best_row is not None:
            # Matched — compare labels normally
            human_role    = best_row["role"]
            label_match   = (sys_role == human_role)
            text_override = seg.get("text_override", False)

            rows.append({
                "call_id":         call_id,
                "start_sec":       sys_start,
                "system_text":     sys_text[:80],
                "human_text":      str(best_row["text"])[:80],
                "text_similarity": round(best_text_sim, 3),
                "combined_score":  round(best_combined, 3),
                "system_role":     sys_role,
                "human_role":      human_role,
                "label_correct":   label_match,
                "matched":         True,
                "text_override":   text_override,
                "override_phrase": seg.get("text_override_phrase", ""),
                "status":          "✅ CORRECT" if label_match else "❌ WRONG",
            })
        else:
            # Unmatched — count as wrong so accuracy is not inflated
            rows.append({
                "call_id":         call_id,
                "start_sec":       sys_start,
                "system_text":     sys_text[:80],
                "human_text":      "",
                "text_similarity": 0.0,
                "combined_score":  round(best_combined, 3),
                "system_role":     sys_role,
                "human_role":      "Unknown",
                "label_correct":   False,
                "matched":         False,
                "text_override":   False,
                "override_phrase": "",
                "status":          "⚠ UNMATCHED",
            })

    return pd.DataFrame(rows)



# ─────────────────────────────────────────────────────────────
# PER-CALL METRICS
# ─────────────────────────────────────────────────────────────

def compute_label_metrics(comparison_df: pd.DataFrame) -> dict:
    """
    Compute Accuracy, Precision, Recall, F1. Positive class = Agent.
    """
    if comparison_df.empty:
        return {}

    tp = len(comparison_df[(comparison_df["system_role"] == "Agent") &
                            (comparison_df["human_role"]  == "Agent")])
    tn = len(comparison_df[(comparison_df["system_role"] == "Customer") &
                            (comparison_df["human_role"]  == "Customer")])
    fp = len(comparison_df[(comparison_df["system_role"] == "Agent") &
                            (comparison_df["human_role"]  == "Customer")])
    fn = len(comparison_df[(comparison_df["system_role"] == "Customer") &
                            (comparison_df["human_role"]  == "Agent")])

    total     = tp + tn + fp + fn
    accuracy  = (tp + tn) / total              if total > 0         else 0.0
    precision = tp / (tp + fp)                 if (tp + fp) > 0     else 0.0
    recall    = tp / (tp + fn)                 if (tp + fn) > 0     else 0.0
    f1        = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0.0

    n_total     = len(comparison_df)
    n_matched   = int(comparison_df["matched"].sum()) if "matched" in comparison_df.columns else n_total
    n_unmatched = n_total - n_matched
    n_correct   = int(comparison_df["label_correct"].sum())
    n_wrong     = n_total - n_correct
    n_override  = int(comparison_df["text_override"].sum()) \
                  if "text_override" in comparison_df.columns else 0
    avg_sim     = float(comparison_df[comparison_df["matched"] == True]["text_similarity"].mean()) \
                  if n_matched > 0 else 0.0

    return {
        "n_total":         n_total,
        "n_matched":       n_matched,
        "n_unmatched":     n_unmatched,
        "n_correct":       n_correct,
        "n_wrong":         n_wrong,
        "n_text_override": n_override,
        "accuracy_pct":    round(accuracy  * 100, 2),
        "precision_pct":   round(precision * 100, 2),
        "recall_pct":      round(recall    * 100, 2),
        "f1_pct":          round(f1        * 100, 2),
        "avg_text_sim":    round(avg_sim,   3),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_accuracy_per_call(metrics_per_call: dict,
                            save_path: str = None) -> str:
    """Bar chart: label accuracy per call."""
    call_ids = list(metrics_per_call.keys())
    accs     = [metrics_per_call[c].get("accuracy_pct", 0) for c in call_ids]
    colors   = [
        "seagreen" if a >= 90 else "orange" if a >= 75 else "tomato"
        for a in accs
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(call_ids) * 1.5), 5))
    bars = ax.bar(call_ids, accs, color=colors, alpha=0.85, edgecolor="white")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.axhline(75,  color="orange", linestyle="--", alpha=0.5, label="75% threshold")
    ax.axhline(90,  color="green",  linestyle="--", alpha=0.5, label="90% threshold")
    ax.set_ylim([0, 115])
    ax.set_ylabel("Label Accuracy (%)")
    ax.set_title("System Label Accuracy vs Human Transcript (per call)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "label_accuracy_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Label accuracy chart saved → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    logger.info("=" * 65)
    logger.info("FYP1 — LABEL COMPARISON: System vs Human Transcript")
    logger.info("=" * 65)

    # Load pipeline results
    if not os.path.isfile(args.results):
        logger.error(
            f"Pipeline results not found: {args.results}\n"
            "Run  python main.py  first."
        )
        sys.exit(1)

    with open(args.results, "r", encoding="utf-8") as f:
        pipeline_data = json.load(f)
    calls = pipeline_data.get("calls", {})

    # Find human transcripts
    available = list_available_human_transcripts()
    if not available:
        logger.error(
            f"No human transcript CSVs found in: {HUMAN_TRANSCRIPTS_DIR}\n"
            f"Create a CSV file named {{call_id}}.csv in that folder.\n"
            f"Required columns: segment_id, role, text\n"
            f"Optional columns: start, end\n"
            f"Example: 1,Agent,Thank you for calling how may I assist,0.0,5.2"
        )
        sys.exit(1)

    logger.info(f"Human transcripts found: {available}")

    # Filter
    if args.call_id:
        if args.call_id not in available:
            logger.error(f"No human transcript for call_id='{args.call_id}'")
            sys.exit(1)
        to_compare = [args.call_id]
    else:
        to_compare      = [c for c in available if c in calls]
        not_in_results  = [c for c in available if c not in calls]
        if not_in_results:
            logger.warning(
                f"These transcripts have no pipeline results yet: {not_in_results}\n"
                f"Run  python main.py  first."
            )

    if not to_compare:
        logger.error(
            "No overlap between human transcripts and pipeline results.\n"
            "Make sure your CSV filenames match your audio filenames exactly."
        )
        sys.exit(1)

    # ── Compare each call ──────────────────────────────────────
    all_metrics  = {}
    report_lines = []

    report_lines.append("=" * 65)
    report_lines.append("FYP1 LABEL COMPARISON REPORT — System vs Human Transcript")
    report_lines.append("=" * 65)

    for cid in to_compare:
        human_df = load_human_transcript(cid)
        if human_df is None:
            continue

        call_data   = calls.get(cid, {})
        system_segs = call_data.get("transcript_hybrid", [])
        if not system_segs:
            system_segs = call_data.get("method3", {}).get("classified", [])
        if not system_segs:
            logger.warning(f"No system output for '{cid}' — skipping")
            continue

        logger.info(
            f"\nComparing: {cid} | "
            f"system_segs={len(system_segs)} | "
            f"human_rows={len(human_df)}"
        )

        comparison_df = match_and_compare(system_segs, human_df, cid)

        if comparison_df.empty:
            logger.warning(
                f"No matches for '{cid}'. "
                f"Check outputs/{cid}_diarized.json to see Whisper's transcription "
                f"and update your human transcript text to be similar."
            )
            continue

        metrics = compute_label_metrics(comparison_df)
        all_metrics[cid] = metrics

        # Save diff CSV
        diff_csv = os.path.join(OUTPUTS_DIR, f"label_comparison_{cid}.csv")
        comparison_df.to_csv(diff_csv, index=False)
        logger.info(f"Diff saved → {diff_csv}")

        # Report section
        sep = "─" * (50 - len(cid))
        report_lines.append(f"\n── {cid} {sep}")
        report_lines.append(
            f"  Total segments   : {metrics['n_total']}  "
            f"(Matched={metrics['n_matched']}  Unmatched={metrics['n_unmatched']})"
        )
        report_lines.append(
            f"  Label accuracy   : {metrics['accuracy_pct']:.1f}%  "
            f"(Correct={metrics['n_correct']}  Wrong={metrics['n_wrong']}  "
            f"Unmatched counted as wrong={metrics['n_unmatched']})"
        )
        report_lines.append(f"  Precision        : {metrics['precision_pct']:.1f}%")
        report_lines.append(f"  Recall           : {metrics['recall_pct']:.1f}%")
        report_lines.append(f"  F1-Score         : {metrics['f1_pct']:.1f}%")
        report_lines.append(
            f"  Confusion        : TP={metrics['tp']} TN={metrics['tn']} "
            f"FP={metrics['fp']} FN={metrics['fn']}"
        )
        report_lines.append(
            f"  Avg text sim     : {metrics['avg_text_sim']:.2f}"
        )
        if metrics.get("n_text_override", 0) > 0:
            report_lines.append(
                f"  Text overrides   : {metrics['n_text_override']} segment(s) "
                f"corrected by strong phrase detection"
            )

        # Wrong segments
        wrong = comparison_df[~comparison_df["label_correct"]]
        if not wrong.empty:
            report_lines.append(f"\n  Wrong labels ({len(wrong)}):")
            for _, row in wrong.iterrows():
                report_lines.append(
                    f"    [{row['start_sec']:.1f}s] "
                    f"System={row['system_role']:8s} Human={row['human_role']:8s} "
                    f"| \"{row['system_text'][:50]}\""
                )
            if args.show_diff:
                print("\n".join(report_lines[-len(wrong)-1:]))
        else:
            report_lines.append("  ✅ All matched segments labelled correctly!")

        # Text override summary
        overridden = comparison_df[comparison_df.get("text_override", False) == True] \
                     if "text_override" in comparison_df.columns else pd.DataFrame()
        if not overridden.empty:
            report_lines.append(f"\n  Text override corrections ({len(overridden)}):")
            for _, row in overridden.iterrows():
                correct_marker = "✅" if row["label_correct"] else "❌"
                report_lines.append(
                    f"    {correct_marker} [{row['start_sec']:.1f}s] → {row['system_role']} "
                    f"(phrase: '{row.get('override_phrase','')}') "
                    f"| \"{row['system_text'][:40]}\""
                )

    # ── Overall summary ──────────────────────────────────────────
    if all_metrics:
        overall_acc = np.mean([m["accuracy_pct"] for m in all_metrics.values()])
        overall_f1  = np.mean([m["f1_pct"]       for m in all_metrics.values()])

        report_lines.append("\n" + "=" * 65)
        report_lines.append("OVERALL SUMMARY")
        report_lines.append("=" * 65)
        report_lines.append(
            f"{'Call ID':<25} {'Accuracy':>8} {'F1':>7} {'Correct':>8} {'Wrong':>6}"
        )
        report_lines.append("-" * 65)
        for cid, m in all_metrics.items():
            report_lines.append(
                f"{cid:<25} {m['accuracy_pct']:>7.1f}% {m['f1_pct']:>6.1f}% "
                f"{m['n_correct']:>8} {m['n_wrong']:>6}"
            )
        report_lines.append("-" * 65)
        report_lines.append(
            f"{'AVERAGE':<25} {overall_acc:>7.1f}% {overall_f1:>6.1f}%"
        )
        report_lines.append("")
        if overall_acc >= 90:
            report_lines.append(
                "✅ Labelling accuracy ≥ 90% — good performance"
            )
        elif overall_acc >= 75:
            report_lines.append(
                "⚠ Labelling accuracy 75–90% — review wrong segments above"
            )
        else:
            report_lines.append(
                "❌ Labelling accuracy below 75% — check diarization output "
                f"in outputs/{{call_id}}_diarized.json"
            )

    # Print + save
    report = "\n".join(report_lines)
    print(report)

    report_path = os.path.join(OUTPUTS_DIR, "2_label_comparison_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"\nReport saved → {report_path}")

    if all_metrics:
        plot_accuracy_per_call(all_metrics)


if __name__ == "__main__":
    main()
