"""
evaluate.py
===========
Standalone evaluation script — runs Phase 4 only.

Reads an existing pipeline_results.json, compares all three methods
against the ground truth CSV, and prints a clean results table
including accuracy, F1, Pearson r, t-test, and efficiency metrics.

Usage:
    python evaluate.py
    python evaluate.py --results outputs/pipeline_results.json
    python evaluate.py --save_report outputs/evaluation_report.txt
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd

os.makedirs("outputs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate")

from config import GROUND_TRUTH_CSV, OUTPUTS_DIR, HUMAN_TRANSCRIPTS_DIR
from evaluation.metrics import (
    compute_classification_metrics,
    compute_paired_ttest,
    compute_pearson_correlation,
    compute_rtf,
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_ttest_boxplot,
)
from evaluation.validator import (
    load_ground_truth,
    validate_call,           # FIX: removed extract_predictions (no longer exists)
    build_qa_comparison_table,
    plot_qa_score_comparison,
)


# ─────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="FYP1 — Standalone Evaluation")
    parser.add_argument(
        "--results",
        default=os.path.join(OUTPUTS_DIR, "pipeline_results.json"),
        help="Path to pipeline_results.json (from main.py)",
    )
    parser.add_argument(
        "--csv", default=GROUND_TRUTH_CSV,
        help="Ground truth CSV path",
    )
    parser.add_argument(
        "--save_report",
        default=os.path.join(OUTPUTS_DIR, "1_evaluation_report.txt"),
        help="Save text report to this path",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# LOAD RESULTS
# ─────────────────────────────────────────────────────────────

def _load_results(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Results file not found: {path}\n"
            "Run  python main.py  first to generate it."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    logger.info("=" * 65)
    logger.info("FYP1 CALL ANALYSIS — EVALUATION REPORT")
    logger.info("=" * 65)

    # Load pipeline results
    data   = _load_results(args.results)
    calls_raw = data.get("calls", {})
    # Handle both list and dict formats
    if isinstance(calls_raw, list):
        calls = {c["call_id"]: c for c in calls_raw if isinstance(c, dict)}
    else:
        calls = calls_raw
    summary = data.get("summary", {})

    if not calls:
        logger.error("No calls found in results JSON.")
        sys.exit(1)

    # Load ground truth CSV
    if not os.path.isfile(args.csv):
        logger.error(f"Ground truth CSV not found: {args.csv}")
        sys.exit(1)

    gt_df = load_ground_truth(args.csv)

    # ── Per-call, per-method evaluation ──────────────────────────
    all_m1, all_m2, all_m3 = [], [], []
    scores_m1_flat, scores_m3_flat = [], []
    system_qa_scores, human_qa_scores = [], []

    for cid, call_data in calls.items():

        # Try loading full human transcript for better evaluation
        ht_path = os.path.join(HUMAN_TRANSCRIPTS_DIR, f"{cid}.csv")
        if os.path.isfile(ht_path):
            import pandas as _pd
            call_gt_df = _pd.read_csv(ht_path)
            if "role" in call_gt_df.columns:
                call_gt_df = call_gt_df.rename(columns={"role": "ground_truth_role"})
        else:
            call_gt_df = gt_df[gt_df["call_id"] == cid]

        for key, method_name, collector in [
            ("method1",    "Method1-Lexical",  all_m1),
            ("method2",    "Method2-Acoustic", all_m2),
            ("method3",    "Method3-LLM",      all_m3),
        ]:
            classified = call_data.get(key, {}).get("classified", [])
            # For LLM — only use segments that were classified by LLM
            if method_name == "Method3-LLM":
                classified = [s for s in classified if s.get("llm_classified")]
            if not classified:
                continue

            # Disable inversion correction for M3 LLM — it's reliable
            auto_correct = (method_name != "Method3-LLM")
            result = validate_call(classified, call_gt_df, cid, method_name,
                                   auto_correct_inversion=auto_correct)
            if result:
                collector.append(result)
                if key == "method1":
                    scores_m1_flat.extend(result.get("per_segment_correct", []))
                if key == "method3":
                    scores_m3_flat.extend(result.get("per_segment_correct", []))

        # Collect QA scores for Pearson correlation
        sys_qa = call_data.get("qa_result", {}).get("qa_score")
        gt_row  = gt_df[gt_df["call_id"] == cid]
        if sys_qa is not None and not gt_row.empty and "human_qa_score" in gt_row.columns:
            human_val = gt_row["human_qa_score"].dropna()
            if not human_val.empty:
                system_qa_scores.append(float(sys_qa))
                human_qa_scores.append(float(human_val.mean()))

    # ── Aggregate metrics across all calls ───────────────────────
    def _agg(results_list):
        if not results_list:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
        return {
            "accuracy":  round(float(np.mean([r["accuracy"]  for r in results_list])), 2),
            "precision": round(float(np.mean([r["precision"] for r in results_list])), 2),
            "recall":    round(float(np.mean([r["recall"]    for r in results_list])), 2),
            "f1":        round(float(np.mean([r["f1"]        for r in results_list])), 2),
        }

    agg_m1     = _agg(all_m1)
    agg_m2     = _agg(all_m2)
    agg_m3     = _agg(all_m3)

    # ── Statistical tests ─────────────────────────────────────────
    n = min(len(scores_m1_flat), len(scores_m3_flat))
    ttest = compute_paired_ttest(scores_m1_flat[:n], scores_m3_flat[:n]) if n >= 2 else {}

    pearson = {}
    if len(system_qa_scores) >= 2:
        pearson = compute_pearson_correlation(system_qa_scores, human_qa_scores)

    # ── Efficiency ────────────────────────────────────────────────
    rtf_result = {}
    if summary.get("total_audio_sec") and summary.get("processing_time_sec"):
        rtf_result = compute_rtf(
            float(summary["total_audio_sec"]),
            float(summary["processing_time_sec"]),
        )

    # ── Build report ──────────────────────────────────────────────
    lines = []
    lines.append("=" * 65)
    lines.append("FYP1 CALL ANALYSIS — EVALUATION REPORT")
    lines.append("=" * 65)
    lines.append(f"\nCalls evaluated  : {len(calls)}")
    if summary.get("total_audio_sec"):
        lines.append(f"Total audio      : {summary['total_audio_sec']:.1f} s")
    if summary.get("processing_time_sec"):
        lines.append(f"Processing time  : {summary['processing_time_sec']:.1f} s")

    lines.append("\n── Classification Performance ───────────────────────────")
    lines.append(f"{'Method':<28} {'Accuracy':>8} {'Precision':>9} {'Recall':>7} {'F1':>6}")
    lines.append("-" * 65)
    for label, m in [
        ("Method 1 (Keyword-Lexical)", agg_m1),
        ("Method 2 (Acoustic DNN)",    agg_m2),
        ("Method 3 (LLM — Llama 3.3)", agg_m3),

    ]:
        lines.append(
            f"{label:<28} "
            f"{m['accuracy']:>7.1f}%"
            f"{m['precision']:>8.1f}%"
            f"{m['recall']:>7.1f}%"
            f"{m['f1']:>6.1f}%"
        )

    lines.append("\n── Paired t-test (Method 4 vs Method 1) ────────────────")
    if ttest:
        lines.append(f"  t-statistic : {ttest.get('t_statistic', 'N/A')}")
        lines.append(f"  p-value     : {ttest.get('p_value', 'N/A')} (α = {ttest.get('alpha', 0.05)})")
        lines.append(f"  Reject H₀   : {ttest.get('reject_null', 'N/A')}")
        lines.append(f"  Conclusion  : {ttest.get('conclusion', 'N/A')}")
    else:
        lines.append("  Not enough samples (need ≥2 calls)")

    lines.append("\n── System vs. Human QA Correlation ─────────────────────")
    if pearson:
        lines.append(f"  Pearson r     : {pearson.get('r', 'N/A')}")
        lines.append(f"  p-value       : {pearson.get('p_value', 'N/A')}")
        lines.append(f"  Interpretation: {pearson.get('interpretation', 'N/A')}")
    else:
        lines.append("  Not enough data (need ≥2 calls with human_qa_score)")

    lines.append("\n── Efficiency (Real-Time Factor) ────────────────────────")
    if rtf_result:
        lines.append(f"  RTF              : {rtf_result.get('rtf', 'N/A')}")
        lines.append(f"  Real-time capable: {rtf_result.get('is_real_time', 'N/A')}")
        lines.append(f"  Speed multiplier : {rtf_result.get('efficiency_multiplier', 'N/A')}×")
    else:
        lines.append("  No timing data available")

    # Label inversion summary
    inverted_calls = [
        cid for method_results in [all_m1, all_m2, all_m3]
        for r in method_results
        for cid in [r.get("call_id", "")]
        if r.get("was_inverted", False)
    ]
    if inverted_calls:
        lines.append(f"\n⚠ Auto-corrected label inversions: {list(set(inverted_calls))}")

    lines.append("\n── Per-Call QA Score Comparison ─────────────────────────")
    if system_qa_scores and human_qa_scores:
        qa_call_ids = []
        for cid in calls:
            sys_qa = calls[cid].get("qa_result", {}).get("qa_score")
            gt_row  = gt_df[gt_df["call_id"] == cid]
            if sys_qa is not None and not gt_row.empty and "human_qa_score" in gt_row.columns:
                if not gt_row["human_qa_score"].dropna().empty:
                    qa_call_ids.append(cid)

        if len(qa_call_ids) == len(system_qa_scores):
            comp_df = build_qa_comparison_table(qa_call_ids, system_qa_scores, human_qa_scores)
            lines.append(comp_df.to_string(index=False))
            plot_qa_score_comparison(comp_df)
    else:
        lines.append("  No human_qa_score in CSV")

    # ── WER + Language Summary ───────────────────────────────
    from analytics.advanced import format_wer_summary
    wer_data = [
        {"call_id": cid,
         "wer": cdata.get("wer", {}),
         "language": cdata.get("language", {})}
        for cid, cdata in calls.items()
    ]
    lines.append(format_wer_summary(wer_data))

    # ── Agent Response Time + Interruptions ───────────────────
    lines.append("\n── Agent Performance Metrics ────────────────────────────")
    lines.append(f"  {'Call ID':<25} {'Lang':<12} {'Resp Time':>10} {'Interrupts':>12} {'Rating'}")
    lines.append("  " + "-" * 70)
    for cid, cdata in sorted(calls.items()):
        rt   = cdata.get("response_time", {})
        intr = cdata.get("interruptions", {})
        lang = cdata.get("language", {}).get("detected_language", "?")
        avg_rt   = f"{rt.get('avg_response_time_sec', 0):.2f}s" if rt.get("avg_response_time_sec") else "N/A"
        n_intr   = intr.get("total_interruptions", 0)
        rating   = rt.get("rating", "N/A")
        lines.append(f"  {cid:<25} {lang:<12} {avg_rt:>10} {n_intr:>12} {rating}")

    # ── Call Outcome Summary ─────────────────────────────────
    lines.append("\n── Call Outcome Summary ─────────────────────────────────")
    outcome_counts = {"Resolved": 0, "Unresolved": 0, "Escalated": 0, "Transferred": 0}
    for cid, call_data in calls.items():
        outcome = call_data.get("call_outcome", {})
        if outcome:
            o = outcome.get("outcome", "Resolved")
            outcome_counts[o] = outcome_counts.get(o, 0) + 1
            emoji = outcome.get("emoji", "✅")
            lines.append(f"  {cid:<25} {emoji} {o}")
    lines.append(f"\n  Summary: Resolved={outcome_counts['Resolved']} | "
                 f"Unresolved={outcome_counts['Unresolved']} | "
                 f"Escalated={outcome_counts['Escalated']} | "
                 f"Transferred={outcome_counts['Transferred']}")

    # ── Rude Behavior Summary ────────────────────────────────
    lines.append("\n── Rude Behavior Warnings ───────────────────────────────")
    has_rude = False
    for cid, call_data in calls.items():
        rude = call_data.get("rude_behavior")
        if not rude:
            continue
        agent_level = rude.get("agent_rudeness_level", "NONE")
        cust_level  = rude.get("customer_rudeness_level", "NONE")
        if agent_level != "NONE" or cust_level != "NONE":
            has_rude = True
            a_count = rude.get("total_agent_incidents", 0)
            c_count = rude.get("total_customer_incidents", 0)
            lines.append(
                f"  {cid:<25} Agent={agent_level:<6} ({a_count} incidents) | "
                f"Customer={cust_level:<6} ({c_count} incidents)"
            )
            for w in rude.get("warnings", []):
                lines.append(f"    {w}")
    if not has_rude:
        lines.append("  No significant rude behavior detected across all calls.")

    lines.append("\n" + "=" * 65)
    report = "\n".join(lines)

    # Print and save
    print(report)
    os.makedirs(os.path.dirname(args.save_report) or ".", exist_ok=True)
    with open(args.save_report, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"\nReport saved → {args.save_report}")

    # Save charts
    plot_accuracy_comparison(agg_m1, agg_m2, agg_m3)
    if n >= 2:
        plot_ttest_boxplot(scores_m1_flat[:n], scores_m3_flat[:n], ttest)

    # ── Save validation summary JSON for dashboard ───────────
    import json as _json
    eval_summary = {
        "aggregated": {
            "m1":     {"accuracy": agg_m1["accuracy"], "f1": agg_m1["f1"],
                       "precision": agg_m1["precision"], "recall": agg_m1["recall"]},
            "m2":     {"accuracy": agg_m2["accuracy"], "f1": agg_m2["f1"],
                       "precision": agg_m2["precision"], "recall": agg_m2["recall"]},
            "m3":     {"accuracy": agg_m3["accuracy"], "f1": agg_m3["f1"],
                       "precision": agg_m3["precision"], "recall": agg_m3["recall"]},
        },
        "ttest": {
            "t_statistic": float(ttest.get("t_statistic", 0)),
            "p_value":     float(ttest.get("p_value", 1)),
            "reject_null": bool(ttest.get("reject_null", False)),
            "conclusion":  str(ttest.get("conclusion", "")),
        },
        "n_calls": len(calls),
        "source": "evaluate.py — full 30-call evaluation",
    }
    eval_json_path = os.path.join(OUTPUTS_DIR, "evaluation_summary.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        _json.dump(eval_summary, f, indent=2)
    logger.info(f"Evaluation summary saved → {eval_json_path}")

    for method_name, results_list in [
        ("Method 1 Lexical",  all_m1),
        ("Method 2 Acoustic", all_m2),
        ("Method 3 LLM",      all_m3),
    ]:
        if results_list:
            tp = sum(r["confusion_matrix"]["TP"] for r in results_list)
            tn = sum(r["confusion_matrix"]["TN"] for r in results_list)
            fp = sum(r["confusion_matrix"]["FP"] for r in results_list)
            fn = sum(r["confusion_matrix"]["FN"] for r in results_list)
            plot_confusion_matrix(
                {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
                method_name=method_name,
            )


if __name__ == "__main__":
    main()
