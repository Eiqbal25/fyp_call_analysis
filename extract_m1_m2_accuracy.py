"""
extract_m1_m2_accuracy.py
=========================
Extracts per-call accuracy for Method 1 (Lexical) and Method 2 (Acoustic)
from pipeline_results.json using the same text-similarity matching logic
as compare_labels.py / validator.py.

Run from project root:
    python extract_m1_m2_accuracy.py

Output:
    outputs/m1_m2_per_call_accuracy.csv
    outputs/m1_m2_per_call_accuracy.txt  (formatted report)
"""

import os
import json
import numpy as np
import pandas as pd
from difflib import SequenceMatcher

# ── CONFIG — paths derived from script location ──────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
PIPELINE_JSON    = os.path.join(BASE_DIR, "outputs", "latest", "pipeline_results.json")
HUMAN_DIR        = os.path.join(BASE_DIR, "human_transcripts")
GROUND_TRUTH_CSV = os.path.join(BASE_DIR, "human_validation_study.csv")
OUTPUT_CSV       = os.path.join(BASE_DIR, "outputs", "latest", "m1_m2_per_call_accuracy.csv")
OUTPUT_TXT       = os.path.join(BASE_DIR, "outputs", "latest", "m1_m2_per_call_accuracy.txt")

SIMILARITY_THRESHOLD = 0.40


# ── TEXT SIMILARITY ───────────────────────────────────────────
def text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


# ── MATCH AND EVALUATE ────────────────────────────────────────
def evaluate_method(segments: list, gt_df: pd.DataFrame,
                    auto_correct_inversion: bool = True) -> dict:
    """
    Match system segments to ground truth by text similarity.
    Returns accuracy, precision, recall, f1.
    """
    if not segments or gt_df.empty:
        return {"accuracy": None, "precision": None, "recall": None, "f1": None}

    matched = []
    used = set()

    for seg in segments:
        sys_text  = seg.get("text", "").strip()
        predicted = seg.get("predicted_role", "Unknown")
        if not sys_text or predicted == "Unknown":
            continue

        best_sim, best_idx, best_gt = 0.0, -1, None
        for idx, row in gt_df.iterrows():
            if idx in used:
                continue
            sim = text_similarity(sys_text, row["text"])
            if sim > best_sim:
                best_sim, best_idx, best_gt = sim, idx, row

        if best_sim >= SIMILARITY_THRESHOLD and best_gt is not None:
            used.add(best_idx)
            gt_role = best_gt["ground_truth_role"]
            matched.append({
                "predicted":    predicted,
                "ground_truth": gt_role,
                "correct":      (predicted == gt_role),
            })

    if not matched:
        return {"accuracy": None, "precision": None, "recall": None, "f1": None}

    # Auto-correct inversion
    if auto_correct_inversion:
        normal_acc   = sum(1 for m in matched if m["correct"]) / len(matched)
        flip         = {"Agent": "Customer", "Customer": "Agent"}
        inverted_acc = sum(
            1 for m in matched
            if flip.get(m["predicted"], m["predicted"]) == m["ground_truth"]
        ) / len(matched)
        if inverted_acc > normal_acc:
            for m in matched:
                m["predicted"] = flip.get(m["predicted"], m["predicted"])
                m["correct"]   = (m["predicted"] == m["ground_truth"])

    # Compute metrics
    tp = sum(1 for m in matched if m["predicted"] == "Agent"    and m["ground_truth"] == "Agent")
    tn = sum(1 for m in matched if m["predicted"] == "Customer" and m["ground_truth"] == "Customer")
    fp = sum(1 for m in matched if m["predicted"] == "Agent"    and m["ground_truth"] == "Customer")
    fn = sum(1 for m in matched if m["predicted"] == "Customer" and m["ground_truth"] == "Agent")

    total     = len(matched)
    accuracy  = round((tp + tn) / total * 100, 1) if total > 0 else None
    precision = round(tp / (tp + fp) * 100, 1)    if (tp + fp) > 0 else 0.0
    recall    = round(tp / (tp + fn) * 100, 1)    if (tp + fn) > 0 else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 1) \
                if (precision + recall) > 0 else 0.0

    return {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "matched":   total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ── MAIN ──────────────────────────────────────────────────────
def main():
    # Load pipeline results
    if not os.path.isfile(PIPELINE_JSON):
        print(f"ERROR: {PIPELINE_JSON} not found. Run main.py first.")
        return

    with open(PIPELINE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    calls_raw = data.get("calls", {})
    if isinstance(calls_raw, list):
        calls = {c["call_id"]: c for c in calls_raw if isinstance(c, dict)}
    else:
        calls = calls_raw

    # Load ground truth
    gt_loaded = False
    gt_df = pd.DataFrame()
    if os.path.isfile(GROUND_TRUTH_CSV):
        gt_df = pd.read_csv(GROUND_TRUTH_CSV)
        gt_df.columns = gt_df.columns.str.lower().str.strip()
        gt_df["ground_truth_role"] = gt_df["ground_truth_role"].str.strip().str.title()
        gt_df["text"] = gt_df["text"].fillna("").astype(str).str.strip()
        gt_loaded = True
        print(f"Loaded ground truth: {len(gt_df)} rows, {gt_df['call_id'].nunique()} calls")
    else:
        print(f"WARNING: Ground truth CSV not found at {GROUND_TRUTH_CSV}")

    # Process each call
    rows = []
    for cid, call_data in sorted(calls.items()):

        # Try per-call human transcript first
        ht_path = os.path.join(HUMAN_DIR, f"{cid}.csv")
        if os.path.isfile(ht_path):
            call_gt = pd.read_csv(ht_path)
            call_gt.columns = call_gt.columns.str.lower().str.strip()
            if "role" in call_gt.columns:
                call_gt = call_gt.rename(columns={"role": "ground_truth_role"})
            call_gt["ground_truth_role"] = call_gt["ground_truth_role"].str.strip().str.title()
            call_gt["text"] = call_gt["text"].fillna("").astype(str).str.strip()
        elif gt_loaded:
            call_gt = gt_df[gt_df["call_id"] == cid].copy()
        else:
            print(f"  SKIP {cid} — no ground truth")
            continue

        if call_gt.empty:
            print(f"  SKIP {cid} — empty ground truth")
            continue

        # Method 1
        m1_segs = call_data.get("method1", {}).get("classified", [])
        m1      = evaluate_method(m1_segs, call_gt, auto_correct_inversion=True)

        # Method 2
        m2_segs = call_data.get("method2", {}).get("classified", [])
        m2      = evaluate_method(m2_segs, call_gt, auto_correct_inversion=True)

        # Method 3 — use llm_classified segments only, no inversion correction
        m3_segs = call_data.get("method3", {}).get("classified", [])
        m3_segs = [s for s in m3_segs if s.get("llm_classified")]
        m3      = evaluate_method(m3_segs, call_gt, auto_correct_inversion=False)

        row = {
            "call_id":      cid,
            "m1_accuracy":  m1["accuracy"],
            "m1_f1":        m1["f1"],
            "m2_accuracy":  m2["accuracy"],
            "m2_f1":        m2["f1"],
            "m3_accuracy":  m3["accuracy"],
            "m3_f1":        m3["f1"],
            "segments":     m1.get("matched", 0),
        }
        rows.append(row)
        print(f"  {cid:<25} M1={m1['accuracy']}%  M2={m2['accuracy']}%  M3={m3['accuracy']}%")

    if not rows:
        print("No results generated.")
        return

    # Save CSV
    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")

    # Build text report
    lines = []
    lines.append("=" * 75)
    lines.append("PER-CALL ACCURACY — Method 1, Method 2, Method 3")
    lines.append("=" * 75)
    lines.append(f"{'Call ID':<25} {'M1 Acc':>8} {'M1 F1':>7} {'M2 Acc':>8} {'M2 F1':>7} {'M3 Acc':>8} {'M3 F1':>7}")
    lines.append("-" * 75)

    for row in rows:
        def fmt(v):
            return f"{v:.1f}%" if v is not None else "  N/A  "
        lines.append(
            f"{row['call_id']:<25} "
            f"{fmt(row['m1_accuracy']):>8} "
            f"{fmt(row['m1_f1']):>7} "
            f"{fmt(row['m2_accuracy']):>8} "
            f"{fmt(row['m2_f1']):>7} "
            f"{fmt(row['m3_accuracy']):>8} "
            f"{fmt(row['m3_f1']):>7}"
        )

    lines.append("-" * 75)

    # Averages
    def avg(col):
        vals = [r[col] for r in rows if r[col] is not None]
        return round(np.mean(vals), 1) if vals else None

    lines.append(
        f"{'AVERAGE':<25} "
        f"{fmt(avg('m1_accuracy')):>8} "
        f"{fmt(avg('m1_f1')):>7} "
        f"{fmt(avg('m2_accuracy')):>8} "
        f"{fmt(avg('m2_f1')):>7} "
        f"{fmt(avg('m3_accuracy')):>8} "
        f"{fmt(avg('m3_f1')):>7}"
    )
    lines.append("=" * 75)

    report = "\n".join(lines)
    print("\n" + report)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
