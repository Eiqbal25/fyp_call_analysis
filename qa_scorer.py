"""
qa_scorer.py
============
Calculates QA scores for customer service calls using an industry-standard
weighted rubric based on:

  - Balto.ai (2025) "Call Center Quality Monitoring Scorecards"
  - Globalify (2026) "Call Center Quality Assurance Guide"
  - Calabrio (2026) "QA Scorecard: 7 Tips for Quality Assurance"

Formula (total = 100 points):
  Resolution     30%  — Was the issue resolved?
  Compliance     25%  — Did agent follow protocol (greeting, closing, ID check)?
  Sentiment      20%  — Emotional quality of interaction
  Communication  15%  — Talk balance + turn-taking
  Professionalism 10% — Agent rudeness (auto-fail if HIGH rudeness)

Auto-fail rule (industry standard):
  Agent rudeness = HIGH → final score capped at 20/100

Usage:
    python qa_scorer.py                          # score all calls in pipeline_results.json
    python qa_scorer.py --update_csv             # also rebuild human_validation_study.csv
"""

import os
import json
import argparse
import pandas as pd
from config import OUTPUTS_DIR, GROUND_TRUTH_CSV


# ── Weight table (must sum to 1.0) ────────────────────────────────────────────
WEIGHTS = {
    "resolution":      0.30,
    "compliance":      0.25,
    "sentiment":       0.20,
    "communication":   0.15,
    "professionalism": 0.10,
}

# ── Auto-fail threshold ────────────────────────────────────────────────────────
AGENT_RUDE_AUTOFAIL_CAP = 20.0   # max score if agent is HIGH rudeness
AGENT_RUDE_MEDIUM_PENALTY = 15.0 # deducted from professionalism if MEDIUM


def score_resolution(call_data: dict) -> float:
    """
    Resolution score (0-100).
    Resolved = 100, Transferred = 70 (partially resolved),
    Escalated = 50, Unresolved = 0.
    """
    outcome = call_data.get("call_outcome", {})
    if isinstance(outcome, dict):
        result = outcome.get("outcome", "Unknown")
    else:
        result = str(outcome)

    mapping = {
        "Resolved":    100.0,
        "Transferred":  70.0,
        "Escalated":    50.0,
        "Unresolved":    0.0,
        "Unknown":      50.0,
    }
    return mapping.get(result, 50.0)


def score_compliance(call_data: dict) -> float:
    """
    Compliance score (0-100).
    Directly uses system compliance_score (already 0-100).
    """
    comp = call_data.get("compliance", {})
    if isinstance(comp, dict):
        return float(comp.get("compliance_score", 0.0))
    return 0.0


def score_sentiment(call_data: dict) -> float:
    """
    Sentiment score (0-100).
    Based on agent sentiment (positive = good) and customer sentiment trend.
    Formula:
      - Agent avg compound: map [-1, +1] → [0, 100]
      - Customer avg compound: if negative, penalize; if positive, reward
      - Emotional mismatch: -10 penalty
    """
    sent = call_data.get("sentiment", {})
    if not isinstance(sent, dict):
        return 50.0

    agent_compound    = float(sent.get("agent_avg_compound",    0.0))
    customer_compound = float(sent.get("customer_avg_compound", 0.0))
    mismatch          = bool(sent.get("emotional_mismatch",     False))

    # Map compound [-1,+1] → [0,100]
    agent_score    = (agent_compound    + 1) / 2 * 100
    customer_score = (customer_compound + 1) / 2 * 100

    # Agent sentiment weighted more heavily (agent controls tone)
    combined = agent_score * 0.6 + customer_score * 0.4

    if mismatch:
        combined -= 10.0

    return max(0.0, min(100.0, combined))


def score_communication(call_data: dict) -> float:
    """
    Communication score (0-100).
    Based on:
      - Talk ratio balance: ideal is 40-60% agent / 40-60% customer
      - Turn-taking balance: equal turns = 100
      - Response time: fast response = good (Excellent/Good/Fair)
    """
    talk  = call_data.get("talk_ratio",    {})
    turns = call_data.get("turn_flow",     {})
    resp  = call_data.get("response_time", {})

    if not isinstance(talk, dict):
        return 50.0

    agent_pct = float(talk.get("agent_talk_pct", 50.0))

    # Talk balance: penalize if agent dominates (>70%) or is too quiet (<20%)
    if 30 <= agent_pct <= 65:
        talk_score = 100.0
    elif 20 <= agent_pct < 30 or 65 < agent_pct <= 75:
        talk_score = 70.0
    else:
        talk_score = 40.0

    # Turn balance
    turn_balance = float(turns.get("turn_balance_score", 0.5)) * 100

    # Response time
    rating_map = {"Excellent": 100, "Good": 80, "Fair": 60, "Poor": 30}
    rt_rating  = resp.get("rating", "Fair") if isinstance(resp, dict) else "Fair"
    rt_score   = rating_map.get(rt_rating, 60)

    return (talk_score * 0.4 + turn_balance * 0.3 + rt_score * 0.3)


def score_professionalism(call_data: dict) -> float:
    """
    Professionalism score (0-100).
    Agent rudeness level:
      NONE   = 100
      LOW    = 80
      MEDIUM = 50  (coaching recommended per industry standard)
      HIGH   = 10  (auto-fail applies separately)
    """
    rude = call_data.get("rude_behavior", {})
    if not isinstance(rude, dict):
        return 100.0

    level = rude.get("agent_rudeness_level", "NONE").upper()
    mapping = {
        "NONE":   100.0,
        "LOW":     80.0,
        "MEDIUM":  50.0,
        "HIGH":    10.0,
    }
    return mapping.get(level, 100.0)


def calculate_qa_score(call_data: dict) -> dict:
    """
    Calculate full QA score for a call using industry-standard weighted rubric.

    Returns dict with:
      qa_score        — final score 0-100
      rating          — Excellent / Good / Fair / Poor / Failed
      breakdown       — per-component raw scores
      weights         — weights used
      auto_failed     — True if agent rudeness triggered auto-fail
    """
    s_resolution     = score_resolution(call_data)
    s_compliance     = score_compliance(call_data)
    s_sentiment      = score_sentiment(call_data)
    s_communication  = score_communication(call_data)
    s_professionalism = score_professionalism(call_data)

    weighted = (
        s_resolution     * WEIGHTS["resolution"]      +
        s_compliance     * WEIGHTS["compliance"]       +
        s_sentiment      * WEIGHTS["sentiment"]        +
        s_communication  * WEIGHTS["communication"]    +
        s_professionalism * WEIGHTS["professionalism"]
    )

    # Auto-fail check
    rude  = call_data.get("rude_behavior", {})
    level = rude.get("agent_rudeness_level", "NONE").upper() if isinstance(rude, dict) else "NONE"
    auto_failed = (level == "HIGH")

    if auto_failed:
        final_score = min(weighted, AGENT_RUDE_AUTOFAIL_CAP)
    else:
        final_score = weighted

    final_score = round(max(0.0, min(100.0, final_score)), 2)

    if auto_failed or final_score < 30:
        rating = "Failed"
    elif final_score < 50:
        rating = "Poor"
    elif final_score < 70:
        rating = "Fair"
    elif final_score < 85:
        rating = "Good"
    else:
        rating = "Excellent"

    return {
        "qa_score":   final_score,
        "rating":     rating,
        "auto_failed": auto_failed,
        "breakdown": {
            "resolution":      round(s_resolution,      2),
            "compliance":      round(s_compliance,      2),
            "sentiment":       round(s_sentiment,       2),
            "communication":   round(s_communication,   2),
            "professionalism": round(s_professionalism, 2),
        },
        "weights": WEIGHTS,
    }


def score_all_calls(pipeline_path: str = None) -> dict:
    """Score all calls in pipeline_results.json."""
    if pipeline_path is None:
        pipeline_path = os.path.join(OUTPUTS_DIR, "pipeline_results.json")

    with open(pipeline_path, encoding="utf-8") as f:
        data = json.load(f)

    calls = data.get("calls", {})
    if isinstance(calls, list):
        calls = {c["call_id"]: c for c in calls if isinstance(c, dict)}

    results = {}
    for cid, call_data in calls.items():
        results[cid] = calculate_qa_score(call_data)

    return results


def update_human_validation_csv(scores: dict, csv_path: str = None):
    """Update human_qa_score in human_validation_study.csv."""
    if csv_path is None:
        csv_path = GROUND_TRUTH_CSV

    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df["human_qa_score"] = df["call_id"].map(
        {cid: v["qa_score"] for cid, v in scores.items()}
    )
    df.to_csv(csv_path, index=False)
    print(f"✅ Updated human_qa_score in {csv_path}")


def print_report(scores: dict):
    """Print QA score report."""
    print("\n" + "=" * 65)
    print("QA SCORE REPORT — Industry-Standard Weighted Rubric")
    print("=" * 65)
    print(f"{'Call ID':<25} {'Score':>6}  {'Rating':<10}  {'Resolution':>10}  {'Compliance':>10}  {'Sentiment':>9}  {'Comms':>6}  {'Prof':>5}  {'AutoFail':>8}")
    print("-" * 110)

    for cid, r in sorted(scores.items()):
        b = r["breakdown"]
        af = "⚠️  YES" if r["auto_failed"] else "NO"
        print(f"{cid:<25} {r['qa_score']:>6.1f}  {r['rating']:<10}  "
              f"{b['resolution']:>10.1f}  {b['compliance']:>10.1f}  "
              f"{b['sentiment']:>9.1f}  {b['communication']:>6.1f}  "
              f"{b['professionalism']:>5.1f}  {af:>8}")

    avg = sum(r["qa_score"] for r in scores.values()) / len(scores)
    print("-" * 110)
    print(f"\nAverage QA Score: {avg:.1f}/100")
    print(f"\nWeights used:")
    for k, v in WEIGHTS.items():
        print(f"  {k:<16} {v*100:.0f}%")
    print()
    print("Reference: Balto.ai (2025), Globalify (2026), Calabrio (2026)")


def main():
    parser = argparse.ArgumentParser(description="Calculate QA scores for call center calls")
    parser.add_argument("--pipeline",    default=None, help="Path to pipeline_results.json")
    parser.add_argument("--update_csv",  action="store_true", help="Update human_validation_study.csv")
    args = parser.parse_args()

    print("Calculating QA scores...")
    scores = score_all_calls(args.pipeline)
    print_report(scores)

    if args.update_csv:
        update_human_validation_csv(scores)


if __name__ == "__main__":
    main()
