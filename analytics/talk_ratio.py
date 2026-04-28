"""
analytics/talk_ratio.py
========================
Phase 3: Automated Call Analytics — Talk Time & Silence Analysis

Implements quantitative metrics from thesis Section 3.5.1:

    R_agent = (Σ d_agent / D_total) × 100

    Silence ratio = (D_total - D_speech) / D_total × 100

    Q_score (composite) = weighted sum of 5 sub-metrics

All values are calculated from actual segment timestamps — no hardcoded numbers.
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

from config import QA_SCORE_WEIGHTS, OUTPUTS_DIR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# TALK-TIME RATIO
# ─────────────────────────────────────────────────────────────

def compute_talk_time_ratio(classified: list[dict],
                             total_duration_sec: float = None) -> dict:
    """
    Compute talk time ratio (TTR) for each role.

    R_role = (Σ d_role / D_total) × 100

    Parameters
    ----------
    classified        : diarized + classified transcript
    total_duration_sec: total call duration (computed from segments if None)

    Returns
    -------
    dict: agent_talk_pct, customer_talk_pct, silence_pct,
          agent_total_sec, customer_total_sec, silence_sec,
          total_duration_sec, interaction_classification
    """
    role_durations = defaultdict(float)

    for seg in classified:
        role     = seg.get("predicted_role", "Unknown")
        duration = float(seg.get("duration", seg.get("end", 0) - seg.get("start", 0)))
        if duration > 0:
            role_durations[role] += duration

    speech_total = sum(role_durations.values())

    if total_duration_sec is None:
        if classified:
            total_duration_sec = max(seg.get("end", 0) for seg in classified)
        else:
            total_duration_sec = speech_total

    if total_duration_sec < 1e-6:
        logger.warning("Zero-duration call detected")
        return {}

    agent_sec    = role_durations.get("Agent",    0.0)
    customer_sec = role_durations.get("Customer", 0.0)
    silence_sec  = max(0.0, total_duration_sec - speech_total)

    agent_pct    = round((agent_sec    / total_duration_sec) * 100, 2)
    customer_pct = round((customer_sec / total_duration_sec) * 100, 2)
    silence_pct  = round((silence_sec  / total_duration_sec) * 100, 2)

    # Classify interaction type (Table 4.7 in thesis)
    if agent_pct >= 85:
        classification = "Agent Monologue (Scripted)"
    elif customer_pct >= 80:
        classification = "Customer Venting"
    elif abs(agent_pct - customer_pct) <= 25:
        classification = "Balanced (Interactive)"
    else:
        classification = "Asymmetric Interaction"

    result = {
        "agent_talk_pct":        agent_pct,
        "customer_talk_pct":     customer_pct,
        "silence_pct":           silence_pct,
        "agent_total_sec":       round(agent_sec,    3),
        "customer_total_sec":    round(customer_sec, 3),
        "silence_sec":           round(silence_sec,  3),
        "total_duration_sec":    round(total_duration_sec, 3),
        "interaction_classification": classification,
    }
    logger.info(f"Talk ratio: Agent={agent_pct}% | Customer={customer_pct}% | "
                f"Silence={silence_pct}% | Type={classification}")
    return result


# ─────────────────────────────────────────────────────────────
# TURN-TAKING FLOW
# ─────────────────────────────────────────────────────────────

def compute_turn_taking(classified: list[dict]) -> dict:
    """
    Compute turn-taking flow metrics.

    Returns
    -------
    dict: total_turns, agent_turns, customer_turns,
          avg_agent_turn_sec, avg_customer_turn_sec,
          turn_balance_score (0–1; 1 = perfectly balanced)
    """
    agent_turns    = []
    customer_turns = []

    prev_role = None
    turn_start = None

    for seg in classified:
        role = seg.get("predicted_role", "Unknown")
        start = seg.get("start", 0)
        end   = seg.get("end",   0)

        if role != prev_role and prev_role is not None:
            # Turn ended
            dur = end - (turn_start or start)
            if prev_role == "Agent":
                agent_turns.append(max(0, dur))
            elif prev_role == "Customer":
                customer_turns.append(max(0, dur))
            turn_start = start
        elif prev_role is None:
            turn_start = start

        prev_role = role

    # Close final turn
    if prev_role and classified:
        last_seg = classified[-1]
        dur = last_seg.get("end", 0) - (turn_start or 0)
        if prev_role == "Agent":
            agent_turns.append(max(0, dur))
        elif prev_role == "Customer":
            customer_turns.append(max(0, dur))

    total_turns    = len(agent_turns) + len(customer_turns)
    avg_agent      = np.mean(agent_turns)    if agent_turns    else 0.0
    avg_customer   = np.mean(customer_turns) if customer_turns else 0.0

    # Turn balance: how equal the turn counts are (1 = perfectly equal)
    if total_turns > 0:
        balance = 1.0 - abs(len(agent_turns) - len(customer_turns)) / total_turns
    else:
        balance = 0.0

    return {
        "total_turns":           total_turns,
        "agent_turns":           len(agent_turns),
        "customer_turns":        len(customer_turns),
        "avg_agent_turn_sec":    round(float(avg_agent),    3),
        "avg_customer_turn_sec": round(float(avg_customer), 3),
        "turn_balance_score":    round(float(balance), 4),
    }


# ─────────────────────────────────────────────────────────────
# COMPOSITE QA SCORE
# ─────────────────────────────────────────────────────────────

def compute_qa_score(talk_ratio: dict,
                      turn_taking: dict,
                      sentiment_result: dict,
                      compliance_result: dict) -> dict:
    """
    Compute composite Automated Quality Score (Q_score ∈ [0, 100]).

    Formula (Section 3.5.1):
        Q_score = Σ (weight_i × sub_score_i) × 100

    Sub-scores (each ∈ [0, 1]):
        1. talk_balance   = how close agent talk% is to ideal ~45-55%
        2. turn_taking    = turn_balance_score from turn analysis
        3. sentiment      = normalized average customer sentiment
        4. compliance     = compliance_score from compliance module
        5. politeness     = proxy from agent sentiment positivity

    Parameters
    ----------
    talk_ratio        : output of compute_talk_time_ratio()
    turn_taking       : output of compute_turn_taking()
    sentiment_result  : output of analytics.sentiment.analyze_sentiment()
    compliance_result : output of analytics.compliance.check_compliance()

    Returns
    -------
    dict: qa_score (0-100), sub_scores, weights
    """
    weights = QA_SCORE_WEIGHTS

    # Sub-score 1: Talk balance (ideal agent:customer = 50:50, tolerance ±20%)
    agent_pct = talk_ratio.get("agent_talk_pct", 50.0)
    ideal_balance = 50.0
    deviation = abs(agent_pct - ideal_balance)
    talk_balance_score = max(0.0, 1.0 - (deviation / 50.0))

    # Sub-score 2: Turn-taking flow
    turn_balance_score = turn_taking.get("turn_balance_score", 0.5)

    # Sub-score 3: Customer sentiment (map [-1, 1] → [0, 1])
    cust_avg = sentiment_result.get("customer_avg_compound", 0.0)
    sentiment_score = (float(cust_avg) + 1.0) / 2.0

    # Sub-score 4: Compliance adherence
    compliance_score = compliance_result.get("compliance_score", 0.0)

    # Sub-score 5: Agent politeness (from agent avg sentiment positivity)
    agent_avg = sentiment_result.get("agent_avg_compound", 0.0)
    politeness_score = (float(agent_avg) + 1.0) / 2.0

    sub_scores = {
        "talk_balance":  round(talk_balance_score,  4),
        "turn_taking":   round(turn_balance_score,  4),
        "sentiment":     round(sentiment_score,     4),
        "compliance":    round(compliance_score,    4),
        "politeness":    round(politeness_score,    4),
    }

    # Weighted sum
    qa_score = 0.0
    for key, weight in weights.items():
        qa_score += weight * sub_scores.get(key, 0.0)
    qa_score = round(qa_score * 100, 2)

    # Rating classification
    if qa_score >= 75:
        rating = "Good"
    elif qa_score >= 55:
        rating = "Fair"
    else:
        rating = "Needs Improvement"

    logger.info(f"QA Score: {qa_score}/100 ({rating})")
    return {
        "qa_score":   qa_score,
        "rating":     rating,
        "sub_scores": sub_scores,
        "weights":    weights,
    }


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_talk_time_distribution(calls_data: list[dict],
                                 save_path: str = None) -> str:
    """
    Stacked bar chart: Agent / Customer / Silence per call.
    calls_data: [{"call_id": str, "talk_ratio": dict}, ...]
    """
    call_ids     = [d["call_id"] for d in calls_data]
    agent_pcts   = [d["talk_ratio"].get("agent_talk_pct",    0) for d in calls_data]
    cust_pcts    = [d["talk_ratio"].get("customer_talk_pct", 0) for d in calls_data]
    silence_pcts = [d["talk_ratio"].get("silence_pct",       0) for d in calls_data]

    x   = np.arange(len(call_ids))
    w   = 0.55
    fig, ax = plt.subplots(figsize=(max(8, len(call_ids) * 1.4), 5))

    p1 = ax.bar(x, agent_pcts,   width=w, label="Agent",    color="#1f77b4", alpha=0.85)
    p2 = ax.bar(x, cust_pcts,    width=w, label="Customer", color="#d62728", alpha=0.85,
                bottom=agent_pcts)
    p3 = ax.bar(x, silence_pcts, width=w, label="Silence",  color="#7f7f7f", alpha=0.5,
                bottom=[a + c for a, c in zip(agent_pcts, cust_pcts)])

    ax.set_xticks(x)
    ax.set_xticklabels(call_ids, rotation=30, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim([0, 115])
    ax.set_title("Talk Time Distribution per Call")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "talk_time_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Talk time distribution plot saved → {save_path}")
    return save_path
