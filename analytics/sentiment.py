"""
analytics/sentiment.py
=======================
Phase 3: Automated Call Analytics — Sentiment Analysis Engine

Implements the VADER-based sentiment module (Section 3.5.2):
  - Per-segment compound score ∈ [-1.0, +1.0]
  - Sentiment trajectory (emotional arc over time)
  - Role-separated analysis: Agent vs. Customer
  - Trend classification: Improving / Declining / Stable

All scores are computed by VADER from actual transcript text —
no hardcoded sentiment values.
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

from config import (
    SENTIMENT_POSITIVE_THRESHOLD,
    SENTIMENT_NEGATIVE_THRESHOLD,
    OUTPUTS_DIR,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# VADER INITIALIZATION
# ─────────────────────────────────────────────────────────────

def _get_vader_analyzer():
    """Lazily initialize and return VADER SentimentIntensityAnalyzer."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
        except ImportError:
            raise ImportError(
                "vaderSentiment not installed. Run: pip install vaderSentiment"
            )
    return SentimentIntensityAnalyzer()


_VADER = None

def get_vader():
    global _VADER
    if _VADER is None:
        _VADER = _get_vader_analyzer()
    return _VADER


# ─────────────────────────────────────────────────────────────
# PER-SEGMENT SENTIMENT SCORING
# ─────────────────────────────────────────────────────────────

def score_sentiment(text: str) -> dict:
    """
    Compute VADER sentiment scores for a text segment.

    Compound score:
        ≥  0.05 → Positive
        ≤ -0.05 → Negative
        else    → Neutral

    Returns
    -------
    dict: compound, positive, negative, neutral, polarity_label
    """
    analyzer = get_vader()
    scores   = analyzer.polarity_scores(text)

    compound = scores["compound"]

    if compound >= SENTIMENT_POSITIVE_THRESHOLD:
        polarity = "Positive"
    elif compound <= SENTIMENT_NEGATIVE_THRESHOLD:
        polarity = "Negative"
    else:
        polarity = "Neutral"

    return {
        "compound":       round(float(compound),         4),
        "positive":       round(float(scores["pos"]),    4),
        "negative":       round(float(scores["neg"]),    4),
        "neutral":        round(float(scores["neu"]),    4),
        "polarity_label": polarity,
    }


# ─────────────────────────────────────────────────────────────
# FULL TRANSCRIPT SENTIMENT ANALYSIS
# ─────────────────────────────────────────────────────────────

def analyze_sentiment(classified: list[dict]) -> dict:
    """
    Run sentiment analysis on every segment and produce:
      - Per-segment compound scores
      - Role-separated averages (Agent vs. Customer)
      - Sentiment trajectory with trend classification
      - Emotional mismatch flag

    Parameters
    ----------
    classified : [{predicted_role, text, start, end, ...}, ...]

    Returns
    -------
    dict:
        segments_with_sentiment    : classified list + sentiment fields
        agent_avg_compound         : float
        customer_avg_compound      : float
        overall_avg_compound       : float
        agent_polarity_label       : str
        customer_polarity_label    : str
        sentiment_trend            : "Improving" | "Declining" | "Stable"
        emotional_mismatch         : bool
        trajectory_timestamps      : [float, ...]
        trajectory_customer_scores : [float, ...]
        trajectory_agent_scores    : [float, ...]
        interpretation             : str
    """
    segments_out = []
    role_scores  = defaultdict(list)
    timestamps   = []
    traj_cust    = []
    traj_agent   = []

    for seg in classified:
        text = seg.get("text", "")
        role = seg.get("predicted_role", "Unknown")
        ts   = seg.get("start", 0)

        sent = score_sentiment(text)
        merged = {**seg, **{
            "sentiment_compound": sent["compound"],
            "sentiment_positive": sent["positive"],
            "sentiment_negative": sent["negative"],
            "sentiment_neutral":  sent["neutral"],
            "polarity_label":     sent["polarity_label"],
        }}
        segments_out.append(merged)
        role_scores[role].append(sent["compound"])

        # Trajectory arrays (chronological)
        timestamps.append(float(ts))
        if role == "Customer":
            traj_cust.append((float(ts), sent["compound"]))
        elif role == "Agent":
            traj_agent.append((float(ts), sent["compound"]))

    # Role averages
    agent_scores    = role_scores.get("Agent",    [])
    customer_scores = role_scores.get("Customer", [])
    all_scores      = agent_scores + customer_scores

    agent_avg    = float(np.mean(agent_scores))    if agent_scores    else 0.0
    customer_avg = float(np.mean(customer_scores)) if customer_scores else 0.0
    overall_avg  = float(np.mean(all_scores))      if all_scores      else 0.0

    # Polarity labels
    def _polarity(score):
        if score >= SENTIMENT_POSITIVE_THRESHOLD:
            return "Positive"
        if score <= SENTIMENT_NEGATIVE_THRESHOLD:
            return "Negative"
        return "Neutral"

    agent_polarity    = _polarity(agent_avg)
    customer_polarity = _polarity(customer_avg)

    # Sentiment trend: compare first-half vs. second-half customer scores
    trend = "Stable"
    if len(customer_scores) >= 4:
        mid   = len(customer_scores) // 2
        first_half  = np.mean(customer_scores[:mid])
        second_half = np.mean(customer_scores[mid:])
        delta = second_half - first_half
        if delta >= 0.05:
            trend = "Improving"
        elif delta <= -0.05:
            trend = "Declining"

    # Emotional mismatch: agent much more positive than customer
    emotional_mismatch = (agent_avg - customer_avg) > 0.2

    # Interpretation
    if trend == "Improving" and customer_avg >= 0.0:
        interpretation = "Successful Service Recovery"
    elif trend == "Declining" or customer_avg <= SENTIMENT_NEGATIVE_THRESHOLD:
        interpretation = "Churn Risk — Unresolved Issue"
    elif emotional_mismatch:
        interpretation = "Emotional Mismatch — Agent Too Cheerful for Context"
    else:
        interpretation = "Routine Interaction"

    logger.info(
        f"Sentiment | Agent={agent_avg:+.3f} ({agent_polarity}) | "
        f"Customer={customer_avg:+.3f} ({customer_polarity}) | "
        f"Trend={trend} | Mismatch={emotional_mismatch}"
    )

    # Build trajectory lists
    traj_cust.sort(key=lambda x: x[0])
    traj_agent.sort(key=lambda x: x[0])

    return {
        "segments_with_sentiment":     segments_out,
        "agent_avg_compound":          round(agent_avg,    4),
        "customer_avg_compound":       round(customer_avg, 4),
        "overall_avg_compound":        round(overall_avg,  4),
        "agent_polarity_label":        agent_polarity,
        "customer_polarity_label":     customer_polarity,
        "sentiment_trend":             trend,
        "emotional_mismatch":          emotional_mismatch,
        "trajectory_customer":         traj_cust,
        "trajectory_agent":            traj_agent,
        "interpretation":              interpretation,
    }


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_sentiment_trajectory(sentiment_result: dict,
                               call_id: str = "",
                               save_path: str = None) -> str:
    """
    Plot sentiment trajectory over time for Agent and Customer.
    Matches the Sentiment Flow visualization in the thesis dashboard.
    """
    traj_cust  = sentiment_result.get("trajectory_customer", [])
    traj_agent = sentiment_result.get("trajectory_agent",    [])

    fig, ax = plt.subplots(figsize=(12, 5))

    if traj_cust:
        ts_c, sc_c = zip(*traj_cust)
        ax.plot(ts_c, sc_c, color="#d62728", marker="o", markersize=4,
                linewidth=2, label="Customer Sentiment")

    if traj_agent:
        ts_a, sc_a = zip(*traj_agent)
        ax.plot(ts_a, sc_a, color="#1f77b4", marker="s", markersize=4,
                linewidth=2, linestyle="--", label="Agent Sentiment")

    # Horizontal bands
    ax.axhspan( 0.05,  1.0,  alpha=0.05, color="green",  label="Positive Zone")
    ax.axhspan(-1.0,  -0.05, alpha=0.05, color="red",    label="Negative Zone")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("VADER Compound Score")
    ax.set_ylim([-1.1, 1.1])
    title = f"Sentiment Trajectory — {call_id}" if call_id else "Sentiment Trajectory"
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path is None:
        safe_id  = call_id.replace(" ", "_").replace("/", "_")
        save_path = os.path.join(OUTPUTS_DIR, f"sentiment_trajectory_{safe_id}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Sentiment trajectory saved → {save_path}")
    return save_path


def plot_sentiment_summary(calls_sentiment: list[dict],
                            save_path: str = None) -> str:
    """
    Multi-call sentiment summary bar chart.
    calls_sentiment: [{"call_id": str, "sentiment": dict}, ...]
    """
    call_ids   = [d["call_id"] for d in calls_sentiment]
    cust_avgs  = [d["sentiment"].get("customer_avg_compound", 0) for d in calls_sentiment]
    agent_avgs = [d["sentiment"].get("agent_avg_compound",    0) for d in calls_sentiment]

    x = np.arange(len(call_ids))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(call_ids) * 1.4), 5))
    ax.bar(x - w/2, agent_avgs,  width=w, label="Agent",    color="#1f77b4", alpha=0.8)
    ax.bar(x + w/2, cust_avgs,   width=w, label="Customer", color="#d62728", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(call_ids, rotation=30, ha="right")
    ax.set_ylabel("Avg VADER Compound Score")
    ax.set_title("Sentiment Analysis Results (Agent vs. Customer)")
    ax.set_ylim([-1.1, 1.1])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "sentiment_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Sentiment summary saved → {save_path}")
    return save_path
