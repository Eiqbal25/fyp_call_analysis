"""
methods/method1_lexical.py
==========================
Method 1: Lexical-Based Speaker Role Classification (Baseline)

Implements the Keyword Density Algorithm from the thesis (Section 3.4.1):

    D = (K / N) × 100

Where:
    D = Lexical Density score
    K = count of matched domain-specific keywords in the segment
    N = total word count of the segment

The speaker with the highest Agent Lexical Density relative to Customer
Lexical Density is labeled "Agent"; the other is labeled "Customer".

Statistics computed:
    - Per-segment keyword densities
    - Confidence scores (normalized softmax-style)
    - Overall accuracy, precision, recall, F1 against ground truth
"""

import os
import re
import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

from config import (
    AGENT_KEYWORDS_FILE,
    CUSTOMER_KEYWORDS_FILE,
    LEXICAL_MIN_WORDS,
    OUTPUTS_DIR,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# KEYWORD LOADING
# ─────────────────────────────────────────────────────────────

def _load_keywords(filepath: str) -> list[str]:
    """Load keyword list from JSON file (all categories merged)."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Keyword file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Flatten all categories into a single list
    all_kws = []
    for category_kws in data.values():
        all_kws.extend([kw.lower().strip() for kw in category_kws])
    return list(set(all_kws))


def load_all_keywords() -> tuple[list[str], list[str]]:
    """
    Returns (agent_keywords, customer_keywords).
    Lazy-loaded on first use — safe to import without keyword files present.
    """
    agent_kws    = _load_keywords(AGENT_KEYWORDS_FILE)
    customer_kws = _load_keywords(CUSTOMER_KEYWORDS_FILE)
    logger.info(f"Loaded {len(agent_kws)} agent keywords, "
                f"{len(customer_kws)} customer keywords")
    return agent_kws, customer_kws


# Lazy globals — populated on first call to _ensure_keywords_loaded()
AGENT_KEYWORDS:    list[str] = []
CUSTOMER_KEYWORDS: list[str] = []


def _ensure_keywords_loaded():
    """Load keyword lists into module globals if not yet initialised."""
    global AGENT_KEYWORDS, CUSTOMER_KEYWORDS
    if not AGENT_KEYWORDS:
        AGENT_KEYWORDS, CUSTOMER_KEYWORDS = load_all_keywords()


# ─────────────────────────────────────────────────────────────
# LEXICAL DENSITY CALCULATION
# ─────────────────────────────────────────────────────────────

def _count_keyword_matches(text: str, keywords: list[str]) -> tuple[int, list[str]]:
    """
    Count how many keywords from the list appear in text.
    Multi-word phrases are matched with substring search.

    Returns (count, matched_keywords)
    """
    text_lower = text.lower()
    matched = []
    for kw in keywords:
        # Use regex word boundary for single words, substring for phrases
        if " " in kw:
            if kw in text_lower:
                matched.append(kw)
        else:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text_lower):
                matched.append(kw)
    return len(matched), matched


def compute_lexical_density(text: str,
                             keywords: list[str]) -> tuple[float, int, int]:
    """
    Compute Lexical Density:  D = (K / N) × 100

    Parameters
    ----------
    text     : transcript text segment
    keywords : list of domain-specific keywords

    Returns
    -------
    density   : float — lexical density score (0-100)
    k_matches : int   — number of matched keywords
    n_words   : int   — total word count
    """
    words   = text.lower().split()
    n_words = len(words)

    if n_words == 0:
        return 0.0, 0, 0

    k_matches, _ = _count_keyword_matches(text, keywords)
    density = (k_matches / n_words) * 100.0
    return round(density, 4), k_matches, n_words


# ─────────────────────────────────────────────────────────────
# CLASSIFICATION PER SEGMENT
# ─────────────────────────────────────────────────────────────

def classify_segment_lexical(text: str) -> dict:
    """
    Classify a single transcript segment as 'Agent' or 'Customer'
    using keyword density comparison.

    Returns
    -------
    dict with keys:
        predicted_role, confidence, agent_density, customer_density,
        agent_matches, customer_matches, n_words, method
    """
    _ensure_keywords_loaded()
    words = text.lower().split()
    n_words = len(words)

    # Insufficient text — low-confidence fallback
    if n_words < LEXICAL_MIN_WORDS:
        return {
            "predicted_role":    "Unknown",
            "confidence":        0.5,
            "agent_density":     0.0,
            "customer_density":  0.0,
            "agent_matches":     0,
            "customer_matches":  0,
            "n_words":           n_words,
            "method":            "lexical",
        }

    agent_density,    agent_k,    _ = compute_lexical_density(text, AGENT_KEYWORDS)
    customer_density, customer_k, _ = compute_lexical_density(text, CUSTOMER_KEYWORDS)

    # Softmax-style confidence: how much one class dominates
    total = agent_density + customer_density
    if total < 1e-9:
        # No keywords matched — use word-count heuristic
        # Agents tend to have more words per turn (scripted)
        predicted_role = "Agent" if n_words > 15 else "Customer"
        confidence = 0.55
    else:
        agent_prob    = agent_density    / total
        customer_prob = customer_density / total

        if agent_density > customer_density:
            predicted_role = "Agent"
            confidence = round(float(agent_prob), 4)
        elif customer_density > agent_density:
            predicted_role = "Customer"
            confidence = round(float(customer_prob), 4)
        else:
            # Tie — default Agent (agents more scripted)
            predicted_role = "Agent"
            confidence = 0.50

    return {
        "predicted_role":   predicted_role,
        "confidence":       confidence,
        "agent_density":    agent_density,
        "customer_density": customer_density,
        "agent_matches":    agent_k,
        "customer_matches": customer_k,
        "n_words":          n_words,
        "method":           "lexical",
    }


# ─────────────────────────────────────────────────────────────
# CLASSIFY FULL DIARIZED TRANSCRIPT
# ─────────────────────────────────────────────────────────────

def classify_transcript_lexical(diarized: list[dict]) -> list[dict]:
    """
    Apply lexical classification to every segment in a diarized transcript.

    Strategy:
      1. Classify each segment independently.
      2. Per speaker_id, aggregate all segment predictions
         (majority vote weighted by confidence).
      3. Assign final stable role per speaker_id.

    Parameters
    ----------
    diarized : [{speaker_id, text, start, end, ...}, ...]

    Returns
    -------
    classified : same list + added keys:
        predicted_role, confidence, agent_density, customer_density,
        agent_matches, customer_matches
    """
    # Step 1: segment-level classification
    classified = []
    for seg in diarized:
        result = classify_segment_lexical(seg.get("text", ""))
        classified.append({**seg, **result})

    # Step 2: speaker-level aggregation (majority vote)
    speaker_votes = defaultdict(lambda: {"Agent": 0.0, "Customer": 0.0})
    for seg in classified:
        spk = seg["speaker_id"]
        role = seg["predicted_role"]
        conf = seg["confidence"]
        if role in speaker_votes[spk]:
            speaker_votes[spk][role] += conf

    # Step 3: assign stable per-speaker role
    speaker_roles = {}
    for spk, votes in speaker_votes.items():
        if votes["Agent"] >= votes["Customer"]:
            speaker_roles[spk] = ("Agent",
                                  round(votes["Agent"] /
                                        max(votes["Agent"] + votes["Customer"], 1e-9), 4))
        else:
            speaker_roles[spk] = ("Customer",
                                  round(votes["Customer"] /
                                        max(votes["Agent"] + votes["Customer"], 1e-9), 4))

    # Step 4: apply stable roles back to segments
    for seg in classified:
        spk = seg["speaker_id"]
        seg["predicted_role"] = speaker_roles[spk][0]
        seg["final_confidence"] = speaker_roles[spk][1]

    logger.info(f"Method 1 classification complete | "
                f"speaker_roles={dict((k, v[0]) for k, v in speaker_roles.items())}")
    return classified


# ─────────────────────────────────────────────────────────────
# KEYWORD FREQUENCY ANALYSIS (Figure 4.3)
# ─────────────────────────────────────────────────────────────

def analyze_keyword_frequency(diarized_classified: list[dict],
                               top_n: int = 15) -> dict:
    """
    Compute top-N keyword frequencies per role for visualization.

    Returns
    -------
    dict: {agent: {kw: count}, customer: {kw: count}}
    """
    freq = {"Agent": defaultdict(int), "Customer": defaultdict(int)}

    for seg in diarized_classified:
        role = seg.get("predicted_role", "Unknown")
        if role not in freq:
            continue
        text = seg.get("text", "")

        # Count agent keyword matches
        _, agent_matched = _count_keyword_matches(text, AGENT_KEYWORDS)
        for kw in agent_matched:
            if role == "Agent":
                freq["Agent"][kw] += 1

        # Count customer keyword matches
        _, cust_matched = _count_keyword_matches(text, CUSTOMER_KEYWORDS)
        for kw in cust_matched:
            if role == "Customer":
                freq["Customer"][kw] += 1

    # Get top-N for each
    result = {}
    for role, counts in freq.items():
        sorted_kws = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        result[role] = dict(sorted_kws)

    return result


def plot_keyword_density(keyword_freq: dict,
                          save_path: str = None) -> str:
    """
    Plot comparative keyword density analysis (Figure 4.3 in thesis).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, role, color in zip(axes,
                                 ["Agent", "Customer"],
                                 ["steelblue", "tomato"]):
        data = keyword_freq.get(role, {})
        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
            continue
        kws    = list(data.keys())
        counts = list(data.values())
        ax.barh(kws, counts, color=color, alpha=0.8)
        ax.set_title(f"Top {role} Keywords (Method 1: Keyword-Based)")
        ax.set_xlabel("Frequency")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)

    plt.suptitle("Comparative Keyword Density Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "keyword_density_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Keyword density plot saved → {save_path}")
    return save_path


def plot_confidence_distribution(classified: list[dict],
                                  save_path: str = None) -> str:
    """
    Plot confidence score distribution (Figure 4.2 in thesis).
    """
    confidences = [seg.get("confidence", 0) for seg in classified]
    mean_conf   = np.mean(confidences) if confidences else 0.0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(confidences, bins=10, range=(0, 1), color="steelblue",
            edgecolor="white", alpha=0.8, label="Segment Confidence")
    ax.axvline(mean_conf, color="red", linestyle="--",
               linewidth=2, label=f"Mean: {mean_conf*100:.2f}%")
    ax.set_title("Confidence Score Distribution - Method 1 (Keyword-Based)")
    ax.set_xlabel("Classification Confidence")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "m1_confidence_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confidence distribution plot saved → {save_path}")
    return save_path
