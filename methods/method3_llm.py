"""
methods/method3_llm.py
Adaptive LLM Classification via Groq + Llama 3.3 70B (FREE)

STRATEGY:
  Good diarization (balance >= 40%): Two-pass
  Broken diarization (balance < 40%): Per-segment with call-type detection

  Per-segment now has TWO modes:
  - Agent-first calls: "First speaker = Agent" (standard inbound)
  - Customer-first calls: "Identify by role/behavior" (emergency, outbound, angry caller)
"""

import json, logging, os, re
from collections import Counter

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq package not installed. Run: pip install groq")

try:
    from config import GROQ_API_KEY, LLM_MODEL
except ImportError:
    GROQ_API_KEY = None
    LLM_MODEL = "llama-3.3-70b-versatile"

DIARIZATION_BALANCE_THRESHOLD = 0.40

PASS1_SYSTEM = """You are analyzing a customer service call. Identify which speaker is the Agent.
The Agent works for the company and ANSWERS the phone. The Customer CALLS IN for help.
The FIRST speaker in the call is almost always the Agent.
Even if the Agent is rude, dismissive, or unprofessional -- they are still the Agent.
Answer with ONLY JSON: {"agent_spk": <number>}"""

SEGMENT_SYSTEM_AGENT_FIRST = """You are an expert at analyzing customer service call transcripts.
Classify each numbered segment as "Agent" or "Customer".

AGENT: Works for the company. Greets callers, verifies identity, references company systems,
offers solutions. CAN BE RUDE -- "Happens all the time", "Obviously you don't", "You have to pay"
are still Agent speech.

CUSTOMER: Calls in for help. Explains their problem, gives personal details, complains,
makes requests. In Malay: "saya nak tanya", "barang saya", "nombor tracking", "saya pelanggan".

ISLAMIC GREETINGS: In outbound Malay calls, the Agent says "Assalamualaikum" first,
and the Customer responds "Waalaikumsalam". Agent introduces company name after greeting.

OUTBOUND SALES: If agent calls customer first, the customer may say "Ada apa ya?",
"Saya ingatkan...", "Bia betul", "Kena berbincang dengan isteri" — these are CUSTOMER responses.

The FIRST segment is always from the Agent (they pick up or make the call).
Respond ONLY with JSON array: [{"id": 1, "role": "Agent"}, {"id": 2, "role": "Customer"}, ...]"""

SEGMENT_SYSTEM_CUSTOMER_FIRST = """You are an expert at analyzing customer service call transcripts.
Classify each numbered segment as "Agent" or "Customer".

IMPORTANT: In this call, the CUSTOMER may speak first (emergency, outbound, angry caller).
Identify roles by BEHAVIOR and CONTENT, not by position:

AGENT characteristics:
- Represents a company (uses company name, "our system", "our policy")
- Calms the caller, verifies identity, offers professional solutions
- Uses phrases like: "my name is X", "how can I help", "let me check", "I can see your account"
- May be outbound caller (sales/retention)

CUSTOMER characteristics:
- Calls for help OR is being called about their account
- Explains a personal problem or situation
- Gives personal info (name, IC, account number, address)
- Expresses frustration, urgency, or gratitude as the person with the problem

Respond ONLY with JSON array: [{"id": 1, "role": "Agent"}, {"id": 2, "role": "Customer"}, ...]"""

def _check_balance(segments):
    spk_counts = Counter(seg.get("speaker_id", 0) for seg in segments)
    if len(spk_counts) < 2:
        return 0.0
    total = sum(spk_counts.values())
    minority = min(spk_counts.values())
    return minority / total


def _detect_call_type(segments, client=None):
    """
    Detect whether Agent or Customer speaks first using local text patterns.
    No API call needed -- saves Groq token quota.
    Returns "Agent" or "Customer".
    """
    if not segments:
        return "Agent"

    # Check first 3 segments for agent-opening patterns
    agent_patterns = [
        r"thank you for calling",
        r"thanks for calling",
        r"my name is",
        r"nama saya",
        r"how can i (help|assist)",
        r"boleh saya bantu",
        r"selamat (pagi|petang|malam|sejahtera)",
        r"terima kasih kerana",
        r"bantuan .*(di sini|here)",
        r"good (morning|afternoon|evening|day)",
        r"hi.*(good|welcome|thank)",
        r"am i speaking (to|with)",
    ]

    customer_patterns = [
        r"tolong saya",
        r"help me",
        r"oh my god",
        r"i think my",
        r"kenapa .*(line|sistem|account)",
        r"you orang",
        r"hello\?",
        r"is anyone",
        r"i (need|want|have a problem)",
        r"my (internet|account|order|package)",
    ]

    text_sample = " ".join(
        seg.get("text", "").lower() for seg in segments[:3]
    )

    agent_score = sum(1 for p in agent_patterns if re.search(p, text_sample))
    customer_score = sum(1 for p in customer_patterns if re.search(p, text_sample))

    if customer_score > agent_score:
        logger.info("Method 3 Call-Type: Customer-first detected (local pattern matching)")
        return "Customer"
    else:
        logger.info("Method 3 Call-Type: Agent-first detected (local pattern matching)")
        return "Agent"


def _two_pass(segments, client, first_spk, all_spks):
    sample_size = min(10, len(segments))
    lines = []
    for i, seg in enumerate(segments[:sample_size]):
        text = seg.get("text", "").strip()
        if text:
            lines.append(
                f"{i+1}. [SPK{seg.get('speaker_id',0)}] [{seg.get('start',0):.1f}s] {text}"
            )

    spk_options = " or ".join(str(s) for s in all_spks)
    prompt = (
        f"Customer service call. Speakers: SPK{all_spks[0]} and SPK{all_spks[1]}.\n\n"
        + "\n".join(lines)
        + f"\n\nWhich speaker_id is the AGENT? First speaker = almost always Agent. "
        f"Answer ONLY: {{\"agent_spk\": {spk_options}}}"
    )

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": PASS1_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0, max_tokens=30,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw).strip()
        agent_spk = int(json.loads(raw).get("agent_spk", first_spk))
        if agent_spk not in all_spks:
            agent_spk = first_spk
        logger.info("Method 3 Two-Pass: Agent=SPK%d", agent_spk)
    except Exception as e:
        logger.error("Method 3 Two-Pass failed: %s -- using SPK%d", e, first_spk)
        agent_spk = first_spk

    customer_spk = [s for s in all_spks if s != agent_spk][0]
    for seg in segments:
        spk = seg.get("speaker_id", 0)
        seg["predicted_role"] = "Agent" if spk == agent_spk else "Customer"
        seg["confidence"] = 0.93
        seg["final_confidence"] = 0.93
        seg["method"] = "llm_twopass"
        seg["llm_classified"] = True

    logger.info("Method 3 Two-Pass: SPK%d=Agent, SPK%d=Customer -> %d segs",
                agent_spk, customer_spk, len(segments))
    return segments


def _per_segment(segments, client, first_speaker="Agent"):
    """
    Per-segment classification with call-type awareness.
    first_speaker: "Agent" for standard inbound, "Customer" for emergency/outbound.
    """
    lines = []
    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"{i+1}. [{seg.get('start',0):.1f}s] {text}")

    # Choose system prompt based on call type
    system_prompt = (
        SEGMENT_SYSTEM_AGENT_FIRST if first_speaker == "Agent"
        else SEGMENT_SYSTEM_CUSTOMER_FIRST
    )

    # Build user prompt based on call type
    if first_speaker == "Agent":
        user_prompt = (
            "Classify this customer service call. "
            "The first segment is from the Agent (they pick up the phone).\n\n"
            + "\n".join(lines)
        )
    else:
        user_prompt = (
            "Classify this customer service call. "
            "The CUSTOMER may speak first in this call -- identify roles by behavior.\n\n"
            + "\n".join(lines)
        )

    logger.info("Method 3 Per-Segment: call_type=%s-first, %d segments",
                first_speaker, len(segments))

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0, max_tokens=2000,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw).strip()
        labels = json.loads(raw)
        role_map = {
            item["id"]: item["role"] for item in labels
            if item.get("role") in ("Agent", "Customer")
        }
        logger.info("Method 3 Per-Segment: parsed %d/%d", len(role_map), len(segments))
    except Exception as e:
        logger.error("Method 3 Per-Segment failed: %s", e)
        return segments

    applied = 0
    for i, seg in enumerate(segments):
        if i + 1 in role_map:
            seg["predicted_role"] = role_map[i + 1]
            seg["confidence"] = 0.90
            seg["final_confidence"] = 0.90
            seg["method"] = "llm_perseg"
            seg["llm_classified"] = True
            applied += 1

    logger.info("Method 3 Per-Segment: %d/%d labelled", applied, len(segments))

    # Short fragment smoothing (both neighbors agree)
    for i in range(1, len(segments) - 1):
        seg = segments[i]
        if not seg.get("llm_classified"):
            continue
        word_count = len(seg.get("text", "").split())
        if word_count <= 4:
            prev_label = segments[i-1].get("predicted_role")
            next_label = segments[i+1].get("predicted_role")
            curr_label = seg.get("predicted_role")
            if prev_label and next_label and prev_label == next_label and curr_label != prev_label:
                logger.info("  Fragment fix [%.1fs]: %s->%s | \"%s\"",
                            seg.get("start", 0), curr_label, prev_label,
                            seg.get("text", "")[:40])
                seg["predicted_role"] = prev_label

    # Inversion check ONLY for agent-first calls
    if first_speaker == "Agent":
        if segments and segments[0].get("llm_classified"):
            if segments[0].get("predicted_role") == "Customer":
                logger.warning("Method 3: Inversion detected -- flipping all labels")
                flip = {"Agent": "Customer", "Customer": "Agent"}
                for seg in segments:
                    if seg.get("llm_classified"):
                        seg["predicted_role"] = flip.get(
                            seg["predicted_role"], seg["predicted_role"])

    return segments


def classify_transcript_llm(segments):
    """Adaptive: two-pass for good diarization, per-segment for broken."""
    if not GROQ_AVAILABLE:
        logger.error("Method 3 failed: run: pip install groq")
        return segments

    api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("Method 3 failed: GROQ_API_KEY not set in .env")
        return segments

    if not segments:
        return segments

    spk_counts = Counter(seg.get("speaker_id", 0) for seg in segments)
    first_spk = segments[0].get("speaker_id", 0)
    all_spks = sorted(spk_counts.keys())
    balance = _check_balance(segments)
    client = Groq(api_key=api_key)

    if len(all_spks) < 2:
        # Single speaker -- detect call type then classify
        logger.warning("Method 3: Only 1 speaker_id (SPK%d) -- detecting call type", first_spk)
        first_speaker = _detect_call_type(segments)
        return _per_segment(segments, client, first_speaker)

    elif balance >= DIARIZATION_BALANCE_THRESHOLD:
        # Good diarization -- use two-pass (always agent-first assumption works here
        # because two-pass reads content to determine which SPK is agent)
        logger.info("Method 3: Good diarization (%.0f%%) -- two-pass", balance * 100)
        return _two_pass(segments, client, first_spk, all_spks)

    else:
        # Broken diarization -- detect call type first, then per-segment
        logger.warning("Method 3: Broken diarization (%.0f%%) -- detecting call type",
                       balance * 100)
        first_speaker = _detect_call_type(segments)
        return _per_segment(segments, client, first_speaker)



def generate_call_summary(segments: list, call_id: str) -> dict:
    """
    Generate a 3-line human-readable summary of the call using Llama.
    Called AFTER classification — does not affect accuracy.
    """
    api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
    if not api_key or not GROQ_AVAILABLE:
        return {"summary": "Summary unavailable.", "topic": "", "outcome": ""}

    # Build condensed transcript (max 30 segments to save tokens)
    lines = []
    step = max(1, len(segments) // 30)
    for seg in segments[::step]:
        role = seg.get("predicted_role", "?")
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"{role}: {text}")

    prompt = (
        "Analyze this customer service call transcript and provide:\n"
        "1. Topic: One sentence describing what the call is about\n"
        "2. Summary: One sentence describing what happened\n"
        "3. Outcome: One sentence describing how it ended\n\n"
        "Transcript:\n" + "\n".join(lines) +
        "\n\nRespond ONLY with JSON: "
        '{{"topic": "...", "summary": "...", "outcome": "..."}}'
    )

    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You summarize customer service calls concisely."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw).strip()
        result = json.loads(raw)
        logger.info(f"Call summary generated for {call_id}")
        return {
            "topic":   result.get("topic", ""),
            "summary": result.get("summary", ""),
            "outcome": result.get("outcome", ""),
        }
    except Exception as e:
        logger.error(f"Call summary failed for {call_id}: {e}")
        return {"topic": "", "summary": "", "outcome": ""}


def is_llm_available():
    api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
    return GROQ_AVAILABLE and bool(api_key)


def get_llm_status():
    api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
    return {
        "available":         is_llm_available(),
        "package_installed": GROQ_AVAILABLE,
        "api_key_set":       bool(api_key),
        "model":             LLM_MODEL,
        "provider":          "Groq (Llama 3.3 70B)",
    }


# ── Utility functions (moved from method4_hybrid) ──
import numpy as np
import matplotlib.pyplot as plt

def compute_confidence_statistics(classified: list[dict]) -> dict:
    confidences = [seg.get("confidence", 0) for seg in classified]
    agreements  = [seg.get("agreement", False) for seg in classified]

    if not confidences:
        return {}

    return {
        "mean_confidence":            round(float(np.mean(confidences)), 4),
        "std_confidence":             round(float(np.std(confidences)),  4),
        "min_confidence":             round(float(np.min(confidences)),  4),
        "max_confidence":             round(float(np.max(confidences)),  4),
        "inter_model_agreement_rate": round(sum(agreements) / max(len(agreements), 1), 4),
        "n_segments":                 len(classified),
    }


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_method_comparison(stats_m1: dict, stats_m2: dict, stats_m3: dict,
                            save_path: str = None) -> str:
    methods = ["Method 1\n(Keyword)", "Method 2\n(Voice)", "Method 3\n(Hybrid)"]
    means   = [stats_m1.get("mean_confidence", 0),
               stats_m2.get("mean_confidence", 0),
               stats_m3.get("mean_confidence", 0)]
    stds    = [stats_m1.get("std_confidence", 0),
               stats_m2.get("std_confidence", 0),
               stats_m3.get("std_confidence", 0)]

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, means, width=0.5, yerr=stds, capsize=8,
                  color=["steelblue", "darkorange", "seagreen"],
                  alpha=0.85, edgecolor="white")
    for bar, mean_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{mean_val:.2f}", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim([0, 1.15])
    ax.set_ylabel("Mean Confidence Score")
    ax.set_title("Comparative Confidence Metrics of Classification Methods")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "method_comparison_confidence.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Method comparison chart saved → {save_path}")
    return save_path

def plot_ensemble_scores(hybrid_classified: list[dict],
                          save_path: str = None) -> str:
    scores     = [seg.get("ensemble_score", 0.5) for seg in hybrid_classified]
    agreements = [seg.get("agreement", False) for seg in hybrid_classified]
    overrides  = [seg.get("text_override", False) for seg in hybrid_classified]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (score, agree, override) in enumerate(zip(scores, agreements, overrides)):
        if override:
            color = "gold"
        elif agree:
            color = "seagreen"
        else:
            color = "tomato"
        ax.scatter(i, score, c=color, alpha=0.8, s=50,
                   zorder=5)

    # Legend proxies
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='seagreen',
               markersize=8, label='Models Agree'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato',
               markersize=8, label='Conflict'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',
               markersize=8, label='Text Override'),
    ]
    ax.legend(handles=legend_elements)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.5,
               label="Decision Boundary")
    ax.axhspan(0.35, 0.65, alpha=0.05, color="yellow")
    ax.set_xlabel("Segment Index")
    ax.set_ylabel("Ensemble Score (>0.5 = Agent)")
    ax.set_title("Method 3: Hybrid Ensemble Scores per Segment\n"
                 "(Green=Agree, Red=Conflict, Gold=Text Override)")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "m3_ensemble_scores.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Ensemble scores plot saved → {save_path}")
    return save_path
