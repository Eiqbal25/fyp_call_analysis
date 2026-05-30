"""
analytics/advanced.py
======================
Advanced Analytics Module

Provides additional analytics beyond core metrics:
  1. Agent Response Time  — how fast agent responds after customer
  2. Interruption Detection — overlapping speech segments
  3. Language Detection — auto-detect English/Malay/Manglish
  4. WER Summary — format WER data for reporting
"""

import re
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 1. AGENT RESPONSE TIME
# ─────────────────────────────────────────────────────────────

def compute_agent_response_time(segments: list) -> dict:
    """
    Measure how quickly the agent responds after the customer stops speaking.
    Low response time = attentive agent.
    High response time = slow, inattentive agent.
    """
    response_times = []

    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]

        if (prev.get("predicted_role") == "Customer" and
                curr.get("predicted_role") == "Agent"):
            prev_end  = prev.get("end",   prev.get("start", 0) + prev.get("duration", 0))
            curr_start = curr.get("start", 0)
            gap = curr_start - prev_end
            if 0 <= gap <= 10:  # ignore gaps > 10s (silence/hold)
                response_times.append(round(gap, 3))

    if not response_times:
        return {
            "avg_response_time_sec": None,
            "min_response_time_sec": None,
            "max_response_time_sec": None,
            "response_count":        0,
            "rating":                "N/A",
        }

    avg = round(sum(response_times) / len(response_times), 2)
    rating = (
        "Excellent" if avg <= 0.5 else
        "Good"      if avg <= 1.5 else
        "Fair"      if avg <= 3.0 else
        "Slow"
    )

    logger.info(
        f"Agent response time: avg={avg}s min={min(response_times)}s "
        f"max={max(response_times)}s ({rating})"
    )

    return {
        "avg_response_time_sec": avg,
        "min_response_time_sec": min(response_times),
        "max_response_time_sec": max(response_times),
        "response_count":        len(response_times),
        "rating":                rating,
    }


# ─────────────────────────────────────────────────────────────
# 2. INTERRUPTION DETECTION
# ─────────────────────────────────────────────────────────────

def detect_interruptions(segments: list) -> dict:
    """
    Detect overlapping speech — when a segment starts before the previous ends.
    Common in heated/rude calls. Agent interruptions are more concerning.
    """
    agent_interruptions    = 0
    customer_interruptions = 0
    interruption_details   = []

    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]

        prev_end   = prev.get("end", prev.get("start", 0) + prev.get("duration", 0))
        curr_start = curr.get("start", 0)

        if curr_start < prev_end:  # overlap detected
            overlap = round(prev_end - curr_start, 2)
            interruptor = curr.get("predicted_role", "Unknown")

            if interruptor == "Agent":
                agent_interruptions += 1
            elif interruptor == "Customer":
                customer_interruptions += 1

            interruption_details.append({
                "timestamp":   round(curr_start, 1),
                "interruptor": interruptor,
                "overlap_sec": overlap,
            })

    total = agent_interruptions + customer_interruptions

    logger.info(
        f"Interruptions: total={total} "
        f"Agent={agent_interruptions} Customer={customer_interruptions}"
    )

    return {
        "total_interruptions":      total,
        "agent_interruptions":      agent_interruptions,
        "customer_interruptions":   customer_interruptions,
        "interruption_details":     interruption_details[:10],  # top 10
        "interruption_rate":        round(total / max(len(segments), 1) * 100, 1),
    }


# ─────────────────────────────────────────────────────────────
# 3. LANGUAGE DETECTION
# ─────────────────────────────────────────────────────────────

MALAY_MARKERS = [
    "saya", "awak", "encik", "cik", "nak", "boleh", "tak",
    "dah", "dengan", "untuk", "yang", "ini", "itu", "ada",
    "selamat", "terima kasih", "maaf", "nombor", "telefon",
    "kami", "kita", "dia", "mereka", "pihak", "barang",
]

MANGLISH_MARKERS = [
    "lah", "loh", "mah", "weh", "kan", "ah", "eh",
    "one", "cannot", "already", "also", "confirm",
    "mana", "macam", "memang", "sikit", "betul",
    "wah", "aiyo", "haiya", "hor", "lor",
]

ENGLISH_MARKERS = [
    "the", "this", "that", "your", "have", "will",
    "please", "thank you", "sorry", "account", "service",
    "calling", "speaking", "transfer", "supervisor",
]


def detect_language(segments: list) -> dict:
    """
    Auto-detect the primary language of the call.
    Returns: English / Malay / Manglish
    """
    all_text = " ".join(s.get("text", "").lower() for s in segments)

    malay_score    = sum(1 for m in MALAY_MARKERS    if m in all_text)
    manglish_score = sum(1 for m in MANGLISH_MARKERS if m in all_text)
    english_score  = sum(1 for m in ENGLISH_MARKERS  if m in all_text)

    # Manglish = mix of English + Malay markers, but needs strong Malay presence
    if manglish_score >= 3 and english_score >= 5 and malay_score >= 3:
        language = "Manglish"
    elif malay_score >= 5 and malay_score > english_score:
        language = "Malay"
    elif english_score >= 5 and malay_score < 3:
        language = "English"
    elif manglish_score >= 2 and malay_score >= 2:
        language = "Manglish"
    else:
        language = "English"

    logger.info(
        f"Language detection: {language} "
        f"(Malay={malay_score} Manglish={manglish_score} English={english_score})"
    )

    return {
        "detected_language": language,
        "malay_score":       malay_score,
        "manglish_score":    manglish_score,
        "english_score":     english_score,
    }


# ─────────────────────────────────────────────────────────────
# 4. WER SUMMARY FORMATTER
# ─────────────────────────────────────────────────────────────

def format_wer_summary(calls_data: list) -> str:
    """
    Format WER results across all calls into a readable table.
    calls_data: [{"call_id": str, "wer": dict, "language": str}, ...]
    """
    lines = [
        "\n── Word Error Rate (WER) per Call ───────────────────────",
        f"  {'Call ID':<25} {'Language':<12} {'WER':>8}  Quality",
        "  " + "-" * 55,
    ]

    wer_values = []
    for call in sorted(calls_data, key=lambda x: x.get("call_id", "")):
        cid      = call.get("call_id", "")
        wer_data = call.get("wer", {})
        lang     = call.get("language", {}).get("detected_language", "Unknown")
        wer_val  = wer_data.get("wer")

        if wer_val is not None:
            wer_values.append(wer_val)
            quality = (
                "Excellent" if wer_val < 10 else
                "Good"      if wer_val < 25 else
                "Fair"      if wer_val < 40 else
                "Poor"
            )
            lines.append(f"  {cid:<25} {lang:<12} {wer_val:>7.1f}%  {quality}")
        else:
            lines.append(f"  {cid:<25} {lang:<12}      N/A  N/A")

    if wer_values:
        avg_wer = sum(wer_values) / len(wer_values)
        lines.append("  " + "-" * 55)
        lines.append(f"  {'AVERAGE':<25} {'':12} {avg_wer:>7.1f}%")

    return "\n".join(lines)
