"""
analytics/rude_behavior.py
===========================
Rude Behavior Detection Module

Detects and warns about rude, unprofessional, or hostile language
from both Agents and Customers in customer service calls.

Severity levels:
  HIGH   - Personal insults, profanity, threats
  MEDIUM - Dismissive, condescending, sarcastic speech
  LOW    - Mild impatience, minor rudeness

Special focus on AGENT rudeness — more serious than customer rudeness
because agents represent the company and have a professional duty.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# RUDE BEHAVIOR KEYWORDS
# ─────────────────────────────────────────────────────────────

# Agent-specific rude patterns (more serious — professional duty)
AGENT_RUDE_HIGH = [
    # English
    "obviously you don't", "not that hard", "should have called",
    "hold your hand", "trouble reading", "go complain",
    "consider it canceled", "wasting their time", "wasting my time",
    "stop wasting my time", "just hang up", "actual customers",
    "desperate for the money", "forgot to check your emails",
    "i don't have time",
    # Malay
    "malas nak layan", "tak payah nak berangan", "ada i kisah",
    "menyemak", "inak saya takut", "report lah", "viral lah viral",
    "takkan benda macam tu pun", "sikit-sikit nak panggil",
    "bukan you sorang je", "duit you yang tak seberapa",
    "you yang tak sabar", "menyusahkan orang lain",
]

AGENT_RUDE_MEDIUM = [
    # English
    "not my problem", "not technical support", "i just handle",
    "read your contract", "printed in bold", "clearly stated",
    "that's how business works", "normal price like everyone else",
    "not that complicated",
    # Malay
    "logiklah", "takkan tu pun", "kena rajin membaca",
    "jangan salahkan", "peak season", "terms and conditions",
    "dah memang tanggungjawab", "pandai-pandailah",
    "bagi alasan", "tak payah nak",
]

# Customer rude patterns
CUSTOMER_RUDE_HIGH = [
    # English
    "are you kidding me", "out of your mind", "absolutely useless",
    "completely incompetent", "blatant lie", "i don't care",
    "unbelievable", "get out of here", "screw this",
    "what the hell", "what is wrong with you",
    # Malay / Manglish
    "otak kau", "kurang ajar", "biadab", "bodoh",
    "tak reti buat kerja", "menyusahkan betul",
    "nak menipu", "pandai-pandai je", "what the",
    "celaka", "bangang",
]

CUSTOMER_RUDE_MEDIUM = [
    # English
    "i've been waiting", "this is unacceptable", "fix this right now",
    "useless", "incompetent", "terrible service", "worst service",
    "speak to your manager", "i want a refund", "this is ridiculous",
    "are you serious", "you people",
    # Malay / Manglish
    "memang teruk", "servis teruk", "tak guna",
    "nak jumpa manager", "nak buat aduan", "macam ni ke servis",
    "menyusahkan", "tak profesional", "you orang ni",
    "lambat sangat", "tak masuk akal",
]


def _check_keywords(text: str, keywords: list) -> list:
    """Return matched keywords found in text."""
    text_lower = text.lower()
    matched = []
    for kw in keywords:
        if kw in text_lower:
            matched.append(kw)
    return matched


def detect_rude_behavior(segments: list[dict]) -> dict:
    """
    Analyse all segments for rude behavior.
    Returns a structured report with per-segment flags and summary.
    """
    agent_incidents = []
    customer_incidents = []

    for seg in segments:
        role = seg.get("predicted_role", "")
        text = seg.get("text", "")
        start = seg.get("start", 0)

        if not text or not role:
            continue

        if role == "Agent":
            high_matches = _check_keywords(text, AGENT_RUDE_HIGH)
            med_matches  = _check_keywords(text, AGENT_RUDE_MEDIUM)

            if high_matches:
                agent_incidents.append({
                    "timestamp": round(start, 1),
                    "severity":  "HIGH",
                    "text":      text[:80],
                    "triggers":  high_matches,
                    "role":      "Agent",
                })
            elif med_matches:
                agent_incidents.append({
                    "timestamp": round(start, 1),
                    "severity":  "MEDIUM",
                    "text":      text[:80],
                    "triggers":  med_matches,
                    "role":      "Agent",
                })

        elif role == "Customer":
            high_matches = _check_keywords(text, CUSTOMER_RUDE_HIGH)
            med_matches  = _check_keywords(text, CUSTOMER_RUDE_MEDIUM)

            if high_matches:
                customer_incidents.append({
                    "timestamp": round(start, 1),
                    "severity":  "HIGH",
                    "text":      text[:80],
                    "triggers":  high_matches,
                    "role":      "Customer",
                })
            elif med_matches:
                customer_incidents.append({
                    "timestamp": round(start, 1),
                    "severity":  "MEDIUM",
                    "text":      text[:80],
                    "triggers":  med_matches,
                    "role":      "Customer",
                })

    # Determine overall rudeness level per speaker
    def _overall_level(incidents):
        if any(i["severity"] == "HIGH" for i in incidents):
            return "HIGH"
        elif incidents:
            return "MEDIUM"
        return "NONE"

    agent_level    = _overall_level(agent_incidents)
    customer_level = _overall_level(customer_incidents)

    # Generate warnings
    warnings = []
    if agent_level == "HIGH":
        warnings.append(
            "⚠️  CRITICAL: Agent displayed highly unprofessional behavior. "
            "Immediate supervisor review required."
        )
    elif agent_level == "MEDIUM":
        warnings.append(
            "⚠️  WARNING: Agent displayed dismissive or condescending behavior. "
            "Coaching recommended."
        )

    if customer_level == "HIGH":
        warnings.append(
            "ℹ️  NOTE: Customer used hostile or abusive language. "
            "Consider escalation protocol."
        )
    elif customer_level == "MEDIUM":
        warnings.append(
            "ℹ️  NOTE: Customer displayed frustrated or impatient behavior."
        )

    if not warnings:
        warnings.append("✅  No significant rude behavior detected.")

    result = {
        "agent_rudeness_level":    agent_level,
        "customer_rudeness_level": customer_level,
        "agent_incidents":         agent_incidents,
        "customer_incidents":      customer_incidents,
        "total_agent_incidents":   len(agent_incidents),
        "total_customer_incidents": len(customer_incidents),
        "warnings":                warnings,
    }

    # Log summary
    logger.info(
        f"Rude behavior | Agent={agent_level} ({len(agent_incidents)} incidents) "
        f"| Customer={customer_level} ({len(customer_incidents)} incidents)"
    )
    for w in warnings:
        logger.info(f"  {w}")

    return result


def format_rude_behavior_report(call_id: str, rude_result: dict) -> str:
    """Format rude behavior result as readable text for the report."""
    lines = [
        f"\n── Rude Behavior Analysis: {call_id} {'─'*(40-len(call_id))}",
        f"  Agent rudeness   : {rude_result['agent_rudeness_level']}",
        f"  Customer rudeness: {rude_result['customer_rudeness_level']}",
    ]

    for w in rude_result["warnings"]:
        lines.append(f"  {w}")

    if rude_result["agent_incidents"]:
        lines.append(f"\n  Agent incidents ({len(rude_result['agent_incidents'])}):")
        for inc in rude_result["agent_incidents"]:
            lines.append(
                f"    [{inc['timestamp']}s] {inc['severity']} | \"{inc['text'][:60]}...\""
            )

    if rude_result["customer_incidents"]:
        lines.append(f"\n  Customer incidents ({len(rude_result['customer_incidents'])}):")
        for inc in rude_result["customer_incidents"]:
            lines.append(
                f"    [{inc['timestamp']}s] {inc['severity']} | \"{inc['text'][:60]}...\""
            )

    return "\n".join(lines)
