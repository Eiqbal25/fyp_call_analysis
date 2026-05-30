"""
analytics/compliance.py
========================
Phase 3: Compliance and Behavioural Flagging

Implements the rule-based Compliance Verification Protocol (Section 3.5.3):
  - Agent SOP checklist: greeting, closing, recorded disclaimer, ID verification
  - Customer risk scanning: legal threats, fraud keywords, escalation requests
  - Automated Compliance Score ∈ [0, 1]
  - Severity tagging: High / Medium / Low / Clean

All checks use keyword spotting on real transcript text — no hardcoded values.
"""

import os
import re
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    COMPLIANCE_GREETING_KEYWORDS,
    COMPLIANCE_CLOSING_KEYWORDS,
    COMPLIANCE_RECORDED_KEYWORDS,
    COMPLIANCE_IDENTITY_KEYWORDS,
    RISK_KEYWORDS,
    OUTPUTS_DIR,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# KEYWORD DETECTION HELPER
# ─────────────────────────────────────────────────────────────

def _text_contains_any(text: str, keywords: list[str]) -> tuple[bool, list[str]]:
    """
    Check if text contains any of the given keywords/phrases.
    Returns (found: bool, matched_keywords: list[str]).
    """
    text_lower = text.lower()
    matched = []
    for kw in keywords:
        if " " in kw:
            if kw in text_lower:
                matched.append(kw)
        else:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                matched.append(kw)
    return len(matched) > 0, matched


# ─────────────────────────────────────────────────────────────
# COMPLIANCE CHECKS
# ─────────────────────────────────────────────────────────────

def check_compliance(classified: list[dict]) -> dict:
    """
    Run compliance verification on a classified transcript.

    Compliance checklist (Agent must satisfy all):
      ✓ Greeting      — opening phrase detected
      ✓ Closing       — closing phrase detected
      ✓ Recorded disc — mandatory recorded-call disclaimer
      ✓ Identity ver  — customer identity confirmed

    Risk scanning (Customer speech):
      ⚠ Legal threats, fraud keywords, escalation requests

    Parameters
    ----------
    classified : [{predicted_role, text, start, end, ...}, ...]

    Returns
    -------
    dict:
        greeting_passed, closing_passed, recorded_passed, identity_passed,
        compliance_score (0-1), compliance_label,
        risk_flag, risk_severity, risk_keywords_found,
        agent_text_combined, customer_text_combined,
        non_compliant_items, notes
    """
    # Separate agent and customer text
    agent_texts    = []
    customer_texts = []

    for seg in classified:
        role = seg.get("predicted_role", "Unknown")
        text = seg.get("text", "")
        if role == "Agent":
            agent_texts.append(text)
        elif role == "Customer":
            customer_texts.append(text)

    agent_combined    = " ".join(agent_texts).lower()
    customer_combined = " ".join(customer_texts).lower()

    # ── Agent compliance checks ──
    greeting_ok,   greeting_found   = _text_contains_any(agent_combined,
                                                           COMPLIANCE_GREETING_KEYWORDS)
    closing_ok,    closing_found    = _text_contains_any(agent_combined,
                                                           COMPLIANCE_CLOSING_KEYWORDS)
    recorded_ok,   recorded_found   = _text_contains_any(agent_combined,
                                                           COMPLIANCE_RECORDED_KEYWORDS)
    identity_ok,   identity_found   = _text_contains_any(agent_combined,
                                                           COMPLIANCE_IDENTITY_KEYWORDS)

    # ── Compliance score ──
    checks_passed = sum([greeting_ok, closing_ok, recorded_ok, identity_ok])
    total_checks  = 4
    compliance_score = round(checks_passed / total_checks, 4)

    if compliance_score == 1.0:
        compliance_label = "Fully Compliant"
    elif compliance_score >= 0.5:
        compliance_label = "Partially Compliant"
    else:
        compliance_label = "Non-Compliant"

    # ── Customer risk scanning ──
    risk_flag, risk_kws_found = _text_contains_any(customer_combined, RISK_KEYWORDS)

    if not risk_flag:
        risk_severity = "Clean"
    elif any(kw in risk_kws_found for kw in ["lawyer", "sue", "police", "legal action", "scam", "fraud"]):
        risk_severity = "High"
    elif any(kw in risk_kws_found for kw in ["manager", "supervisor", "complaint", "cancel", "refund"]):
        risk_severity = "Medium"
    else:
        risk_severity = "Low"

    # Non-compliant items for reporting
    non_compliant = []
    if not greeting_ok:
        non_compliant.append("Missing Greeting")
    if not closing_ok:
        non_compliant.append("Missing Closing")
    if not recorded_ok:
        non_compliant.append("Missing Recorded Disclaimer")
    if not identity_ok:
        non_compliant.append("Missing Identity Verification")

    # Notes
    notes = []
    if not greeting_ok:
        notes.append("⚠ Agent did not open with a standard greeting — may indicate late recording start.")
    if risk_severity == "High":
        notes.append("🚨 HIGH RISK: Customer used legal/fraud language — escalate immediately.")
    if risk_severity == "Medium":
        notes.append("⚠ MEDIUM RISK: Customer requested supervisor or mentioned complaint.")

    logger.info(
        f"Compliance: score={compliance_score:.2f} ({compliance_label}) | "
        f"risk={risk_severity} | non_compliant={non_compliant}"
    )

    return {
        "greeting_passed":       greeting_ok,
        "closing_passed":        closing_ok,
        "recorded_passed":       recorded_ok,
        "identity_passed":       identity_ok,
        "greeting_keywords":     greeting_found,
        "closing_keywords":      closing_found,
        "recorded_keywords":     recorded_found,
        "identity_keywords":     identity_found,
        "compliance_score":      compliance_score,
        "compliance_label":      compliance_label,
        "checks_passed":         checks_passed,
        "total_checks":          total_checks,
        "risk_flag":             risk_flag,
        "risk_severity":         risk_severity,
        "risk_keywords_found":   risk_kws_found,
        "non_compliant_items":   non_compliant,
        "notes":                 notes,
    }


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_compliance_summary(calls_compliance: list[dict],
                             save_path: str = None) -> str:
    """
    Bar chart of compliance component adherence rates across calls.
    calls_compliance: [{"call_id": str, "compliance": dict}, ...]
    """
    components = ["Greeting", "Closing", "Recorded\nDisclaimer", "Identity\nVerification"]
    keys       = ["greeting_passed", "closing_passed", "recorded_passed", "identity_passed"]

    rates = []
    for key in keys:
        n_passed = sum(1 for d in calls_compliance if d["compliance"].get(key, False))
        rates.append(n_passed / len(calls_compliance) * 100 if calls_compliance else 0)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(components, rates, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"],
                  alpha=0.85, edgecolor="white", width=0.5)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=11)

    ax.set_ylim([0, 115])
    ax.set_ylabel("Adherence Rate (%)")
    ax.set_title("Compliance Component Adherence Across All Calls")
    ax.axhline(80, color="red", linestyle="--", alpha=0.4, label="80% threshold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "compliance_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Compliance summary chart saved → {save_path}")
    return save_path


def plot_risk_severity_distribution(calls_compliance: list[dict],
                                     save_path: str = None) -> str:
    """Pie chart of risk severity distribution across calls."""
    severity_counts = {"Clean": 0, "Low": 0, "Medium": 0, "High": 0}
    for d in calls_compliance:
        sev = d["compliance"].get("risk_severity", "Clean")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    # Only plot non-zero
    labels = [k for k, v in severity_counts.items() if v > 0]
    sizes  = [v for v in severity_counts.values() if v > 0]
    colors = {"Clean": "green", "Low": "gold", "Medium": "orange", "High": "red"}
    pie_colors = [colors.get(l, "gray") for l in labels]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=labels, colors=pie_colors, autopct="%1.1f%%",
           startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax.set_title("Risk Severity Distribution Across Calls")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "risk_severity_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Risk severity chart saved → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────
# CALL OUTCOME DETECTION
# ─────────────────────────────────────────────────────────────

OUTCOME_RESOLVED_KEYWORDS = [
    "thank you for your help", "problem solved", "issue resolved",
    "that's all", "that works", "perfect thank you", "great thank you",
    "alright thank you", "okay thank you", "sounds good",
    "terima kasih", "dah selesai", "okay dah", "okaylah",
    "submission success", "done", "selesai", "boleh dah",
    "alhamdulillah", "thanks so much", "lifesaver",
]

OUTCOME_UNRESOLVED_KEYWORDS = [
    "still not working", "not fixed", "still the same problem",
    "nothing changed", "didn't help", "useless", "waste of time",
    "cancel my service", "cancel my account", "i'm canceling",
    "tak selesai", "masih tak boleh", "still cannot",
    "biadab", "kurang ajar", "report kes",
]

OUTCOME_ESCALATED_KEYWORDS = [
    "speak to your manager", "speak to a supervisor", "want a manager",
    "transfer me to", "escalate this", "file a complaint",
    "report this", "corporate office", "head office",
    "cakap dengan manager", "nak jumpa pengurus", "buat aduan rasmi",
    "nak buat report", "panggil bos",
]

OUTCOME_TRANSFERRED_KEYWORDS = [
    "transferring you now", "let me transfer", "i will transfer",
    "connecting you to", "putting you through", "one moment while i transfer",
    "hold on and do not hang up", "stay on the line",
    "tunggu sekejap saya pindahkan", "saya sambungkan",
]


def detect_call_outcome(segments: list[dict]) -> dict:
    """
    Detect overall call outcome based on transcript content.
    Returns: Resolved / Unresolved / Escalated / Transferred
    """
    # Look at last 30% of segments for resolution signals
    n = len(segments)
    end_segs = segments[int(n * 0.7):]
    all_text  = " ".join(s.get("text", "").lower() for s in segments)
    end_text  = " ".join(s.get("text", "").lower() for s in end_segs)

    _, transferred_kws = _text_contains_any(all_text, OUTCOME_TRANSFERRED_KEYWORDS)
    _, escalated_kws   = _text_contains_any(all_text, OUTCOME_ESCALATED_KEYWORDS)
    _, unresolved_kws  = _text_contains_any(all_text, OUTCOME_UNRESOLVED_KEYWORDS)
    _, resolved_kws    = _text_contains_any(end_text,  OUTCOME_RESOLVED_KEYWORDS)

    # Priority: Transferred > Escalated > Unresolved > Resolved
    if transferred_kws:
        outcome = "Transferred"
        emoji   = "📞"
        triggers = transferred_kws
    elif escalated_kws and unresolved_kws:
        outcome = "Escalated"
        emoji   = "🔄"
        triggers = escalated_kws
    elif unresolved_kws and not resolved_kws:
        outcome = "Unresolved"
        emoji   = "❌"
        triggers = unresolved_kws
    elif resolved_kws:
        outcome = "Resolved"
        emoji   = "✅"
        triggers = resolved_kws
    else:
        outcome = "Resolved"
        emoji   = "✅"
        triggers = []

    logger.info(f"Call outcome: {emoji} {outcome} | triggers={triggers[:2]}")

    return {
        "outcome":  outcome,
        "emoji":    emoji,
        "triggers": triggers[:3],
    }
