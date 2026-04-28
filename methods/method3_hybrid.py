"""
methods/method3_hybrid.py
==========================
Method 3: Hybrid Ensemble Fusion — UPGRADED for near-100% labelling accuracy

Core formula (Section 3.4.3):
    S_ensemble = (P_lex × C_lex × α + P_ac × C_ac × β) / (α + β)

UPGRADES over original:
  1. Speaker Anchoring       — Lock Agent = first speaker who uses a greeting keyword.
                               Eliminates the random label-flip from Resemblyzer.
  2. Dynamic α/β Weights     — Weights adjust per-call based on which model is more
                               confident. If Method 1 has clear keywords → α rises.
  3. Global Consistency      — After fusion, enforce exactly 1 Agent + 1 Customer.
  4. Segment-Level Override  — After speaker-level vote, scan every segment's text
                               for strong role-indicator phrases. If found, override
                               that segment's label directly. This fixes diarization
                               "leakage" where Agent speech is assigned to the Customer's
                               speaker_id cluster (and vice versa).
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

from config import (
    HYBRID_ALPHA,
    HYBRID_BETA,
    HYBRID_CONFLICT_PENALTY,
    HYBRID_DYNAMIC_WEIGHTS,
    HYBRID_MIN_ALPHA,
    HYBRID_MAX_ALPHA,
    SPEAKER_ANCHOR_WINDOW,
    COMPLIANCE_GREETING_KEYWORDS,
    COMPLIANCE_CLOSING_KEYWORDS,
    OUTPUTS_DIR,
)

logger = logging.getLogger(__name__)

_ROLE_TO_PROB = {"Agent": 1.0, "Customer": 0.0, "Unknown": 0.5}

# ─────────────────────────────────────────────────────────────
# STRONG PHRASE DICTIONARIES (for Upgrade 4)
# ─────────────────────────────────────────────────────────────

# Phrases that PROVE a segment is Agent regardless of diarization assignment.
# Selected to be distinctive — phrases a customer would never say.
_STRONG_AGENT_PHRASES = [
    # ── Universal English agent service actions ───────────────
    # These phrases appear ONLY in agent speech in any customer service call.
    # An agent is the one who pulls up accounts, checks systems, takes actions.
    "do you have your",
    "can i pull up",
    "i can pull up",
    "let me pull up",
    "let me check your",
    "let me verify",
    "i can schedule",
    "shall i proceed",
    "i am processing",
    "you will receive",
    "is there anything else",
    "anything else i can",
    "my pleasure",
    "it is my pleasure",
    "it's my pleasure",
    "looking at your",
    "i can see your",
    "i understand the urgency",
    "you are fully covered",
    "you are covered",
    "zero dollar deductible",
    "mobile technician",
    "confirmation text",
    "i can confirm",
    "for quality assurance",
    "recorded for quality",
    "this call is being recorded",
    "give me one moment",
    "one moment please",
    "bear with me",
    "let me check that for you",
    "i will transfer you",
    "let me transfer you",
    "i can transfer you",
    "i will connect you",
    "placing you on hold",
    "i am going to put you on hold",
    "estimated time",
    "our technicians are",
    "we can offer you",
    "i sincerely apologize",
    "i apologize for",
    "run a diagnostic",
    "line error",
    "i will escalate",
    # ── Universal Malay agent service actions ─────────────────
    # These are phrases only a customer service agent would say.
    "boleh saya dapatkan",
    "boleh saya semak",
    "saya akan semak",
    "saya akan uruskan",
    "saya akan proses",
    "saya akan sambungkan",
    "saya akan hantar",
    "maaf encik",
    "maaf cik",
    "minta maaf atas",
    "harap bersabar",
    "prosedur keselamatan",
    "memerlukan nombor",
    "untuk pengesahan",
    "saya nampak",
    "berdasarkan rekod",
    "mengikut rekod",
    "proses bayaran",
    "bayaran balik",
    "saya faham perasaan",
    "saya boleh bantu",
    "ada apa apa lagi",
    "ada apa-apa lagi",
    "kami akan",
    "maklum balas",
    "pegawai atasan",
    "pihak pengurusan",
]
_STRONG_CUSTOMER_PHRASES = [
    # ── Universal English customer phrases ───────────────────
    # These phrases appear ONLY in customer speech.
    # A customer is the one describing problems, expressing frustration,
    # giving their information, and asking for things.
    "i am calling because",
    "i'm calling because",
    "i have a problem",
    "i have an issue",
    "i need help",
    "i want to",
    "i would like to",
    "can you help me",
    "i am not happy",
    "i'm not happy",
    "i am frustrated",
    "this is unacceptable",
    "i want to speak to",
    "i want to talk to",
    "i want a manager",
    "speak to your supervisor",
    "i am not paying",
    "i did not",
    "i did not buy",
    "i did not order",
    "i live alone",
    "i checked my",
    "i have a contract",
    "my account number",
    "yes just a second",
    "yes please",
    "that works perfectly",
    "that would be a relief",
    "i am really worried",
    "i'm really worried",
    "i have a meeting",
    "i work from home",
    "i was out of town",
    "canceling my service",
    "cancel my account",
    "do not put me on hold",
    "do not dare",
    # ── Universal Malay customer phrases ─────────────────────
    "saya nak tanya",
    "saya nak",
    "saya nak refund",
    "saya nak cancel",
    "saya pesan",
    "saya dah",
    "saya tak terima",
    "saya tak dapat",
    "saya bayar",
    "saya tak beli",
    "saya pelanggan",
    "tak masuk akal",
    "ini tak betul",
    "kenapa pula",
    "boleh tolong",
    "nak buat aduan",
    "nak cakap dengan pengurus",
    "nak cakap dengan",
    "tak nak berurusan",
    "terima kasih ya",
    "itu saja",
    "tak ada dah",
]

# Strong agent-anchoring phrases used in first-turn detection
_ANCHOR_AGENT_PHRASES = [
    "thank you for calling",
    "thanks for calling",
    "good morning",
    "good afternoon",
    "good evening",
    "how may i assist",
    "how can i help",
    "how may i help",
    "welcome to",
    "you have reached",
    "this is",
    "my name is",
    "speaking",
    # Malay
    "selamat pagi",
    "selamat petang",
    "terima kasih kerana",
    "boleh saya bantu",
    "saya dari",
    "pusat bantuan",
    "khidmat pelanggan",
]

_ANCHOR_CUSTOMER_PHRASES = [
    "i need help",
    "i have a problem",
    "i have an issue",
    "i want to",
    "i would like to",
    "can you help me",
    "i am calling",
    "i called because",
    # Malay
    "saya nak",
    "boleh tolong",
    "ada masalah",
    "tak dapat",
    "saya nak tanya",
]


# ─────────────────────────────────────────────────────────────
# UPGRADE 1 — SPEAKER ANCHORING
# ─────────────────────────────────────────────────────────────

def anchor_speaker_roles(diarized: list[dict]) -> dict:
    """
    Determine which speaker_id is Agent and which is Customer
    by analysing the first SPEAKER_ANCHOR_WINDOW segments.

    Logic (priority order):
      1. Any speaker using a strong greeting phrase → Agent
      2. Any speaker using a strong customer phrase → Customer
      3. Speaker with more total words (scripted = Agent) → Agent
      4. Fallback: speaker 0 = Agent

    Returns
    -------
    anchor_map : {speaker_id: "Agent" | "Customer"}
    """
    if not diarized:
        return {}

    speakers = sorted(set(seg["speaker_id"] for seg in diarized))

    if len(speakers) == 1:
        logger.warning("Only 1 speaker found — labelling as Agent by default")
        return {speakers[0]: "Agent"}

    window = diarized[:SPEAKER_ANCHOR_WINDOW]
    agent_score = defaultdict(float)

    for seg in window:
        spk  = seg["speaker_id"]
        text = seg.get("text", "").lower().strip()

        for phrase in _ANCHOR_AGENT_PHRASES:
            if phrase in text:
                agent_score[spk] += 2.0

        for phrase in COMPLIANCE_GREETING_KEYWORDS:
            if phrase in text:
                agent_score[spk] += 1.5

        for phrase in COMPLIANCE_CLOSING_KEYWORDS:
            if phrase in text:
                agent_score[spk] += 0.5

        for phrase in _ANCHOR_CUSTOMER_PHRASES:
            if phrase in text:
                agent_score[spk] -= 1.5

    spk_scores = {spk: agent_score.get(spk, 0.0) for spk in speakers}
    logger.info(f"Speaker anchor scores: {spk_scores}")

    best_agent_spk = max(spk_scores, key=lambda s: spk_scores[s])

    # Tie-break: use word count
    if len(set(spk_scores.values())) == 1:
        word_counts = defaultdict(int)
        for seg in window:
            word_counts[seg["speaker_id"]] += len(seg.get("text", "").split())
        best_agent_spk = max(word_counts, key=lambda s: word_counts[s])
        logger.info(f"Anchor tie — using word count: agent={best_agent_spk}")

    anchor_map = {
        spk: "Agent" if spk == best_agent_spk else "Customer"
        for spk in speakers
    }
    logger.info(f"Speaker anchor result: {anchor_map}")
    return anchor_map


# ─────────────────────────────────────────────────────────────
# UPGRADE 2 — DYNAMIC WEIGHT CALCULATION
# ─────────────────────────────────────────────────────────────

def compute_dynamic_weights(lexical_classified: list[dict],
                             acoustic_classified: list[dict]) -> tuple[float, float]:
    """
    Adjust α (lexical) and β (acoustic) based on per-call model confidence.
    Returns (alpha, beta) normalised so they sum to 1.0.
    """
    if not HYBRID_DYNAMIC_WEIGHTS:
        return HYBRID_ALPHA, HYBRID_BETA

    lex_confs = [seg.get("confidence", 0.5) for seg in lexical_classified
                 if seg.get("predicted_role", "Unknown") != "Unknown"]
    ac_confs  = [seg.get("confidence", 0.5) for seg in acoustic_classified
                 if seg.get("predicted_role", "Unknown") != "Unknown"]

    lex_mean = float(np.mean(lex_confs)) if lex_confs else 0.5
    ac_mean  = float(np.mean(ac_confs))  if ac_confs  else 0.5

    total = lex_mean + ac_mean
    if total < 1e-6:
        return HYBRID_ALPHA, HYBRID_BETA

    raw_alpha = lex_mean / total
    alpha     = max(HYBRID_MIN_ALPHA, min(HYBRID_MAX_ALPHA, raw_alpha))
    beta      = 1.0 - alpha

    logger.info(
        f"Dynamic weights | lex_conf={lex_mean:.3f} ac_conf={ac_mean:.3f} "
        f"→ α={alpha:.3f} β={beta:.3f}"
    )
    return round(alpha, 4), round(beta, 4)


# ─────────────────────────────────────────────────────────────
# ENSEMBLE FUSION — PER SEGMENT
# ─────────────────────────────────────────────────────────────

def fuse_predictions(lexical_result: dict,
                     acoustic_result: dict,
                     alpha: float = HYBRID_ALPHA,
                     beta: float  = HYBRID_BETA) -> dict:
    """
    Fuse lexical and acoustic predictions:
        S_ensemble = (P_lex × C_lex × α + P_ac × C_ac × β) / (α + β)
    """
    p_lex = _ROLE_TO_PROB.get(lexical_result.get("predicted_role", "Unknown"), 0.5)
    c_lex = float(lexical_result.get("confidence", 0.5))

    p_ac  = _ROLE_TO_PROB.get(acoustic_result.get("predicted_role", "Unknown"), 0.5)
    c_ac  = float(acoustic_result.get("confidence", 0.5))

    lex_contrib = p_lex * c_lex * alpha
    ac_contrib  = p_ac  * c_ac  * beta
    s_ensemble  = (lex_contrib + ac_contrib) / (alpha + beta)

    lex_role  = lexical_result.get("predicted_role", "Unknown")
    ac_role   = acoustic_result.get("predicted_role", "Unknown")
    agreement = (lex_role == ac_role and lex_role != "Unknown")

    if s_ensemble > 0.5:
        predicted_role = "Agent"
        raw_confidence = s_ensemble
    elif s_ensemble < 0.5:
        predicted_role = "Customer"
        raw_confidence = 1.0 - s_ensemble
    else:
        predicted_role = ac_role if ac_role != "Unknown" else "Agent"
        raw_confidence = 0.50

    if not agreement:
        raw_confidence = max(0.0, raw_confidence - HYBRID_CONFLICT_PENALTY)

    return {
        "predicted_role":   predicted_role,
        "ensemble_score":   round(float(s_ensemble), 4),
        "confidence":       round(float(raw_confidence), 4),
        "agreement":        agreement,
        "lex_contribution": round(float(lex_contrib), 4),
        "ac_contribution":  round(float(ac_contrib),  4),
        "lexical_role":     lex_role,
        "acoustic_role":    ac_role,
        "method":           "hybrid",
    }


# ─────────────────────────────────────────────────────────────
# UPGRADE 3 — GLOBAL CONSISTENCY ENFORCEMENT
# ─────────────────────────────────────────────────────────────

def enforce_global_consistency(hybrid: list[dict],
                                anchor_map: dict) -> list[dict]:
    """
    Ensure exactly 1 Agent + 1 Customer per call.
    If both speakers end up with the same label, override using anchor_map.
    """
    if not anchor_map:
        return hybrid

    speaker_scores = defaultdict(lambda: {"Agent": 0.0, "Customer": 0.0})
    for seg in hybrid:
        spk  = seg["speaker_id"]
        role = seg.get("predicted_role", "Unknown")
        conf = seg.get("final_confidence", seg.get("confidence", 0.5))
        if role in speaker_scores[spk]:
            speaker_scores[spk][role] += conf

    speaker_labels = {
        spk: "Agent" if scores["Agent"] >= scores["Customer"] else "Customer"
        for spk, scores in speaker_scores.items()
    }

    labels = list(speaker_labels.values())
    if len(set(labels)) < 2 and len(speaker_labels) >= 2:
        logger.warning(
            f"Global consistency violation: all speakers labelled '{labels[0]}'. "
            f"Overriding with anchor map: {anchor_map}"
        )
        final_roles = anchor_map
    else:
        final_roles = speaker_labels

    for seg in hybrid:
        spk = seg["speaker_id"]
        if spk in final_roles:
            seg["predicted_role"]   = final_roles[spk]
            seg["final_confidence"] = round(
                speaker_scores[spk].get(final_roles[spk], 0.5) /
                max(sum(speaker_scores[spk].values()), 1e-9), 4
            )

    logger.info(
        f"Global consistency enforced | final_roles="
        f"{dict((k, v) for k, v in final_roles.items())}"
    )
    return hybrid


# ─────────────────────────────────────────────────────────────
# UPGRADE 4 — SEGMENT-LEVEL TEXT OVERRIDE
# ─────────────────────────────────────────────────────────────

def segment_level_text_override(hybrid: list[dict]) -> list[dict]:
    """
    After speaker-level classification, scan every segment's text for
    strong role-indicator phrases. If a phrase is found that contradicts
    the current label, override that segment's label.

    WHY THIS IS NEEDED:
    Diarization sometimes assigns Agent speech to the Customer's speaker_id
    cluster (and vice versa), especially at speaker turn transitions.
    The speaker-level majority vote then uses these wrong assignments and
    can produce incorrect final labels.

    This step uses text semantics as a final correction layer — certain
    phrases (e.g. "Is there anything else I can assist you with?") are
    unambiguously Agent speech regardless of which speaker cluster they
    were assigned to by Resemblyzer.

    Only overrides when a STRONG phrase is detected. Does not change
    speaker_id, only predicted_role for that individual segment.
    """
    corrected = 0

    for seg in hybrid:
        text      = seg.get("text", "").lower().strip()
        curr_role = seg.get("predicted_role", "Unknown")

        if not text:
            continue

        # Check strong agent signals
        agent_hit = next(
            (p for p in _STRONG_AGENT_PHRASES if p in text), None
        )
        if agent_hit and curr_role != "Agent":
            logger.info(
                f"  Text override [{seg.get('start', 0):.1f}s]: "
                f"{curr_role} → Agent  (phrase: '{agent_hit}')\n"
                f"  Text: \"{text[:70]}\""
            )
            seg["predicted_role"]       = "Agent"
            seg["final_confidence"]     = 0.92
            seg["text_override"]        = True
            seg["text_override_phrase"] = agent_hit
            corrected += 1
            continue   # no need to check customer phrases

        # Check strong customer signals
        cust_hit = next(
            (p for p in _STRONG_CUSTOMER_PHRASES if p in text), None
        )
        if cust_hit and curr_role != "Customer":
            logger.info(
                f"  Text override [{seg.get('start', 0):.1f}s]: "
                f"{curr_role} → Customer  (phrase: '{cust_hit}')\n"
                f"  Text: \"{text[:70]}\""
            )
            seg["predicted_role"]       = "Customer"
            seg["final_confidence"]     = 0.92
            seg["text_override"]        = True
            seg["text_override_phrase"] = cust_hit
            corrected += 1

    if corrected:
        logger.info(f"Segment text override: {corrected} segment(s) corrected")
    else:
        logger.info("Segment text override: no corrections needed")

    return hybrid


# ─────────────────────────────────────────────────────────────
# MAIN: CLASSIFY FULL TRANSCRIPT
# ─────────────────────────────────────────────────────────────

def classify_transcript_hybrid(lexical_classified: list[dict],
                                acoustic_classified: list[dict]) -> list[dict]:
    """
    Upgraded 5-step hybrid fusion pipeline:
      Step 1 — Speaker anchoring       (who is Agent/Customer?)
      Step 2 — Dynamic α/β             (which model to trust more?)
      Step 3 — Segment fusion          (S_ensemble per segment)
      Step 4 — Global consistency      (enforce 1 Agent + 1 Customer)
      Step 5 — Segment text override   (fix diarization leakage)
    """
    if len(lexical_classified) != len(acoustic_classified):
        logger.warning(
            f"Segment count mismatch: lex={len(lexical_classified)}, "
            f"ac={len(acoustic_classified)} — aligning by index"
        )
        n = min(len(lexical_classified), len(acoustic_classified))
        lexical_classified  = lexical_classified[:n]
        acoustic_classified = acoustic_classified[:n]

    if not lexical_classified:
        return []

    # ── Step 1: Speaker anchoring ────────────────────────────────
    anchor_map = anchor_speaker_roles(lexical_classified)

    # ── Step 2: Dynamic weights ──────────────────────────────────
    alpha, beta = compute_dynamic_weights(lexical_classified, acoustic_classified)

    # ── Step 3: Segment-level fusion ─────────────────────────────
    hybrid = []
    for lex_seg, ac_seg in zip(lexical_classified, acoustic_classified):
        fusion = fuse_predictions(lex_seg, ac_seg, alpha=alpha, beta=beta)

        merged = {**lex_seg}
        merged.update({
            "acoustic_role":       ac_seg.get("predicted_role", "Unknown"),
            "acoustic_confidence": ac_seg.get("confidence", 0.5),
            "agent_prob":          ac_seg.get("agent_prob", 0.5),
            "customer_prob":       ac_seg.get("customer_prob", 0.5),
            "lexical_role":        lex_seg.get("predicted_role", "Unknown"),
            "lexical_confidence":  lex_seg.get("confidence", 0.5),
            "predicted_role":      fusion["predicted_role"],
            "confidence":          fusion["confidence"],
            "ensemble_score":      fusion["ensemble_score"],
            "agreement":           fusion["agreement"],
            "lex_contribution":    fusion["lex_contribution"],
            "ac_contribution":     fusion["ac_contribution"],
            "alpha_used":          alpha,
            "beta_used":           beta,
            "text_override":       False,
            "text_override_phrase": None,
            "method":              "hybrid",
        })
        hybrid.append(merged)

    # ── Step 4: Global consistency ───────────────────────────────
    hybrid = enforce_global_consistency(hybrid, anchor_map)

    # ── Step 5: Segment-level text override ──────────────────────
    hybrid = segment_level_text_override(hybrid)

    # Summary log
    n_agree    = sum(1 for s in hybrid if s.get("agreement", False))
    n_override = sum(1 for s in hybrid if s.get("text_override", False))
    agree_rate = n_agree / len(hybrid) * 100 if hybrid else 0

    role_summary = {}
    for seg in hybrid:
        spk  = seg["speaker_id"]
        role = seg["predicted_role"]
        role_summary[spk] = role

    logger.info(
        f"Method 3 Hybrid complete | segments={len(hybrid)} | "
        f"agreement_rate={agree_rate:.1f}% | "
        f"text_overrides={n_override} | "
        f"α={alpha:.3f} β={beta:.3f} | "
        f"final_roles={dict(set(role_summary.items()))}"
    )
    return hybrid


# ─────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────

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
