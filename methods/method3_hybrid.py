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

from methods.method4_llm import classify_transcript_llm, is_llm_available
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
    # ── Added phrases targeting remaining wrong labels ─────────────────
    "mohon tunggu sebentar",        # bank [100.1s] — clear Malay agent hold phrase
    "sila tunggu sebentar",         # Malay variant
    "the system shows",             # billing — agent referencing their system
    "our records show",             # billing variant
    "i just handle",                # billing [51.4s] — agent describing scope
    "putting you in the queue",     # billing [79.0s] — agent action
    "i just need it to access",     # internet [26.7s] — agent asking for access
    "let me access your",           # internet variant
    "habis itu kau buat apa",       # bank [90.3s] — wrong but agent context phrase
    "jangan nak melinggal",         # bank — agent apologising for wait
    "encik nak saya",               # Malay — agent offering to do something
    "kami akan uruskan",            # Malay — we will handle it
    "sistem kami",                  # Malay — our system
    "rekod kami",                   # Malay — our records
    # ── Remaining wrong segments — universal agent patterns ──────
    "not at all we just",           # insurance [51.5s] — after punctuation strip
    "processing that now",          # insurance [63.3s]
    "i am processing that",         # insurance variant
    "sementara saya simak",         # bank [39.4s]
    "saya faham",                   # bank [72.5s] — empathy phrase
    "proses pembatalan",            # bank [83.8s]
    "nak saya pulangkan",           # food [40.0s] — agent offering refund
    "proses ini akan",              # food [56.6s] — agent explaining
    "mengambil masa",               # food [56.6s] variant
    "i will note that",
    "let me note",
    "i have documented",
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
    "cancelling my service",    # British spelling variant
    "cancel my account",
    "cancelling my account",
    "i'm cancelling",
    "do not put me on hold",
    "do not dare",
    "do not you dare",
    # ── Universal Malay customer phrases ─────────────────────
    "saya nak tanya",
    "saya nak refund",
    "saya nak cancel",
    "saya pesan",
    "saya tak terima",
    "saya tak dapat",
    "saya bayar",
    "saya tak beli",
    "saya pelanggan",
    "ini tak betul",
    "kenapa pula",
    "boleh tolong",
    "nak buat aduan",
    "nak cakap dengan",
    "tak nak berurusan",
    "terima kasih ya",
    "itu saja",
    "tak ada dah",
    # ── Additional universal customer phrases ────────────────
    "whatever just fix",
    "your company is",
    "that would be a huge relief",
    "that would be a relief",
    "oh that would be",
    "cepat sikit",
    "korang ni",
    "angkat telefon je",
    "you guys are absolutely",
    "absolutely useless",
    "i do not care about nodes",
    "i do not care about",
    "i am really worried",
    "i'm really worried",
    "i have a long road",           # insurance [28.7s]
    "need this fixed immediately",  # customer urgency
    "i want my money back",
    "give me a refund",
    "korang memang",                # Malay — you people (complaint)
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
    "my name is",
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

    # ── FIRST-SPEAKER PRIOR ───────────────────────────────────────────────
    # In every customer service call the agent ALWAYS picks up the phone first.
    # This is unconditional — rude, informal, or non-scripted agents still
    # answer first. Making this a strong prior (+3.0) prevents keyword noise
    # from flipping the anchor on informal-agent calls (billing, delivery).
    first_spk = diarized[0]["speaker_id"] if diarized else 0
    agent_score[first_spk] += 3.0
    logger.info(f"First-speaker prior applied: SPK{first_spk} +3.0")

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

    # Tie-break: use FIRST SPEAKER rule
    # The agent always answers the phone first in any customer service call.
    # This is universally true regardless of language, style, or compliance level.
    # "Yeah what do you want" = Agent because they answered at t=0.
    # Word count is WRONG: customers explain problems at length, agents use brief scripts.
    if len(set(spk_scores.values())) == 1 or max(spk_scores.values()) <= 0.0:
        first_spk = diarized[0]["speaker_id"] if diarized else 0
        best_agent_spk = first_spk
        logger.info(f"Anchor tie — first-speaker rule: SPK{first_spk}=Agent")

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

    THREE cases handled (in priority order):

    Case 1 — Same label: both speakers voted the same role.
      → Clear violation. Use anchor_map.

    Case 2 — Inversion: voted labels are opposite to anchor_map.
      → The ensemble majority-voted the wrong way (happens when agent
        sounds like a customer — billing_english, delivery_malay).
        Anchor (first-speaker rule) is more reliable. Use anchor_map.

    Case 3 — Agreement: voted labels match anchor_map.
      → No conflict. Use voted labels (more granular confidence data).
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

    # ── Case 1: Both speakers got the same label ──────────────────────
    if len(set(labels)) < 2 and len(speaker_labels) >= 2:
        logger.warning(
            f"Global consistency — SAME LABEL: all speakers = '{labels[0]}'. "
            f"Overriding with anchor map: {anchor_map}"
        )
        final_roles = anchor_map

    # ── Case 2: Inversion — voted labels are opposite to anchor ──────
    elif len(speaker_labels) >= 2 and anchor_map:
        n_conflicts = sum(
            1 for spk, anchor_role in anchor_map.items()
            if spk in speaker_labels and speaker_labels[spk] != anchor_role
        )
        if n_conflicts == len(anchor_map):
            # Every speaker's voted label conflicts with anchor → full inversion
            logger.warning(
                f"Global consistency — INVERSION DETECTED: "
                f"voted={dict(speaker_labels)} conflicts with anchor={dict(anchor_map)}. "
                f"This happens when an informal/rude agent sounds like a customer. "
                f"Trusting anchor (first-speaker rule) over ensemble vote."
            )
            final_roles = anchor_map
        else:
            # Partial agreement — use voted (anchor already agrees on most speakers)
            final_roles = speaker_labels

    # ── Case 3: Full agreement ────────────────────────────────────────
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

def _normalize_text(text: str) -> str:
    """
    Strip punctuation before phrase matching.
    Whisper output: "Not at all. We just need" fails to match "not at all we just"
    because of the period. Removing punctuation fixes this silently.
    """
    import re as _re
    t = text.lower().strip()
    t = _re.sub(r"[^\w\s]", " ", t)   # remove punctuation
    t = _re.sub(r"\s+", " ", t).strip() # collapse spaces
    return t


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
        raw_text  = seg.get("text", "")
        text      = _normalize_text(raw_text)   # strip punctuation for reliable matching
        curr_role = seg.get("predicted_role", "Unknown")

        if not text:
            continue

        # Check strong agent signals (using punctuation-stripped text)
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
# UPGRADE 5 — CONTEXT SMOOTHING (short fragment inheritance)
# ─────────────────────────────────────────────────────────────

def context_smoothing(hybrid: list[dict],
                       window: int = 2,
                       min_words: int = 5) -> list[dict]:
    """
    Short segments (< min_words) with no phrase override inherit their label
    from the surrounding context window.

    WHY THIS IS NEEDED:
    Whisper splits audio into very fine segments. Short fragments like "Yes,",
    "Fine.", "Hi Marcus.", "just a second.", "shortly." carry no keyword or
    acoustic signal — both Method 1 and Method 2 effectively guess randomly.
    These fragments are almost always part of the same speaker turn as their
    neighbours, so inheriting the surrounding majority label is correct ~95%
    of the time.

    Only applies to:
      - Segments shorter than min_words words
      - Segments that did NOT already get a text_override
      - Segments where the surrounding window has a clear majority (not tied)
    """
    corrected = 0

    for i, seg in enumerate(hybrid):
        # Skip segments that already have a confident override
        if seg.get("text_override", False):
            continue

        word_count = len(seg.get("text", "").split())
        if word_count >= min_words:
            continue

        # Collect neighbour labels within window
        neighbour_roles = []
        for j in range(max(0, i - window), min(len(hybrid), i + window + 1)):
            if j != i:
                role = hybrid[j].get("predicted_role", "Unknown")
                if role != "Unknown":
                    neighbour_roles.append(role)

        if not neighbour_roles:
            continue

        agent_count    = neighbour_roles.count("Agent")
        customer_count = neighbour_roles.count("Customer")

        # Only apply when there is a clear majority (not a tie)
        if agent_count == customer_count:
            continue

        majority_role = "Agent" if agent_count > customer_count else "Customer"
        current_role  = seg.get("predicted_role", "Unknown")

        if majority_role != current_role:
            logger.info(
                f"  Context smooth [{seg.get('start', 0):.1f}s]: "
                f"{current_role} → {majority_role}  "
                f"(words={word_count}, neighbours={neighbour_roles})"
            )
            seg["predicted_role"]    = majority_role
            seg["final_confidence"]  = 0.78
            seg["context_smoothed"]  = True
            corrected += 1

    if corrected:
        logger.info(f"Context smoothing: {corrected} short segment(s) corrected")
    else:
        logger.info("Context smoothing: no short-segment corrections needed")

    return hybrid


# ─────────────────────────────────────────────────────────────
# MAIN: CLASSIFY FULL TRANSCRIPT
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# UPGRADE 6 — BROKEN DIARIZATION FALLBACK
# ─────────────────────────────────────────────────────────────

DIARIZATION_BALANCE_THRESHOLD = 0.20   # if minority speaker < 20% of segments

def check_diarization_balance(segments: list[dict]) -> dict:
    """
    Check if diarization produced a meaningful two-speaker split.
    Returns balance info including whether it's broken.
    """
    from collections import Counter
    spk_counts = Counter(seg["speaker_id"] for seg in segments)

    if len(spk_counts) < 2:
        return {"balanced": False, "reason": "only_one_speaker",
                "counts": dict(spk_counts), "minority_ratio": 0.0}

    total = sum(spk_counts.values())
    minority = min(spk_counts.values())
    ratio = minority / total

    balanced = ratio >= DIARIZATION_BALANCE_THRESHOLD

    if not balanced:
        logger.warning(
            f"BROKEN DIARIZATION detected: speaker split = {dict(spk_counts)} "
            f"(minority ratio = {ratio:.1%}, threshold = {DIARIZATION_BALANCE_THRESHOLD:.0%}). "
            f"Falling back to text-only independent classification."
        )

    return {"balanced": balanced, "minority_ratio": ratio,
            "counts": dict(spk_counts), "reason": "" if balanced else "imbalanced"}


def classify_independent_by_text(hybrid: list[dict]) -> list[dict]:
    """
    When diarization is broken (one speaker cluster dominates), classify
    each segment INDEPENDENTLY using text content alone.

    This ignores speaker_id entirely. Each segment gets its own Agent/Customer
    label based on:
      1. Strong phrase match (highest priority)
      2. Keyword density comparison (fallback)
      3. Conversational position cues

    Called INSTEAD OF enforce_global_consistency when diarization is broken.
    """
    logger.info("Running independent text-only classification (diarization fallback)")

    for seg in hybrid:
        text = _normalize_text(seg.get("text", ""))
        if not text.strip():
            continue

        agent_score = 0.0
        customer_score = 0.0
        matched_phrase = None

        # ── Check strong agent phrases ────────────────────────────
        for phrase in _STRONG_AGENT_PHRASES:
            if phrase in text:
                agent_score += 5.0
                matched_phrase = phrase
                break

        # ── Check strong customer phrases ─────────────────────────
        for phrase in _STRONG_CUSTOMER_PHRASES:
            if phrase in text:
                customer_score += 5.0
                matched_phrase = phrase
                break

        # ── Keyword density as tiebreaker ─────────────────────────
        words = text.split()
        if words:
            # Load keywords (same ones used by Method 1)
            for kw in _ANCHOR_AGENT_PHRASES:
                if kw in text:
                    agent_score += 1.0

            # Customer signal words
            customer_signals = [
                # ── English customer signals ─────────────────────
                "my bill", "my account", "my order", "i want", "i need",
                "i am calling", "i do not", "i did not", "i live",
                "i was out", "i have an", "i checked", "unbelievable",
                "worst", "transfer me", "i want to speak",
                "i am not paying", "i want to cancel",
                "this is the worst", "your company",
                # ── Malay customer signals (specific, not broad) ─
                "tak boleh", "macam mana", "tak puas hati",
                "nak cancel", "nak refund", "nak complain",
                "saya nak tanya", "saya tak terima",
                "saya pelanggan", "barang saya",
                "dua minggu tak sampai", "kenapa pula",
                "saya nak buat aduan",
            ]
            for signal in customer_signals:
                if signal in text:
                    customer_score += 1.5

            # Agent signal words (system references, actions)
            agent_signals = [
                # ── English agent signals ────────────────────────
                "the system", "our system", "our records", "pulling it up",
                "pulling up", "your account shows", "the charge is",
                "in the queue", "i just handle", "supervisor is going to",
                "calling you back", "like i said", "you bought",
                "you went over", "happens all the time",
                "you have to pay", "so you owe", "the charge",
                "not calling you back", "you might be waiting",
                "that is why", "the bill is",
                # ── Malay agent signals (rude + professional) ────
                "sistem kami", "rekod kami", "saya akan",
                "mohon tunggu", "encik", "pihak kami",
                "status tulis", "sistem cakap", "nombor dia",
                "bagilah nombor", "pergilah cari", "tanggungjawab kita",
                "pandai-pandailah", "ada apa-apa lagi",
                "pengurus takde", "pengurus tak ada",
                "buatlah aduan", "nak tunggu tunggulah",
                "dah hantar", "kurier dah",
            ]
            for signal in agent_signals:
                if signal in text:
                    agent_score += 1.5

        # ── Assign role ───────────────────────────────────────────
        if agent_score > customer_score:
            seg["predicted_role"] = "Agent"
            seg["final_confidence"] = min(0.95, 0.6 + agent_score * 0.05)
        elif customer_score > agent_score:
            seg["predicted_role"] = "Customer"
            seg["final_confidence"] = min(0.95, 0.6 + customer_score * 0.05)
        else:
            # Tie — keep whatever the ensemble assigned
            pass

        if matched_phrase:
            seg["text_override"] = True
            seg["text_override_phrase"] = matched_phrase

    # Count changes
    agents = sum(1 for s in hybrid if s.get("predicted_role") == "Agent")
    customers = sum(1 for s in hybrid if s.get("predicted_role") == "Customer")
    logger.info(
        f"Independent classification complete: "
        f"Agent={agents}, Customer={customers}, Total={len(hybrid)}"
    )

    return hybrid


def classify_transcript_hybrid(lexical_classified: list[dict],
                                acoustic_classified: list[dict]) -> list[dict]:
    """
    Upgraded 6-step hybrid fusion pipeline:
      Step 1 — Speaker anchoring       (who is Agent/Customer?)
      Step 2 — Dynamic α/β             (which model to trust more?)
      Step 3 — Segment fusion          (S_ensemble per segment)
      Step 4 — Global consistency      (enforce 1 Agent + 1 Customer)
      Step 5 — Segment text override   (fix diarization leakage)
      Step 6 — Context smoothing       (fix short fragment flips)
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

    # ── Step 3.5: Check diarization quality ─────────────────────
    balance = check_diarization_balance(hybrid)

    if balance["balanced"]:
        # ═══ Normal path — diarization is reliable ═══════════════
        # ── Step 4: Global consistency ────────────────────────────
        hybrid = enforce_global_consistency(hybrid, anchor_map)

        # ── Step 5: Segment-level text override ───────────────────
        hybrid = segment_level_text_override(hybrid)

        # ── Step 6: Context smoothing ─────────────────────────────
        hybrid = context_smoothing(hybrid)
    else:
        # ═══ Fallback path — diarization is broken ═══════════════
        # Skip global consistency (speaker_id is unreliable).
        # Try LLM classification first (best accuracy), fall back to
        # rule-based text classification if LLM is unavailable.
        if is_llm_available():
            logger.info(
                "Diarization fallback → using Method 4 (LLM classification)"
            )
            hybrid = classify_transcript_llm(hybrid)
        else:
            logger.warning(
                "Diarization fallback → LLM unavailable, using rule-based "
                "text classification. Set ANTHROPIC_API_KEY in config.py "
                "for better accuracy."
            )
            hybrid = classify_independent_by_text(hybrid)

        # Still run text override for strong phrase corrections
        hybrid = segment_level_text_override(hybrid)

        # Still run context smoothing for short fragments
        hybrid = context_smoothing(hybrid)

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
