"""
methods/method4_llm.py
========================
Method 4 — LLM-Based Speaker Role Classification (FREE via Google Gemini)

Uses Google's Gemini model to classify each transcript segment as Agent
or Customer based on conversational context.

COST: FREE — Google AI Studio provides free API access to Gemini models.

SETUP:
  1. pip install google-generativeai
  2. Get a free API key at: https://aistudio.google.com/app/apikey
  3. Set GEMINI_API_KEY in config.py

WHY THIS EXISTS:
  Methods 1-3 rely on keywords, acoustic features, and ensemble fusion.
  They fail when:
    - The agent speaks informally/rudely (no SOP keywords to detect)
    - Diarization is broken (both speakers in one cluster)
    - Short fragments have no signal

  An LLM understands conversational CONTEXT — it knows "The system does
  not lie, so you have to pay it" is an agent speaking even without any
  keyword list.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# ── Try to import Google Generative AI ────────────────────────
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning(
        "google-generativeai package not installed. "
        "Run: pip install google-generativeai"
    )

try:
    from config import GEMINI_API_KEY, LLM_MODEL
except ImportError:
    GEMINI_API_KEY = None
    LLM_MODEL = "gemini-2.0-flash"


# ─────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert at analyzing customer service call transcripts.
Your task is to label each numbered segment as either "Agent" or "Customer".

RULES:
- The Agent works at the company handling the call
- The Customer is the person calling for help, to complain, or to inquire
- Agents typically: greet callers, verify identity, reference company systems/policies, offer solutions, transfer calls, close calls
- Customers typically: explain problems, provide personal info, express frustration, ask questions, make requests
- An agent can be rude, informal, or unprofessional — they are still the Agent
- In customer service calls, the person who PICKS UP the phone is always the Agent
- Short fragments like "Yes", "Fine", "Whatever" should be assigned based on conversational context

Respond with ONLY a valid JSON array. No explanation, no markdown, no backticks.
Each element must have "id" (the segment number) and "role" ("Agent" or "Customer").
Example: [{"id": 1, "role": "Agent"}, {"id": 2, "role": "Customer"}]"""


# ─────────────────────────────────────────────────────────────
# CORE CLASSIFICATION
# ─────────────────────────────────────────────────────────────

def classify_transcript_llm(segments: list[dict]) -> list[dict]:
    """
    Classify each transcript segment using Google Gemini (free).

    Parameters
    ----------
    segments : list of dicts with at least "text", "start", "end" keys.

    Returns
    -------
    Same segments with "predicted_role", "confidence", and "method" added.
    Returns original segments unchanged if LLM is unavailable.
    """
    if not GEMINI_AVAILABLE:
        logger.error("Method 4 failed: google-generativeai package not installed")
        return segments

    if not GEMINI_API_KEY:
        logger.error("Method 4 failed: GEMINI_API_KEY not set in config.py")
        return segments

    if not segments:
        return segments

    # ── Build transcript text ─────────────────────────────────
    transcript_lines = []
    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        start = seg.get("start", 0)
        if text:
            transcript_lines.append(f"{i+1}. [{start:.1f}s] {text}")

    transcript_text = "\n".join(transcript_lines)

    user_prompt = f"""{SYSTEM_PROMPT}

Classify each segment in this customer service call transcript:

{transcript_text}"""

    # ── Call Gemini ───────────────────────────────────────────
    logger.info(f"Method 4: Sending {len(segments)} segments to LLM ({LLM_MODEL})...")

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(LLM_MODEL)

        response = model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=2000,
            ),
        )

        raw_text = response.text.strip()
        logger.info(f"Method 4: LLM response received ({len(raw_text)} chars)")

    except Exception as e:
        logger.error(f"Method 4: LLM API call failed: {e}")
        return segments

    # ── Parse response ────────────────────────────────────────
    try:
        cleaned = raw_text
        cleaned = re.sub(r'^```json\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        cleaned = cleaned.strip()

        labels = json.loads(cleaned)

        if not isinstance(labels, list):
            raise ValueError(f"Expected list, got {type(labels)}")

        role_map = {}
        for item in labels:
            seg_id = item.get("id", 0)
            role = item.get("role", "Unknown")
            if role in ("Agent", "Customer"):
                role_map[seg_id] = role

        logger.info(f"Method 4: Parsed {len(role_map)} labels from LLM")

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Method 4: Failed to parse LLM response: {e}")
        logger.debug(f"Raw response: {raw_text[:500]}")
        return segments

    # ── Apply labels ──────────────────────────────────────────
    applied = 0
    for i, seg in enumerate(segments):
        seg_id = i + 1
        if seg_id in role_map:
            seg["predicted_role"] = role_map[seg_id]
            seg["confidence"] = 0.90
            seg["final_confidence"] = 0.90
            seg["method"] = "llm"
            seg["llm_classified"] = True
            applied += 1

    logger.info(
        f"Method 4 complete: {applied}/{len(segments)} segments classified by LLM"
    )
    return segments


# ─────────────────────────────────────────────────────────────
# AVAILABILITY CHECK
# ─────────────────────────────────────────────────────────────

def is_llm_available() -> bool:
    """Check if Method 4 can be used."""
    return GEMINI_AVAILABLE and bool(GEMINI_API_KEY)


def get_llm_status() -> dict:
    """Return status info for logging/dashboard."""
    return {
        "available": is_llm_available(),
        "package_installed": GEMINI_AVAILABLE,
        "api_key_set": bool(GEMINI_API_KEY),
        "model": LLM_MODEL,
    }
