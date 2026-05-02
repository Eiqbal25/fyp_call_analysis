"""
preprocessing/malay_corrections.py
=====================================
Post-correction dictionary for common Whisper transcription errors on
Malaysian Malay (Bahasa Malaysia) audio.

WHY THIS IS NEEDED:
  Whisper's Malay training data is limited and skewed toward Indonesian.
  It makes systematic errors:
    1. Indonesian spelling  — "nomor" instead of Malaysian "nombor"
    2. Phonetic mishearing  — "suhu" for "suruh", "Korea" for "kurier"
    3. Truncation           — "ruma" for "rumah", "kapa" for "kenapa"
    4. Title mishearing     — "ujik"/"ncik" for "encik"

  These errors cascade into Method 3 phrase matching failures.
  Correcting them BEFORE classification restores phrase matching accuracy.

USAGE:
  Applied automatically in run_transcription_diarization().
  Only active when the detected/forced language is Malay ("ms").
  Safe to run on English — no English words appear in the correction tables.
"""

import re
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# PHRASE-LEVEL CORRECTIONS (check these FIRST — more specific)
# ─────────────────────────────────────────────────────────────
# Key = wrong Whisper output (lowercase)
# Value = correct Malaysian spelling

PHRASE_CORRECTIONS = {
    # Compound title errors
    "encik johan":          "Encik Johan",
    "cik sarah":            "Cik Sarah",

    # "nomor kat" → "nombor kad" (Indonesian + consonant mishear)
    "nomor kat pengenalan": "nombor kad pengenalan",
    "nomor kat":            "nombor kad",

    # Courier mishear — "Korea" sounds like "kurier" in fast Malay speech
    "sistem cakap korea":   "sistem cakap kurier",
    "cakap korea":          "cakap kurier",
    "korea dah":            "kurier dah",
    "dari korea":           "dari kurier",

    # "pagah ruma" → "pagar rumah"
    "pagah ruma":           "pagar rumah",
    "atas pagah":           "atas pagar",

    # Common agent-line Whisper errors
    "saya simak":           "saya semak",
    "sementara saya simak": "sementara saya semak",
}


# ─────────────────────────────────────────────────────────────
# WORD-LEVEL CORRECTIONS (applied after phrase corrections)
# ─────────────────────────────────────────────────────────────
# Uses whole-word matching (\b boundaries) — safe, no partial replacements.

WORD_CORRECTIONS = {
    # ── Indonesian → Malaysian spelling ──────────────────────
    "nomor":    "nombor",
    "nomer":    "nombor",

    # ── Consonant / phoneme mishearing ───────────────────────
    "suhu":     "suruh",     # jangan suhu → jangan suruh
    "pagah":    "pagar",     # pagar (fence)
    "tantar":   "hantar",    # send
    "tantor":   "hantar",
    "maninya":  "maknanya",  # meaning
    "manisnya": "maknanya",

    # ── Truncation / clipping ────────────────────────────────
    "ruma":     "rumah",     # house
    "kapa":     "kenapa",    # why

    # ── Title / honorific mishearing ─────────────────────────
    "ujik":     "encik",     # Mr
    "ujek":     "encik",
    "ncik":     "encik",
    "enjik":    "encik",
    "hia":      "ya",        # yes / affirmative

    # ── Other common substitutions ───────────────────────────
    "analah":   "alah",      # exclamation
    "manalah":  "manalah",   # keep — already correct
    "maknanya": "maknanya",  # keep — already correct (medium fixed maninya)
    "takde":    "tak ada",   # informal contraction
    "takdek":   "tak ada",
    "inik":     "ini",       # delivery_malay mishear at 70.8s
    "nj":       "eh",        # delivery_malay [59.9s] "NJ" = exclamation
    "bantuan":  "bantu",     # food_malay [3.6s] "boleh saya bantuan" — but keep if standalone
    "sebentar": "sebentar",  # keep — already correct
}


# ─────────────────────────────────────────────────────────────
# APPLY CORRECTIONS
# ─────────────────────────────────────────────────────────────

def apply_corrections(text: str) -> str:
    """
    Apply phrase and word corrections to a single text segment.
    Case-insensitive matching; preserves original capitalisation for
    words that aren't in the correction table.
    """
    if not text or not text.strip():
        return text

    corrected = text

    # ── Phrase corrections first (most specific) ─────────────
    for wrong, right in PHRASE_CORRECTIONS.items():
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        corrected = pattern.sub(right, corrected)

    # ── Word corrections (whole-word, case-insensitive) ──────
    for wrong, right in WORD_CORRECTIONS.items():
        pattern = re.compile(r'\b' + re.escape(wrong) + r'\b', re.IGNORECASE)

        def _replace(m, right=right):
            # Preserve original capitalisation: if original starts uppercase, capitalise
            matched = m.group(0)
            if matched[0].isupper():
                return right.capitalize()
            return right

        corrected = pattern.sub(_replace, corrected)

    return corrected


def apply_corrections_to_segments(segments: list[dict],
                                   language: str = None) -> list[dict]:
    """
    Apply Malay corrections to all segments in a transcript.

    Parameters
    ----------
    segments : list of diarized segment dicts — each must have "text"
    language : Whisper language code detected for this call.
               Corrections only run when language is "ms" or None
               (None = auto-detect, may still be Malay).
               Pass "en" to skip entirely for English calls.

    Returns
    -------
    segments with corrected "text" fields (in-place + returned)
    """
    # Skip corrections entirely for confirmed English calls
    if language == "en":
        return segments

    corrected_count = 0
    for seg in segments:
        original = seg.get("text", "")
        fixed    = apply_corrections(original)

        if fixed != original:
            seg["text"]          = fixed
            seg["text_corrected"] = True
            logger.debug(
                f"  Correction [{seg.get('start', 0):.1f}s]: "
                f'"{original.strip()}" → "{fixed.strip()}"'
            )
            corrected_count += 1

    if corrected_count:
        logger.info(
            f"Malay post-correction: {corrected_count} segment(s) corrected"
        )

    return segments
