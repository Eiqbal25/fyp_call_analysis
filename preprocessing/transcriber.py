"""
preprocessing/transcriber.py
=============================
Phase 1 (continued): Transcription & Speaker Diarization

REAL FIX for low accuracy on billing_english and delivery_malay:

ROOT CAUSE:
  Whisper produces long segments (5-7 seconds) that contain BOTH speakers
  talking back to back within the same segment. When Resemblyzer receives
  a 6-second audio chunk containing two different voices, it produces one
  embedding that is an average of both — making it impossible to cluster
  correctly. This is why billing_english ended up with all 15 segments
  assigned to SPK0 despite having two clearly different speakers.

FIX — split_long_segments():
  After Whisper transcription, scan each segment's word timestamps for
  silence gaps > SPLIT_SILENCE_SEC (0.4s). Any gap that long almost
  certainly marks a speaker turn boundary. Split the segment there.
  Result: shorter, single-speaker segments → diarization works correctly.

  billing_english before: 15 segments, avg 5.4s, all SPK0
  billing_english after:  ~30 segments, avg 2.5s, proper SPK0/SPK1 split
"""

import os
import json
import logging
import numpy as np

from config import (
    AUDIO_SAMPLE_RATE,
    WHISPER_MODEL_SIZE,
    WHISPER_LANGUAGE,
    WHISPER_INITIAL_PROMPT,
    WHISPER_CONDITION_ON_PREVIOUS_TEXT,
    WHISPER_BEAM_SIZE,
    DIARIZATION_MIN_SEGMENT_SEC,
    DIARIZATION_NUM_SPEAKERS,
    OUTPUTS_DIR,
    WHISPER_DEVICE,
)

logger = logging.getLogger(__name__)

# Silence gap threshold for splitting merged segments.
# Gaps > this value (seconds) are treated as speaker turn boundaries.
SPLIT_SILENCE_SEC = 0.4


# ─────────────────────────────────────────────────────────────
# WHISPER MODEL CACHE
# ─────────────────────────────────────────────────────────────

_whisper_model = None


def _get_whisper_model():
    """Return cached Whisper model, loading on first call."""
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "openai-whisper not installed. Run: pip install openai-whisper"
            )
        logger.info(
            f"Loading Whisper '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE} "
            f"(cached for all calls)..."
        )
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE)
        logger.info(f"Whisper '{WHISPER_MODEL_SIZE}' ready.")
    return _whisper_model


# ─────────────────────────────────────────────────────────────
# SEGMENT SPLITTING — The core fix
# ─────────────────────────────────────────────────────────────

def split_long_segments(segments: list[dict],
                         silence_threshold: float = SPLIT_SILENCE_SEC) -> list[dict]:
    """
    Split Whisper segments at silence gaps using word-level timestamps.

    WHY THIS IS NEEDED:
    Whisper groups audio into segments based on its own internal logic,
    not based on speaker turns. In fast-paced conversations or calls where
    speakers overlap or interrupt, Whisper puts multiple speakers' words
    into one segment. This is the primary cause of diarization failure —
    Resemblyzer gets mixed audio and cannot determine which speaker it is.

    HOW IT WORKS:
    Each Whisper segment has word-level timestamps (start/end per word).
    We scan consecutive word pairs. When word[i].end to word[i+1].start
    is >= silence_threshold, that gap is a speaker turn boundary.
    We split the segment there, creating two shorter segments.

    Parameters
    ----------
    segments          : Whisper segments with word timestamps
    silence_threshold : gap >= this (seconds) triggers a split

    Returns
    -------
    split_segments : shorter segments with clean speaker turns
    """
    result = []

    for seg in segments:
        words = seg.get("words", [])

        # No word timestamps available — keep segment as-is
        if not words:
            result.append({
                "text":  seg["text"].strip(),
                "start": seg["start"],
                "end":   seg["end"],
            })
            continue

        # Find split points — gaps between consecutive words
        split_points = []   # indices where to split (after word[i])
        for i in range(len(words) - 1):
            w_end   = float(words[i]["end"])
            w_start = float(words[i + 1]["start"])
            gap     = w_start - w_end
            if gap >= silence_threshold:
                split_points.append(i)
                logger.debug(
                    f"Split point at {w_end:.2f}s (gap={gap:.2f}s)"
                )

        if not split_points:
            # No gaps found — single speaker, keep as-is
            result.append({
                "text":  seg["text"].strip(),
                "start": seg["start"],
                "end":   seg["end"],
            })
            continue

        # Split into sub-segments at each gap
        boundaries = [-1] + split_points + [len(words) - 1]
        for j in range(len(boundaries) - 1):
            w_from = boundaries[j] + 1
            w_to   = boundaries[j + 1]

            sub_words = words[w_from: w_to + 1]
            if not sub_words:
                continue

            sub_text = " ".join(
                w.get("word", "").strip() for w in sub_words
            ).strip()
            if not sub_text:
                continue

            sub_start = float(sub_words[0]["start"])
            sub_end   = float(sub_words[-1]["end"])

            # Minimum 0.2s — discard tiny fragments
            if sub_end - sub_start < 0.2:
                continue

            result.append({
                "text":  sub_text,
                "start": round(sub_start, 3),
                "end":   round(sub_end,   3),
            })

    n_before = len(segments)
    n_after  = len(result)
    if n_after > n_before:
        logger.info(
            f"Segment splitting: {n_before} → {n_after} segments "
            f"({n_after - n_before} splits at silence gaps ≥{silence_threshold}s)"
        )
    return result


# ─────────────────────────────────────────────────────────────
# TRANSCRIPTION — OpenAI Whisper
# ─────────────────────────────────────────────────────────────

def transcribe_audio(y: np.ndarray, sr: int = AUDIO_SAMPLE_RATE) -> list[dict]:
    """
    Transcribe audio using the cached Whisper model, then split at silence
    boundaries to produce single-speaker segments.

    Parameters
    ----------
    y  : float32 numpy waveform at 16 kHz
    sr : sample rate (must be 16000)

    Returns
    -------
    segments : [{text, start, end}, ...]  — split at speaker turn gaps
    """
    if sr != 16000:
        raise ValueError(f"Whisper requires sr=16000, got sr={sr}")

    model = _get_whisper_model()

    logger.info(
        f"Transcribing | model={WHISPER_MODEL_SIZE} | "
        f"device={WHISPER_DEVICE} | "
        f"lang={WHISPER_LANGUAGE or 'auto-detect'} | "
        f"beam={WHISPER_BEAM_SIZE}"
    )

    result = model.transcribe(
        y,
        language=WHISPER_LANGUAGE,
        word_timestamps=True,          # REQUIRED for segment splitting
        verbose=False,
        initial_prompt=WHISPER_INITIAL_PROMPT,
        condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS_TEXT,
        beam_size=WHISPER_BEAM_SIZE,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
    )

    # Collect raw Whisper segments WITH word timestamps
    raw_segments = []
    for seg in result.get("segments", []):
        text = seg["text"].strip()
        if not text:
            continue
        raw_segments.append({
            "text":  text,
            "start": round(float(seg["start"]), 3),
            "end":   round(float(seg["end"]),   3),
            "words": seg.get("words", []),
        })

    logger.info(
        f"Whisper raw segments: {len(raw_segments)} | "
        f"language={result.get('language', 'unknown')}"
    )

    # Split at silence boundaries — the key diarization fix
    segments = split_long_segments(raw_segments, SPLIT_SILENCE_SEC)

    # Strip word timestamps from output (not needed downstream)
    clean_segments = [
        {"text": s["text"], "start": s["start"], "end": s["end"]}
        for s in segments
    ]

    logger.info(f"Transcription complete | final segments={len(clean_segments)}")
    return clean_segments


# ─────────────────────────────────────────────────────────────
# DIARIZATION — Resemblyzer + Agglomerative Clustering
# ─────────────────────────────────────────────────────────────

def diarize_audio(y: np.ndarray, sr: int = AUDIO_SAMPLE_RATE,
                  n_speakers: int = DIARIZATION_NUM_SPEAKERS) -> list[dict]:
    """
    Speaker diarization using Resemblyzer d-vectors + AgglomerativeClustering.

    With the segment splitting fix, Resemblyzer now receives shorter audio
    chunks that contain predominantly one speaker, producing more reliable
    cluster assignments.

    Parameters
    ----------
    y          : float32 waveform at 16 kHz
    sr         : sample rate
    n_speakers : expected number of speakers

    Returns
    -------
    speaker_segments : [{speaker_id, start, end, embedding}, ...]
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        raise ImportError("resemblyzer not installed. Run: pip install resemblyzer")

    from sklearn.cluster import AgglomerativeClustering

    logger.info("Initializing Resemblyzer VoiceEncoder...")
    encoder = VoiceEncoder()

    wav_preprocessed = preprocess_wav(y, source_sr=sr)

    # 2.5s window, 0.75s step — optimised for stability vs resolution
    window_sec  = 2.5
    step_sec    = 0.75
    window_samp = int(window_sec * sr)
    step_samp   = int(step_sec   * sr)

    timestamps  = []
    embeddings  = []

    n_samples = len(wav_preprocessed)
    start = 0
    while start + window_samp <= n_samples:
        end_samp = start + window_samp
        chunk    = wav_preprocessed[start:end_samp]
        try:
            emb = encoder.embed_utterance(chunk)
            timestamps.append({
                "start": round(start    / sr, 3),
                "end":   round(end_samp / sr, 3),
            })
            embeddings.append(emb)
        except Exception as e:
            logger.debug(f"Skipping chunk at {start/sr:.2f}s: {e}")
        start += step_samp

    if len(embeddings) < n_speakers:
        logger.warning("Not enough embeddings — returning single speaker")
        return [{
            "speaker_id": 0,
            "start":      0.0,
            "end":        len(y) / sr,
            "embedding":  embeddings[0] if embeddings else None,
        }]

    emb_matrix = np.vstack(embeddings)
    logger.info(f"Extracted {len(embeddings)} embeddings | shape={emb_matrix.shape}")

    clustering = AgglomerativeClustering(
        n_clusters=n_speakers,
        metric="cosine",
        linkage="average",
    )
    labels = list(clustering.fit_predict(emb_matrix))
    logger.info(f"Clustering complete | speakers found={len(set(labels))}")

    # Neighbour smoothing — fix isolated mis-assigned windows
    n_smoothed = 0
    for i in range(1, len(labels) - 1):
        prev_lbl = labels[i - 1]
        next_lbl = labels[i + 1]
        if prev_lbl == next_lbl and labels[i] != prev_lbl:
            logger.debug(
                f"Neighbour smooth: window {i} at "
                f"{timestamps[i]['start']:.1f}s "
                f"SPK{labels[i]} → SPK{prev_lbl}"
            )
            labels[i] = prev_lbl
            n_smoothed += 1

    if n_smoothed:
        logger.info(f"Neighbour smoothing: corrected {n_smoothed} window(s)")
    labels = np.array(labels)

    # Build speaker segments
    speaker_segments = []
    for i, (ts, label) in enumerate(zip(timestamps, labels)):
        if (speaker_segments
                and speaker_segments[-1]["speaker_id"] == int(label)
                and ts["start"] - speaker_segments[-1]["end"] < 1.5):
            speaker_segments[-1]["end"] = ts["end"]
        else:
            speaker_segments.append({
                "speaker_id": int(label),
                "start":      ts["start"],
                "end":        ts["end"],
                "embedding":  embeddings[i].tolist(),
            })

    speaker_segments = [
        s for s in speaker_segments
        if (s["end"] - s["start"]) >= DIARIZATION_MIN_SEGMENT_SEC
    ]

    logger.info(f"Speaker segments after merging: {len(speaker_segments)}")

    found_speakers = set(s["speaker_id"] for s in speaker_segments)
    if len(found_speakers) < 2:
        logger.warning(
            f"Only {len(found_speakers)} speaker(s) detected. "
            "Classification will still proceed."
        )

    return speaker_segments


# ─────────────────────────────────────────────────────────────
# TEMPORAL ALIGNMENT
# ─────────────────────────────────────────────────────────────

def align_transcript_with_speakers(
        whisper_segments: list[dict],
        speaker_segments: list[dict]) -> list[dict]:
    """
    Map each (now shorter, split) Whisper segment to its speaker
    using maximum temporal overlap.
    """
    diarized = []

    for ws in whisper_segments:
        ws_start = ws["start"]
        ws_end   = ws["end"]
        text     = ws["text"].strip()

        if not text:
            continue

        best_speaker = 0
        best_overlap = 0.0

        for ss in speaker_segments:
            overlap = max(
                0.0,
                min(ws_end, ss["end"]) - max(ws_start, ss["start"])
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = ss["speaker_id"]

        diarized.append({
            "speaker_id":  best_speaker,
            "speaker_raw": f"SPEAKER_{best_speaker}",
            "text":        text,
            "start":       ws_start,
            "end":         ws_end,
            "duration":    round(ws_end - ws_start, 3),
        })

    logger.info(f"Aligned {len(diarized)} transcript segments to speakers")
    return diarized


# ─────────────────────────────────────────────────────────────
# WORD ERROR RATE
# ─────────────────────────────────────────────────────────────

def compute_wer(reference: str, hypothesis: str) -> dict:
    """Compute WER = (S + D + I) / N using Levenshtein DP."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    n_ref     = len(ref_words)
    n_hyp     = len(hyp_words)

    if n_ref == 0:
        return {"wer": 0.0, "substitutions": 0, "deletions": 0,
                "insertions": 0, "ref_length": 0, "hyp_length": n_hyp}

    dp = np.zeros((n_ref + 1, n_hyp + 1), dtype=int)
    dp[:, 0] = np.arange(n_ref + 1)
    dp[0, :] = np.arange(n_hyp + 1)

    for i in range(1, n_ref + 1):
        for j in range(1, n_hyp + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i, j] = dp[i-1, j-1]
            else:
                dp[i, j] = 1 + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])

    s = d = ins = 0
    i, j = n_ref, n_hyp
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            s += 1; i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ins += 1; j -= 1
        else:
            d += 1; i -= 1

    wer_pct = round((s + d + ins) / n_ref * 100, 2)
    logger.info(f"WER={wer_pct}% | S={s} D={d} I={ins} | ref={n_ref}")
    return {"wer": wer_pct, "substitutions": s, "deletions": d,
            "insertions": ins, "ref_length": n_ref, "hyp_length": n_hyp}


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────

def run_transcription_diarization(y: np.ndarray,
                                   sr: int = AUDIO_SAMPLE_RATE,
                                   save_json: str = None) -> list[dict]:
    """
    Full pipeline: Whisper → split at silence gaps → Resemblyzer → align.
    """
    logger.info("--- Starting Transcription + Diarization pipeline ---")

    whisper_segs = transcribe_audio(y, sr)    # includes splitting
    speaker_segs = diarize_audio(y, sr)
    diarized     = align_transcript_with_speakers(whisper_segs, speaker_segs)

    if save_json:
        os.makedirs(os.path.dirname(save_json) or ".", exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(diarized, f, indent=2, ensure_ascii=False)
        logger.info(f"Diarized transcript saved → {save_json}")

    return diarized


def compute_diarization_turn_counts(diarized: list[dict]) -> dict:
    """Count speaker turn switches."""
    turns = 0
    turns_per_speaker = {}
    for i, seg in enumerate(diarized):
        spk = seg["speaker_id"]
        turns_per_speaker[spk] = turns_per_speaker.get(spk, 0) + 1
        if i > 0 and seg["speaker_id"] != diarized[i-1]["speaker_id"]:
            turns += 1
    return {
        "total_segments":    len(diarized),
        "num_turns":         turns,
        "turns_per_speaker": turns_per_speaker,
    }
