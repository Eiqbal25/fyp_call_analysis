"""
preprocessing/transcriber.py
=============================
Phase 1 (continued): Transcription & Speaker Diarization

UPGRADE: Resemblyzer → pyannote.audio
======================================

WHY RESEMBLYZER FAILED on billing_english and delivery_malay:
  Resemblyzer uses a sliding window (2.5s) to extract d-vector embeddings,
  then clusters them with AgglomerativeClustering. When the very first window
  captures BOTH speakers talking (agent says something, customer immediately
  replies within the same 2.5s window), Resemblyzer embeds that mixed-voice
  block and uses it as the centroid for SPK0. Every subsequent segment then
  gets attracted to this contaminated centroid regardless of whose voice it is.
  Result: 30/31 segments in billing_english = SPK0. Completely unusable.

WHY pyannote.audio IS BETTER:
  pyannote uses a trained neural segmentation model
  (pyannote/segmentation-3.0) that was specifically designed for:
  - Overlapping speech detection
  - Rapid turn-taking (agents interrupting customers)
  - Short utterances (< 1 second)
  - Noisy telephone-quality audio
  It processes the entire audio at once with learned speaker representations,
  not a naive sliding window average.
  Expected DER improvement: ~30% → ~8% on typical call recordings.

SETUP REQUIRED (one time only):
  1. pip install pyannote.audio
  2. Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1
  3. Accept license at https://huggingface.co/pyannote/segmentation-3.0
  4. Add your HuggingFace token to config.py:
     HUGGINGFACE_TOKEN = "hf_your_token_here"

FALLBACK:
  If pyannote fails for any reason (no token, no internet, model load error),
  the system automatically falls back to the old Resemblyzer diarization.
  Processing continues — you will see a WARNING in the log.
"""

import os
import json
import logging
import tempfile
import numpy as np
import soundfile as sf

from config import (
    AUDIO_SAMPLE_RATE,
    WHISPER_MODEL_SIZE,
    WHISPER_LANGUAGE,
    WHISPER_LANGUAGE_MAP,
    WHISPER_INITIAL_PROMPT,
    WHISPER_CONDITION_ON_PREVIOUS_TEXT,
    WHISPER_BEAM_SIZE,
    DIARIZATION_MIN_SEGMENT_SEC,
    DIARIZATION_NUM_SPEAKERS,
    OUTPUTS_DIR,
    CALLS_DIR,
    get_call_dir,
    WHISPER_DEVICE,
    HUGGINGFACE_TOKEN,
)
from preprocessing.malay_corrections import apply_corrections_to_segments

logger = logging.getLogger(__name__)

# Gap between words (seconds) that triggers a segment split
SPLIT_SILENCE_SEC = 0.35


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
# PYANNOTE PIPELINE CACHE
# ─────────────────────────────────────────────────────────────

_pyannote_pipeline = None
_pyannote_failed   = False   # if True, use Resemblyzer fallback


def _get_pyannote_pipeline():
    """
    Return cached pyannote diarization pipeline.
    Returns None if pyannote is unavailable or token is missing.
    """
    global _pyannote_pipeline, _pyannote_failed

    if _pyannote_failed:
        return None

    if _pyannote_pipeline is not None:
        return _pyannote_pipeline

    if not HUGGINGFACE_TOKEN:
        logger.warning(
            "HUGGINGFACE_TOKEN not set in config.py. "
            "Falling back to Resemblyzer diarization. "
            "To use pyannote, add: HUGGINGFACE_TOKEN = 'hf_...' to config.py"
        )
        _pyannote_failed = True
        return None

    try:
        from pyannote.audio import Pipeline
        from huggingface_hub import login as hf_login
        import torch

        hf_login(token=HUGGINGFACE_TOKEN, add_to_git_credential=False)

        logger.info("Loading pyannote speaker-diarization-3.1 (first load — cached)...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
        
        )

        # Move to GPU if available
        if WHISPER_DEVICE in ("cuda", "mps"):
            import torch
            device = torch.device(WHISPER_DEVICE)
            pipeline = pipeline.to(device)
            logger.info(f"pyannote pipeline moved to {WHISPER_DEVICE}")

        _pyannote_pipeline = pipeline
        logger.info("pyannote speaker-diarization-3.1 ready.")
        return _pyannote_pipeline

    except Exception as e:
        logger.warning(
            f"pyannote failed to load: {e}\n"
            "Falling back to Resemblyzer diarization."
        )
        _pyannote_failed = True
        return None


# ─────────────────────────────────────────────────────────────
# SEGMENT SPLITTING (keep for Whisper transcript quality)
# ─────────────────────────────────────────────────────────────

def split_long_segments(segments: list[dict],
                         silence_threshold: float = SPLIT_SILENCE_SEC) -> list[dict]:
    """
    Split Whisper segments at silence gaps using word-level timestamps.
    This improves temporal alignment even when pyannote handles diarization.
    """
    result = []

    for seg in segments:
        words = seg.get("words", [])

        if not words:
            result.append({
                "text":  seg["text"].strip(),
                "start": seg["start"],
                "end":   seg["end"],
            })
            continue

        split_points = []
        for i in range(len(words) - 1):
            w_end   = float(words[i]["end"])
            w_start = float(words[i + 1]["start"])
            if w_start - w_end >= silence_threshold:
                split_points.append(i)

        if not split_points:
            result.append({
                "text":  seg["text"].strip(),
                "start": seg["start"],
                "end":   seg["end"],
            })
            continue

        boundaries = [-1] + split_points + [len(words) - 1]
        for j in range(len(boundaries) - 1):
            w_from    = boundaries[j] + 1
            w_to      = boundaries[j + 1]
            sub_words = words[w_from: w_to + 1]
            if not sub_words:
                continue
            sub_text  = " ".join(w.get("word", "").strip() for w in sub_words).strip()
            sub_start = float(sub_words[0]["start"])
            sub_end   = float(sub_words[-1]["end"])
            if not sub_text or sub_end - sub_start < 0.15:
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
            f"Segment splitting: {n_before} → {n_after} "
            f"({n_after - n_before} splits)"
        )
    return result


# ─────────────────────────────────────────────────────────────
# TRANSCRIPTION — OpenAI Whisper
# ─────────────────────────────────────────────────────────────

def transcribe_audio(y: np.ndarray, sr: int = AUDIO_SAMPLE_RATE,
                     language: str = None) -> list[dict]:
    """
    Transcribe audio using the cached Whisper model.

    Returns
    -------
    segments : [{text, start, end}, ...]
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

    # Use per-call language if provided, otherwise fall back to config
    lang_to_use = language if language is not None else WHISPER_LANGUAGE
    logger.info(f"  Language: {lang_to_use or 'auto-detect'}")

    result = model.transcribe(
        y,
        language=lang_to_use,
        word_timestamps=True,
        verbose=False,
        initial_prompt=WHISPER_INITIAL_PROMPT,
        condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS_TEXT,
        beam_size=WHISPER_BEAM_SIZE,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
    )

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
        f"Whisper raw: {len(raw_segments)} segments | "
        f"language={result.get('language', 'unknown')}"
    )

    segments     = split_long_segments(raw_segments, SPLIT_SILENCE_SEC)
    clean_segs   = [
        {"text": s["text"], "start": s["start"], "end": s["end"]}
        for s in segments
    ]
    logger.info(f"Transcription complete | {len(clean_segs)} segments after splitting")
    return clean_segs


# ─────────────────────────────────────────────────────────────
# DIARIZATION — pyannote.audio (PRIMARY)
# ─────────────────────────────────────────────────────────────

def diarize_audio_pyannote(y: np.ndarray, sr: int = AUDIO_SAMPLE_RATE,
                            n_speakers: int = DIARIZATION_NUM_SPEAKERS) -> list[dict]:
    """
    Speaker diarization using pyannote/speaker-diarization-3.1.

    pyannote returns a timeline of {speaker_label: time_range} annotations.
    We convert these to the same [{speaker_id, start, end}] format as before
    so the rest of the pipeline is completely unchanged.

    Parameters
    ----------
    y          : float32 waveform at 16 kHz
    sr         : sample rate
    n_speakers : expected number of speakers (2 for Agent + Customer)

    Returns
    -------
    speaker_segments : [{speaker_id, start, end}]
    """
    pipeline = _get_pyannote_pipeline()
    if pipeline is None:
        # Fallback to Resemblyzer
        return diarize_audio_resemblyzer(y, sr, n_speakers)

    # pyannote requires a WAV file or dict with waveform tensor
    # We use a temporary WAV file to avoid torch tensor conversion issues
    try:
        import torch
        from pyannote.audio import Audio

        logger.info(
            f"Running pyannote diarization | "
            f"n_speakers={n_speakers} | "
            f"duration={len(y)/sr:.1f}s"
        )

        # Write temp WAV — pyannote accepts file paths
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, y, sr)

        try:
            # Run diarization with known number of speakers for better accuracy
            diarization = pipeline(
                tmp_path,
                num_speakers=n_speakers,
            )
        finally:
            os.unlink(tmp_path)   # clean up temp file

        # Convert pyannote annotation to [{speaker_id, start, end}] format
        # pyannote uses string labels like "SPEAKER_00", "SPEAKER_01"
        # We map these to integer IDs 0, 1
        label_to_id = {}
        speaker_segments = []

        for turn, _, label in diarization.itertracks(yield_label=True):
            if label not in label_to_id:
                label_to_id[label] = len(label_to_id)

            spk_id = label_to_id[label]
            start  = round(float(turn.start), 3)
            end    = round(float(turn.end),   3)

            if end - start < DIARIZATION_MIN_SEGMENT_SEC:
                continue

            # Merge consecutive same-speaker segments (gap < 0.5s)
            if (speaker_segments
                    and speaker_segments[-1]["speaker_id"] == spk_id
                    and start - speaker_segments[-1]["end"] < 0.5):
                speaker_segments[-1]["end"] = end
            else:
                speaker_segments.append({
                    "speaker_id": spk_id,
                    "start":      start,
                    "end":        end,
                })

        found = set(s["speaker_id"] for s in speaker_segments)
        logger.info(
            f"pyannote diarization complete | "
            f"{len(speaker_segments)} segments | "
            f"speakers found={len(found)} | "
            f"label_map={label_to_id}"
        )

        if len(found) < 2:
            logger.warning(
                f"pyannote found only {len(found)} speaker(s). "
                "This may indicate a mono-channel recording or very short audio."
            )

        return speaker_segments

    except Exception as e:
        logger.error(
            f"pyannote diarization failed: {e}\n"
            "Falling back to Resemblyzer."
        )
        return diarize_audio_resemblyzer(y, sr, n_speakers)


# ─────────────────────────────────────────────────────────────
# DIARIZATION — Resemblyzer (FALLBACK)
# ─────────────────────────────────────────────────────────────

def diarize_audio_resemblyzer(y: np.ndarray, sr: int = AUDIO_SAMPLE_RATE,
                               n_speakers: int = DIARIZATION_NUM_SPEAKERS) -> list[dict]:
    """
    Fallback: Resemblyzer + AgglomerativeClustering.
    Used when pyannote is unavailable.
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        raise ImportError("resemblyzer not installed. Run: pip install resemblyzer")

    from sklearn.cluster import AgglomerativeClustering

    logger.info("Using Resemblyzer diarization (fallback)...")
    encoder          = VoiceEncoder()
    wav_preprocessed = preprocess_wav(y, source_sr=sr)

    window_sec  = 2.5
    step_sec    = 0.75
    window_samp = int(window_sec * sr)
    step_samp   = int(step_sec   * sr)

    timestamps  = []
    embeddings  = []
    n_samples   = len(wav_preprocessed)
    start       = 0

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
        return [{"speaker_id": 0, "start": 0.0, "end": len(y) / sr}]

    emb_matrix = np.vstack(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=n_speakers, metric="cosine", linkage="average"
    )
    labels = list(clustering.fit_predict(emb_matrix))

    # Neighbour smoothing
    for i in range(1, len(labels) - 1):
        if labels[i-1] == labels[i+1] and labels[i] != labels[i-1]:
            labels[i] = labels[i-1]
    labels = np.array(labels)

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
            })

    speaker_segments = [
        s for s in speaker_segments
        if (s["end"] - s["start"]) >= DIARIZATION_MIN_SEGMENT_SEC
    ]

    logger.info(
        f"Resemblyzer fallback complete | "
        f"{len(speaker_segments)} segments | "
        f"speakers={set(s['speaker_id'] for s in speaker_segments)}"
    )
    return speaker_segments


# ─────────────────────────────────────────────────────────────
# TEMPORAL ALIGNMENT — Whisper text + Speaker segments
# ─────────────────────────────────────────────────────────────

def align_transcript_with_speakers(
        whisper_segments: list[dict],
        speaker_segments: list[dict]) -> list[dict]:
    """
    Assign each Whisper text segment to a speaker using max temporal overlap.
    Works identically whether speaker_segments came from pyannote or Resemblyzer.
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
                                   save_json: str = None,
                                   language: str = None) -> list[dict]:
    """
    Full pipeline:
      1. Whisper transcription + word-level segment splitting
      2. pyannote.audio diarization (falls back to Resemblyzer if unavailable)
      3. Temporal alignment of transcription + speaker labels
      4. Malay post-correction (only when language="ms" or auto-detect)

    Parameters
    ----------
    language : Whisper language code for this specific call.
               Detected automatically from filename in main.py.
               "ms" = force Malay, "en" = force English, None = auto-detect.
    """
    logger.info("--- Starting Transcription + Diarization pipeline ---")

    whisper_segs = transcribe_audio(y, sr, language=language)
    speaker_segs = diarize_audio_pyannote(y, sr)    # uses pyannote or fallback
    diarized     = align_transcript_with_speakers(whisper_segs, speaker_segs)

    # Apply Malay post-correction before saving
    diarized = apply_corrections_to_segments(diarized, language=language)

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


# Backward-compatibility alias — keeps preprocessing/__init__.py working
diarize_audio = diarize_audio_pyannote
