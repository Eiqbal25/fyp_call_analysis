"""
preprocessing/
==============
Phase 1: Data Acquisition and Pre-processing

Modules:
    audio_processor  — DSP: noise reduction, normalization, SNR calculation
    transcriber      — Whisper ASR + Resemblyzer diarization + WER
"""

from preprocessing.audio_processor import preprocess_audio, compute_snr
from preprocessing.transcriber import (
    transcribe_audio,
    diarize_audio,
    align_transcript_with_speakers,
    run_transcription_diarization,
    compute_wer,
)

__all__ = [
    "preprocess_audio",
    "compute_snr",
    "transcribe_audio",
    "diarize_audio",
    "align_transcript_with_speakers",
    "run_transcription_diarization",
    "compute_wer",
]
