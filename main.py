"""
main.py
=======
Entry point for the FYP1 Call Analysis System.

Orchestrates the full pipeline:
  Phase 1  → Preprocessing (noise reduction, normalization, transcription, diarization)
  Phase 2  → Role Detection (Method 1, 2, 3)
  Phase 3  → Analytics (talk ratio, sentiment, compliance, QA score)
  Phase 4  → Evaluation (metrics, t-test, Pearson correlation)
  Phase 5  → Export results JSON, CSV, and per-call transcripts for dashboard

Usage:
    python main.py
    python main.py --skip_acoustic                   # faster, no GPU needed
    python main.py --whisper_model small             # better accuracy on Manglish
    python main.py --call_id airasia_call            # process one file only
    python main.py --data_dir path/to/audio/

Results are saved to:
    outputs/pipeline_results.json   → dashboard input
    outputs/analytics_summary.csv   → spreadsheet-friendly report
    outputs/{call_id}_transcript.txt → human-readable diarized transcripts
"""

import os
import sys
import json
import time
import glob
import logging
import argparse
import numpy as np

# ── FIX: Create outputs/ BEFORE FileHandler is instantiated ─────
os.makedirs("outputs", exist_ok=True)

# ── Logging setup ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")

# ── Local imports ─────────────────────────────────────────────────
from config import (
    DATA_DIR, OUTPUTS_DIR, GROUND_TRUTH_CSV, AUDIO_SAMPLE_RATE,
)
import config as _cfg   # allow runtime override of WHISPER_MODEL_SIZE

from preprocessing.audio_processor import (
    preprocess_audio, plot_waveform_comparison, plot_spectrogram_comparison,
)
from preprocessing.transcriber import (
    run_transcription_diarization, compute_diarization_turn_counts, compute_wer,
)
from methods.method1_lexical import (
    classify_transcript_lexical, analyze_keyword_frequency,
    plot_keyword_density, plot_confidence_distribution,
)
from methods.method2_acoustic import classify_transcript_acoustic
from methods.method3_hybrid import (
    classify_transcript_hybrid, compute_confidence_statistics,
    plot_method_comparison, plot_ensemble_scores,
)
from analytics.talk_ratio import (
    compute_talk_time_ratio, compute_turn_taking, compute_qa_score,
    plot_talk_time_distribution,
)
from analytics.sentiment import (
    analyze_sentiment, plot_sentiment_trajectory, plot_sentiment_summary,
)
from analytics.compliance import (
    check_compliance, plot_compliance_summary, plot_risk_severity_distribution,
)
from evaluation.metrics import compute_classification_metrics, compute_rtf
from utils.file_utils import save_json, save_transcript_txt, save_analytics_csv


# ─────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="FYP1 Call Analysis System — Full Pipeline"
    )
    parser.add_argument(
        "--data_dir", default=DATA_DIR,
        help="Directory containing .wav/.mp3 audio files (default: data/)")
    parser.add_argument(
        "--skip_validation", action="store_true",
        help="Skip ground truth validation (if CSV not available)")
    parser.add_argument(
        "--skip_acoustic", action="store_true",
        help="Skip Method 2 DNN — runs faster, no GPU required")
    parser.add_argument(
        "--output_json",
        default=os.path.join(OUTPUTS_DIR, "pipeline_results.json"),
        help="Output JSON path for the Streamlit dashboard")
    parser.add_argument(
        "--whisper_model", default=None,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Override Whisper model size (default from config.py)")
    parser.add_argument(
        "--call_id", default=None,
        help="Process only this specific call (filename without extension)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────
# FIND AUDIO FILES
# ─────────────────────────────────────────────────────────────────

def find_audio_files(data_dir: str, call_id_filter: str = None) -> list[str]:
    """
    Find all .wav and .mp3 files in data_dir.
    If call_id_filter is set, return only the matching file.
    """
    files = []
    for ext in ("*.wav", "*.mp3", "*.WAV", "*.MP3"):
        files.extend(glob.glob(os.path.join(data_dir, ext)))
    files = sorted(set(files))

    if call_id_filter:
        files = [f for f in files
                 if os.path.splitext(os.path.basename(f))[0] == call_id_filter]
        if not files:
            logger.error(
                f"No audio file found for call_id='{call_id_filter}' in '{data_dir}'"
            )
    logger.info(f"Found {len(files)} audio file(s) in '{data_dir}'")
    return files


# ─────────────────────────────────────────────────────────────────
# PROCESS ONE CALL
# ─────────────────────────────────────────────────────────────────

def process_call(audio_path: str,
                 skip_acoustic: bool = False,
                 reference_transcripts: dict = None) -> dict:
    """
    Full 5-phase pipeline for a single audio file.

    Parameters
    ----------
    audio_path            : path to .wav/.mp3
    skip_acoustic         : skip Method 2 DNN classification
    reference_transcripts : {call_id: str} ground-truth text for WER (optional)

    Returns
    -------
    dict — all results, JSON-serialisable
    """
    call_id = os.path.splitext(os.path.basename(audio_path))[0]
    logger.info(f"\n{'='*60}\nProcessing: {call_id}\n{'='*60}")
    result = {"call_id": call_id, "audio_path": audio_path}

    # ── Phase 1A: DSP Preprocessing ──────────────────────────────
    logger.info("Phase 1: Audio preprocessing...")
    prep = preprocess_audio(audio_path, save_cleaned=True)
    y, sr = prep["y_clean"], prep["sr"]

    result["preprocessing"] = {
        "duration_sec":       prep["duration_sec"],
        "snr_before_db":      prep["snr_before_db"],
        "snr_after_db":       prep["snr_after_db"],
        "snr_improvement_db": prep["snr_improvement_db"],
    }

    plot_waveform_comparison(
        prep["y_raw"], y, sr,
        title=f"Waveform Comparison — {call_id}",
        save_path=os.path.join(OUTPUTS_DIR, f"{call_id}_waveform.png"),
    )
    plot_spectrogram_comparison(
        prep["y_raw"], y, sr,
        title=f"Spectrogram — {call_id}",
        save_path=os.path.join(OUTPUTS_DIR, f"{call_id}_spectrogram.png"),
    )

    # ── Phase 1B: Transcription + Diarization ────────────────────
    logger.info("Phase 1: Transcription + Diarization...")
    diarized = run_transcription_diarization(
        y, sr,
        save_json=os.path.join(OUTPUTS_DIR, f"{call_id}_diarized.json"),
    )
    turn_stats = compute_diarization_turn_counts(diarized)
    result["diarization"] = {
        "num_segments":      len(diarized),
        "num_turns":         turn_stats["num_turns"],
        "turns_per_speaker": turn_stats["turns_per_speaker"],
    }

    if not diarized:
        logger.error(f"No segments produced for {call_id} — skipping")
        return result

    # WER calculation (only when reference transcript is available and has good coverage)
    wer_result = {}
    if reference_transcripts and call_id in reference_transcripts:
        hypothesis   = " ".join(seg.get("text", "") for seg in diarized)
        reference    = reference_transcripts[call_id]
        # Only calculate WER when reference has at least 20 words
        # (too few reference words gives meaningless WER like 935%)
        ref_words = reference.split()
        if len(ref_words) >= 20:
            wer_result = compute_wer(reference, hypothesis)
            logger.info(f"WER for {call_id}: {wer_result.get('wer', 'N/A')}%")
        else:
            logger.info(
                f"WER skipped for {call_id} — reference too short "
                f"({len(ref_words)} words). Add more rows to human_validation_study.csv "
                f"for meaningful WER."
            )
    result["wer"] = wer_result

    # ── Phase 2A: Method 1 — Lexical ─────────────────────────────
    logger.info("Phase 2: Method 1 — Lexical classification...")
    m1_classified = classify_transcript_lexical(diarized)

    kw_freq = analyze_keyword_frequency(m1_classified)
    plot_keyword_density(
        kw_freq,
        save_path=os.path.join(OUTPUTS_DIR, f"{call_id}_m1_keywords.png"),
    )
    plot_confidence_distribution(
        m1_classified,
        save_path=os.path.join(OUTPUTS_DIR, f"{call_id}_m1_confidence.png"),
    )
    m1_stats = compute_confidence_statistics(m1_classified)
    result["method1"] = {
        "classified": _serialize(m1_classified),
        "stats":      m1_stats,
    }

    # ── Phase 2B: Method 2 — Acoustic DNN ────────────────────────
    m2_classified = []
    # Auto-skip Method 2 if no trained model exists (avoids random-weight garbage)
    from config import ACOUSTIC_MODEL_PATH
    if not skip_acoustic and not os.path.isfile(ACOUSTIC_MODEL_PATH):
        logger.warning(
            "Method 2 skipped — no trained model at models/acoustic_model.pth\n"
            "  Run  python train.py  first to enable acoustic classification.\n"
            "  Continuing with Method 1 + Method 3 (lexical + hybrid)."
        )
        skip_acoustic = True

    if not skip_acoustic:
        logger.info("Phase 2: Method 2 — Acoustic DNN classification...")
        speaker_audio = {}
        for seg in diarized:
            spk   = seg["speaker_id"]
            start = int(seg["start"] * sr)
            end   = int(seg["end"]   * sr)
            chunk = y[max(0, start): min(len(y), end)]
            if spk not in speaker_audio:
                speaker_audio[spk] = chunk
            else:
                speaker_audio[spk] = np.concatenate([speaker_audio[spk], chunk])
        m2_classified = classify_transcript_acoustic(diarized, speaker_audio)
    else:
        logger.info("Phase 2: Method 2 — Skipped (--skip_acoustic flag)")
        for seg in diarized:
            m2_classified.append({
                **seg,
                "predicted_role":   "Unknown",
                "confidence":       0.5,
                "agent_prob":       0.5,
                "customer_prob":    0.5,
                "final_confidence": 0.5,
                "method":           "acoustic_skipped",
            })

    m2_stats = compute_confidence_statistics(m2_classified)
    result["method2"] = {
        "classified": _serialize(m2_classified),
        "stats":      m2_stats,
    }

    # ── Phase 2C: Method 3 — Hybrid Ensemble ─────────────────────
    logger.info("Phase 2: Method 3 — Hybrid Ensemble Fusion...")
    m3_classified = classify_transcript_hybrid(m1_classified, m2_classified)

    m3_stats = compute_confidence_statistics(m3_classified)
    plot_ensemble_scores(
        m3_classified,
        save_path=os.path.join(OUTPUTS_DIR, f"{call_id}_m3_ensemble.png"),
    )
    result["method3"] = {
        "classified": _serialize(m3_classified),
        "stats":      m3_stats,
    }
    plot_method_comparison(
        m1_stats, m2_stats, m3_stats,
        save_path=os.path.join(OUTPUTS_DIR, f"{call_id}_method_comparison.png"),
    )

    # ── Phase 3: Analytics ───────────────────────────────────────
    logger.info("Phase 3: Analytics...")
    sentiment_result  = analyze_sentiment(m3_classified)
    m3_with_sentiment = sentiment_result["segments_with_sentiment"]

    talk_ratio        = compute_talk_time_ratio(m3_with_sentiment, prep["duration_sec"])
    turn_flow         = compute_turn_taking(m3_with_sentiment)
    compliance_result = check_compliance(m3_with_sentiment)
    qa_result         = compute_qa_score(talk_ratio, turn_flow,
                                          sentiment_result, compliance_result)

    result["sentiment"]         = {k: v for k, v in sentiment_result.items()
                                   if k != "segments_with_sentiment"}
    result["talk_ratio"]        = talk_ratio
    result["turn_flow"]         = turn_flow
    result["compliance"]        = compliance_result
    result["qa_result"]         = qa_result
    result["transcript_hybrid"] = _serialize(m3_with_sentiment)

    # Sentiment trajectory plot
    plot_sentiment_trajectory(
        sentiment_result, call_id,
        save_path=os.path.join(OUTPUTS_DIR, f"{call_id}_sentiment.png"),
    )

    # Export human-readable transcript
    save_transcript_txt(
        m3_with_sentiment,
        os.path.join(OUTPUTS_DIR, f"{call_id}_transcript.txt"),
    )

    logger.info(
        f"✓ {call_id} | QA={qa_result['qa_score']}/100 ({qa_result['rating']}) | "
        f"Compliance={compliance_result['compliance_score']*100:.0f}% | "
        f"Risk={compliance_result['risk_severity']}"
    )
    return result


# ─────────────────────────────────────────────────────────────────
# SERIALIZATION HELPER
# ─────────────────────────────────────────────────────────────────

def _serialize(classified: list[dict]) -> list[dict]:
    """Strip numpy arrays and convert numpy scalars for JSON serialisation."""
    out = []
    for seg in classified:
        row = {}
        for k, v in seg.items():
            if isinstance(v, np.ndarray):
                continue
            elif isinstance(v, (np.integer, np.int64)):
                row[k] = int(v)
            elif isinstance(v, (np.floating, np.float32, np.float64)):
                row[k] = float(v)
            elif isinstance(v, np.bool_):
                row[k] = bool(v)
            else:
                row[k] = v
        out.append(row)
    return out


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Apply Whisper model CLI override
    if args.whisper_model:
        _cfg.WHISPER_MODEL_SIZE = args.whisper_model
        logger.info(f"Whisper model overridden → '{args.whisper_model}'")

    logger.info("=" * 70)
    logger.info("FYP1 CALL ANALYSIS SYSTEM")
    logger.info("Speaker Role Detection & Segmented Analysis in Customer Service Calls")
    logger.info("=" * 70)

    audio_files = find_audio_files(args.data_dir, call_id_filter=args.call_id)
    if not audio_files:
        logger.warning(
            f"No audio files found in '{args.data_dir}'.\n"
            "Place .wav or .mp3 files in the data/ folder and run again."
        )
        return

    # Load reference transcripts for WER (from ground truth CSV if available)
    reference_transcripts = {}
    if os.path.isfile(GROUND_TRUTH_CSV):
        try:
            import pandas as pd
            gt_df = pd.read_csv(GROUND_TRUTH_CSV)
            gt_df.columns = gt_df.columns.str.lower().str.strip()
            if "text" in gt_df.columns and "call_id" in gt_df.columns:
                for cid, grp in gt_df.groupby("call_id"):
                    reference_transcripts[cid] = " ".join(
                        grp["text"].dropna().astype(str).tolist()
                    )
        except Exception as e:
            logger.warning(f"Could not load reference transcripts for WER: {e}")

    pipeline_start  = time.time()
    all_results     = {}
    total_audio_sec = 0.0

    for audio_path in audio_files:
        call_result = process_call(
            audio_path,
            skip_acoustic=args.skip_acoustic,
            reference_transcripts=reference_transcripts,
        )
        cid = call_result["call_id"]
        all_results[cid] = call_result
        total_audio_sec += call_result.get("preprocessing", {}).get("duration_sec", 0)

    pipeline_end    = time.time()
    processing_time = pipeline_end - pipeline_start

    # Efficiency metrics
    efficiency = compute_rtf(total_audio_sec, processing_time)
    logger.info(
        f"\nPipeline complete | {len(all_results)} call(s) | "
        f"audio={total_audio_sec:.1f}s | processing={processing_time:.1f}s | "
        f"RTF={efficiency.get('rtf','?')} | "
        f"{efficiency.get('efficiency_multiplier','?')}× faster than real-time"
    )

    # Cross-call summary plots (only when >1 call)
    if len(all_results) > 1:
        plot_talk_time_distribution([
            {"call_id": cid, "talk_ratio": r.get("talk_ratio", {})}
            for cid, r in all_results.items()
        ])
        plot_sentiment_summary([
            {"call_id": cid, "sentiment": r.get("sentiment", {})}
            for cid, r in all_results.items()
        ])
        calls_comp = [
            {"call_id": cid, "compliance": r.get("compliance", {})}
            for cid, r in all_results.items()
        ]
        plot_compliance_summary(calls_comp)
        plot_risk_severity_distribution(calls_comp)

    # Phase 4: Validation against ground truth
    validation_results = {}
    if not args.skip_validation and os.path.isfile(GROUND_TRUTH_CSV):
        logger.info("\nPhase 4: Validation against ground truth...")
        try:
            from evaluation.validator import run_validation
            validation_results = run_validation(
                calls_m1=[all_results[cid].get("method1", {}).get("classified", [])
                           for cid in all_results],
                calls_m2=[all_results[cid].get("method2", {}).get("classified", [])
                           for cid in all_results],
                calls_m3=[all_results[cid].get("method3", {}).get("classified", [])
                           for cid in all_results],
                call_ids=list(all_results.keys()),
                system_qa_scores=[all_results[cid].get("qa_result", {}).get("qa_score", 50)
                                   for cid in all_results],
                processing_time_sec=processing_time,
                total_audio_duration_sec=total_audio_sec,
            )
        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)
    elif not os.path.isfile(GROUND_TRUTH_CSV):
        logger.info(f"Validation skipped — no CSV at: {GROUND_TRUTH_CSV}")

    # Phase 5: Export all outputs
    output_payload = {
        "summary": {
            "n_calls":               len(all_results),
            "total_audio_sec":       round(total_audio_sec,  2),
            "processing_time_sec":   round(processing_time,  2),
            "rtf":                   efficiency.get("rtf"),
            "efficiency_multiplier": efficiency.get("efficiency_multiplier"),
        },
        "calls":              all_results,
        "validation_results": validation_results,
    }

    csv_path = os.path.join(OUTPUTS_DIR, "analytics_summary.csv")
    save_analytics_csv(all_results, csv_path)
    save_json(output_payload, args.output_json)

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Results saved:")
    logger.info(f"   JSON      → {args.output_json}")
    logger.info(f"   CSV       → {csv_path}")
    logger.info(f"   Transcripts → {OUTPUTS_DIR}/*_transcript.txt")
    logger.info(f"   Plots     → {OUTPUTS_DIR}/")
    logger.info(f"\n   Launch dashboard:  streamlit run dashboard/app.py")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
