"""
fix_speakers.py
===============
Post-processing script that reassigns speaker labels using voice embeddings.

Problem: pyannote outputs all segments as SPEAKER_0 for difficult calls.
Solution: Extract d-vectors from audio for each segment, cluster into 2 groups,
          then assign Agent/Customer based on first-segment rule.

Usage:
    python fix_speakers.py --call_id eng_rudeagt_01
    python fix_speakers.py --call_id my_sales_01
    python fix_speakers.py --all   # fix all calls with only 1 speaker

This updates the *_diarized.json files in outputs/latest/
"""

import os
import json
import argparse
import logging
import numpy as np
import librosa

import torch
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from config import OUTPUTS_DIR, DATA_DIR, CALLS_DIR, COLAB_TRANSCRIPTS_DIR, get_call_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_audio(call_id: str) -> tuple:
    """Load WAV file for a call."""
    for ext in [".wav", ".mp3"]:
        path = os.path.join(DATA_DIR, f"{call_id}{ext}")
        if os.path.exists(path):
            y, sr = librosa.load(path, sr=16000, mono=True)
            return y, sr
    raise FileNotFoundError(f"No audio found for {call_id} in {DATA_DIR}")


def extract_embeddings(y: np.ndarray, sr: int, segments: list,
                       encoder: VoiceEncoder) -> np.ndarray:
    """Extract d-vector embedding for each segment."""
    embeddings = []

    for seg in segments:
        start = seg.get("start", 0)
        end   = seg.get("end", start + seg.get("duration", 1))

        # Extract audio slice
        start_sample = int(start * sr)
        end_sample   = int(end   * sr)
        audio_slice  = y[start_sample:end_sample]

        # Need at least 0.5s of audio
        if len(audio_slice) < sr * 0.5:
            # Pad short segments
            audio_slice = np.pad(audio_slice, (0, max(0, int(sr * 0.5) - len(audio_slice))))

        try:
            wav = preprocess_wav(audio_slice, source_sr=sr)
            if len(wav) < 160:  # too short
                embeddings.append(np.zeros(256))
            else:
                emb = encoder.embed_utterance(wav)
                embeddings.append(emb)
        except Exception:
            embeddings.append(np.zeros(256))

    return np.array(embeddings)


def cluster_speakers(embeddings: np.ndarray, n_clusters: int = 2) -> np.ndarray:
    """Cluster embeddings into n_clusters groups."""
    # Remove zero embeddings (short segments) — they'll be assigned later
    valid_mask = np.any(embeddings != 0, axis=1)

    if valid_mask.sum() < n_clusters:
        logger.warning("Not enough valid segments to cluster — returning all 0")
        return np.zeros(len(embeddings), dtype=int)

    # Normalize
    scaler = StandardScaler()
    valid_embs = scaler.fit_transform(embeddings[valid_mask])

    # Try Agglomerative first (more robust for audio)
    try:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels_valid = clusterer.fit_predict(valid_embs)
    except Exception:
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_valid = clusterer.fit_predict(valid_embs)

    # Assign clusters back
    labels = np.zeros(len(embeddings), dtype=int)
    labels[valid_mask] = labels_valid

    # Assign invalid (short) segments to nearest cluster
    if not valid_mask.all():
        for i, valid in enumerate(valid_mask):
            if not valid:
                # Find nearest valid neighbor
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    nearest = valid_indices[np.argmin(np.abs(valid_indices - i))]
                    labels[i] = labels[nearest]

    return labels


def detect_first_speaker(segments: list) -> str:
    """
    Detect whether Agent or Customer speaks first.
    Uses same logic as method3_llm._detect_call_type.
    """
    import re
    if not segments:
        return "Agent"

    agent_patterns = [
        r"thank you for calling", r"thanks for calling",
        r"my name is", r"nama saya", r"how can i (help|assist)",
        r"boleh saya bantu", r"selamat (pagi|petang|malam)",
        r"terima kasih kerana", r"good (morning|afternoon|evening|day)",
        r"assalamualaikum.*saya", r"welcome to",
    ]
    customer_patterns = [
        r"tolong saya", r"help me", r"help!", r"oh my god",
        r"i think my", r"emergency", r"kecemasan",
        r"i just got", r"my (house|car|phone)",
        r"finally", r"i've been (on hold|waiting)",
    ]

    first_text = " ".join(s.get("text", "").lower() for s in segments[:3])

    for p in customer_patterns:
        if re.search(p, first_text):
            return "Customer"
    for p in agent_patterns:
        if re.search(p, first_text):
            return "Agent"

    return "Agent"  # default


def assign_roles(segments: list, labels: np.ndarray,
                 call_type: str = "auto") -> list:
    """
    Assign Agent/Customer roles based on cluster labels.
    Detects call type automatically if call_type="auto".
    """
    if len(segments) == 0:
        return segments

    if call_type == "auto":
        first_speaker = detect_first_speaker(segments)
    else:
        first_speaker = call_type

    # Cluster of first segment = first_speaker
    first_cluster = labels[0]

    updated = []
    for i, seg in enumerate(segments):
        cluster = labels[i]
        if cluster == first_cluster:
            role        = first_speaker
            speaker_id  = 0 if first_speaker == "Agent" else 1
            speaker_raw = "SPEAKER_0" if first_speaker == "Agent" else "SPEAKER_1"
        else:
            role        = "Customer" if first_speaker == "Agent" else "Agent"
            speaker_id  = 1 if first_speaker == "Agent" else 0
            speaker_raw = "SPEAKER_1" if first_speaker == "Agent" else "SPEAKER_0"

        updated_seg = dict(seg)
        updated_seg["speaker_id"]  = speaker_id
        updated_seg["speaker_raw"] = speaker_raw
        updated.append(updated_seg)

    logger.info(f"  Call type: {first_speaker}-first | Cluster {first_cluster} → {first_speaker}")
    return updated


def fix_call(call_id: str, encoder: VoiceEncoder, force: bool = False) -> bool:
    """Fix speaker labels for a single call."""
    # Search order: colab_transcripts/ → calls/<id>/ → outputs/latest/
    json_path = os.path.join(COLAB_TRANSCRIPTS_DIR, f"{call_id}_diarized.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(get_call_dir(call_id), f"{call_id}_diarized.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(OUTPUTS_DIR, f"{call_id}_diarized.json")

    if not os.path.exists(json_path):
        logger.error(f"JSON not found: {json_path}")
        return False

    with open(json_path, encoding="utf-8") as f:
        segments = json.load(f)

    # Check if already multi-speaker
    speaker_ids = set(s.get("speaker_id", 0) for s in segments)
    if len(speaker_ids) > 1 and not force:
        logger.info(f"{call_id}: Already has {len(speaker_ids)} speakers — skipping")
        return False

    logger.info(f"{call_id}: {len(segments)} segments — fixing speaker labels...")

    # Load audio
    try:
        y, sr = load_audio(call_id)
    except FileNotFoundError as e:
        logger.error(str(e))
        return False

    # Extract embeddings
    logger.info(f"  Extracting voice embeddings...")
    embeddings = extract_embeddings(y, sr, segments, encoder)

    # Cluster
    logger.info(f"  Clustering into 2 speakers...")
    labels = cluster_speakers(embeddings)

    cluster_counts = {0: (labels == 0).sum(), 1: (labels == 1).sum()}
    logger.info(f"  Cluster 0: {cluster_counts[0]} segs | Cluster 1: {cluster_counts[1]} segs")

    # Check balance — if very imbalanced, clustering probably failed
    minority = min(cluster_counts.values())
    majority = max(cluster_counts.values())
    balance  = minority / max(majority, 1)

    if balance < 0.1:
        logger.warning(f"  Very imbalanced clusters ({balance:.2f}) — voices too similar to separate")
        return False

    # Assign roles
    updated = assign_roles(segments, labels)

    # Backup original
    backup_path = json_path.replace(".json", "_original.json")
    if not os.path.exists(backup_path):
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        logger.info(f"  Backup saved → {backup_path}")

    # Save updated
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)

    logger.info(f"  ✅ {call_id}: speaker labels updated")
    return True


def get_single_speaker_calls() -> list:
    """Find all calls in colab_transcripts/ with only SPEAKER_0."""
    single = []
    scan_dir = COLAB_TRANSCRIPTS_DIR if os.path.isdir(COLAB_TRANSCRIPTS_DIR) else OUTPUTS_DIR
    for f in os.listdir(scan_dir):
        if not f.endswith("_diarized.json") or "_original" in f:
            continue
        call_id = f.replace("_diarized.json", "")
        with open(os.path.join(scan_dir, f), encoding="utf-8") as fp:
            segs = json.load(fp)
        if isinstance(segs, dict):
            segs = segs.get("segments", segs.get("utterances", []))
        speaker_ids = set(s.get("speaker_id", 0) for s in segs)
        if len(speaker_ids) == 1:
            single.append(call_id)
    return sorted(single)


def main():
    parser = argparse.ArgumentParser(description="Fix speaker labels using voice embeddings")
    parser.add_argument("--call_id", type=str, help="Specific call to fix")
    parser.add_argument("--all",     action="store_true", help="Fix all single-speaker calls")
    parser.add_argument("--force",   action="store_true", help="Fix even if already multi-speaker")
    args = parser.parse_args()

    if not args.call_id and not args.all:
        print("Usage:")
        print("  python fix_speakers.py --call_id eng_rudeagt_01")
        print("  python fix_speakers.py --all")
        return

    # Load voice encoder
    logger.info("Loading Resemblyzer VoiceEncoder...")
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = VoiceEncoder(device=device)
    logger.info(f"✅ VoiceEncoder ready on {device}")

    if args.all:
        # Fix all calls in colab_transcripts/ (force mode skips the single-speaker filter)
        scan_dir = COLAB_TRANSCRIPTS_DIR if os.path.isdir(COLAB_TRANSCRIPTS_DIR) else OUTPUTS_DIR
        all_found = sorted([
            f.replace("_diarized.json", "")
            for f in os.listdir(scan_dir)
            if f.endswith("_diarized.json") and "_original" not in f
        ])
        if args.force:
            calls = all_found
        else:
            # Only fix calls that don't already have an _original backup
            calls = [
                c for c in all_found
                if not os.path.exists(os.path.join(scan_dir, f"{c}_diarized_original.json"))
            ]
            already_fixed = [c for c in all_found if c not in calls]
            if already_fixed:
                logger.info(f"Skipping {len(already_fixed)} already-fixed calls (use --force to redo)")
        logger.info(f"Processing {len(calls)} calls: {calls}")
    else:
        calls = [args.call_id]

    fixed = 0
    for call_id in calls:
        success = fix_call(call_id, encoder, force=args.force)
        if success:
            fixed += 1

    print(f"\n✅ Fixed {fixed}/{len(calls)} calls")
    if fixed > 0:
        print("\nNow rerun:")
        print("  python main.py --skip_transcription --call_id <call_id>")
        print("  (or python main.py --skip_transcription for all)")


if __name__ == "__main__":
    main()
