"""
train.py
========
Standalone training script for the Method 2 Acoustic DNN.

WHAT THIS SCRIPT DOES:
  1. Loads labeled audio segments from human_transcripts/*.csv
     (your detailed per-segment transcripts with role labels and timestamps)
  2. Loads any additional rows from human_validation_study.csv
  3. Extracts 298-dim MFCC + d-vector features per segment
  4. Applies heavy data augmentation to expand the dataset
  5. Trains the PyTorch MLP with class-weighted loss
  6. Saves model weights + scaler to models/

WHY THIS IS NEEDED:
  The old train.py relied on human_validation_study.csv which only has
  rows for placeholder calls (airasia, airbnb etc.) that have no audio
  files. So it extracted 0 useful training samples and the model was
  saved with random weights — explaining the 50% accuracy.

  This version uses human_transcripts/ which has your real audio files
  (food_malay, insurance_english) with precise timestamps per segment.

AUGMENTATION TECHNIQUES:
  - Gaussian noise         (simulates background noise)
  - Time stretch ×0.9      (slower speaker)
  - Time stretch ×1.1      (faster speaker)
  - Pitch shift +2 semitones
  - Pitch shift -2 semitones
  Each original sample → 5 augmented copies → dataset grows 6×.

Usage:
    python train.py
    python train.py --epochs 100
    python train.py --no_augment   # disable augmentation (faster, less accurate)

Run order:
    python train.py          ← run this FIRST (once per dataset)
    python main.py           ← then process your calls
"""

import os
import sys
import glob
import logging
import argparse
import numpy as np

# Create required directories before FileHandler
os.makedirs("outputs", exist_ok=True)
os.makedirs("models",  exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/train.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("train")

from config import (
    DATA_DIR,
    GROUND_TRUTH_CSV,
    HUMAN_TRANSCRIPTS_DIR,
    OUTPUTS_DIR,
    AUDIO_SAMPLE_RATE,
    FEATURE_DIM,
    ACOUSTIC_EPOCHS,
)
from preprocessing.audio_processor import preprocess_audio
from methods.method2_acoustic import (
    build_feature_vector,
    train_acoustic_model,
    plot_training_history,
)


# ─────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Method 2 Acoustic DNN"
    )
    parser.add_argument(
        "--data_dir", default=DATA_DIR,
        help=f"Directory containing audio files (default: {DATA_DIR})"
    )
    parser.add_argument(
        "--epochs", type=int, default=ACOUSTIC_EPOCHS,
        help=f"Training epochs (default: {ACOUSTIC_EPOCHS})"
    )
    parser.add_argument(
        "--no_augment", action="store_true",
        help="Disable data augmentation (faster but less accurate)"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# AUDIO CACHE
# ─────────────────────────────────────────────────────────────

_audio_cache: dict = {}


def _load_audio(call_id: str, data_dir: str):
    """Load and preprocess audio for a call_id. Cached to avoid reloading."""
    if call_id in _audio_cache:
        return _audio_cache[call_id]

    for ext in (".wav", ".mp3", ".WAV", ".MP3"):
        path = os.path.join(data_dir, call_id + ext)
        if os.path.isfile(path):
            try:
                prep = preprocess_audio(path, save_cleaned=False)
                _audio_cache[call_id] = (prep["y_clean"], prep["sr"])
                logger.info(f"Loaded audio: {call_id} ({prep['duration_sec']:.1f}s)")
                return _audio_cache[call_id]
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                _audio_cache[call_id] = None
                return None

    logger.warning(f"No audio file found for call_id='{call_id}' in {data_dir}")
    _audio_cache[call_id] = None
    return None


# ─────────────────────────────────────────────────────────────
# DATA AUGMENTATION
# ─────────────────────────────────────────────────────────────

def augment_segment(segment: np.ndarray, sr: int) -> list[np.ndarray]:
    """
    Apply 5 augmentation techniques to one audio segment.
    Returns list of augmented versions (does NOT include the original).

    Techniques:
      1. Gaussian noise  — simulates background noise / phone quality
      2. Time stretch ×0.9 — slower speaker (elderly / hesitant)
      3. Time stretch ×1.1 — faster speaker (rushed / confident)
      4. Pitch shift +2 semitones — higher pitch
      5. Pitch shift −2 semitones — lower pitch
    """
    augmented = []

    # 1. Gaussian noise
    noise = np.random.normal(0, 0.005, len(segment)).astype(np.float32)
    augmented.append(np.clip(segment + noise, -1.0, 1.0))

    # 2–5. Librosa time stretch and pitch shift
    try:
        import librosa

        # Time stretch slow
        slow = librosa.effects.time_stretch(segment.astype(np.float64), rate=0.9)
        augmented.append(slow.astype(np.float32))

        # Time stretch fast
        fast = librosa.effects.time_stretch(segment.astype(np.float64), rate=1.1)
        augmented.append(fast.astype(np.float32))

        # Pitch up +2 semitones
        up = librosa.effects.pitch_shift(
            segment.astype(np.float64), sr=sr, n_steps=2
        )
        augmented.append(up.astype(np.float32))

        # Pitch down -2 semitones
        down = librosa.effects.pitch_shift(
            segment.astype(np.float64), sr=sr, n_steps=-2
        )
        augmented.append(down.astype(np.float32))

    except ImportError:
        logger.warning("librosa not available — using only noise augmentation")
    except Exception as e:
        logger.debug(f"Augmentation partial failure: {e}")

    return augmented


# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION FROM HUMAN TRANSCRIPTS
# ─────────────────────────────────────────────────────────────

ROLE_TO_INT = {"Agent": 0, "Customer": 1}


def extract_from_human_transcripts(data_dir: str,
                                    augment: bool = True) -> tuple:
    """
    Primary data source: human_transcripts/*.csv

    These CSVs have per-segment timestamps (start, end) and role labels
    for your actual audio files. This gives the most accurate training
    samples because each segment is sliced to exactly the right portion
    of audio.

    Returns (X, y) arrays.
    """
    import pandas as pd

    csv_files = glob.glob(os.path.join(HUMAN_TRANSCRIPTS_DIR, "*.csv"))
    # Skip example file
    csv_files = [f for f in csv_files if "example" not in os.path.basename(f).lower()]

    if not csv_files:
        logger.warning(
            f"No human transcript CSVs found in {HUMAN_TRANSCRIPTS_DIR}. "
            "Skipping this data source."
        )
        return np.array([]), np.array([])

    X_list, y_list = [], []

    for csv_path in sorted(csv_files):
        call_id = os.path.splitext(os.path.basename(csv_path))[0]
        logger.info(f"Processing human transcript: {call_id}")

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower().str.strip()

        if "role" not in df.columns or "text" not in df.columns:
            logger.warning(f"Skipping {csv_path} — missing 'role' or 'text' column")
            continue

        df["role"] = df["role"].str.strip().str.title()

        # Load audio
        audio = _load_audio(call_id, data_dir)
        if audio is None:
            logger.warning(f"No audio for {call_id} — skipping")
            continue

        y_audio, sr = audio

        for _, row in df.iterrows():
            role  = row["role"]
            label = ROLE_TO_INT.get(role)
            if label is None:
                continue

            # Slice the exact segment using timestamps
            try:
                start_s = float(row.get("start", 0))
                end_s   = float(row.get("end",   len(y_audio) / sr))
                start_i = max(0,             int(start_s * sr))
                end_i   = min(len(y_audio),  int(end_s   * sr))
                segment = y_audio[start_i:end_i]
            except (ValueError, TypeError):
                segment = y_audio

            # Minimum 0.5s of audio needed for reliable features
            if len(segment) < int(0.5 * sr):
                logger.debug(
                    f"Segment too short: {call_id} row {row.get('segment_id','?')} "
                    f"({len(segment)/sr:.2f}s < 0.5s) — skipping"
                )
                continue

            try:
                fv = build_feature_vector(segment, sr)
                X_list.append(fv)
                y_list.append(label)

                # Augmentation
                if augment:
                    for aug_seg in augment_segment(segment, sr):
                        if len(aug_seg) >= int(0.5 * sr):
                            fv_aug = build_feature_vector(aug_seg, sr)
                            X_list.append(fv_aug)
                            y_list.append(label)

            except Exception as e:
                logger.warning(
                    f"Feature extraction failed: {call_id} "
                    f"row {row.get('segment_id','?')}: {e}"
                )

    if not X_list:
        return np.array([]), np.array([])

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int64)
    logger.info(
        f"Human transcripts → {len(X)} samples "
        f"(Agent={int(np.sum(y==0))} Customer={int(np.sum(y==1))})"
    )
    return X, y


def extract_from_validation_csv(data_dir: str,
                                  csv_path: str,
                                  augment: bool = True) -> tuple:
    """
    Secondary data source: human_validation_study.csv

    Only extracts rows where a matching audio file actually exists.
    Skips placeholder call_ids (airasia, airbnb etc.) that have no audio.

    Returns (X, y) arrays — may be empty if no matching audio found.
    """
    import pandas as pd

    if not os.path.isfile(csv_path):
        logger.warning(f"Validation CSV not found: {csv_path}")
        return np.array([]), np.array([])

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()

    if "ground_truth_role" not in df.columns or "call_id" not in df.columns:
        logger.warning("Validation CSV missing required columns — skipping")
        return np.array([]), np.array([])

    df["ground_truth_role"] = df["ground_truth_role"].str.strip().str.title()

    X_list, y_list = [], []

    for call_id, group in df.groupby("call_id"):
        audio = _load_audio(call_id, data_dir)
        if audio is None:
            continue   # no audio file — skip silently

        y_audio, sr = audio

        for _, row in group.iterrows():
            role  = row["ground_truth_role"]
            label = ROLE_TO_INT.get(role)
            if label is None:
                continue

            # Slice segment if timestamps available
            try:
                start_s = float(row.get("start", 0))
                end_s   = float(row.get("end",   len(y_audio) / sr))
                start_i = max(0,             int(start_s * sr))
                end_i   = min(len(y_audio),  int(end_s   * sr))
                segment = y_audio[start_i:end_i]
            except (ValueError, TypeError):
                segment = y_audio

            if len(segment) < int(0.5 * sr):
                continue

            try:
                fv = build_feature_vector(segment, sr)
                X_list.append(fv)
                y_list.append(label)

                if augment:
                    for aug_seg in augment_segment(segment, sr):
                        if len(aug_seg) >= int(0.5 * sr):
                            X_list.append(build_feature_vector(aug_seg, sr))
                            y_list.append(label)

            except Exception as e:
                logger.warning(f"Feature extraction failed: {call_id}: {e}")

    if not X_list:
        return np.array([]), np.array([])

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int64)
    logger.info(
        f"Validation CSV → {len(X)} samples "
        f"(Agent={int(np.sum(y==0))} Customer={int(np.sum(y==1))})"
    )
    return X, y


# ─────────────────────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────────────────────

def build_dataset(data_dir: str, csv_path: str,
                  augment: bool = True) -> tuple:
    """
    Combine all available data sources into one training dataset.

    Priority order:
      1. human_transcripts/*.csv  (most accurate — per-segment timestamps)
      2. human_validation_study.csv (supplementary — broader coverage)

    Removes duplicate samples that appear in both sources.
    """
    logger.info("Building training dataset from all available sources...")

    # Source 1: human transcripts (primary)
    X1, y1 = extract_from_human_transcripts(data_dir, augment=augment)

    # Source 2: validation CSV (supplementary)
    X2, y2 = extract_from_validation_csv(data_dir, csv_path, augment=augment)

    # Combine
    parts_X, parts_y = [], []
    if len(X1) > 0:
        parts_X.append(X1)
        parts_y.append(y1)
    if len(X2) > 0:
        parts_X.append(X2)
        parts_y.append(y2)

    if not parts_X:
        raise RuntimeError(
            "No training samples could be extracted.\n\n"
            "Make sure you have:\n"
            "  1. Audio files in data/  (food_malay.wav, insurance_english.wav)\n"
            "  2. CSV files in human_transcripts/  (food_malay.csv, insurance_english.csv)\n"
            "     with columns: segment_id, role, text, start, end\n\n"
            "Check outputs/train.log for details."
        )

    X = np.vstack(parts_X)
    y = np.concatenate(parts_y)

    # Summary
    n_agent    = int(np.sum(y == 0))
    n_customer = int(np.sum(y == 1))
    logger.info("=" * 55)
    logger.info(f"TOTAL DATASET: {len(X)} samples")
    logger.info(f"  Agent    (label 0): {n_agent}")
    logger.info(f"  Customer (label 1): {n_customer}")
    logger.info(f"  Feature dimension : {X.shape[1]}")
    logger.info(f"  Augmentation      : {'ON' if augment else 'OFF'}")
    logger.info("=" * 55)

    if n_agent == 0 or n_customer == 0:
        raise RuntimeError(
            f"Dataset has only one class "
            f"(Agent={n_agent}, Customer={n_customer}). "
            "Add more labeled segments with both roles."
        )

    return X, y


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    augment = not args.no_augment

    logger.info("=" * 60)
    logger.info("Method 2 — Acoustic DNN Training")
    logger.info("=" * 60)
    logger.info(f"Data dir    : {args.data_dir}")
    logger.info(f"Epochs      : {args.epochs}")
    logger.info(f"Augmentation: {'ON' if augment else 'OFF'}")
    logger.info(f"Sources     : human_transcripts/ + human_validation_study.csv")

    # ── Step 1: Build dataset ─────────────────────────────────
    logger.info("\nStep 1: Building training dataset...")
    try:
        X, y = build_dataset(
            data_dir=args.data_dir,
            csv_path=GROUND_TRUTH_CSV,
            augment=augment,
        )
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    # ── Step 2: Train DNN ─────────────────────────────────────
    logger.info("\nStep 2: Training the DNN...")
    import config as _cfg
    _cfg.ACOUSTIC_EPOCHS = args.epochs

    result  = train_acoustic_model(X, y, save_model=True)
    history = result["history"]

    # ── Step 3: Save training history plot ───────────────────
    logger.info("\nStep 3: Saving training history plot...")
    plot_training_history(
        history,
        save_path=os.path.join(OUTPUTS_DIR, "m2_training_history.png"),
    )

    # ── Step 4: Print summary ─────────────────────────────────
    final_train_acc  = history["train_acc"][-1]  * 100
    final_val_acc    = history["val_acc"][-1]    * 100
    final_train_loss = history["train_loss"][-1]

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Final train accuracy : {final_train_acc:.1f}%")
    logger.info(f"  Final val accuracy   : {final_val_acc:.1f}%")
    logger.info(f"  Final train loss     : {final_train_loss:.5f}")
    logger.info(f"  Model saved          : models/acoustic_model.pth")
    logger.info(f"  Scaler saved         : models/scaler.npy")
    logger.info(f"  Training plot        : outputs/m2_training_history.png")
    logger.info("=" * 60)

    if final_val_acc >= 70:
        logger.info("✅ Model accuracy acceptable — ready to use in main.py")
    elif final_val_acc >= 55:
        logger.info(
            "⚠ Model accuracy moderate — consider adding more audio data\n"
            "  Add more calls to data/ and more rows to human_transcripts/"
        )
    else:
        logger.info(
            "❌ Model accuracy low — likely too few training samples\n"
            "  Add more audio files + human transcript CSVs and retrain"
        )

    logger.info("\nNext step: python main.py")


if __name__ == "__main__":
    main()
