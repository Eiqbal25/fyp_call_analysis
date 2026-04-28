"""
preprocessing/audio_processor.py
=================================
Phase 1: Data Acquisition and Pre-processing

Handles all Digital Signal Processing (DSP) operations:
  - Audio loading and resampling to 16 kHz mono
  - Spectral Gating Noise Reduction
  - Amplitude Normalization
  - Signal-to-Noise Ratio (SNR) calculation

All statistics (SNR before/after) are computed mathematically — no hardcoded values.
"""

import os
import logging
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import noisereduce as nr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    AUDIO_SAMPLE_RATE,
    NOISE_STATIONARY_PROP,
    AUDIO_NORMALIZE_PEAK,
    OUTPUTS_DIR,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# CORE DSP FUNCTIONS
# ─────────────────────────────────────────────────────────────

def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    """
    Load an audio file (wav/mp3) and return waveform + sample rate.
    Does NOT resample yet — raw load only.

    Returns
    -------
    y   : np.ndarray  — raw waveform
    sr  : int         — original sample rate
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    y, sr = librosa.load(file_path, sr=None, mono=True)
    logger.info(f"Loaded '{os.path.basename(file_path)}' | sr={sr} Hz | "
                f"duration={len(y)/sr:.2f}s | samples={len(y)}")
    return y, sr


def resample_audio(y: np.ndarray, original_sr: int,
                   target_sr: int = AUDIO_SAMPLE_RATE) -> np.ndarray:
    """
    Resample waveform to target_sr (default 16 kHz for Whisper + Resemblyzer).
    """
    if original_sr == target_sr:
        return y
    y_resampled = librosa.resample(y, orig_sr=original_sr, target_sr=target_sr)
    logger.info(f"Resampled {original_sr} Hz → {target_sr} Hz")
    return y_resampled


def reduce_noise(y: np.ndarray, sr: int = AUDIO_SAMPLE_RATE) -> np.ndarray:
    """
    Apply Spectral Gating Noise Reduction.

    Uses the first NOISE_STATIONARY_PROP fraction of the audio to estimate
    the stationary noise profile, then suppresses it across the entire signal.

    Parameters
    ----------
    y  : raw waveform (float32, range ~[-1, 1])
    sr : sample rate

    Returns
    -------
    y_clean : noise-reduced waveform
    """
    n_noise_samples = max(1, int(len(y) * NOISE_STATIONARY_PROP))
    noise_profile = y[:n_noise_samples]

    y_clean = nr.reduce_noise(
        y=y,
        sr=sr,
        y_noise=noise_profile,
        stationary=True,
        prop_decrease=1.0,
    )
    logger.info(f"Noise reduction applied | noise_profile_samples={n_noise_samples}")
    return y_clean.astype(np.float32)


def normalize_amplitude(y: np.ndarray,
                        peak: float = AUDIO_NORMALIZE_PEAK) -> np.ndarray:
    """
    Normalize signal amplitude so max(|y|) == peak.
    Scales signal to [-peak, +peak] to remove volume disparities.
    """
    max_val = np.max(np.abs(y))
    if max_val < 1e-9:
        logger.warning("Near-silent audio detected — skipping normalization")
        return y
    y_norm = (y / max_val) * peak
    logger.info(f"Amplitude normalized | max_amplitude={max_val:.4f} → {peak:.4f}")
    return y_norm.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# SNR CALCULATION (mathematically derived)
# ─────────────────────────────────────────────────────────────

def compute_snr(y: np.ndarray, sr: int = AUDIO_SAMPLE_RATE,
                noise_frac: float = NOISE_STATIONARY_PROP) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR) in decibels.

    SNR (dB) = 10 * log10( P_signal / P_noise )

    Signal power  = mean of squared values across entire signal
    Noise power   = mean of squared values of the estimated noise floor
                    (first noise_frac * N samples used as noise reference)

    Parameters
    ----------
    y          : waveform array
    sr         : sample rate
    noise_frac : fraction of audio to treat as noise reference

    Returns
    -------
    snr_db : float
    """
    n_noise = max(1, int(len(y) * noise_frac))
    noise_segment = y[:n_noise]

    power_signal = np.mean(y ** 2)
    power_noise  = np.mean(noise_segment ** 2)

    if power_noise < 1e-12:
        logger.warning("Noise power near zero — returning SNR = inf")
        return float("inf")

    snr_db = 10.0 * np.log10(power_signal / power_noise)
    return round(float(snr_db), 2)


# ─────────────────────────────────────────────────────────────
# FULL PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────

def preprocess_audio(file_path: str,
                     save_cleaned: bool = True) -> dict:
    """
    Full preprocessing pipeline for a single audio file.

    Steps:
      1. Load raw audio
      2. Resample to 16 kHz mono
      3. Compute SNR (before cleaning)
      4. Apply spectral noise reduction
      5. Normalize amplitude
      6. Compute SNR (after cleaning)
      7. Optionally save cleaned file

    Parameters
    ----------
    file_path    : path to raw .wav / .mp3
    save_cleaned : if True, saves cleaned .wav to OUTPUTS_DIR

    Returns
    -------
    dict with keys:
        file_path, cleaned_path, y_raw, y_clean, sr,
        duration_sec, snr_before_db, snr_after_db, snr_improvement_db
    """
    logger.info(f"=== Preprocessing: {os.path.basename(file_path)} ===")

    # 1. Load
    y_raw, sr_orig = load_audio(file_path)

    # 2. Resample
    y_resampled = resample_audio(y_raw, sr_orig, AUDIO_SAMPLE_RATE)

    # 3. SNR before cleaning
    snr_before = compute_snr(y_resampled, AUDIO_SAMPLE_RATE)
    logger.info(f"SNR before cleaning: {snr_before:.2f} dB")

    # 4. Noise reduction
    y_clean = reduce_noise(y_resampled, AUDIO_SAMPLE_RATE)

    # 5. Normalize
    y_clean = normalize_amplitude(y_clean)

    # 6. SNR after cleaning
    snr_after = compute_snr(y_clean, AUDIO_SAMPLE_RATE)
    logger.info(f"SNR after cleaning:  {snr_after:.2f} dB | "
                f"Improvement: +{snr_after - snr_before:.2f} dB")

    # 7. Save
    cleaned_path = None
    if save_cleaned:
        stem = os.path.splitext(os.path.basename(file_path))[0]
        cleaned_path = os.path.join(OUTPUTS_DIR, f"{stem}_cleaned.wav")
        sf.write(cleaned_path, y_clean, AUDIO_SAMPLE_RATE)
        logger.info(f"Saved cleaned audio → {cleaned_path}")

    duration_sec = len(y_clean) / AUDIO_SAMPLE_RATE

    return {
        "file_path":          file_path,
        "cleaned_path":       cleaned_path,
        "y_raw":              y_resampled,
        "y_clean":            y_clean,
        "sr":                 AUDIO_SAMPLE_RATE,
        "duration_sec":       round(duration_sec, 3),
        "snr_before_db":      snr_before,
        "snr_after_db":       snr_after,
        "snr_improvement_db": round(snr_after - snr_before, 2),
    }


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_waveform_comparison(y_raw: np.ndarray, y_clean: np.ndarray,
                              sr: int, title: str = "Waveform Comparison",
                              save_path: str = None) -> str:
    """
    Plot raw vs. cleaned waveform side-by-side (matches Figure 3.2 in thesis).
    Saves PNG to save_path or OUTPUTS_DIR.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    time_axis = np.linspace(0, len(y_raw) / sr, num=len(y_raw))

    axes[0].plot(time_axis, y_raw, color="gray", linewidth=0.5)
    axes[0].set_title("Raw Acoustic Signal (Noisy & Unnormalized)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_ylim([-1.1, 1.1])
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_axis, y_clean, color="steelblue", linewidth=0.5)
    axes[1].set_title("Processed Signal (Noise Reduced & Normalized)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylim([-1.1, 1.1])
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "waveform_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Waveform comparison saved → {save_path}")
    return save_path


def plot_spectrogram_comparison(y_raw: np.ndarray, y_clean: np.ndarray,
                                 sr: int, title: str = "Spectrogram Comparison",
                                 save_path: str = None) -> str:
    """
    Plot mel-spectrogram before/after cleaning (matches Figure 4.1 in thesis).
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for ax, y, label, cmap in zip(
            axes,
            [y_raw, y_clean],
            ["Before: Raw Audio Input (High Noise Floor)",
             "After: Spectral Noise Gating Applied (Reduced Noise)"],
            ["magma", "viridis"]):

        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis="time",
                                        y_axis="hz", ax=ax, cmap=cmap)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "spectrogram_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Spectrogram comparison saved → {save_path}")
    return save_path
