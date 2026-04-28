"""
methods/method2_acoustic.py
============================
Method 2: Acoustic-Based Speaker Role Classification (Deep Learning)

Architecture (thesis Section 3.4.2):
  - Feature vector: 40 MFCC mean + 40 MFCC std + 256 d-vector = 336 dims
    → padded/truncated to FEATURE_DIM (298)
  - MLP: Input(298) → 256 → 128 → 64 → 32 → 2 (Agent/Customer)

FIXES applied in this version:
  FIX 1 — Scaler applied during inference.
           Previously the StandardScaler was only used during training but
           raw un-scaled features were passed to the model at inference time.
           This alone caused the model to output garbage regardless of training.

  FIX 2 — VoiceEncoder cached at module level.
           Previously a new VoiceEncoder instance was created for every single
           segment (~0.5s overhead per segment). Now loaded once and reused.

  FIX 3 — Class-weighted CrossEntropyLoss during training.
           Prevents the model collapsing to the majority class when Agent and
           Customer sample counts are unequal.

  FIX 4 — Reduced Dropout from 0.3 → 0.1 for small datasets.
           0.3 dropout on a network with <20 training samples drops too many
           activations, preventing the model from learning any patterns.
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    AUDIO_SAMPLE_RATE,
    MFCC_N_MFCC,
    MFCC_HOP_LENGTH,
    MFCC_N_FFT,
    FEATURE_DIM,
    ACOUSTIC_HIDDEN_DIMS,
    ACOUSTIC_DROPOUT,
    ACOUSTIC_EPOCHS,
    ACOUSTIC_LR,
    ACOUSTIC_BATCH_SIZE,
    ACOUSTIC_MODEL_PATH,
    OUTPUTS_DIR,
    DEVICE,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# FIX 2 — VOICE ENCODER CACHE
# ─────────────────────────────────────────────────────────────

# Cached VoiceEncoder — loaded once at module level, reused for all segments.
# Old behaviour: new VoiceEncoder() called for every segment → ~0.5s overhead each.
_voice_encoder = None


def _get_voice_encoder():
    """Return cached VoiceEncoder, loading on first call."""
    global _voice_encoder
    if _voice_encoder is None:
        try:
            from resemblyzer import VoiceEncoder
        except ImportError:
            raise ImportError("resemblyzer not installed. Run: pip install resemblyzer")
        logger.info("Loading Resemblyzer VoiceEncoder (cached)...")
        _voice_encoder = VoiceEncoder()
        logger.info("VoiceEncoder ready.")
    return _voice_encoder


# ─────────────────────────────────────────────────────────────
# PYTORCH MODEL DEFINITION
# ─────────────────────────────────────────────────────────────

def _build_model(dropout_rate: float = None):
    """Build the DNN model. Returns PyTorch nn.Module."""
    try:
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch not installed. Run: pip install torch")

    # FIX 4: Use provided rate or fall back to config value.
    # For small datasets, caller should pass a lower rate (0.1 instead of 0.3).
    rate = dropout_rate if dropout_rate is not None else ACOUSTIC_DROPOUT

    class SpeakerRoleDNN(nn.Module):
        """
        MLP for binary speaker role classification.
        Input(FEATURE_DIM) → 256 → 128 → 64 → 32 → 2 (Agent/Customer)
        ReLU + Dropout after layers 1 & 2.
        """
        def __init__(self, input_dim, hidden_dims, dr):
            super().__init__()
            layers = []
            prev   = input_dim
            for i, h in enumerate(hidden_dims):
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                if i < 2:
                    layers.append(nn.Dropout(dr))
                prev = h
            layers.append(nn.Linear(prev, 2))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    return SpeakerRoleDNN(FEATURE_DIM, ACOUSTIC_HIDDEN_DIMS, rate)


# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_mfcc_features(y: np.ndarray,
                           sr: int = AUDIO_SAMPLE_RATE) -> np.ndarray:
    """
    Extract 80-dim MFCC feature vector: [mean(40 MFCCs), std(40 MFCCs)].
    Captures voice timbre — stable across the full duration of a speaker turn.
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa not installed. Run: pip install librosa")

    if len(y) < MFCC_N_FFT:
        y = np.pad(y, (0, MFCC_N_FFT - len(y)))

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=MFCC_N_MFCC,
        hop_length=MFCC_HOP_LENGTH,
        n_fft=MFCC_N_FFT,
    )
    return np.concatenate([np.mean(mfcc, axis=1),
                           np.std(mfcc,  axis=1)])   # (80,)


def extract_dvector_features(y: np.ndarray,
                               sr: int = AUDIO_SAMPLE_RATE) -> np.ndarray:
    """
    Extract 256-dim Resemblyzer d-vector (vocal identity embedding).
    Uses cached VoiceEncoder — no overhead after first call.
    """
    try:
        from resemblyzer import preprocess_wav
    except ImportError:
        raise ImportError("resemblyzer not installed. Run: pip install resemblyzer")

    encoder = _get_voice_encoder()
    wav     = preprocess_wav(y, source_sr=sr)

    if len(wav) < 160:
        return np.zeros(256, dtype=np.float32)

    try:
        return encoder.embed_utterance(wav).astype(np.float32)
    except Exception as e:
        logger.warning(f"d-vector extraction failed: {e} — returning zeros")
        return np.zeros(256, dtype=np.float32)


def build_feature_vector(y: np.ndarray,
                          sr: int = AUDIO_SAMPLE_RATE) -> np.ndarray:
    """
    Combine MFCC (80) + d-vector (256) = 336 dims → pad/truncate to FEATURE_DIM (298).
    """
    combined = np.concatenate([
        extract_mfcc_features(y, sr),      # (80,)
        extract_dvector_features(y, sr),   # (256,)
    ]).astype(np.float32)                  # (336,)

    if len(combined) < FEATURE_DIM:
        combined = np.pad(combined, (0, FEATURE_DIM - len(combined)))
    else:
        combined = combined[:FEATURE_DIM]

    return combined                        # (298,)


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def train_acoustic_model(X: np.ndarray,
                          y_labels: np.ndarray,
                          save_model: bool = True) -> dict:
    """
    Train the DNN on extracted feature vectors.

    Fixes applied:
      FIX 3 — Class-weighted loss (handles imbalanced Agent/Customer counts)
      FIX 4 — Reduced dropout for small datasets (<50 samples → 0.1)

    Parameters
    ----------
    X          : (N, FEATURE_DIM) feature matrix
    y_labels   : (N,) integer labels  {0=Agent, 1=Customer}
    save_model : save weights to ACOUSTIC_MODEL_PATH

    Returns
    -------
    dict: history, model, scaler_path
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}")

    n_samples = len(X)
    logger.info(
        f"Training acoustic DNN | samples={n_samples} | "
        f"features={X.shape[1]} | epochs={ACOUSTIC_EPOCHS}"
    )

    # ── Normalise features (MUST match inference-time normalisation) ──────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Save scaler — loaded during inference so features are on the same scale
    scaler_path = os.path.join(
        os.path.dirname(ACOUSTIC_MODEL_PATH), "scaler.npy"
    )
    np.save(scaler_path, {"mean": scaler.mean_, "scale": scaler.scale_})
    logger.info(f"Scaler saved → {scaler_path}")

    # ── FIX 4: Adjust dropout based on dataset size ───────────────────────
    # Small dataset (< 50 samples) → dropout 0.1 to prevent under-fitting.
    # Larger dataset → use configured ACOUSTIC_DROPOUT (default 0.3).
    dropout_rate = 0.1 if n_samples < 50 else ACOUSTIC_DROPOUT
    logger.info(f"Dropout rate: {dropout_rate} "
                f"({'small dataset' if n_samples < 50 else 'normal'})")

    # ── Train / val split ─────────────────────────────────────────────────
    # Need at least 2 samples per class to stratify.
    n_agent    = int(np.sum(y_labels == 0))
    n_customer = int(np.sum(y_labels == 1))
    can_split  = (n_samples >= 4 and n_agent >= 2 and n_customer >= 2)

    if can_split:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_scaled, y_labels,
            test_size=0.2, random_state=42,
            stratify=y_labels,
        )
    else:
        logger.warning(
            f"Too few samples to split (Agent={n_agent}, Customer={n_customer}). "
            "Using full dataset for both train and val."
        )
        X_tr, y_tr   = X_scaled, y_labels
        X_val, y_val = X_scaled, y_labels

    X_tr_t  = torch.FloatTensor(X_tr).to(DEVICE)
    y_tr_t  = torch.LongTensor(y_tr).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.LongTensor(y_val).to(DEVICE)

    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=min(ACOUSTIC_BATCH_SIZE, len(X_tr)),
        shuffle=True,
    )

    # ── FIX 3: Class-weighted loss ────────────────────────────────────────
    # Prevent the model from collapsing to always predicting the majority class.
    # weight[c] = total_samples / (n_classes × count[c])
    n_total   = len(y_tr)
    n_classes = 2
    w_agent    = n_total / (n_classes * max(int(np.sum(y_tr == 0)), 1))
    w_customer = n_total / (n_classes * max(int(np.sum(y_tr == 1)), 1))
    class_weights = torch.FloatTensor([w_agent, w_customer]).to(DEVICE)
    logger.info(
        f"Class weights — Agent: {w_agent:.3f}, Customer: {w_customer:.3f}"
    )

    model     = _build_model(dropout_rate=dropout_rate).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)   # FIX 3
    optimizer = optim.Adam(model.parameters(), lr=ACOUSTIC_LR,
                           weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    model.train()
    for epoch in range(ACOUSTIC_EPOCHS):
        epoch_loss = correct = total = 0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(y_batch)
            correct    += (torch.argmax(logits, 1) == y_batch).sum().item()
            total      += len(y_batch)

        scheduler.step()
        train_loss = epoch_loss / total
        train_acc  = correct   / total

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss   = criterion(val_logits, y_val_t).item()
            val_preds  = torch.argmax(val_logits, 1)
            val_acc    = (val_preds == y_val_t).float().mean().item()
        model.train()

        history["train_loss"].append(round(train_loss, 5))
        history["train_acc"].append(round(train_acc,  4))
        history["val_loss"].append(round(val_loss,    5))
        history["val_acc"].append(round(val_acc,      4))

        if (epoch + 1) % 20 == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{ACOUSTIC_EPOCHS} | "
                f"train_loss={train_loss:.4f} acc={train_acc*100:.1f}% | "
                f"val_loss={val_loss:.4f} val_acc={val_acc*100:.1f}%"
            )

    if save_model:
        os.makedirs(os.path.dirname(ACOUSTIC_MODEL_PATH), exist_ok=True)
        torch.save(model.state_dict(), ACOUSTIC_MODEL_PATH)
        logger.info(f"Model saved → {ACOUSTIC_MODEL_PATH}")

    return {"history": history, "model": model, "scaler_path": scaler_path}


# ─────────────────────────────────────────────────────────────
# FIX 1 — SCALER LOADING FOR INFERENCE
# ─────────────────────────────────────────────────────────────

# Cached scaler data loaded from models/scaler.npy
_scaler_cache: dict = None


def _load_scaler() -> dict:
    """
    Load and cache the StandardScaler parameters saved during training.
    Returns dict with 'mean' and 'scale' arrays.
    """
    global _scaler_cache
    if _scaler_cache is not None:
        return _scaler_cache

    scaler_path = os.path.join(
        os.path.dirname(ACOUSTIC_MODEL_PATH), "scaler.npy"
    )
    if not os.path.isfile(scaler_path):
        logger.warning(
            f"Scaler not found at {scaler_path}. "
            "Features will NOT be normalised — predictions may be inaccurate."
        )
        return None

    try:
        data = np.load(scaler_path, allow_pickle=True).item()
        _scaler_cache = {
            "mean":  np.array(data["mean"],  dtype=np.float32),
            "scale": np.array(data["scale"], dtype=np.float32),
        }
        logger.debug("Scaler loaded and cached.")
        return _scaler_cache
    except Exception as e:
        logger.warning(f"Failed to load scaler: {e}")
        return None


def _apply_scaler(features: np.ndarray) -> np.ndarray:
    """
    Apply StandardScaler normalisation: (x - mean) / scale.
    This MUST match exactly what was done during training.
    """
    scaler = _load_scaler()
    if scaler is None:
        return features   # fallback: return unchanged

    mean  = scaler["mean"]
    scale = scaler["scale"]

    # Avoid division by zero on constant features
    safe_scale = np.where(scale < 1e-8, 1.0, scale)
    return (features - mean) / safe_scale


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────

# Cached inference model
_inference_model = None


def _get_inference_model():
    """Load and cache the trained model for inference."""
    global _inference_model
    if _inference_model is not None:
        return _inference_model

    import torch
    model = _build_model()
    if os.path.isfile(ACOUSTIC_MODEL_PATH):
        model.load_state_dict(
            torch.load(ACOUSTIC_MODEL_PATH, map_location=DEVICE)
        )
        logger.info("Loaded trained acoustic model for inference.")
    else:
        logger.warning(
            "No trained acoustic model found. "
            "Run  python train.py  first. "
            "Using random weights until then."
        )
    model = model.to(DEVICE)
    model.eval()
    _inference_model = model
    return _inference_model


def classify_segment_acoustic(feature_vector: np.ndarray,
                               model=None) -> dict:
    """
    Classify a single feature vector as Agent or Customer.

    FIX 1 applied here: feature_vector is normalised with the saved scaler
    before being passed to the model.

    Parameters
    ----------
    feature_vector : raw np.ndarray shape (FEATURE_DIM,) from build_feature_vector()
    model          : optional pre-loaded PyTorch model

    Returns
    -------
    dict: predicted_role, confidence, agent_prob, customer_prob, method
    """
    import torch
    import torch.nn.functional as F

    if model is None:
        model = _get_inference_model()

    # FIX 1: Normalise with the same scaler used during training
    scaled = _apply_scaler(feature_vector)

    model.eval()
    x = torch.FloatTensor(scaled).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    agent_prob    = float(probs[0])
    customer_prob = float(probs[1])

    if agent_prob > customer_prob:
        predicted_role = "Agent"
        confidence     = round(agent_prob, 4)
    else:
        predicted_role = "Customer"
        confidence     = round(customer_prob, 4)

    return {
        "predicted_role": predicted_role,
        "confidence":     confidence,
        "agent_prob":     round(agent_prob,    4),
        "customer_prob":  round(customer_prob, 4),
        "method":         "acoustic",
    }


def classify_transcript_acoustic(diarized: list[dict],
                                   audio_segments: dict,
                                   model=None) -> list[dict]:
    """
    Classify all segments in a diarized transcript using acoustic features.

    Parameters
    ----------
    diarized       : [{speaker_id, text, start, end, ...}]
    audio_segments : {speaker_id: np.ndarray}  — full audio per speaker
    model          : optional pre-loaded PyTorch model

    Returns
    -------
    classified : list with acoustic classification fields added
    """
    from collections import defaultdict

    # Use cached model for all segments in this call
    if model is None:
        model = _get_inference_model()

    classified = []
    for seg in diarized:
        spk   = seg["speaker_id"]
        y_seg = audio_segments.get(spk)

        if y_seg is None or len(y_seg) < 1600:   # < 0.1s
            result = {
                "predicted_role": "Unknown",
                "confidence":     0.5,
                "agent_prob":     0.5,
                "customer_prob":  0.5,
                "method":         "acoustic",
            }
        else:
            fv     = build_feature_vector(y_seg)
            result = classify_segment_acoustic(fv, model)

        classified.append({**seg, **result})

    # Speaker-level majority vote (weighted by confidence)
    speaker_votes = defaultdict(lambda: {"Agent": 0.0, "Customer": 0.0})
    for seg in classified:
        spk  = seg["speaker_id"]
        role = seg["predicted_role"]
        conf = seg["confidence"]
        if role in speaker_votes[spk]:
            speaker_votes[spk][role] += conf

    speaker_roles = {}
    for spk, votes in speaker_votes.items():
        total = votes["Agent"] + votes["Customer"]
        if total < 1e-9:
            speaker_roles[spk] = ("Unknown", 0.5)
        elif votes["Agent"] >= votes["Customer"]:
            speaker_roles[spk] = ("Agent",
                                  round(votes["Agent"] / total, 4))
        else:
            speaker_roles[spk] = ("Customer",
                                  round(votes["Customer"] / total, 4))

    for seg in classified:
        spk = seg["speaker_id"]
        seg["predicted_role"]   = speaker_roles[spk][0]
        seg["final_confidence"] = speaker_roles[spk][1]

    logger.info(
        f"Method 2 acoustic complete | "
        f"roles={dict((k, v[0]) for k, v in speaker_roles.items())}"
    )
    return classified


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_training_history(history: dict, save_path: str = None) -> str:
    """Plot training and validation accuracy/loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history["train_acc"]) + 1)

    axes[0].plot(epochs, history["train_acc"],
                 label="Train Accuracy", color="steelblue")
    axes[0].plot(epochs, history["val_acc"],
                 label="Val Accuracy", color="tomato", linestyle="--")
    axes[0].set_title("Training & Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_loss"],
                 label="Train Loss", color="steelblue")
    axes[1].plot(epochs, history["val_loss"],
                 label="Val Loss", color="tomato", linestyle="--")
    axes[1].set_title("Training & Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Method 2: Acoustic DNN Training History", fontsize=13)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "m2_training_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Training history plot saved → {save_path}")
    return save_path
