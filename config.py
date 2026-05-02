"""
config.py
=========
Central configuration file for the FYP1 Call Analysis System.
All paths, constants, and hyperparameters are defined here.
"""

import os

# ─────────────────────────────────────────────
# BASE PATHS
# ─────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
KEYWORDS_DIR = os.path.join(BASE_DIR, "keywords")

# ─────────────────────────────────────────────
# RUN MANAGEMENT
# ─────────────────────────────────────────────
# Change RUN_NAME to save results into a named folder you can keep.
#
#   "latest"  →  outputs/latest/   (always overwritten — good for quick runs)
#   "run_01"  →  outputs/run_01/   (archived — share this folder with Claude)
#   "run_02"  →  outputs/run_02/   (another archived run)
#
# Usage:
#   Just edit RUN_NAME below, then run:  python main.py
#
# To share with Claude for analysis, zip or upload only:
#   outputs/<RUN_NAME>/evaluation_report.txt
#   outputs/<RUN_NAME>/label_comparison_report.txt
#   outputs/<RUN_NAME>/analytics_summary.csv
#   + any specific .py file you want to improve
#
RUN_NAME    = "latest"
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", RUN_NAME)

# Ensure all output directories exist
for _d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────
# AUDIO PREPROCESSING
# ─────────────────────────────────────────────
AUDIO_SAMPLE_RATE     = 16000   # Hz — optimal for Whisper & Resemblyzer
AUDIO_CHANNELS        = 1       # Mono
AUDIO_NORMALIZE_PEAK  = 1.0     # Amplitude normalization ceiling
NOISE_STATIONARY_PROP = 0.1     # Fraction of audio used for noise profile

# ─────────────────────────────────────────────
# TRANSCRIPTION (OpenAI Whisper)
# ─────────────────────────────────────────────

# Model size — UPGRADED from "base" to "small"
# base  : ~74M params, WER ~44% on Malaysian English, ~1 min/call on GPU
# small : ~244M params, WER ~28% on Malaysian English, ~2 min/call on GPU
# medium: ~769M params, WER ~22% on Malaysian English, ~4 min/call on GPU
WHISPER_MODEL_SIZE = "small"

# ─────────────────────────────────────────────
# HUGGINGFACE TOKEN (required for pyannote.audio)
# ─────────────────────────────────────────────
# Get your token from: https://huggingface.co/settings/tokens
# Accept model licenses at:
#   https://huggingface.co/pyannote/speaker-diarization-3.1
#   https://huggingface.co/pyannote/segmentation-3.0
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", None)

# ─────────────────────────────────────────────────────────────
# METHOD 4 — LLM CLASSIFICATION (FREE — Google Gemini)
# ─────────────────────────────────────────────────────────────
# Set your Google AI Studio API key to enable LLM-based classification.
# Get a FREE key at: https://aistudio.google.com/apikey
# Cost: $0.00 — Gemini Flash free tier (15 req/min, 1M tokens/day).
# Leave as None to disable Method 4.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)  # e.g. "AIzaSy..."
LLM_MODEL = "gemini-2.0-flash"


# Language — None = auto-detect each call.
# Set to "ms" to force Malay, "en" to force English.
# Leave as None for mixed-language calls (auto-detect per call).
WHISPER_LANGUAGE = None

# Per-call language detection — maps filename suffix to Whisper language code.
# If the call_id contains "_malay" the transcriber forces Malay (ms).
# If it contains "_english" it forces English (en).
# Any other name → None (auto-detect per Whisper).
# You can extend this map freely — e.g. {"_mandarin": "zh"}.
WHISPER_LANGUAGE_MAP = {
    "_malay":   "ms",
    "_english": "en",
}

# Initial prompt — primes Whisper to expect Malaysian English patterns.
# This reduces errors on Manglish words, code-switching, and
# Malaysian-accented English that "base" frequently mistranscribed.
# Whisper uses this as context for the first audio window only.
WHISPER_INITIAL_PROMPT = (
    "This is a customer service call recording in Malaysian English or Malay. "
    "Common words include: boleh, lah, encik, cik, saya, nak, tak, ya, "
    "okay, refund, account, policy, booking, order, delivery, complaint."
)

# Prevent Whisper from hallucinating repeated phrases on noisy/Manglish audio.
# When True, Whisper feeds its own previous output back as context — this
# causes the "hallucination loop" problem (e.g. repeating the same sentence).
# Setting to False makes each segment transcribed independently.
WHISPER_CONDITION_ON_PREVIOUS_TEXT = False

# Beam search size — small model benefits from beam_size=5 over greedy (=1).
# Higher = better accuracy, slightly slower. 5 is the recommended default.
WHISPER_BEAM_SIZE = 5

# ─────────────────────────────────────────────
# GPU / DEVICE CONFIGURATION (auto-detected)
# ─────────────────────────────────────────────
try:
    import torch as _torch
    if _torch.cuda.is_available():
        DEVICE   = "cuda"
        GPU_NAME = _torch.cuda.get_device_name(0)
    elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
        DEVICE   = "mps"
        GPU_NAME = "Apple Silicon MPS"
    else:
        DEVICE   = "cpu"
        GPU_NAME = "CPU (no GPU detected)"
except ImportError:
    DEVICE   = "cpu"
    GPU_NAME = "CPU (PyTorch not installed)"

# Whisper runs on same device as PyTorch
WHISPER_DEVICE = DEVICE

# ─────────────────────────────────────────────
# SPEAKER DIARIZATION (Resemblyzer)
# ─────────────────────────────────────────────
DIARIZATION_MIN_SEGMENT_SEC = 0.5   # Minimum segment duration to keep
DIARIZATION_NUM_SPEAKERS    = 2     # Expected speakers per call (Agent + Customer)
EMBEDDING_DIM               = 256   # Resemblyzer d-vector dimensionality

# ─────────────────────────────────────────────
# METHOD 1 — LEXICAL CLASSIFIER
# ─────────────────────────────────────────────
AGENT_KEYWORDS_FILE    = os.path.join(KEYWORDS_DIR, "agent_keywords.json")
CUSTOMER_KEYWORDS_FILE = os.path.join(KEYWORDS_DIR, "customer_keywords.json")
LEXICAL_MIN_WORDS      = 3   # Minimum words needed to classify a segment

# ─────────────────────────────────────────────
# METHOD 2 — ACOUSTIC DNN CLASSIFIER (PyTorch)
# ─────────────────────────────────────────────
MFCC_N_MFCC          = 40
MFCC_HOP_LENGTH      = 512
MFCC_N_FFT           = 2048
FEATURE_DIM          = 298   # MFCCs(40×mean+std=80) + d-vector(256) — padded
ACOUSTIC_HIDDEN_DIMS = [256, 128, 64, 32]
ACOUSTIC_DROPOUT     = 0.3
ACOUSTIC_EPOCHS      = 50
ACOUSTIC_LR          = 1e-3
ACOUSTIC_BATCH_SIZE  = 8
ACOUSTIC_MODEL_PATH  = os.path.join(MODELS_DIR, "acoustic_model.pth")

# ─────────────────────────────────────────────
# METHOD 3 — HYBRID ENSEMBLE FUSION
# ─────────────────────────────────────────────
HYBRID_ALPHA           = 0.4    # Weight for lexical signal (α)
HYBRID_BETA            = 0.6    # Weight for acoustic signal (β)
HYBRID_CONFLICT_PENALTY = 0.15

# Method 3 upgrades — speaker anchoring + dynamic weights
SPEAKER_ANCHOR_WINDOW  = 8      # Check first N segments to detect Agent by greeting
HYBRID_DYNAMIC_WEIGHTS = True   # Adjust alpha/beta per-call based on model confidence
HYBRID_MIN_ALPHA       = 0.25   # Minimum lexical weight when acoustic dominates
HYBRID_MAX_ALPHA       = 0.65   # Maximum lexical weight when lexical dominates

# ─────────────────────────────────────────────
# HUMAN TRANSCRIPT COMPARISON
# ─────────────────────────────────────────────
HUMAN_TRANSCRIPTS_DIR            = os.path.join(BASE_DIR, "human_transcripts")
os.makedirs(HUMAN_TRANSCRIPTS_DIR, exist_ok=True)
LABEL_MATCH_SIMILARITY_THRESHOLD = 0.35   # Min fuzzy text similarity to accept a match

# ─────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────
SENTIMENT_POSITIVE_THRESHOLD = 0.05    # VADER compound score
SENTIMENT_NEGATIVE_THRESHOLD = -0.05

# Compliance SOP keywords (mandatory agent phrases)
COMPLIANCE_GREETING_KEYWORDS = [
    "thank you for calling", "good morning", "good afternoon",
    "good evening", "welcome", "how may i assist", "how can i help",
]
COMPLIANCE_CLOSING_KEYWORDS = [
    "thank you", "have a great day", "is there anything else",
    "goodbye", "take care", "have a nice day",
]
COMPLIANCE_RECORDED_KEYWORDS = [
    "call is being recorded", "this call may be recorded",
    "recorded for quality", "monitoring purposes",
]
COMPLIANCE_IDENTITY_KEYWORDS = [
    "may i have your name", "can i verify", "date of birth",
    "account number", "ic number", "identity",
]

# Customer risk / high-severity keywords
RISK_KEYWORDS = [
    "scam", "fraud", "lawyer", "sue", "police", "report",
    "complaint", "manager", "supervisor", "legal action",
    "threaten", "angry", "unacceptable", "cancel", "refund",
]

# QA Score weights (must sum to 1.0)
QA_SCORE_WEIGHTS = {
    "talk_balance":  0.25,
    "turn_taking":   0.20,
    "sentiment":     0.25,
    "compliance":    0.20,
    "politeness":    0.10,
}

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
GROUND_TRUTH_CSV = os.path.join(BASE_DIR, "human_validation_study.csv")
TTEST_ALPHA      = 0.05   # Significance level for paired t-test

# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────
DASHBOARD_TITLE = "Automated Call Analysis & Quality Assurance System"
DASHBOARD_PORT  = 8501
AGENT_COLOR     = "#1f77b4"   # Blue
CUSTOMER_COLOR  = "#d62728"   # Red
