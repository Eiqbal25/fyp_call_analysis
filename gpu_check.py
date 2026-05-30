"""
gpu_check.py
============
System readiness check before running the pipeline.
Run this once to confirm GPU, packages, API keys, and data files are ready.

Usage:
    python gpu_check.py
"""

import sys
import os
import glob
import subprocess
import platform

print("=" * 62)
print("FYP1 CALL ANALYSIS — SYSTEM READINESS CHECK")
print("=" * 62)

# ── Python & OS ───────────────────────────────────────────────
print(f"\n Python : {sys.version.split()[0]}")
print(f" OS     : {platform.system()} {platform.release()}")

# ── GPU / PyTorch ─────────────────────────────────────────────
print("\n── GPU / PyTorch ─────────────────────────────────────────")
try:
    import torch
    print(f" PyTorch         : ✅ {torch.__version__}")
    print(f" CUDA available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f" GPU             : {torch.cuda.get_device_name(0)}")
        print(f" CUDA version    : {torch.version.cuda}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f" GPU memory      : {mem:.1f} GB")
        a = torch.randn(1000, 1000, device="cuda")
        b = torch.randn(1000, 1000, device="cuda")
        _ = torch.mm(a, b)
        print(f" GPU compute     : ✅ PASSED")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f" Apple MPS GPU   : ✅ Available")
    else:
        print(f" ⚠  No GPU — will use CPU (slower but works)")
        print(f"   Note: Whisper transcription runs on Colab (T4 GPU) — not needed locally.")

except ImportError:
    print(" ❌ PyTorch not installed")
    print("    Managed by Conda — activate fyp2 environment first")

# ── Whisper ASR ───────────────────────────────────────────────
print("\n── Whisper ASR ───────────────────────────────────────────")
try:
    import whisper
    print(f" openai-whisper  : ✅ Installed")
    try:
        from config import WHISPER_MODEL_SIZE, WHISPER_DEVICE
        print(f" Model size      : {WHISPER_MODEL_SIZE}")
        print(f" Will run on     : {WHISPER_DEVICE}")
        print(f" Note            : Transcription runs on Colab T4 GPU (--skip_transcription locally)")
    except Exception:
        pass
except ImportError:
    print(" ❌ openai-whisper not installed — run: pip install openai-whisper")

# ── pyannote.audio (Primary Diarization) ─────────────────────
print("\n── pyannote.audio (Primary Diarization) ─────────────────")
try:
    import pyannote.audio
    print(f" pyannote.audio  : ✅ Installed ({pyannote.audio.__version__})")

    # Check HuggingFace token
    hf_token = os.environ.get("HUGGINGFACE_TOKEN", "")
    if not hf_token:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("HUGGINGFACE_TOKEN"):
                        hf_token = line.split("=", 1)[-1].strip().strip('"')

    if hf_token and hf_token.startswith("hf_"):
        print(f" HF Token        : ✅ Set (hf_...{hf_token[-4:]})")
    else:
        print(f" HF Token        : ❌ Not set")
        print(f"   Add to .env:  HUGGINGFACE_TOKEN=hf_your_token_here")
        print(f"   Get token at: https://huggingface.co/settings/tokens")
        print(f"   Accept licenses:")
        print(f"     https://huggingface.co/pyannote/speaker-diarization-3.1")
        print(f"     https://huggingface.co/pyannote/segmentation-3.0")

except ImportError:
    print(" ❌ pyannote.audio not installed — pip install pyannote.audio")
    print("    System will fall back to Resemblyzer (lower accuracy)")

# ── Resemblyzer (Fallback + fix_speakers.py) ─────────────────
print("\n── Resemblyzer (Fallback + fix_speakers.py) ─────────────")
try:
    from resemblyzer import VoiceEncoder
    print(f" resemblyzer     : ✅ Installed")
    print(f"   Used by: fix_speakers.py (re-clusters single-speaker diarizations)")
except ImportError:
    print(f" resemblyzer     : ❌ Not installed — pip install resemblyzer")
    print(f"   Required for fix_speakers.py and Method 2 d-vector features")

# ── Groq API (Method 3 — LLM Hybrid) ─────────────────────────
print("\n── Groq API (Method 3 — LLM Hybrid) ─────────────────────")
try:
    import groq
    print(f" groq            : ✅ Installed ({groq.__version__})")

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("GROQ_API_KEY"):
                        groq_key = line.split("=", 1)[-1].strip().strip('"')

    if groq_key and groq_key.startswith("gsk_"):
        masked = groq_key[:8] + "..." + groq_key[-4:]
        print(f" Groq API key   : ✅ Set ({masked})")
        print(f" LLM model      : llama-3.3-70b-versatile (free tier)")
        print(f" Daily limit    : 100,000 tokens (~15 calls/day)")
    else:
        print(f" Groq API key   : ❌ Not set")
        print(f"   Add to .env: GROQ_API_KEY=gsk_your_key_here")
        print(f"   Get free key at: https://console.groq.com")
        print(f"   Without this, Method 3 (LLM Hybrid) will be disabled")
except ImportError:
    print(f" groq            : ❌ Not installed — pip install groq")

# ── Audio Processing ─────────────────────────────────────────
print("\n── Audio Processing ──────────────────────────────────────")
for pkg in ["librosa", "soundfile", "noisereduce"]:
    try:
        __import__(pkg)
        print(f" {pkg:15s} : ✅")
    except ImportError:
        print(f" {pkg:15s} : ❌ — pip install {pkg}")

# ── ffmpeg ────────────────────────────────────────────────────
print("\n── ffmpeg (required for audio loading) ───────────────────")
result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
if result.returncode == 0:
    version_line = result.stdout.split("\n")[0]
    print(f" ffmpeg          : ✅ {version_line[:55]}")
else:
    print(" ffmpeg          : ❌ Not found in PATH")
    print("   Windows: https://ffmpeg.org → download → add to PATH")
    print("   macOS:   brew install ffmpeg")
    print("   Ubuntu:  sudo apt install ffmpeg")

# ── Other Packages ────────────────────────────────────────────
print("\n── Other Packages ────────────────────────────────────────")
packages = ["sklearn", "scipy", "pandas", "numpy",
            "matplotlib", "plotly", "streamlit",
            "vaderSentiment", "tqdm", "dotenv"]
for pkg in packages:
    import_name = "vaderSentiment.vaderSentiment" if pkg == "vaderSentiment" else pkg
    display_name = pkg
    try:
        mod = __import__(import_name.split(".")[0])
        ver = getattr(mod, "__version__", "?")
        print(f" {display_name:20s}: ✅ {ver}")
    except ImportError:
        print(f" {display_name:20s}: ❌ — pip install {pkg.replace('dotenv','python-dotenv')}")

# ── Data Files ────────────────────────────────────────────────
print("\n── Data Files ────────────────────────────────────────────")
try:
    from config import DATA_DIR, GROUND_TRUTH_CSV, HUMAN_TRANSCRIPTS_DIR, COLAB_TRANSCRIPTS_DIR

    audio_files = (glob.glob(os.path.join(DATA_DIR, "*.wav")) +
                   glob.glob(os.path.join(DATA_DIR, "*.mp3")))
    if audio_files:
        print(f" Audio files     : ✅ {len(audio_files)} file(s) in data/")
        for f in sorted(audio_files)[:6]:
            print(f"   • {os.path.basename(f)}")
        if len(audio_files) > 6:
            print(f"   ... and {len(audio_files)-6} more")
    else:
        print(f" Audio files     : ❌ No .wav/.mp3 in data/")
        print(f"   Add your audio recordings to the data/ folder")

    colab_jsons = glob.glob(os.path.join(COLAB_TRANSCRIPTS_DIR, "*_diarized.json"))
    colab_jsons = [f for f in colab_jsons if "_original" not in f]
    if colab_jsons:
        print(f" Colab transcripts: ✅ {len(colab_jsons)} diarized JSON(s) in colab_transcripts/")
    else:
        print(f" Colab transcripts: ⚠  None found — run Colab notebook first")

    import pandas as pd
    if os.path.isfile(GROUND_TRUTH_CSV):
        df = pd.read_csv(GROUND_TRUTH_CSV)
        calls = df["call_id"].nunique() if "call_id" in df.columns else "?"
        print(f" Ground truth CSV: ✅ {len(df)} rows across {calls} calls")
    else:
        print(f" Ground truth CSV: ❌ {GROUND_TRUTH_CSV} not found")

    ht_files = glob.glob(os.path.join(HUMAN_TRANSCRIPTS_DIR, "*.csv"))
    ht_files = [f for f in ht_files if "example" not in os.path.basename(f).lower()]
    if ht_files:
        print(f" Human transcripts: ✅ {len(ht_files)} CSV(s) in human_transcripts/")
    else:
        print(f" Human transcripts: ⚠  None found in human_transcripts/")
        print(f"   Needed for: python compare_labels.py")

except Exception as e:
    print(f" ❌ Could not check data files: {e}")

# ── Active Configuration ──────────────────────────────────────
print("\n── Active Configuration ──────────────────────────────────")
try:
    from config import (DEVICE, GPU_NAME, WHISPER_MODEL_SIZE,
                        HYBRID_ALPHA, HYBRID_BETA, LLM_MODEL, RUN_NAME)
    print(f" Device          : {DEVICE} ({GPU_NAME})")
    print(f" Whisper model   : {WHISPER_MODEL_SIZE} (runs on Colab)")
    print(f" LLM model       : {LLM_MODEL}")
    print(f" Hybrid α / β    : {HYBRID_ALPHA} / {HYBRID_BETA}")
    print(f" Output folder   : outputs/{RUN_NAME}/")
except Exception as e:
    print(f" ❌ Could not load config: {e}")

# ── Run Order ─────────────────────────────────────────────────
print("\n" + "=" * 62)
print("CORRECT RUN ORDER:")
print("=" * 62)
print("  1.  python gpu_check.py                 ← you are here")
print("  2.  pip install -r requirements.txt      ← install packages")
print("  3.  python train.py                      ← train Method 2 DNN (once)")
print("  4.  [Colab] FYP_Transcribe_Colab.ipynb   ← transcribe on GPU")
print("  5.  python preflight_check.py --day 1    ← validate Day 1 files")
print("  6.  run_day.bat 1                        ← process Day 1 (15 calls)")
print("  7.  python preflight_check.py --day 2    ← validate Day 2 files")
print("  8.  run_day.bat 2                        ← process Day 2 + evaluate")
print("  9.  streamlit run dashboard/app.py       ← open dashboard")
print("")
print("  Single call (quick test):")
print("  python main.py --skip_transcription --call_id eng_prof_01")
print("")
print("  Skip Method 2 DNN (faster, no GPU needed locally):")
print("  python main.py --skip_transcription --skip_acoustic")
print("")
print("  Check processing status:")
print("  python run_tracker.py")
print("=" * 62)
