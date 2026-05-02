"""
gpu_check.py
============
System readiness check before running the pipeline.
Run this once to confirm GPU, packages, token, and data files are ready.

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
    try:
        from config import HUGGINGFACE_TOKEN
        if HUGGINGFACE_TOKEN and HUGGINGFACE_TOKEN.startswith("hf_"):
            print(f" HF Token        : ✅ Set (hf_...{HUGGINGFACE_TOKEN[-4:]})")
        else:
            print(f" HF Token        : ❌ Not set in config.py")
            print(f"   Add: HUGGINGFACE_TOKEN = 'hf_your_token_here'")
            print(f"   Get token at: https://huggingface.co/settings/tokens")
    except ImportError:
        print(f" HF Token        : ❌ Cannot read config.py")

    # Check model licenses
    print(f" Model licenses  : Accept at:")
    print(f"   https://huggingface.co/pyannote/speaker-diarization-3.1")
    print(f"   https://huggingface.co/pyannote/segmentation-3.0")

except ImportError:
    print(" ❌ pyannote.audio not installed")
    print("    Run: pip install pyannote.audio")
    print("    System will fall back to Resemblyzer (lower accuracy)")

# ── Resemblyzer (Fallback Diarization) ───────────────────────
print("\n── Resemblyzer (Fallback Diarization) ───────────────────")
try:
    from resemblyzer import VoiceEncoder
    print(f" resemblyzer     : ✅ Installed (fallback ready)")
except ImportError:
    print(f" resemblyzer     : ⚠  Not installed")
    print(f"    Only needed as fallback. Run: pip install resemblyzer")

# ── Audio Processing ─────────────────────────────────────────
print("\n── Audio Processing ──────────────────────────────────────")
for pkg in ["librosa", "soundfile", "noisereduce"]:
    try:
        __import__(pkg)
        print(f" {pkg:15s} : ✅")
    except ImportError:
        print(f" {pkg:15s} : ❌ — pip install {pkg}")

# ── ffmpeg ────────────────────────────────────────────────────
print("\n── ffmpeg (required for .mp3 files) ──────────────────────")
result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
if result.returncode == 0:
    version_line = result.stdout.split("\n")[0]
    print(f" ffmpeg          : ✅ {version_line[:50]}")
else:
    print(" ffmpeg          : ❌ Not found in PATH")
    print("   Windows: https://ffmpeg.org → download → add to PATH")
    print("   macOS:   brew install ffmpeg")
    print("   Ubuntu:  sudo apt install ffmpeg")

# ── Other Packages ────────────────────────────────────────────
print("\n── Other Packages ────────────────────────────────────────")
packages = ["sklearn", "scipy", "pandas", "numpy",
            "matplotlib", "plotly", "streamlit", "vaderSentiment"]
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "?")
        print(f" {pkg:20s}: ✅ {ver}")
    except ImportError:
        print(f" {pkg:20s}: ❌ — pip install {pkg}")

# ── Data Files ────────────────────────────────────────────────
print("\n── Data Files ────────────────────────────────────────────")
try:
    from config import DATA_DIR, GROUND_TRUTH_CSV, HUMAN_TRANSCRIPTS_DIR

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

    import pandas as pd
    if os.path.isfile(GROUND_TRUTH_CSV):
        df = pd.read_csv(GROUND_TRUTH_CSV)
        calls = df["call_id"].nunique()
        print(f" Ground truth CSV: ✅ {len(df)} rows across {calls} calls")
    else:
        print(f" Ground truth CSV: ❌ {GROUND_TRUTH_CSV} not found")

    ht_files = glob.glob(os.path.join(HUMAN_TRANSCRIPTS_DIR, "*.csv"))
    ht_files = [f for f in ht_files if "example" not in os.path.basename(f).lower()]
    if ht_files:
        print(f" Human transcripts: ✅ {len(ht_files)} CSV(s) in human_transcripts/")
        for f in sorted(ht_files):
            print(f"   • {os.path.basename(f)}")
    else:
        print(f" Human transcripts: ⚠  None found in human_transcripts/")
        print(f"   Needed for: python compare_labels.py")

except Exception as e:
    print(f" ❌ Could not check data files: {e}")

# ── Active Configuration ──────────────────────────────────────
print("\n── Active Configuration ──────────────────────────────────")
try:
    from config import DEVICE, GPU_NAME, WHISPER_MODEL_SIZE, HYBRID_ALPHA, HYBRID_BETA
    print(f" Device          : {DEVICE} ({GPU_NAME})")
    print(f" Whisper model   : {WHISPER_MODEL_SIZE}")
    print(f" Hybrid α / β    : {HYBRID_ALPHA} / {HYBRID_BETA}")
except Exception as e:
    print(f" ❌ Could not load config: {e}")

# ── Run Order ─────────────────────────────────────────────────
print("\n" + "=" * 62)
print("CORRECT RUN ORDER:")
print("=" * 62)
print("  1. python gpu_check.py              ← you are here")
print("  2. pip install -r requirements.txt  ← install packages")
print("  3. python train.py                  ← train Method 2 DNN")
print("  4. python main.py                   ← process all calls")
print("  5. python evaluate.py               ← accuracy vs ground truth")
print("  6. python compare_labels.py         ← per-segment label diff")
print("  7. streamlit run dashboard/app.py   ← open dashboard")
print("")
print("  Fast run (no Method 2 DNN needed):")
print("  python main.py --skip_acoustic")
print("")
print("  Process one specific call only:")
print("  python main.py --call_id food_malay")
print("=" * 62)
