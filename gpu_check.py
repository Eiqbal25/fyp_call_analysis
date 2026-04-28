"""
gpu_check.py
============
Check GPU availability and print a full system readiness report.
Run this before train.py or main.py to confirm everything is set up.

Usage:
    python gpu_check.py
"""

import sys
import os

print("=" * 60)
print("FYP1 SYSTEM READINESS CHECK")
print("=" * 60)

# ── Python version ────────────────────────────────────────────
import platform
print(f"\n Python : {sys.version}")
print(f" OS     : {platform.system()} {platform.release()}")

# ── GPU / PyTorch ─────────────────────────────────────────────
print("\n── GPU / PyTorch ────────────────────────────────────────")
try:
    import torch
    print(f" PyTorch version : {torch.__version__}")
    print(f" CUDA available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f" GPU name        : {torch.cuda.get_device_name(0)}")
        print(f" CUDA version    : {torch.version.cuda}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f" GPU memory      : {mem:.1f} GB")

        # Quick compute test
        a = torch.randn(1000, 1000, device="cuda")
        b = torch.randn(1000, 1000, device="cuda")
        c = torch.mm(a, b)
        print(f" GPU compute test: ✅ PASSED")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f" Apple MPS GPU   : ✅ Available")
    else:
        print(f" ⚠ No GPU found — will use CPU (slower but works)")
        print(f"   To use your NVIDIA GPU, install the CUDA version of PyTorch:")
        print(f"   pip install torch --index-url https://download.pytorch.org/whl/cu118")

except ImportError:
    print(" ❌ PyTorch NOT installed — run: pip install torch")

# ── Whisper ───────────────────────────────────────────────────
print("\n── Whisper ASR ──────────────────────────────────────────")
try:
    import whisper
    print(f" openai-whisper  : ✅ Installed")
    from config import WHISPER_MODEL_SIZE, WHISPER_DEVICE
    print(f" Model size      : {WHISPER_MODEL_SIZE}")
    print(f" Will run on     : {WHISPER_DEVICE}")
except ImportError:
    print(" ❌ openai-whisper NOT installed — run: pip install openai-whisper")

# ── Resemblyzer ───────────────────────────────────────────────
print("\n── Resemblyzer (Speaker Diarization) ───────────────────")
try:
    from resemblyzer import VoiceEncoder
    print(f" resemblyzer     : ✅ Installed")
except ImportError:
    print(" ❌ resemblyzer NOT installed — run: pip install resemblyzer")

# ── Audio ─────────────────────────────────────────────────────
print("\n── Audio Processing ─────────────────────────────────────")
for pkg in ["librosa", "soundfile", "noisereduce"]:
    try:
        __import__(pkg)
        print(f" {pkg:15s}: ✅")
    except ImportError:
        print(f" {pkg:15s}: ❌ — run: pip install {pkg}")

# ── ffmpeg ────────────────────────────────────────────────────
print("\n── ffmpeg (required for .mp3 files) ─────────────────────")
import subprocess
result = subprocess.run(
    ["ffmpeg", "-version"], capture_output=True, text=True
)
if result.returncode == 0:
    version_line = result.stdout.split("\n")[0]
    print(f" ffmpeg          : ✅ {version_line}")
else:
    print(" ffmpeg          : ❌ NOT found")
    print("   Windows: download from https://ffmpeg.org and add to PATH")
    print("   Mac:     brew install ffmpeg")
    print("   Ubuntu:  sudo apt install ffmpeg")

# ── Other packages ────────────────────────────────────────────
print("\n── Other Packages ───────────────────────────────────────")
other_packages = [
    "sklearn", "scipy", "pandas", "numpy",
    "matplotlib", "plotly", "streamlit", "vaderSentiment"
]
for pkg in other_packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "?")
        print(f" {pkg:20s}: ✅ {ver}")
    except ImportError:
        print(f" {pkg:20s}: ❌ — run: pip install {pkg}")

# ── Data files ────────────────────────────────────────────────
print("\n── Data Files ───────────────────────────────────────────")
import glob
from config import DATA_DIR, GROUND_TRUTH_CSV

audio_files = glob.glob(os.path.join(DATA_DIR, "*.wav")) + \
              glob.glob(os.path.join(DATA_DIR, "*.mp3"))
if audio_files:
    print(f" Audio files     : ✅ {len(audio_files)} file(s) in data/")
    for f in audio_files[:5]:
        print(f"   • {os.path.basename(f)}")
    if len(audio_files) > 5:
        print(f"   ... and {len(audio_files)-5} more")
else:
    print(f" Audio files     : ❌ No .wav/.mp3 files in data/")
    print(f"   Place your call recordings in the data/ folder")

if os.path.isfile(GROUND_TRUTH_CSV):
    import pandas as pd
    df = pd.read_csv(GROUND_TRUTH_CSV)
    print(f" Ground truth    : ✅ {GROUND_TRUTH_CSV} ({len(df)} rows)")
else:
    print(f" Ground truth    : ⚠ Not found — run label_tool.py after main.py")

# ── Config device summary ─────────────────────────────────────
print("\n── Active Configuration ─────────────────────────────────")
try:
    from config import DEVICE, GPU_NAME, WHISPER_MODEL_SIZE
    print(f" Active device   : {DEVICE} ({GPU_NAME})")
    print(f" Whisper model   : {WHISPER_MODEL_SIZE}")
except Exception as e:
    print(f" Could not load config: {e}")

print("\n" + "=" * 60)
print("Run order:")
print("  1. python gpu_check.py       ← you are here")
print("  2. pip install -r requirements.txt")
print("  3. python main.py --skip_acoustic   ← first run (fast)")
print("  4. python label_tool.py             ← label your data")
print("  5. python train.py                  ← train Method 2 DNN")
print("  6. python main.py                   ← full run with GPU")
print("  7. python evaluate.py               ← see true accuracy")
print("  8. streamlit run dashboard/app.py   ← open dashboard")
print("=" * 60)
