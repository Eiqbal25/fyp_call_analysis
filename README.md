# Speaker Role Detection and Segmented Analysis in Customer Service Calls

**FYP1 — Muhammad Eiqbal Bin Hasbollah (2216911)**  
Department of Mechatronics Engineering, IIUM · January 2026

---

## Overview

An automated **Call Analysis System** that processes raw audio recordings of customer service calls and produces structured quality assurance (QA) intelligence.

The system solves the "who spoke when" problem through speaker diarization, then classifies each speaker as **Agent** or **Customer** using three progressively sophisticated methods, and finally extracts actionable analytics including talk-time ratios, sentiment trajectories, and compliance flags.

| Method | Approach | Accuracy |
|--------|----------|----------|
| Method 1 | Keyword Density (Lexical baseline) | ~30% |
| Method 2 | Acoustic DNN (MFCC + d-vector) | ~50% |
| **Method 3** | **Hybrid Ensemble Fusion (proposed)** | **~92%** |

**Key result:** The system processes audio **21.7× faster** than human review (RTF = 0.11), enabling 100% call coverage instead of the industry-standard <5% random sampling.

---

## Project Structure

```
fyp1_call_analysis/
│
├── config.py                   # Central configuration (paths, hyperparameters)
│
├── main.py                     # ★ Entry point — run the full pipeline
│
├── preprocessing/
│   ├── __init__.py
│   ├── audio_processor.py      # DSP: noise reduction, normalization, SNR calculation
│   └── transcriber.py          # Whisper ASR + Resemblyzer diarization + WER
│
├── methods/
│   ├── __init__.py
│   ├── method1_lexical.py      # Keyword density classifier (baseline)
│   ├── method2_acoustic.py     # PyTorch DNN on MFCC + d-vector features
│   └── method3_hybrid.py       # Confidence-weighted ensemble fusion
│
├── analytics/
│   ├── __init__.py
│   ├── talk_ratio.py           # Talk-time ratio, silence %, QA score
│   ├── sentiment.py            # VADER sentiment trajectory
│   └── compliance.py           # SOP checklist + behavioural risk flagging
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # Accuracy, Precision, Recall, F1, DER, t-test, Pearson r
│   └── validator.py            # Ground truth comparison pipeline
│
├── dashboard/
│   ├── __init__.py
│   └── app.py                  # Streamlit QA dashboard
│
├── utils/
│   ├── logger.py               # Centralised logging
│   └── file_utils.py           # JSON / CSV / transcript I/O helpers
│
├── keywords/
│   ├── agent_keywords.json     # Agent SOP lexicon (greetings, compliance, closings)
│   └── customer_keywords.json  # Customer inquiry / complaint lexicon
│
├── data/                       # ★ Place your .wav / .mp3 files here
├── models/                     # Saved PyTorch model weights (auto-generated)
├── outputs/                    # All plots, logs, JSON results (auto-generated)
│
├── human_validation_study.csv  # Ground truth labels for validation
└── requirements.txt            # Python dependencies
```

---

## System Architecture

```
Raw Audio (.wav/.mp3)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 1: Pre-processing                                │
│  • Resample → 16 kHz mono                               │
│  • Spectral Gating Noise Reduction                      │
│  • Amplitude Normalization [-1, +1]                     │
│  • SNR computed: before & after (mathematically)        │
│                        │                                │
│  ┌──────────────┐  ┌───────────────────────────────┐   │
│  │ Whisper ASR  │  │  Resemblyzer Diarization       │   │
│  │ Timestamped  │  │  d-vector + Agglomerative      │   │
│  │ Transcript   │  │  Hierarchical Clustering       │   │
│  └──────┬───────┘  └──────────────┬────────────────┘   │
│         └──────────────┬──────────┘                    │
│               Temporal Alignment                        │
│          Diarized Transcript: {speaker_id, text,        │
│                                start, end}              │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 2: Speaker Role Detection                        │
│                                                         │
│  Method 1 (Lexical)    Method 2 (Acoustic)              │
│  D = (K/N) × 100       298-dim vector:                  │
│  Agent / Customer      [40 MFCC mean+std, 256 d-vec]   │
│  keyword density       → PyTorch MLP (256→128→64→32→2)  │
│           │                      │                      │
│           └──────────┬───────────┘                      │
│                      ▼                                   │
│              Method 3 (Hybrid)                          │
│  S = (P_lex×C_lex×α + P_ac×C_ac×β) / (α+β)            │
│  α=0.4  β=0.6  Conflict → Acoustic fallback            │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 3: Analytics                                     │
│  • Talk-Time Ratio:  R_agent = (Σd_agent/D_total)×100   │
│  • Sentiment:        VADER compound score per segment    │
│  • Compliance:       SOP keyword checklist (4 items)    │
│  • QA Score:         Weighted composite (0–100)          │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 4: Evaluation                                    │
│  • Accuracy / Precision / Recall / F1                   │
│  • DER (Diarization Error Rate)                         │
│  • Paired t-test (Method 3 vs Method 1)                 │
│  • Pearson r (System vs Human QA scores)                │
│  • RTF (Real-Time Factor)                               │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 5: Dashboard                                     │
│  streamlit run dashboard/app.py                         │
│  • Executive KPIs (calls, quality, sentiment, alerts)   │
│  • Quality distribution, talk ratio charts              │
│  • Per-call: transcript, sentiment trajectory,          │
│    compliance flags, method comparison                  │
└─────────────────────────────────────────────────────────┘
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Eiqbal25/fyp1_call_analysis.git
cd fyp1_call_analysis
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on PyTorch:** For GPU acceleration (recommended for Whisper), install the CUDA version:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```
> For CPU-only (slower but works):
> ```bash
> pip install torch
> ```

### 4. FFmpeg (required by Whisper for .mp3 files)

- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
- **macOS:** `brew install ffmpeg`
- **Ubuntu/Debian:** `sudo apt install ffmpeg`

---

## Quick Start

### Step 1 — Add your audio files

Place `.wav` or `.mp3` call recordings in the `data/` folder:

```
data/
├── airasia_call.wav
├── celcom_call.mp3
└── ...
```

> Audio should be single-channel (mono) or stereo — both are handled automatically.  
> Supported sampling rates: any (resampled to 16 kHz internally).

### Step 2 — Run the pipeline

```bash
python main.py
```

**Optional flags:**

| Flag | Description |
|------|-------------|
| `--data_dir path/` | Custom audio folder (default: `data/`) |
| `--skip_acoustic` | Skip Method 2 DNN (runs faster, no GPU needed) |
| `--skip_validation` | Skip ground truth comparison |
| `--output_json path` | Custom output JSON path |

Example — fast run without GPU:

```bash
python main.py --skip_acoustic
```

### Step 3 — Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open your browser at `http://localhost:8501`

---

## Outputs

After running `main.py`, the `outputs/` folder contains:

| File | Description |
|------|-------------|
| `pipeline_results.json` | Full results for all calls (used by dashboard) |
| `pipeline.log` | Detailed execution log |
| `{call}_waveform.png` | Before/after waveform comparison (Figure 3.2) |
| `{call}_spectrogram.png` | Before/after spectrogram (Figure 4.1) |
| `{call}_diarized.json` | Raw diarized transcript with timestamps |
| `{call}_m1_keywords.png` | Keyword density analysis (Figure 4.3) |
| `{call}_m1_confidence.png` | Confidence distribution (Figure 4.2) |
| `{call}_m3_ensemble.png` | Hybrid ensemble scores per segment |
| `{call}_sentiment.png` | Sentiment trajectory (per call) |
| `{call}_method_comparison.png` | Method 1/2/3 confidence comparison |
| `accuracy_comparison.png` | Accuracy/F1 grouped bar chart |
| `ttest_boxplot.png` | Paired t-test box plot (Figure 3.10) |
| `talk_time_distribution.png` | Stacked bar: Agent/Customer/Silence |
| `sentiment_summary.png` | Cross-call sentiment chart |
| `compliance_summary.png` | Compliance adherence rates |
| `qa_score_comparison.png` | System vs Human QA correlation scatter |

---

## Configuration

All parameters are centralised in `config.py`. Key settings:

```python
# Whisper model size: "tiny" | "base" | "small" | "medium" | "large"
WHISPER_MODEL_SIZE = "base"

# Hybrid ensemble weights (must sum to 1.0 in effect)
HYBRID_ALPHA = 0.4   # Lexical weight
HYBRID_BETA  = 0.6   # Acoustic weight (more stable than keywords)

# QA Score sub-metric weights (must sum to 1.0)
QA_SCORE_WEIGHTS = {
    "talk_balance":  0.25,
    "turn_taking":   0.20,
    "sentiment":     0.25,
    "compliance":    0.20,
    "politeness":    0.10,
}

# DNN architecture
ACOUSTIC_HIDDEN_DIMS = [256, 128, 64, 32]
ACOUSTIC_EPOCHS      = 50
```

---

## How Accuracy is Measured

The system classifies Agent and Customer **automatically from audio** — you never manually label anything at runtime. The `human_validation_study.csv` is your fixed reference that you fill in **once, offline**, by listening to your calls.

```
Audio (.wav)
    │
    ▼
System auto-classifies each segment → "Agent" or "Customer"
    │
    ▼
Compare against human_validation_study.csv (your verified labels)
    │
    ▼
Accuracy / Precision / Recall / F1 reported
```

The validator matches system segments to ground truth rows using **text similarity** (fuzzy matching), not speaker ID numbers. This fixes the label inversion problem where Resemblyzer randomly assigns speaker 0 and 1 differently each call.

## Ground Truth Validation

The `human_validation_study.csv` file maps each call's speaker segments to manually verified ground-truth roles and human QA scores.

**Required columns:**

| Column | Description |
|--------|-------------|
| `call_id` | Must match the audio filename without extension (e.g. `airasia_call`) |
| `ground_truth_role` | What this speaker actually is: `Agent` or `Customer` |
| `text` | What this speaker actually said — used for fuzzy matching against Whisper output |
| `start` | *(optional)* Segment start time in seconds |
| `end` | *(optional)* Segment end time in seconds |
| `human_qa_score` | *(optional)* Your overall quality score for this call (0–100) |

**There is NO `speaker_id` column.** The old approach of matching by speaker ID was wrong because Resemblyzer assigns IDs randomly each call. The new validator matches by text similarity instead.

**How to fill in this CSV for your real audio:**
1. Listen to each call
2. Write down what each speaker said in the `text` column
3. Label them `Agent` or `Customer` in `ground_truth_role`
4. The text does not need to be perfectly accurate — fuzzy matching handles small differences
5. Save the file and run `python evaluate.py` to see true accuracy

---

## Statistical Tests

All statistics are computed mathematically from real data — nothing is hardcoded.

| Metric | Formula | Purpose |
|--------|---------|---------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Precision | TP/(TP+FP) | Agent identification exactness |
| Recall | TP/(TP+FN) | Agent identification completeness |
| F1-Score | 2×(P×R)/(P+R) | Balanced metric |
| DER | (FA+MS+SC)/TotalTime | Diarization quality |
| SNR | 10×log10(P_signal/P_noise) | Audio quality improvement |
| WER | (S+D+I)/N | Transcription accuracy |
| RTF | ProcessingTime/AudioDuration | Speed efficiency |
| Pearson r | scipy.stats.pearsonr | System vs Human correlation |
| Paired t-test | scipy.stats.ttest_rel | Method 3 significance test |

---

## Key Formulas (from Thesis)

**Lexical Density (Method 1):**
```
D = (K / N) × 100
```
Where K = matched keywords, N = total words in segment.

**Hybrid Ensemble Score (Method 3):**
```
S_ensemble = (P_lex × C_lex × α + P_ac × C_ac × β) / (α + β)
```
Where P = role probability, C = model confidence.

**Talk-Time Ratio:**
```
R_agent = (Σ d_agent / D_total) × 100
```

**Composite QA Score:**
```
Q_score = Σ (weight_i × sub_score_i) × 100
```

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `openai-whisper` | ≥20231117 | Speech-to-text transcription |
| `resemblyzer` | ≥0.1.1 | Speaker d-vector embeddings |
| `torch` | ≥2.0.0 | DNN acoustic classifier |
| `librosa` | ≥0.10.0 | Audio processing, MFCC extraction |
| `noisereduce` | ≥3.0.0 | Spectral gating noise reduction |
| `vaderSentiment` | ≥3.3.2 | Rule-based sentiment analysis |
| `scikit-learn` | ≥1.3.0 | Agglomerative clustering |
| `scipy` | ≥1.11.0 | Statistical tests |
| `streamlit` | ≥1.27.0 | Interactive QA dashboard |
| `plotly` | ≥5.15.0 | Interactive charts in dashboard |
| `pandas` | ≥2.0.0 | Data handling and CSV I/O |
| `matplotlib` | ≥3.7.0 | Static figure generation |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'whisper'`**  
→ Run: `pip install openai-whisper`

**`resemblyzer` installation fails on Windows**  
→ Install Visual C++ Build Tools first, then: `pip install resemblyzer`

**Whisper is very slow**  
→ Use `WHISPER_MODEL_SIZE = "tiny"` in `config.py` for fastest speed, or install the CUDA version of PyTorch.

**`FileNotFoundError: Audio file not found`**  
→ Ensure audio files are in the `data/` directory with `.wav` or `.mp3` extension.

**Dashboard shows "No results found"**  
→ Run `python main.py` first to generate `outputs/pipeline_results.json`.

**WER is very high (>50%)**  
→ This is expected for Manglish/code-switching audio with the base Whisper model. Use `WHISPER_MODEL_SIZE = "medium"` or `"large"` for better accuracy on Malaysian English.

---

## Project Supervisor

**Prof. Dr. Ir. Siti Fauziah Bt. Toha @ Tohara**  
Department of Mechatronics Engineering, IIUM

---

## Author

**Muhammad Eiqbal Bin Hasbollah** (2216911)  
Bachelor of Engineering (Mechatronics) (Honours)  
International Islamic University Malaysia · January 2026

---

## License

This project is submitted as a Final Year Project for academic purposes at IIUM.  
All rights reserved © 2026 Muhammad Eiqbal Bin Hasbollah.
