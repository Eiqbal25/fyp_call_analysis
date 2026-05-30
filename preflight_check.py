"""
preflight_check.py
==================
Validates everything is ready before running main.py for Day 1 or Day 2.

Checks:
  1. All *_diarized.json in colab_transcripts/
  2. All .wav in data/
  3. All .csv in human_transcripts/
  4. Groq API key is set
  5. fix_speakers already run (checks for original backup)
  6. Which calls are already done vs pending
  7. Groq daily token estimate

Usage:
    python preflight_check.py           # check all 30 calls
    python preflight_check.py --day 1   # check day 1 calls only
    python preflight_check.py --day 2   # check day 2 calls only
"""

import os
import sys
import json
import argparse
from datetime import datetime

# ── Call lists — dynamically discovered from colab_transcripts/ ──
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import discover_calls as _discover
_schedule = _discover()
DAY1      = _schedule["day1"]
DAY2      = _schedule["day2"]
ALL_CALLS = _schedule["all"]

# ── Colors for terminal output ────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}✅{RESET} {msg}")
def fail(msg):  print(f"  {RED}❌{RESET} {msg}")
def warn(msg):  print(f"  {YELLOW}⚠️ {RESET} {msg}")
def info(msg):  print(f"  {BLUE}ℹ️ {RESET} {msg}")


def check_files(calls: list, base_dir: str, ext: str, label: str) -> tuple:
    """Check that all expected files exist. Returns (found, missing)."""
    found, missing = [], []
    for cid in calls:
        path = os.path.join(base_dir, f"{cid}{ext}")
        if os.path.exists(path):
            found.append(cid)
        else:
            missing.append(cid)
    return found, missing


def check_fix_speakers(calls: list, colab_dir: str) -> tuple:
    """
    Check fix_speakers has been run.
    Passes if either:
      - _diarized_original.json backup exists (fix was applied), OR
      - diarized.json exists and already has 2 speakers (fix not needed)
    """
    done, not_done = [], []
    for cid in calls:
        # Pass 1: backup exists = fix was applied
        orig = os.path.join(colab_dir, f"{cid}_diarized_original.json")
        if os.path.exists(orig):
            done.append(cid)
            continue

        # Pass 2: check if diarized.json already has 2 speakers
        diar = os.path.join(colab_dir, f"{cid}_diarized.json")
        if os.path.exists(diar):
            try:
                import json as _json
                with open(diar, encoding="utf-8") as f:
                    segs = _json.load(f)
                if isinstance(segs, dict):
                    segs = segs.get("segments", segs.get("utterances", []))
                speaker_ids = set(s.get("speaker_id", s.get("speaker", 0)) for s in segs)
                if len(speaker_ids) >= 2:
                    done.append(cid)
                    continue
            except Exception:
                pass

        not_done.append(cid)
    return done, not_done


def check_already_processed(calls: list, pipeline_path: str) -> tuple:
    """Check which calls already exist in pipeline_results.json."""
    if not os.path.exists(pipeline_path):
        return [], calls
    with open(pipeline_path, encoding="utf-8") as f:
        data = json.load(f)
    existing = data.get("calls", {})
    if isinstance(existing, list):
        existing = {c["call_id"]: c for c in existing if isinstance(c, dict)}
    done     = [c for c in calls if c in existing]
    pending  = [c for c in calls if c not in existing]
    return done, pending


def load_tracker(tracker_path: str) -> dict:
    if os.path.exists(tracker_path):
        with open(tracker_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def init_tracker(calls: list, tracker_path: str, day: int):
    """Create or update run_tracker.json with pending entries."""
    tracker = load_tracker(tracker_path)
    changed = False
    for cid in calls:
        if cid not in tracker:
            tracker[cid] = {"status": "pending", "day": day, "time": None}
            changed = True
    if changed:
        with open(tracker_path, "w", encoding="utf-8") as f:
            json.dump(tracker, f, indent=2)


def estimate_groq_tokens(calls: list, colab_dir: str) -> int:
    """Rough estimate of Groq tokens needed based on segment count."""
    total_segments = 0
    for cid in calls:
        path = os.path.join(colab_dir, f"{cid}_diarized.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Handle both list format and dict with "segments" key
        if isinstance(data, list):
            segs = data
        elif isinstance(data, dict):
            segs = data.get("segments", data.get("utterances", []))
        else:
            segs = []
        total_segments += len(segs)
    # ~80 tokens per segment (prompt + response average for Llama 3.3)
    return total_segments * 80


def main():
    parser = argparse.ArgumentParser(description="Pre-flight check before running main.py")
    parser.add_argument("--day", type=str, default="all",
                        help="Which day to check: 1, 2, or all (default: all)")
    args = parser.parse_args()

    BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR        = os.path.join(BASE_DIR, "data")
    COLAB_DIR       = os.path.join(BASE_DIR, "colab_transcripts")
    HT_DIR          = os.path.join(BASE_DIR, "human_transcripts")
    OUTPUTS_DIR     = os.path.join(BASE_DIR, "outputs", "latest")
    PIPELINE_JSON   = os.path.join(OUTPUTS_DIR, "pipeline_results.json")
    TRACKER_PATH    = os.path.join(BASE_DIR, "run_tracker.json")
    GROQ_LIMIT      = 100_000  # daily token limit (free tier)

    if args.day == "1":
        calls    = DAY1
        day_num  = 1
    elif args.day == "2":
        calls    = DAY2
        day_num  = 2
    else:
        calls    = ALL_CALLS
        day_num  = 0

    print()
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  FYP1 Pre-flight Check — Day {args.day}{RESET}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{BOLD}{'='*60}{RESET}")
    print()

    errors   = 0
    warnings = 0

    # ── 1. Colab transcripts ─────────────────────────────────
    print(f"{BOLD}[1] Colab transcripts (colab_transcripts/*.json){RESET}")
    found_json, missing_json = check_files(calls, COLAB_DIR, "_diarized.json", "diarized JSON")
    if missing_json:
        fail(f"{len(missing_json)}/{len(calls)} diarized JSONs missing:")
        for cid in missing_json:
            print(f"       {RED}→ {cid}_diarized.json{RESET}")
        errors += 1
    else:
        ok(f"{len(found_json)}/{len(calls)} diarized JSONs found")

    # ── 2. WAV files ─────────────────────────────────────────
    print()
    print(f"{BOLD}[2] Raw audio (data/*.wav){RESET}")
    found_wav, missing_wav = check_files(calls, DATA_DIR, ".wav", "WAV")
    if missing_wav:
        fail(f"{len(missing_wav)}/{len(calls)} WAV files missing:")
        for cid in missing_wav:
            print(f"       {RED}→ {cid}.wav{RESET}")
        errors += 1
    else:
        ok(f"{len(found_wav)}/{len(calls)} WAV files found")

    # ── 3. Human transcripts ─────────────────────────────────
    print()
    print(f"{BOLD}[3] Human transcripts (human_transcripts/*.csv){RESET}")
    found_csv, missing_csv = check_files(calls, HT_DIR, ".csv", "CSV")
    if missing_csv:
        warn(f"{len(missing_csv)}/{len(calls)} human transcript CSVs missing:")
        for cid in missing_csv:
            print(f"       {YELLOW}→ {cid}.csv{RESET}")
        warnings += 1
    else:
        ok(f"{len(found_csv)}/{len(calls)} human transcript CSVs found")

    # ── 4. fix_speakers check ────────────────────────────────
    print()
    print(f"{BOLD}[4] fix_speakers.py status{RESET}")
    fixed, not_fixed = check_fix_speakers(calls, COLAB_DIR)
    if not_fixed:
        warn(f"{len(not_fixed)} call(s) have single-speaker diarization:")
        for cid in not_fixed:
            print(f"       {YELLOW}→ {cid} (single-speaker — voices too similar for pyannote){RESET}")
        print(f"       {YELLOW}Try: python fix_speakers.py --call_id <id> --force{RESET}")
        print(f"       {YELLOW}These calls will still run — M3 LLM handles them via text{RESET}")
        warnings += 1
    else:
        ok(f"fix_speakers confirmed for all {len(calls)} calls")

    # ── 5. Groq API key ──────────────────────────────────────
    print()
    print(f"{BOLD}[5] Groq API key{RESET}")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        # Try loading from .env
        env_path = os.path.join(BASE_DIR, ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("GROQ_API_KEY"):
                        groq_key = line.split("=", 1)[-1].strip().strip('"')
    if groq_key:
        masked = groq_key[:8] + "..." + groq_key[-4:]
        ok(f"Groq API key found ({masked})")
    else:
        fail("GROQ_API_KEY not set — set in .env or environment")
        errors += 1

    # ── 6. Token estimate ────────────────────────────────────
    print()
    print(f"{BOLD}[6] Groq token estimate{RESET}")
    estimated = estimate_groq_tokens(calls, COLAB_DIR)
    if estimated > GROQ_LIMIT:
        fail(f"Estimated tokens: {estimated:,} — EXCEEDS daily limit of {GROQ_LIMIT:,}")
        fail(f"Split into 2 days of 15 calls each")
        errors += 1
    elif estimated > GROQ_LIMIT * 0.8:
        warn(f"Estimated tokens: {estimated:,} — close to daily limit ({GROQ_LIMIT:,})")
        warnings += 1
    else:
        ok(f"Estimated tokens: {estimated:,} / {GROQ_LIMIT:,} ({estimated/GROQ_LIMIT*100:.0f}% of limit)")

    # ── 7. Already processed ─────────────────────────────────
    print()
    print(f"{BOLD}[7] Pipeline status{RESET}")
    done_calls, pending_calls = check_already_processed(calls, PIPELINE_JSON)
    if done_calls:
        warn(f"{len(done_calls)} calls already in pipeline_results.json:")
        for cid in done_calls:
            print(f"       {YELLOW}→ {cid} (will be overwritten){RESET}")
    if pending_calls:
        info(f"{len(pending_calls)} calls pending:")
        for cid in pending_calls:
            print(f"       {BLUE}→ {cid}{RESET}")
    if not done_calls:
        ok("No calls already processed — clean run")

    # ── 8. Backup check ──────────────────────────────────────
    print()
    print(f"{BOLD}[8] Backup{RESET}")
    backup_path = os.path.join(BASE_DIR, "best_results_backup.zip")
    if os.path.exists(backup_path):
        size_mb = os.path.getsize(backup_path) / 1024 / 1024
        mtime   = datetime.fromtimestamp(os.path.getmtime(backup_path))
        ok(f"Backup found: best_results_backup.zip ({size_mb:.1f} MB, {mtime.strftime('%Y-%m-%d %H:%M')})")
    else:
        if not os.path.exists(PIPELINE_JSON):
            ok("First run — no backup needed yet")
        else:
            warn("No backup found — recommended before running:")
            print(f"       {YELLOW}powershell Compress-Archive -Path outputs\\latest\\* -DestinationPath best_results_backup.zip{RESET}")
            warnings += 1

    # ── Init run_tracker.json ────────────────────────────────
    if day_num > 0:
        init_tracker(calls, TRACKER_PATH, day_num)
        ok(f"run_tracker.json initialised for Day {day_num}")

    # ── Final verdict ────────────────────────────────────────
    print()
    print(f"{BOLD}{'='*60}{RESET}")
    if errors == 0 and warnings == 0:
        print(f"{GREEN}{BOLD}  All checks passed — ready to run!{RESET}")
        print(f"  Command: run_day.bat {args.day}")
    elif errors == 0:
        print(f"{YELLOW}{BOLD}  Ready with {warnings} warning(s) — proceed carefully{RESET}")
        print(f"  Command: run_day.bat {args.day}")
    else:
        print(f"{RED}{BOLD}  {errors} error(s) found — fix before running{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print()

    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
