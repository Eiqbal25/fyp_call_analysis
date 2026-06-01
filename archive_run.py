"""
archive_run.py
==============
Archives current run results and prepares for a fresh run.

Usage:
    python archive_run.py --save 1    # Save current results as Run 1
    python archive_run.py --save 2    # Save current results as Run 2
    python archive_run.py --save 3    # Save current results as Run 3
    python archive_run.py --summary   # Show average accuracy across saved runs
    python archive_run.py --use 2     # Set Run 2 as final pipeline_results.json
"""

import os
import json
import shutil
import argparse
from datetime import datetime

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "latest")
RUNS_DIR    = os.path.join(BASE_DIR, "outputs", "runs")

# Key files to archive per run
ARCHIVE_FILES = [
    "pipeline_results.json",
    "label_accuracy_summary.json",
    "evaluation_summary.json",
    "2_label_comparison_report.txt",
    "1_evaluation_report.txt",
    "3_analytics_summary.csv",
]

DAY1 = ["eng_prof_01","eng_prof_02","eng_prof_03","eng_prof_04","eng_prof_05",
        "eng_rudeagt_01","eng_rudeagt_02","eng_rudeagt_03","eng_rudeagt_04",
        "eng_rudecust_01","eng_rudecust_02","eng_rudecust_03","eng_rudecust_04",
        "eng_rudecust_05","long_01"]
DAY2 = ["manglish_01","manglish_02","manglish_03","manglish_04","manglish_05",
        "my_prof_01","my_prof_02","my_prof_03","my_prof_04","my_prof_05",
        "my_rude_01","my_rude_02","my_rude_03","my_sales_01","my_sales_02"]
ALL_CALLS = DAY1 + DAY2


def save_run(run_num: int):
    """Archive current results as Run N."""
    run_dir = os.path.join(RUNS_DIR, f"run{run_num}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Saving current results as Run {run_num}")
    print(f"  → {run_dir}")
    print(f"{'='*55}")

    saved = 0
    for fname in ARCHIVE_FILES:
        src = os.path.join(OUTPUTS_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(run_dir, fname))
            print(f"  ✅ {fname}")
            saved += 1
        else:
            print(f"  ⚠️  {fname} not found — skipping")

    # Save timestamp
    meta = {
        "run": run_num,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "files_saved": saved,
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  {saved}/{len(ARCHIVE_FILES)} files saved")

    # Ask to clear for fresh run
    if saved > 0:
        print(f"\n{'='*55}")
        print("  Ready for fresh run!")
        print("  Run Day 1: run_day.bat 1")
        print(f"{'='*55}\n")


def show_summary():
    """Show per-call accuracy across all saved runs."""
    runs = {}
    for i in [1, 2, 3]:
        run_dir = os.path.join(RUNS_DIR, f"run{i}")
        label_path = os.path.join(run_dir, "label_accuracy_summary.json")
        if os.path.exists(label_path):
            with open(label_path, encoding="utf-8") as f:
                runs[i] = json.load(f)

    if not runs:
        print("No runs found. Save at least one run first:")
        print("  python archive_run.py --save 1")
        return

    print()
    print("=" * 72)
    print("  Results Comparison Across Runs")
    print("=" * 72)
    print(f"  {'Call ID':<25} ", end="")
    for r in sorted(runs.keys()):
        print(f"{'Run'+str(r):>8}", end="")
    print(f"  {'Avg':>8}  {'Best':>8}")
    print("-" * 72)

    call_avgs = {}
    for cid in ALL_CALLS:
        scores = []
        print(f"  {cid:<25} ", end="")
        for r in sorted(runs.keys()):
            acc = runs[r].get(cid, {}).get("accuracy_pct", None)
            if acc is not None:
                scores.append(acc)
                print(f"{acc:>7.1f}%", end="")
            else:
                print(f"{'N/A':>8}", end="")
        if scores:
            avg  = sum(scores) / len(scores)
            best = max(scores)
            print(f"  {avg:>7.1f}%  {best:>7.1f}%")
            call_avgs[cid] = {"avg": avg, "best": best, "scores": scores}
        else:
            print()

    print("-" * 72)
    if call_avgs:
        overall_avg  = sum(v["avg"]  for v in call_avgs.values()) / len(call_avgs)
        overall_best = sum(v["best"] for v in call_avgs.values()) / len(call_avgs)
        print(f"  {'AVERAGE':<25} ", end="")
        for r in sorted(runs.keys()):
            run_avg = sum(
                runs[r].get(c, {}).get("accuracy_pct", 0)
                for c in ALL_CALLS if c in runs[r]
            ) / len([c for c in ALL_CALLS if c in runs[r]])
            print(f"{run_avg:>7.1f}%", end="")
        print(f"  {overall_avg:>7.1f}%  {overall_best:>7.1f}%")

    print()
    # Which run is best?
    best_run = None
    best_score = 0
    for r, data in runs.items():
        scores = [v.get("accuracy_pct", 0) for v in data.values()
                  if isinstance(v, dict)]
        if scores:
            avg = sum(scores) / len(scores)
            if avg > best_score:
                best_score = avg
                best_run = r

    if best_run:
        print(f"  Best run: Run {best_run} ({best_score:.1f}% average)")
        print(f"  To use: python archive_run.py --use {best_run}")
    print("=" * 72)


def use_run(run_num: int):
    """Restore a saved run as the active pipeline_results.json."""
    run_dir = os.path.join(RUNS_DIR, f"run{run_num}")
    if not os.path.exists(run_dir):
        print(f"❌ Run {run_num} not found at {run_dir}")
        return

    print(f"\nRestoring Run {run_num} as active results...")

    for fname in ARCHIVE_FILES:
        src = os.path.join(run_dir, fname)
        dst = os.path.join(OUTPUTS_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✅ {fname}")

    print(f"\n✅ Run {run_num} is now active")
    print("Run: streamlit run dashboard\\app.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save",    type=int, choices=[1,2,3], help="Save current results as Run N")
    parser.add_argument("--summary", action="store_true",       help="Compare all saved runs")
    parser.add_argument("--use",     type=int, choices=[1,2,3], help="Restore Run N as active")
    args = parser.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True)

    if args.save:
        save_run(args.save)
    elif args.summary:
        show_summary()
    elif args.use:
        use_run(args.use)
    else:
        print("Usage:")
        print("  python archive_run.py --save 1    # Archive current as Run 1")
        print("  python archive_run.py --save 2    # Archive current as Run 2")
        print("  python archive_run.py --save 3    # Archive current as Run 3")
        print("  python archive_run.py --summary   # Compare all 3 runs")
        print("  python archive_run.py --use 2     # Restore Run 2 as active")


if __name__ == "__main__":
    main()
