"""
run_tracker.py
==============
Manages run_tracker.json — tracks status of every call across runs.

Status values:
  pending   — not yet processed
  done      — processed successfully
  failed    — processing failed (can retry)
  skipped   — intentionally skipped

Usage (as module):
    from run_tracker import update_tracker, print_tracker_status

Usage (as script):
    python run_tracker.py            # show status of all calls
    python run_tracker.py --reset    # reset all to pending
    python run_tracker.py --failed   # show only failed calls
"""

import os
import json
import argparse
from datetime import datetime

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TRACKER_PATH = os.path.join(BASE_DIR, "run_tracker.json")

import sys
sys.path.insert(0, BASE_DIR)
from config import discover_calls as _discover

_schedule = _discover()
DAY1      = _schedule["day1"]
DAY2      = _schedule["day2"]
ALL_CALLS = _schedule["all"]


def load_tracker() -> dict:
    if os.path.exists(TRACKER_PATH):
        with open(TRACKER_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_tracker(tracker: dict):
    with open(TRACKER_PATH, "w", encoding="utf-8") as f:
        json.dump(tracker, f, indent=2)


def update_tracker(call_id: str, status: str, day: int = 0, error_msg: str = None):
    """
    Update a single call's status in run_tracker.json.
    Called by main.py after each call completes or fails.
    """
    tracker = load_tracker()
    tracker[call_id] = {
        "status":    status,
        "day":       day,
        "time":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        "error":     error_msg,
    }
    save_tracker(tracker)


def get_pending_calls(day: int = 0) -> list:
    """Return list of calls not yet done for a given day (0 = all days)."""
    tracker = load_tracker()
    if day == 1:
        calls = DAY1
    elif day == 2:
        calls = DAY2
    else:
        calls = ALL_CALLS

    return [
        c for c in calls
        if tracker.get(c, {}).get("status") not in ("done",)
    ]


def print_tracker_status(day: int = 0, failed_only: bool = False):
    tracker = load_tracker()
    calls   = DAY1 if day == 1 else DAY2 if day == 2 else ALL_CALLS

    STATUS_ICON = {
        "done":    "✅",
        "failed":  "❌",
        "pending": "⏳",
        "skipped": "⏭️ ",
    }

    print()
    print(f"{'='*62}")
    header = f"Run Tracker — Day {day}" if day else "Run Tracker — All Calls"
    print(f"  {header}")
    print(f"{'='*62}")

    done = failed = pending = 0
    current_day = None

    for cid in calls:
        entry   = tracker.get(cid, {"status": "pending", "day": 0, "time": None})
        status  = entry.get("status", "pending")
        time    = entry.get("time", "")
        call_day = entry.get("day", 0)

        if failed_only and status != "failed":
            continue

        icon = STATUS_ICON.get(status, "?")
        time_str = f"  {time}" if time else ""

        # Day separator
        if call_day != current_day and call_day > 0:
            print(f"\n  Day {call_day}:")
            current_day = call_day

        print(f"  {icon}  {cid:<22} {status:<8}{time_str}")

        if status == "done":    done += 1
        elif status == "failed": failed += 1
        else:                    pending += 1

    print()
    print(f"{'─'*62}")
    print(f"  Done: {done}  |  Failed: {failed}  |  Pending: {pending}  |  Total: {len(calls)}")
    print(f"{'='*62}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Show or manage run tracker")
    parser.add_argument("--day",    type=int, default=0, help="Filter by day (1 or 2)")
    parser.add_argument("--reset",  action="store_true",  help="Reset all calls to pending")
    parser.add_argument("--failed", action="store_true",  help="Show only failed calls")
    args = parser.parse_args()

    if args.reset:
        tracker = {}
        day_map = {c: 1 for c in DAY1}
        day_map.update({c: 2 for c in DAY2})
        for cid in ALL_CALLS:
            tracker[cid] = {"status": "pending", "day": day_map[cid], "time": None, "error": None}
        save_tracker(tracker)
        print("✅ run_tracker.json reset — all calls set to pending")
        return

    print_tracker_status(day=args.day, failed_only=args.failed)


if __name__ == "__main__":
    main()
