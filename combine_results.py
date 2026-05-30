"""
combine_results.py
==================
Combines two or more pipeline_results.json files into one.

Usage:
    python combine_results.py                          # combine day1.json + day2.json → pipeline_results.json
    python combine_results.py file1.json file2.json    # custom files
    python combine_results.py file1.json file2.json file3.json  # multiple files

Output:
    outputs/latest/pipeline_results.json (merged)
    Backup of original saved as pipeline_results_backup.json
"""

import json
import os
import sys
import shutil
from config import OUTPUTS_DIR

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def combine_pipelines(files: list) -> dict:
    """Merge multiple pipeline_results.json files into one."""
    merged_calls      = {}
    merged_validation = {}
    base_meta         = None

    for path in files:
        if not os.path.exists(path):
            print(f"  ⚠️  Skipping missing file: {path}")
            continue

        data = load_json(path)
        calls = data.get("calls", [])

        # Handle both list and dict formats
        if isinstance(calls, dict):
            calls_list = list(calls.values())
        else:
            calls_list = calls

        for call in calls_list:
            if isinstance(call, dict) and "call_id" in call:
                merged_calls[call["call_id"]] = call

        # Merge validation results
        val = data.get("validation_results", {})
        for method, method_calls in val.items():
            if method not in merged_validation:
                merged_validation[method] = {}
            merged_validation[method].update(method_calls)

        # Use first file's metadata as base
        if base_meta is None:
            base_meta = {k: v for k, v in data.items()
                         if k not in ("calls", "validation_results")}

        print(f"  ✅ Loaded {path}: {len(calls_list)} calls")

    if base_meta is None:
        base_meta = {}

    result = {
        **base_meta,
        "calls":              list(merged_calls.values()),
        "validation_results": merged_validation,
    }
    result["total_calls"] = len(merged_calls)

    return result


def main():
    output_path = os.path.join(OUTPUTS_DIR, "pipeline_results.json")

    # Determine input files
    if len(sys.argv) > 1:
        input_files = sys.argv[1:]
    else:
        # Default: look for day1.json and day2.json in outputs/latest
        day1 = os.path.join(OUTPUTS_DIR, "day1.json")
        day2 = os.path.join(OUTPUTS_DIR, "day2.json")
        if os.path.exists(day1) and os.path.exists(day2):
            input_files = [day1, day2]
        else:
            print("Usage:")
            print("  python combine_results.py file1.json file2.json")
            print()
            print("Or save your two runs as:")
            print(f"  {day1}")
            print(f"  {day2}")
            print("Then run: python combine_results.py")
            return

    print(f"\n=== Combining {len(input_files)} pipeline result files ===")
    for f in input_files:
        print(f"  Input: {f}")

    merged = combine_pipelines(input_files)

    # Backup existing pipeline_results.json if it exists
    if os.path.exists(output_path):
        backup = output_path.replace(".json", "_backup.json")
        shutil.copy(output_path, backup)
        print(f"\n  📦 Backup saved → {backup}")

    save_json(merged, output_path)

    print(f"\n✅ Combined {merged['total_calls']} calls → {output_path}")
    print("\nNow run:")
    print("  python compare_labels.py")
    print("  python evaluate.py")
    print("  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
