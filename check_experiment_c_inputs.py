"""
check_experiment_c_inputs.py

Locates and prints the structure of the files Experiment C needs, so the
prep step that builds internal_fold_metrics.csv / external_fold_auc.csv
can be written against your ACTUAL keys instead of a guessed schema.

Run from repo root. Prints nothing destructive -- read-only.
"""

import os
import json
import glob

CANDIDATE_GLOBS = [
    "outputs/cv/sel_*_folds.json",          # Phase 2 selection outputs
    "outputs/cv/*scale*.json",              # Experiment A (scale_incoherence_test.py) output
    "outputs/cv/*gap*.json",
    "outputs/cv/*incoherence*.json",
    "outputs/adni_external/*.json",         # ADNI per-fold external results
    "outputs/adni_external/*.csv",
]


def summarize_json(path, max_depth=3):
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"    [could not parse as JSON: {e}]")
        return

    def describe(obj, depth=0, prefix=""):
        indent = "    " * (depth + 1)
        if depth > max_depth:
            print(f"{indent}{prefix}... (truncated, deeper than max_depth)")
            return
        if isinstance(obj, dict):
            print(f"{indent}{prefix}dict with keys: {list(obj.keys())}")
            for k, v in list(obj.items())[:5]:
                describe(v, depth + 1, prefix=f"'{k}': ")
        elif isinstance(obj, list):
            print(f"{indent}{prefix}list, length={len(obj)}")
            if obj:
                describe(obj[0], depth + 1, prefix="[0] = ")
        else:
            print(f"{indent}{prefix}{type(obj).__name__} = {obj!r}"[:200])

    describe(data)


def summarize_csv(path):
    try:
        import pandas as pd
        head = pd.read_csv(path, nrows=5)
        print(f"    columns: {list(head.columns)}")
        print(f"    first row: {head.iloc[0].to_dict() if len(head) else '(empty)'}")
    except Exception as e:
        print(f"    [could not read as CSV: {e}]")


def main():
    found_any = False
    for pattern in CANDIDATE_GLOBS:
        matches = glob.glob(pattern)
        for path in matches:
            found_any = True
            print(f"\n=== {path} ===")
            if path.endswith(".json"):
                summarize_json(path)
            elif path.endswith(".csv"):
                summarize_csv(path)

    if not found_any:
        print("No files matched the expected patterns:")
        for p in CANDIDATE_GLOBS:
            print(f"  {p}")
        print("\nRun this from your repo root. If your Experiment A / ADNI "
              "external outputs live under different paths or names, tell me "
              "the actual paths and I'll adjust the glob patterns.")
        return

    print("\n" + "=" * 70)
    print("Paste the printed structure back and I'll write "
          "build_experiment_c_inputs.py to merge these into the two CSVs "
          "experiment_c_analysis.py expects.")


if __name__ == "__main__":
    main()
