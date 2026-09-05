"""
find_experiment_a_and_expand_json.py

Two jobs, run from repo root:

1) Search the ENTIRE repo tree for the script that produced the
   Experiment A / scale-incoherence / gap-contribution numbers. We search
   by file content (keywords), not just filename, in case it was renamed
   or is a .ipynb / inline cell.

2) Fully expand 'by_criterion' and 'history' for fold 1 of
   sel_fusion_folds.json, since the earlier inspector truncated those two
   keys. This tells us whether both auc- and neg_brier-selected metrics
   live in one JSON file per mode (needed to build internal_fold_metrics.csv
   correctly).
"""

import os
import json
import glob

REPO_ROOT = "."

# ---- Part 1: find the Experiment A script -------------------------------

FILENAME_HINTS = ["scale", "incoherence", "gap", "rank_transform", "airola", "experiment_a", "experiment_c"]
CONTENT_HINTS = ["permutation", "rank_transform", "pooled_auc", "gap_contribution",
                 "scale_offset", "incoherence"]

def search_repo():
    print("=" * 70)
    print("SEARCHING REPO FOR EXPERIMENT A SCRIPT")
    print("=" * 70)

    candidates = set()

    # filename match
    for root, dirs, files in os.walk(REPO_ROOT):
        # skip venv / git / node_modules to keep this fast
        dirs[:] = [d for d in dirs if d not in (".git", "venv", "__pycache__", "node_modules")]
        for fname in files:
            if fname.endswith((".py", ".ipynb")):
                lower = fname.lower()
                if any(h in lower for h in FILENAME_HINTS):
                    candidates.add(os.path.join(root, fname))

    # content match (only .py, ipynb JSON parsing is messier)
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "venv", "__pycache__", "node_modules")]
        for fname in files:
            if fname.endswith(".py"):
                path = os.path.join(root, fname)
                try:
                    with open(path, "r", errors="ignore") as f:
                        content = f.read()
                except Exception:
                    continue
                hits = [h for h in CONTENT_HINTS if h in content]
                if hits:
                    candidates.add(path)

    if not candidates:
        print("No candidate script found by filename or content keywords.")
        print("It may only exist as inline code from a past chat session, "
              "not saved as a file. If so, tell me and I'll define the "
              "gap-contribution / scale-offset formulas fresh, documented "
              "clearly as a NEW definition (not a reproduction of the old "
              "numbers), so nothing gets silently mismatched.")
    else:
        print(f"Found {len(candidates)} candidate file(s):")
        for c in sorted(candidates):
            print(f"  - {c}")
        print("\nOpen/paste the one that looks right.")


# ---- Part 2: fully expand by_criterion / history for one fold -----------

def expand_fold_detail(path="outputs/cv/sel_fusion_folds.json"):
    print("\n" + "=" * 70)
    print(f"FULL EXPANSION OF FOLD 1 IN {path}")
    print("=" * 70)

    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    with open(path) as f:
        data = json.load(f)

    fold0 = data["folds"][0]
    print(f"Top-level keys in fold entry: {list(fold0.keys())}\n")

    if "by_criterion" in fold0:
        bc = fold0["by_criterion"]
        print(f"'by_criterion' type: {type(bc).__name__}")
        if isinstance(bc, dict):
            print(f"'by_criterion' keys: {list(bc.keys())}")
            for crit_name, crit_data in bc.items():
                print(f"\n  --- by_criterion['{crit_name}'] ---")
                if isinstance(crit_data, dict):
                    print(f"  keys: {list(crit_data.keys())}")
                    for k, v in crit_data.items():
                        if isinstance(v, dict):
                            print(f"    '{k}': dict with keys {list(v.keys())}")
                        elif isinstance(v, list):
                            print(f"    '{k}': list, length={len(v)}, first={v[0] if v else None}")
                        else:
                            print(f"    '{k}': {type(v).__name__} = {v}")
        else:
            print(f"'by_criterion' value: {bc}")
    else:
        print("No 'by_criterion' key present on this fold entry.")

    if "history" in fold0:
        h = fold0["history"]
        print(f"\n'history' type: {type(h).__name__}")
        if isinstance(h, dict):
            print(f"'history' keys: {list(h.keys())}")
        elif isinstance(h, list):
            print(f"'history' length: {len(h)}, first entry: {h[0] if h else None}")
    else:
        print("No 'history' key present on this fold entry.")

    # also confirm predictions list lengths line up
    preds = fold0.get("predictions", {})
    if isinstance(preds, dict):
        lens = {k: len(v) if isinstance(v, list) else "not a list" for k, v in preds.items()}
        print(f"\n'predictions' field lengths: {lens}")


if __name__ == "__main__":
    search_repo()
    expand_fold_detail()
