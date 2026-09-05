"""
build_experiment_c_inputs.py

Builds the two CSVs that experiment_c_analysis.py expects:
    internal_fold_metrics.csv   (mode, criterion, fold, int_auc, int_brier,
                                 int_scale_offset, int_gap_contribution)
    external_fold_auc.csv       (mode, criterion, fold, ext_auc)

WHAT IS REUSED vs WHAT IS NEW
-----------------------------
REUSED (imported, not reimplemented, so it cannot drift from Experiment A):
    load_oasis, gap_from_folds  from scale_incoherence_test.py

NEW (defined here for the first time -- scale_incoherence_test.py computes the
gap per (mode, criterion) GROUP, never per fold, so these two per-fold
quantities did not previously exist anywhere):

  int_gap_contribution
      Leave-one-fold-out delta:  gap(all 5 folds) - gap(the other 4 folds).
      Positive => removing this fold SHRINKS the gap => this fold inflates
      the incoherence. Uses the imported gap_from_folds, so it is the same
      gap statistic Experiment A reports, just recomputed on subsets.
      Known weakness: LOO deltas over 5 folds are inherently high-variance.
      That is a property of having 5 folds, not a defect of the definition.

  int_scale_offset
      This fold's BALANCED mean predicted probability minus the grand
      balanced mean across all 5 folds.
      Balanced = mean of (mean prob among CN, mean prob among AD), so that a
      fold whose test split happens to hold more AD subjects is not scored as
      "shifted" merely because of case-mix. OASIS-3 folds hold disjoint
      subjects, so the raw mean would confound scale with composition.
      This is precisely the between-fold quantity that the rank transform in
      scale_incoherence_test.py destroys.

Run from repo root, after scale_incoherence_test.py is present.
"""

import os
import json
import numpy as np
import pandas as pd

from scale_incoherence_test import load_oasis, gap_from_folds, MODES

CV_DIR = "outputs/cv"
ADNI_RESULTS = "outputs/adni_external/adni_external_results.json"
CRITERIA = ["auc", "neg_brier"]
MODE_KEYS = ["clinical", "imaging", "fusion"]
STUB = {"clinical": "sel_clinical", "imaging": "sel_imaging", "fusion": "sel_fusion"}


# --------------------------------------------------------------- new metrics
def balanced_mean_prob(probs, labels):
    """Mean of per-class mean probabilities; None if a class is absent."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    parts = []
    for cls in (0, 1):
        m = labels == cls
        if not m.any():
            return None
        parts.append(probs[m].mean())
    return float(np.mean(parts))


def scale_offsets(fold_probs, fold_labels):
    """Per-fold balanced mean prob minus the grand balanced mean."""
    per_fold = []
    for p, y in zip(fold_probs, fold_labels):
        bm = balanced_mean_prob(p, y)
        if bm is None:
            raise ValueError("A fold test split is missing one class entirely.")
        per_fold.append(bm)
    grand = float(np.mean(per_fold))
    return [float(v - grand) for v in per_fold]


def gap_contributions(fold_probs, fold_labels):
    """Leave-one-fold-out delta on the imported gap statistic."""
    full = gap_from_folds(fold_probs, fold_labels)
    if full is None:
        raise ValueError("gap_from_folds returned None on the full set "
                         "(a fold likely has only one class).")
    gap_all = full[2]

    contribs = []
    for k in range(len(fold_probs)):
        sub_p = [p for i, p in enumerate(fold_probs) if i != k]
        sub_y = [y for i, y in enumerate(fold_labels) if i != k]
        loo = gap_from_folds(sub_p, sub_y)
        if loo is None:
            raise ValueError(f"gap_from_folds returned None with fold {k+1} removed.")
        contribs.append(float(gap_all - loo[2]))
    return contribs, float(gap_all)


# --------------------------------------------------------------- internal
def build_internal():
    rows = []
    for mk in MODE_KEYS:
        path = os.path.join(CV_DIR, f"{STUB[mk]}_folds.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")
        data = json.load(open(path, encoding="utf-8"))

        for crit in CRITERIA:
            loaded = load_oasis(mk, crit)
            if loaded is None:
                raise ValueError(f"load_oasis returned None for {mk}/{crit}")
            fold_probs, fold_labels = loaded

            offsets = scale_offsets(fold_probs, fold_labels)
            contribs, gap_all = gap_contributions(fold_probs, fold_labels)

            print(f"  {mk:9s} {crit:10s} group gap = {gap_all:+.4f}   "
                  f"LOO contributions = "
                  f"{', '.join(f'{c:+.4f}' for c in contribs)}")

            for i, f in enumerate(data["folds"]):
                test = f["by_criterion"][crit]["test"]
                rows.append({
                    "mode": mk,
                    "criterion": crit,
                    "fold": int(f["fold"]),
                    "int_auc": float(test["auc"]),
                    "int_brier": float(test["brier"]),
                    "int_scale_offset": offsets[i],
                    "int_gap_contribution": contribs[i],
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- external
def build_external():
    if not os.path.exists(ADNI_RESULTS):
        raise FileNotFoundError(f"Missing {ADNI_RESULTS}")
    entries = json.load(open(ADNI_RESULTS, encoding="utf-8"))

    full_to_short = {v: k for k, v in MODES.items()}  # clinical_only -> clinical

    rows = []
    n_skipped_summary = 0
    for e in entries:
        mode_full = e.get("mode")
        crit = e.get("criterion")
        if mode_full not in full_to_short or crit not in CRITERIA:
            continue  # extra modes/criteria in the file are ignored by design

        # The results file mixes per-fold rows with aggregate rows whose
        # 'fold' field is a string such as 'pooled_summary'. Those aggregates
        # are a different statistic (cross-fold pooled-style AUC over all five
        # models) and must NOT be treated as a sixth fold. Skip them
        # explicitly rather than letting int() decide.
        fold_raw = e.get("fold")
        if not isinstance(fold_raw, (int, np.integer)):
            n_skipped_summary += 1
            continue

        rows.append({
            "mode": full_to_short[mode_full],
            "criterion": crit,
            "fold": int(fold_raw),
            "ext_auc": float(e["auc"]),
        })

    if n_skipped_summary:
        print(f"  (skipped {n_skipped_summary} non-fold aggregate rows, "
              f"e.g. pooled_summary)")

    df = pd.DataFrame(rows)
    dupes = df.duplicated(subset=["mode", "criterion", "fold"]).sum()
    if dupes:
        raise ValueError(f"{dupes} duplicate (mode, criterion, fold) rows in "
                         f"{ADNI_RESULTS} -- refusing to guess which to keep.")
    return df


def main():
    print("Building internal fold metrics from OASIS-3 CV JSON...")
    internal = build_internal()

    print("\nBuilding external fold AUCs from ADNI results JSON...")
    external = build_external()

    expected = len(MODE_KEYS) * len(CRITERIA) * 5
    for name, df in [("internal", internal), ("external", external)]:
        if len(df) != expected:
            raise ValueError(f"{name} has {len(df)} rows, expected {expected} "
                             f"(3 modes x 2 criteria x 5 folds). Inspect before "
                             f"proceeding rather than forcing a merge.")

    internal.to_csv("internal_fold_metrics.csv", index=False)
    external.to_csv("external_fold_auc.csv", index=False)

    print(f"\nWrote internal_fold_metrics.csv ({len(internal)} rows)")
    print(f"Wrote external_fold_auc.csv ({len(external)} rows)")
    print("\nNext:  python experiment_c_analysis.py")


if __name__ == "__main__":
    main()