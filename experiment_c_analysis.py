"""
experiment_c_analysis.py

Experiment C: does an OASIS-3-internal, no-external-data signal
(internal AUC, Brier score, probability scale offset, or per-fold-vs-
pooled gap contribution) predict a fold-model's ADNI external AUC?

Honest framing this script enforces:
  - You have 5 folds. A correlation on 5 points is close to meaningless
    on its own. Pooling 3 modes x 2 criteria gives 30 rows, but they are
    NOT independent (same 5 folds, same subjects, reused across modes/
    criteria) -- so any pooled p-value is invalid and is labelled as such.
  - The decisive evidence is whether the SIGN of each within-group
    correlation is consistent across all 6 (mode, criterion) groups.
    One correlation flipping sign across groups is disqualifying for a
    "this predicts external failure" claim, regardless of pooled rho.
  - A permutation null (shuffling external AUC within each group only,
    so fold/group structure survives) is reported alongside the naive
    pooled rho so nobody downstream mistakes noise for signal.

--------------------------------------------------------------------
INPUT CONTRACT -- adjust paths and column names below to match your files.
This script does NOT invent your data; it expects two tidy CSVs:

1) INTERNAL_CSV -- OASIS-3, one row per (mode, criterion, fold):
     mode                  'clinical' | 'imaging' | 'fusion'
     criterion             'auc' | 'neg_brier'
     fold                  1..5
     int_auc               fold's own internal test-split AUC
     int_brier             fold's own internal Brier score
     int_scale_offset      this fold's probability-scale deviation from
                            the group (however you defined it for
                            Experiment A / the rank-transform diagnostic)
     int_gap_contribution  this fold's individual contribution to the
                            per-fold-vs-pooled AUC gap

2) EXTERNAL_CSV -- ADNI, one row per (mode, criterion, fold):
     mode, criterion, fold, ext_auc

Expected total rows after merge: 3 modes x 2 criteria x 5 folds = 30.
The script hard-fails (rather than silently proceeding) if that count
is wrong, if any key is duplicated, or if any predictor column has NaNs.
--------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

INTERNAL_CSV = "internal_fold_metrics.csv"   # <-- point at your real file
EXTERNAL_CSV = "external_fold_auc.csv"       # <-- point at your real file

N_PERM = 20000
RNG_SEED = 42
PREDICTORS = ["int_auc", "int_brier", "int_scale_offset", "int_gap_contribution"]
KEY = ["mode", "criterion", "fold"]


def load_and_merge():
    internal = pd.read_csv(INTERNAL_CSV)
    external = pd.read_csv(EXTERNAL_CSV)

    for df, name in [(internal, INTERNAL_CSV), (external, EXTERNAL_CSV)]:
        missing = set(KEY) - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing key columns: {missing}")

    merged = internal.merge(external, on=KEY, how="inner", validate="one_to_one")

    expected_rows = 3 * 2 * 5
    if len(merged) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} merged rows (3 modes x 2 criteria x 5 "
            f"folds), got {len(merged)}. Check for missing fold-models, typo'd "
            f"mode/criterion labels, or duplicate keys."
        )

    for col in PREDICTORS + ["ext_auc"]:
        if col not in merged.columns:
            raise ValueError(f"Merged frame missing required column: {col}")
        if merged[col].isna().any():
            raise ValueError(f"Column '{col}' has NaNs after merge -- fix inputs first.")

    return merged


def within_group_table(merged):
    rows = []
    for (mode, criterion), g in merged.groupby(["mode", "criterion"]):
        if len(g) != 5:
            raise ValueError(f"Group {mode}/{criterion} has {len(g)} rows, expected 5.")
        row = {"mode": mode, "criterion": criterion, "n": len(g)}
        for pred in PREDICTORS:
            rho, p = spearmanr(g[pred], g["ext_auc"])
            row[f"{pred}_rho"] = rho
            row[f"{pred}_p_nominal_n5_use_with_care"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def sign_consistency(within_df):
    print("\n=== SIGN CONSISTENCY ACROSS THE 6 GROUPS (the decisive table) ===")
    verdicts = {}
    for pred in PREDICTORS:
        signs = np.sign(within_df[f"{pred}_rho"]).astype(int).tolist()
        nonzero = [s for s in signs if s != 0]
        consistent = len(set(nonzero)) <= 1
        verdicts[pred] = consistent
        print(f"{pred:22s} signs={signs}  sign_consistent={consistent}")
    return verdicts


def pooled_correlation(merged):
    rows = []
    for pred in PREDICTORS:
        rho, p_nominal = spearmanr(merged[pred], merged["ext_auc"])
        rows.append({
            "predictor": pred,
            "pooled_rho_n30_nonindependent": rho,
            "pooled_p_nominal_INVALID_do_not_report": p_nominal,
        })
    return pd.DataFrame(rows)


def permutation_null(merged, n_perm=N_PERM, seed=RNG_SEED):
    """
    Shuffle ext_auc WITHIN each (mode, criterion) group only, preserving
    the 5-fold block structure. Answers: 'if fold identity carried no
    real link between the internal predictor and external AUC, what
    pooled |rho| would this exact group structure produce from noise
    alone?' Same logic as the Airola permutation control in Experiment A.
    This does not manufacture independence -- it gives an honest floor.
    """
    rng = np.random.default_rng(seed)
    idx_by_group = [
        g.index.to_numpy() for _, g in merged.groupby(["mode", "criterion"])
    ]
    ext = merged["ext_auc"].to_numpy()
    null_rhos = {pred: np.empty(n_perm) for pred in PREDICTORS}

    for i in range(n_perm):
        shuffled = ext.copy()
        for idxs in idx_by_group:
            shuffled[idxs] = ext[rng.permutation(idxs)]
        for pred in PREDICTORS:
            rho, _ = spearmanr(merged[pred].to_numpy(), shuffled)
            null_rhos[pred][i] = rho

    rows = []
    for pred in PREDICTORS:
        observed_rho, _ = spearmanr(merged[pred], merged["ext_auc"])
        null = null_rhos[pred]
        p_perm = (np.sum(np.abs(null) >= abs(observed_rho)) + 1) / (n_perm + 1)
        rows.append({
            "predictor": pred,
            "observed_pooled_rho": observed_rho,
            "null_5th_pct": np.percentile(null, 5),
            "null_95th_pct": np.percentile(null, 95),
            "p_perm_two_sided": p_perm,
        })
    return pd.DataFrame(rows)


def main():
    merged = load_and_merge()

    within_df = within_group_table(merged)
    print("=== WITHIN-GROUP (n=5 per group) SPEARMAN RHO ===")
    print(within_df.to_string(index=False))

    verdicts = sign_consistency(within_df)

    pooled_df = pooled_correlation(merged)
    print("\n=== POOLED (n=30, NON-INDEPENDENT -- nominal p is INVALID) ===")
    print(pooled_df.to_string(index=False))

    perm_df = permutation_null(merged)
    print("\n=== PERMUTATION NULL (group structure preserved) ===")
    print(perm_df.to_string(index=False))

    within_df.to_csv("experiment_c_within_group.csv", index=False)
    pooled_df.to_csv("experiment_c_pooled.csv", index=False)
    perm_df.to_csv("experiment_c_permutation_null.csv", index=False)

    print("\nWrote: experiment_c_within_group.csv, experiment_c_pooled.csv, "
          "experiment_c_permutation_null.csv")

    print("\n=== BOTTOM LINE ===")
    any_consistent = any(verdicts.values())
    if not any_consistent:
        print("No predictor is sign-consistent across all 6 groups. "
              "Do not claim an internal-gap-forecasts-external-failure result. "
              "The only defensible discussion-section sentence is the pooled "
              "int_auc rho, explicitly caveated as n=30 non-independent.")
    else:
        consistent_preds = [p for p, ok in verdicts.items() if ok]
        print(f"Sign-consistent across all 6 groups: {consistent_preds}. "
              "Still report with the n=5-per-group and non-independence "
              "caveats -- sign consistency is necessary, not sufficient, "
              "for a real effect.")


if __name__ == "__main__":
    main()
