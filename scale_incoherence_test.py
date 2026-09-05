"""
Is the per-fold/pooled gap genuine cross-model incoherence, or the known
negative bias of pooled AUC?

The threat
----------
Pooling predictions from different folds to compute a single AUC is documented
to carry a negative bias (Airola et al. 2009/2011; Parker et al. 2007; Forman &
Scholz 2010; Smith et al. 2014). Tsamardinos et al. (2018) state the mechanism
directly: pooling combines predictions from different models, so the scores must
be on a comparable scale for AUC to be meaningful.

So a reviewer can say: "your gap is a known estimator artefact, not a finding."
This script exists to answer that, not to argue around it.

The two explanations make DIFFERENT predictions
-----------------------------------------------
  (a) SCALE INCOHERENCE (this paper's claim): the five fold-models rank subjects
      well individually but place probabilities on mutually incompatible scales.
      Pooling then compares incomparable numbers.
  (b) ESTIMATOR BIAS (Airola): pooling is negatively biased as an estimator,
      independent of whether the models agree on scale.

Three tests separate them:

1. RANK TRANSFORM. Convert each fold's probabilities to within-fold ranks in
   [0,1] before pooling. This destroys between-fold scale differences while
   preserving every within-fold ordering exactly. Per-fold AUC is mathematically
   unchanged (AUC depends only on ranks).
     - if the gap COLLAPSES  -> the gap was scale incoherence  -> (a)
     - if the gap SURVIVES   -> something rank-invariant causes it -> (b)

2. PERMUTATION NULL. Randomly reassign which fold each prediction came from,
   keeping the subject-label-probability triples intact, and recompute the gap
   many times. Shuffling destroys any real between-fold difference while leaving
   the pooling operation and the fold sizes untouched. The resulting
   distribution is the gap you would see from pooling ALONE. That is the bias
   floor, measured on this data rather than assumed.

3. CORRECTED GAP = observed gap - null floor, reported with a percentile-based
   p-value. This is the number that should go in the paper.

Honest note on what a null result would mean: if the gap survives the rank
transform and sits inside the permutation null, the paper's mechanism
explanation is wrong and the claim must be rewritten. That outcome is reported
as-is; the script does not favour either answer.

Inputs
------
OASIS-3: outputs/cv/sel_{clinical,imaging,fusion}_folds.json  (out-of-fold -
  each subject appears in exactly one fold's test set)
ADNI:    outputs/adni_external/adni_preds_{mode}_fold{k}_{criterion}.csv
  (cross-fold pooled-style - every subject is scored by all five fold-models)

The two cohorts need different handling and the script keeps them separate
rather than pretending they are the same statistic.

Output
------
outputs/scale_test/scale_incoherence_results.json
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

CV_DIR = "outputs/cv"
ADNI_DIR = "outputs/adni_external"
OUT_DIR = "outputs/scale_test"
MODES = {"clinical": "clinical_only", "imaging": "imaging_only", "fusion": "fusion"}
CRITERIA = ["auc", "neg_brier"]
N_PERM = 2000
SEED = 42


# ----------------------------------------------------------------- helpers
def rank_normalise(p):
    """Map probabilities to within-fold ranks scaled to (0,1).

    Strictly monotonic within the fold, so every within-fold ordering - and
    therefore the per-fold AUC - is preserved exactly. Only the between-fold
    scale is destroyed. Ties get average ranks, matching how AUC treats them.
    """
    if len(p) < 2:
        return np.full(len(p), 0.5)
    return (rankdata(p, method="average") - 0.5) / len(p)


def gap_from_folds(fold_probs, fold_labels, transform=None):
    """per-fold mean AUC, pooled AUC, gap. `transform` applied per fold."""
    per_fold, pooled_p, pooled_y = [], [], []
    for p, y in zip(fold_probs, fold_labels):
        if len(set(y)) < 2:
            return None
        per_fold.append(roc_auc_score(y, p))
        pooled_p.append(transform(p) if transform else p)
        pooled_y.append(y)
    pf = float(np.mean(per_fold))
    pooled = float(roc_auc_score(np.concatenate(pooled_y),
                                 np.concatenate(pooled_p)))
    return pf, pooled, pf - pooled


def permutation_null(fold_probs, fold_labels, n_perm=N_PERM, seed=SEED):
    """Gap distribution when fold membership carries no information.

    All (probability, label) pairs are pooled and randomly re-split into folds
    of the ORIGINAL sizes. The pooling operation, the fold count and the fold
    sizes are all preserved; only the association between a prediction and the
    model that produced it is destroyed. Any gap remaining is attributable to
    the estimator, not to between-model differences.
    """
    rng = np.random.default_rng(seed)
    all_p = np.concatenate(fold_probs)
    all_y = np.concatenate(fold_labels)
    sizes = [len(p) for p in fold_probs]
    n = len(all_p)

    gaps = []
    for _ in range(n_perm):
        idx = rng.permutation(n)
        start, fp, fy = 0, [], []
        for s in sizes:
            sel = idx[start:start + s]
            fp.append(all_p[sel])
            fy.append(all_y[sel])
            start += s
        out = gap_from_folds(fp, fy)
        if out is not None:
            gaps.append(out[2])
    return np.array(gaps)


def analyse(name, fold_probs, fold_labels, note):
    raw = gap_from_folds(fold_probs, fold_labels)
    if raw is None:
        return None
    pf, pooled, gap = raw

    rank = gap_from_folds(fold_probs, fold_labels, transform=rank_normalise)
    pf_r, pooled_r, gap_r = rank

    null = permutation_null(fold_probs, fold_labels)
    null_mean = float(null.mean())
    null_p95 = float(np.percentile(null, 95))
    # one-sided: how often does pooling alone produce a gap this large?
    p_val = float((null >= gap).mean())

    closed = 1.0 - (gap_r / gap) if abs(gap) > 1e-9 else float("nan")

    res = {
        "analysis": name,
        "note": note,
        "per_fold_mean_auc": pf,
        "pooled_auc_raw": pooled,
        "gap_raw": gap,
        "per_fold_mean_auc_rank": pf_r,
        "pooled_auc_rank": pooled_r,
        "gap_rank_transformed": gap_r,
        "fraction_of_gap_closed_by_rank_transform": closed,
        "null_gap_mean": null_mean,
        "null_gap_p95": null_p95,
        "null_gap_sd": float(null.std()),
        "gap_minus_null_mean": gap - null_mean,
        "permutation_p_value": p_val,
        "n_permutations": int(len(null)),
        "seed": SEED,
    }

    print(f"  {name}")
    print(f"    per-fold {pf:.4f}  pooled {pooled:.4f}  gap {gap:.4f}")
    print(f"    rank-transformed pooled {pooled_r:.4f}  gap {gap_r:.4f}   "
          f"({closed*100:.0f}% of gap closed)")
    print(f"    permutation null: mean {null_mean:.4f}  p95 {null_p95:.4f}  "
          f"p = {p_val:.4f}")
    print(f"    corrected gap (observed - null): {gap - null_mean:.4f}")
    # per-fold AUC must be unchanged by a within-fold monotonic transform
    if abs(pf - pf_r) > 1e-9:
        print(f"    WARNING: per-fold AUC changed under rank transform "
              f"({pf:.6f} -> {pf_r:.6f}) - it should be invariant. "
              f"Check for ties or a bug.")
    return res


# ----------------------------------------------------------------- loaders
def load_oasis(mode_key, criterion):
    """OASIS-3 out-of-fold: each subject appears in exactly one fold."""
    stub = {"clinical": "sel_clinical", "imaging": "sel_imaging",
            "fusion": "sel_fusion"}[mode_key]
    path = os.path.join(CV_DIR, f"{stub}_folds.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    fp, fy, seen = [], [], set()
    for f in d["folds"]:
        pr = f["by_criterion"][criterion]["predictions"]
        subs = pr["subject"]
        if seen & set(subs):
            raise ValueError(f"{stub}/{criterion}: a subject appears in more "
                             f"than one fold's test set - not out-of-fold")
        seen |= set(subs)
        fp.append(np.asarray(pr["prob_ad"], dtype=float))
        fy.append(np.asarray(pr["label"], dtype=int))
    return fp, fy


def load_adni(mode_full, criterion):
    """ADNI: every subject scored by all five fold-models (correlated rows)."""
    fp, fy, ref = [], [], None
    for k in range(1, 6):
        path = os.path.join(ADNI_DIR,
                            f"adni_preds_{mode_full}_fold{k}_{criterion}.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path).sort_values("subject").reset_index(drop=True)
        if ref is None:
            ref = list(df["subject"])
        elif list(df["subject"]) != ref:
            raise ValueError(f"{mode_full}/{criterion}: fold {k} covers a "
                             f"different subject set - refusing to pool")
        fp.append(df["prob"].to_numpy(float))
        fy.append(df["label"].to_numpy(int))
    return fp, fy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--n_perm", type=int, default=N_PERM)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results = []

    print("=" * 74)
    print("OASIS-3  (true out-of-fold: one prediction per subject)")
    print("=" * 74)
    for mk, mfull in MODES.items():
        for crit in CRITERIA:
            loaded = load_oasis(mk, crit)
            if loaded is None:
                continue
            r = analyse(f"oasis/{mk}/{crit}", *loaded,
                        note="true out-of-fold; each subject in exactly one "
                             "fold test set")
            if r:
                results.append(r)

    print()
    print("=" * 74)
    print("ADNI  (cross-fold pooled-style: 5 correlated rows per subject)")
    print("=" * 74)
    for mk, mfull in MODES.items():
        for crit in CRITERIA:
            loaded = load_adni(mfull, crit)
            if loaded is None:
                continue
            r = analyse(f"adni/{mk}/{crit}", *loaded,
                        note="cross-fold pooled-style; every subject scored by "
                             "all five fold-models, so rows are correlated and "
                             "this is a coherence diagnostic, not a pooled AUC")
            if r:
                results.append(r)

    path = os.path.join(args.out_dir, "scale_incoherence_results.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)

    print()
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    print("  If the rank transform closes most of the gap AND the observed gap")
    print("  sits far above the permutation null, the gap reflects between-fold")
    print("  SCALE INCOHERENCE, not the estimator bias of pooling.")
    print("  If the gap survives the rank transform, or sits inside the null,")
    print("  the mechanism explanation in the paper is wrong.\n")
    for r in results:
        if abs(r["gap_raw"]) < 0.02:
            continue
        verdict = ("scale incoherence"
                   if r["fraction_of_gap_closed_by_rank_transform"] > 0.5
                   and r["permutation_p_value"] < 0.05
                   else "NOT explained by scale - re-examine")
        print(f"  {r['analysis']:28s} gap {r['gap_raw']:.4f} -> "
              f"{r['gap_rank_transformed']:.4f} after rank, "
              f"null {r['null_gap_mean']:.4f}, p={r['permutation_p_value']:.4f}"
              f"   [{verdict}]")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
