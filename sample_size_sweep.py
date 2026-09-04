"""
Sample-size sweep: is the per-fold/pooled gap an artefact of test-set size?

The objection this answers
-------------------------
"You trained on 365 subjects. The instability you report is a small-data
artefact - evaluate on more subjects and it will disappear."

Method
------
Subsample the ADNI predictions at increasing n and recompute both the per-fold
mean AUC and the cross-fold pooled-style AUC at each size. If the gap shrinks
toward zero as n grows, it was measurement noise. If it stays flat, it is a
stable property of the fold-models themselves - more evaluation data measures it
more precisely rather than making it go away.

No new inference is run. All 30 ADNI prediction CSVs already exist on disk, so
this is pure resampling of numbers already computed. The identical subject
subset is used across all five folds within a draw, which is required: the
pooled-style statistic compares the five fold-models' probabilities on the SAME
subjects, so drawing different subsets per fold would confound subject
composition with model disagreement.

What this does NOT establish
----------------------------
It says nothing about whether training on more data would fix the instability.
That would need retraining on progressively larger training sets, which on ADNI
would burn the clean external test set. The claim supported here is narrower and
must be stated as such: the gap is not an artefact of TEST-set size.

Design choices
--------------
- Sampling is WITHOUT replacement: genuine subsets of the real cohort, not
  bootstrap resamples.
- Stratified by label. ADNI is CN-heavy (1:1.97); an unstratified draw at n=100
  could land with too few AD subjects for a stable AUC.
- N_REPEATS draws per size, reported as mean and 5th-95th percentile. A single
  draw at small n could land anywhere by chance, so a one-shot curve would be
  noise.
- The largest size is the full cohort, which has exactly one possible draw, so
  it is evaluated once and its spread is zero by construction.

Output
  outputs/sample_size/sample_size_results.json
  outputs/sample_size/sample_size_curve.png
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

PREDS_DIR = "outputs/adni_external"
OUT_DIR = "outputs/sample_size"
MODES = ["clinical_only", "imaging_only", "fusion"]
CRITERIA = ["auc", "neg_brier"]
SIZES = [100, 200, 400, 800, None]      # None = full cohort
N_REPEATS = 20
SEED = 42


def load_fold_predictions(mode, criterion):
    """Load all five folds' ADNI predictions, indexed by subject.

    Asserts the five folds cover an identical subject set - the pooled-style
    statistic is meaningless if they do not, and a silent mismatch would
    produce a plausible-looking but wrong number.
    """
    frames = []
    for k in range(1, 6):
        path = os.path.join(PREDS_DIR, f"adni_preds_{mode}_fold{k}_{criterion}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing {path} - run inference_adni.py first")
        df = pd.read_csv(path)[["subject", "label", "prob"]]
        if df["subject"].duplicated().any():
            raise ValueError(f"{path}: duplicate subjects")
        frames.append(df.set_index("subject"))

    ref = set(frames[0].index)
    for k, f in enumerate(frames[1:], start=2):
        if set(f.index) != ref:
            raise ValueError(
                f"{mode}/{criterion}: fold 1 and fold {k} cover different "
                f"subject sets - refusing to pool")
    # labels must also agree fold to fold
    for k, f in enumerate(frames[1:], start=2):
        if not (f.loc[frames[0].index, "label"].values == frames[0]["label"].values).all():
            raise ValueError(f"{mode}/{criterion}: labels differ between "
                             f"fold 1 and fold {k}")
    return frames


def stratified_subsample(subjects, labels, n, rng):
    """Draw n subjects without replacement, preserving class proportions."""
    idx_ad = np.where(labels == 1)[0]
    idx_cn = np.where(labels == 0)[0]
    frac = n / len(labels)
    n_ad = max(2, int(round(len(idx_ad) * frac)))
    n_cn = max(2, n - n_ad)
    n_ad = min(n_ad, len(idx_ad))
    n_cn = min(n_cn, len(idx_cn))
    pick = np.concatenate([rng.choice(idx_ad, n_ad, replace=False),
                           rng.choice(idx_cn, n_cn, replace=False)])
    return subjects[pick]


def gap_for_subset(frames, subset):
    """per-fold mean AUC, pooled-style AUC, and their gap on a subject subset."""
    per_fold, pooled_p, pooled_y = [], [], []
    for f in frames:
        sub = f.loc[subset]
        y, p = sub["label"].values, sub["prob"].values
        if len(set(y)) < 2:
            return None
        per_fold.append(roc_auc_score(y, p))
        pooled_p.append(p)
        pooled_y.append(y)
    pf = float(np.mean(per_fold))
    pooled = float(roc_auc_score(np.concatenate(pooled_y),
                                 np.concatenate(pooled_p)))
    return pf, pooled, pf - pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_dir", default=PREDS_DIR)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = []

    for mode in MODES:
        for criterion in CRITERIA:
            frames = load_fold_predictions(mode, criterion)
            subjects = frames[0].index.values
            labels = frames[0]["label"].values
            n_total = len(subjects)
            print(f"\n{mode} [{criterion}]  n={n_total} "
                  f"({int(labels.sum())} AD / {int((1-labels).sum())} CN)")

            for size in SIZES:
                n = n_total if size is None else size
                if n > n_total:
                    continue
                # the full cohort has exactly one possible draw
                reps = 1 if n == n_total else args.repeats
                rng = np.random.default_rng(SEED)

                gaps, pfs, pooleds = [], [], []
                for _ in range(reps):
                    subset = (subjects if n == n_total
                              else stratified_subsample(subjects, labels, n, rng))
                    out = gap_for_subset(frames, subset)
                    if out is None:
                        continue
                    pf, pooled, gap = out
                    pfs.append(pf); pooleds.append(pooled); gaps.append(gap)

                if not gaps:
                    continue
                row = {
                    "mode": mode, "criterion": criterion,
                    "n": int(n), "n_repeats": len(gaps),
                    "gap_mean": float(np.mean(gaps)),
                    "gap_sd": float(np.std(gaps)),
                    "gap_p5": float(np.percentile(gaps, 5)),
                    "gap_p95": float(np.percentile(gaps, 95)),
                    "per_fold_mean_auc": float(np.mean(pfs)),
                    "pooled_style_auc": float(np.mean(pooleds)),
                }
                results.append(row)
                print(f"  n={n:>5}  gap {row['gap_mean']:.4f} "
                      f"±{row['gap_sd']:.4f}  "
                      f"[{row['gap_p5']:.4f}, {row['gap_p95']:.4f}]  "
                      f"(per-fold {row['per_fold_mean_auc']:.4f}, "
                      f"pooled {row['pooled_style_auc']:.4f}, "
                      f"{row['n_repeats']} draws)")

    out_json = os.path.join(args.out_dir, "sample_size_results.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved to {out_json}")

    # ---- figure -------------------------------------------------------
    df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, len(MODES), figsize=(4.2 * len(MODES), 3.8),
                             sharey=True)
    colors = {"auc": "#c0392b", "neg_brier": "#2471a3"}
    for ax, mode in zip(np.atleast_1d(axes), MODES):
        for crit in CRITERIA:
            d = df[(df["mode"] == mode) & (df["criterion"] == crit)].sort_values("n")
            if d.empty:
                continue
            ax.plot(d["n"], d["gap_mean"], "o-", color=colors[crit],
                    label=crit, markersize=4)
            ax.fill_between(d["n"], d["gap_p5"], d["gap_p95"],
                            color=colors[crit], alpha=0.18)
        ax.axhline(0, color="grey", lw=0.8, ls=":")
        ax.set_title(mode, fontsize=10)
        ax.set_xlabel("ADNI subjects sampled")
        ax.set_xscale("log")
    np.atleast_1d(axes)[0].set_ylabel("per-fold mean AUC − pooled-style AUC")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Does the per-fold/pooled gap close with more evaluation data?\n"
                 "Shaded band = 5th-95th percentile over draws", fontsize=10)
    fig.tight_layout()
    fig_path = os.path.join(args.out_dir, "sample_size_curve.png")
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {fig_path}")

    # ---- the headline read -------------------------------------------
    print("\n" + "=" * 74)
    print("DOES THE GAP CLOSE WITH MORE EVALUATION DATA?")
    print("=" * 74)
    for mode in MODES:
        for crit in CRITERIA:
            d = df[(df["mode"] == mode) & (df["criterion"] == crit)].sort_values("n")
            if len(d) < 2:
                continue
            small, large = d.iloc[0], d.iloc[-1]
            change = large["gap_mean"] - small["gap_mean"]
            print(f"  {mode:14s} {crit:10s} "
                  f"n={int(small['n']):>4} gap {small['gap_mean']:.4f}  ->  "
                  f"n={int(large['n']):>4} gap {large['gap_mean']:.4f}   "
                  f"change {change:+.4f}")
    print("\nA gap that stays roughly flat as n grows is a property of the")
    print("fold-models, not a small-sample measurement artefact. Note this")
    print("concerns TEST-set size only - it says nothing about whether more")
    print("TRAINING data would fix the instability.")


if __name__ == "__main__":
    main()
