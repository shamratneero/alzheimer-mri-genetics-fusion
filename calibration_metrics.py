"""
calibration_metrics.py

Computes the five-metric calibration panel (ECE, AECE, MCE, OE, Brier) and
per-fold reliability diagrams, for both cohorts, from predictions already on
disk. No GPU, no re-inference.

WHY FIVE METRICS
----------------
Dawood et al. (2023, Medical Image Analysis) report ECE, AECE, OE, MCE and
Brier, and show that conclusions about calibration can differ depending on
which metric you look at. Reporting two metrics against a competitor who
reports five is an easy reviewer objection. This closes that gap.

SCOPE, AND WHY IT IS ASYMMETRIC
-------------------------------
OASIS-3: all four selection criteria (auc, auc_minus_ece, gated_bacc,
         neg_brier) -- all four were run in cross-validation.
ADNI:    auc and neg_brier only -- only those two checkpoints were pushed
         through external inference.
This asymmetry is stated rather than hidden. Fabricating the missing two
would require an inference run that was never done.

CONVENTION -- READ THIS BEFORE ADDING A METRIC
----------------------------------------------
train.py's expected_calibration_error compares, within each bin, the mean
predicted probability against the POSITIVE RATE (fraction of the bin that is
truly class 1). An earlier independently-written ECE used per-bin ACCURACY
instead. Those are different quantities and the discrepancy was a real bug.

Therefore:
  - ECE is IMPORTED from train.py, never reimplemented here.
  - AECE, MCE and OE are defined below on the SAME positive-rate convention,
    so all four are mutually comparable.
Do not add a metric to this file on the accuracy convention.

METRIC DEFINITIONS (all on positive-rate convention)
----------------------------------------------------
  ECE   equal-WIDTH bins; weighted mean |mean_prob - positive_rate|.
        Weakness: with skewed probability distributions most samples land in
        one or two bins and the rest are near-empty and noisy.
  AECE  equal-MASS bins (adaptive); same weighted mean. Every bin holds the
        same number of samples, which is what makes it robust to the skew
        that distorts ECE. This is why Dawood report both.
  MCE   the WORST bin's |mean_prob - positive_rate|. Sensitive to a single
        sparse bin, so it is reported alongside ECE/AECE, never alone.
  OE    overconfidence error: sum over bins of
        weight * mean_prob * max(mean_prob - positive_rate, 0).
        Penalises only confident-and-wrong, which is the direction that
        matters clinically. Under-confidence contributes zero.
  Brier mean squared error of the probability. Decomposes into calibration
        and refinement, so a lower Brier does not by itself imply better
        calibration (Bella et al. 2012; Flach 2008) -- reported as one member
        of the panel, not as the arbiter.

RELIABILITY DIAGRAMS
--------------------
One panel per (mode, criterion), with all five fold-models overlaid as
separate curves rather than aggregated into one. Aggregating would average
away the between-fold disagreement, which is the phenomenon under study.
Curves that scatter under one criterion and tighten under another are the
visual form of the paper's argument.

Run from repo root.
Outputs:
  outputs/calibration/calibration_metrics.csv
  outputs/calibration/calibration_metrics.json
  figures_calibration/reliability_{cohort}_{mode}_{criterion}.png
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Imported, not reimplemented -- see CONVENTION note above.
from train import expected_calibration_error

CV_DIR = "outputs/cv"
ADNI_DIR = "outputs/adni_external"
OUT_DIR = "outputs/calibration"
FIG_DIR = "figures_calibration"

STUB = {"clinical": "sel_clinical", "imaging": "sel_imaging", "fusion": "sel_fusion"}
MODE_FULL = {"clinical": "clinical_only", "imaging": "imaging_only", "fusion": "fusion"}
OASIS_CRITERIA = ["auc", "auc_minus_ece", "gated_bacc", "neg_brier"]
ADNI_CRITERIA = ["auc", "neg_brier"]
N_BINS = 10


# --------------------------------------------------------------- metrics
def _bin_stats(probs, labels, edges):
    """Per-bin (weight, mean_prob, positive_rate) for non-empty bins."""
    out = []
    n = len(probs)
    for lo, hi, is_last in zip(edges[:-1], edges[1:],
                               [False] * (len(edges) - 2) + [True]):
        m = (probs >= lo) & (probs <= hi) if is_last else (probs >= lo) & (probs < hi)
        if not m.any():
            continue
        out.append((m.sum() / n, float(probs[m].mean()), float(labels[m].mean())))
    return out


def adaptive_ece(probs, labels, n_bins=N_BINS):
    """Equal-MASS binning: quantile edges so each bin holds ~n/n_bins samples."""
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, float)
    edges = np.quantile(probs, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)          # collapse duplicate quantiles
    if len(edges) < 2:                # degenerate: all probabilities identical
        return float(abs(probs.mean() - labels.mean()))
    return float(sum(w * abs(mp - pr) for w, mp, pr in _bin_stats(probs, labels, edges)))


def max_calibration_error(probs, labels, n_bins=N_BINS):
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    stats = _bin_stats(probs, labels, edges)
    return float(max((abs(mp - pr) for _, mp, pr in stats), default=0.0))


def overconfidence_error(probs, labels, n_bins=N_BINS):
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return float(sum(w * mp * max(mp - pr, 0.0)
                     for w, mp, pr in _bin_stats(probs, labels, edges)))


def brier(probs, labels):
    return float(np.mean((np.asarray(probs, float) - np.asarray(labels, float)) ** 2))


def all_metrics(probs, labels):
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, int)
    return {
        "ece": float(expected_calibration_error(probs, labels)),
        "aece": adaptive_ece(probs, labels),
        "mce": max_calibration_error(probs, labels),
        "oe": overconfidence_error(probs, labels),
        "brier": brier(probs, labels),
        "n": int(len(probs)),
        "positive_rate": float(labels.mean()),
        "mean_prob": float(probs.mean()),
    }


# --------------------------------------------------------------- loaders
def load_oasis(mode_key, criterion):
    path = os.path.join(CV_DIR, f"{STUB[mode_key]}_folds.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    folds = []
    for f in d["folds"]:
        bc = f.get("by_criterion", {})
        if criterion not in bc:
            return None
        pr = bc[criterion]["predictions"]
        folds.append((int(f["fold"]),
                      np.asarray(pr["prob_ad"], float),
                      np.asarray(pr["label"], int)))
    return folds


def load_adni(mode_key, criterion):
    folds = []
    for k in range(1, 6):
        path = os.path.join(
            ADNI_DIR, f"adni_preds_{MODE_FULL[mode_key]}_fold{k}_{criterion}.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        folds.append((k, df["prob"].to_numpy(float), df["label"].to_numpy(int)))
    return folds


# --------------------------------------------------------------- figure
def reliability_figure(folds, cohort, mode_key, criterion, n_bins=N_BINS):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect", zorder=1)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres, curves = [], []
    for fold_id, probs, labels in folds:
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (probs >= lo) & (probs < hi) if hi < 1.0 else (probs >= lo) & (probs <= hi)
            if not m.any():
                continue
            xs.append(probs[m].mean())
            ys.append(labels[m].mean())
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.2, alpha=0.85,
                label=f"fold {fold_id}", zorder=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed positive rate")
    ax.set_title(f"{cohort} / {mode_key} / {criterion}", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"reliability_{cohort}_{mode_key}_{criterion}.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--n_bins", type=int, default=N_BINS)
    ap.add_argument("--no_figures", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows, figures = [], []

    for cohort, loader, criteria in [
        ("oasis", load_oasis, OASIS_CRITERIA),
        ("adni", load_adni, ADNI_CRITERIA),
    ]:
        print("=" * 72)
        print(f"{cohort.upper()}   criteria: {', '.join(criteria)}")
        print("=" * 72)

        for mode_key in ["clinical", "imaging", "fusion"]:
            for crit in criteria:
                folds = loader(mode_key, crit)
                if folds is None:
                    print(f"  [missing] {mode_key}/{crit} -- skipped")
                    continue

                for fold_id, probs, labels in folds:
                    m = all_metrics(probs, labels)
                    m.update(cohort=cohort, mode=mode_key,
                             criterion=crit, fold=fold_id)
                    rows.append(m)

                sub = [r for r in rows
                       if r["cohort"] == cohort and r["mode"] == mode_key
                       and r["criterion"] == crit]
                print(f"  {mode_key:9s} {crit:14s} "
                      f"ECE {np.mean([r['ece'] for r in sub]):.4f}  "
                      f"AECE {np.mean([r['aece'] for r in sub]):.4f}  "
                      f"MCE {np.mean([r['mce'] for r in sub]):.4f}  "
                      f"OE {np.mean([r['oe'] for r in sub]):.4f}  "
                      f"Brier {np.mean([r['brier'] for r in sub]):.4f}")

                if not args.no_figures:
                    figures.append(
                        reliability_figure(folds, cohort, mode_key, crit, args.n_bins))

    if not rows:
        print("\nNo predictions loaded. Check paths and run from repo root.")
        return

    df = pd.DataFrame(rows)[
        ["cohort", "mode", "criterion", "fold", "n", "positive_rate",
         "mean_prob", "ece", "aece", "mce", "oe", "brier"]
    ].sort_values(["cohort", "mode", "criterion", "fold"])

    csv_path = os.path.join(args.out_dir, "calibration_metrics.csv")
    json_path = os.path.join(args.out_dir, "calibration_metrics.json")
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print(f"\nWrote {csv_path}  ({len(df)} rows)")
    print(f"Wrote {json_path}")
    if figures:
        print(f"Wrote {len(figures)} reliability figures to {FIG_DIR}/")
        print("\nREMINDER: outputs/* and *.png are gitignored by default. "
              "Add '!outputs/calibration/' and '!figures_calibration/*.png' "
              "or these will silently never be committed.")


if __name__ == "__main__":
    main()
