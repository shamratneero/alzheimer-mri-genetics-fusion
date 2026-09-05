"""
oe_degeneracy_check.py

QUESTION
--------
Does overconfidence error (OE) give a LOW (good-looking) score to degenerate
fold-models that predict a single class for every subject?

WHY THIS MATTERS
----------------
The paper already shows that temperature scaling can make a one-class model
look well calibrated (fusion fold 4: fitted T = 145,474, ECE 0.4353 -> 0.0342,
balanced accuracy 0.5000 before and after, 0 of 73 predicted AD both times).
If OE has the same blind spot, that is a second, independent instance of the
same pattern -- a standard reliability measure that a broken model passes --
which is a stronger claim than either instance alone.

THE MECHANISM BEING TESTED
--------------------------
OE = sum over bins of  weight * mean_prob * max(mean_prob - positive_rate, 0)

The max(...) makes it ASYMMETRIC: only bins where the model is more confident
than the outcome rate contribute. Under-confidence contributes exactly zero.

So the prediction splits by DIRECTION of degeneracy:
  all-CN degenerate  -> probabilities near 0, so mean_prob < positive_rate in
                        most bins, so OE ~ 0. LOOKS EXCELLENT. This is the
                        blind spot.
  all-AD degenerate  -> probabilities near 1 with positive_rate < 1, so
                        mean_prob > positive_rate, so OE is LARGE. Caught.

Lumping both directions together would wash the effect out, so they are
reported separately.

HOW THIS CAN FALSIFY THE HYPOTHESIS
-----------------------------------
If all-CN degenerate folds do NOT have systematically lower OE than working
folds, the hypothesis is wrong and the script says so. It does not assume the
answer. With a handful of degenerate folds this is a descriptive comparison,
not a significance test, and it is labelled as such.

Run from repo root, after calibration_metrics.py.
Output: outputs/calibration/oe_degeneracy_check.csv
"""

import os
import json
import numpy as np
import pandas as pd

from calibration_metrics import (
    overconfidence_error, adaptive_ece, max_calibration_error, brier,
    load_oasis, load_adni, OASIS_CRITERIA, ADNI_CRITERIA,
)
from train import expected_calibration_error

OUT_DIR = "outputs/calibration"
THRESHOLD = 0.5


def balanced_accuracy(probs, labels, thr=THRESHOLD):
    pred = (np.asarray(probs) >= thr).astype(int)
    labels = np.asarray(labels, int)
    recalls = []
    for cls in (0, 1):
        m = labels == cls
        if not m.any():
            return float("nan")
        recalls.append(float((pred[m] == cls).mean()))
    return float(np.mean(recalls))


def degeneracy(probs, labels, thr=THRESHOLD):
    """Returns (is_degenerate, direction) at the given decision threshold."""
    pred = (np.asarray(probs) >= thr).astype(int)
    n_pos = int(pred.sum())
    if n_pos == 0:
        return True, "all_CN"
    if n_pos == len(pred):
        return True, "all_AD"
    return False, "working"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    for cohort, loader, criteria in [
        ("oasis", load_oasis, OASIS_CRITERIA),
        ("adni", load_adni, ADNI_CRITERIA),
    ]:
        for mode_key in ["clinical", "imaging", "fusion"]:
            for crit in criteria:
                folds = loader(mode_key, crit)
                if folds is None:
                    continue
                for fold_id, probs, labels in folds:
                    is_deg, direction = degeneracy(probs, labels)
                    rows.append({
                        "cohort": cohort,
                        "mode": mode_key,
                        "criterion": crit,
                        "fold": fold_id,
                        "n": len(probs),
                        "n_pred_AD": int((probs >= THRESHOLD).sum()),
                        "bal_acc": balanced_accuracy(probs, labels),
                        "degenerate": is_deg,
                        "direction": direction,
                        "mean_prob": float(np.mean(probs)),
                        "positive_rate": float(np.mean(labels)),
                        "oe": overconfidence_error(probs, labels),
                        "ece": float(expected_calibration_error(probs, labels)),
                        "aece": adaptive_ece(probs, labels),
                        "mce": max_calibration_error(probs, labels),
                        "brier": brier(probs, labels),
                    })

    if not rows:
        print("No predictions loaded. Run from repo root.")
        return

    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "oe_degeneracy_check.csv")
    df.to_csv(path, index=False)

    print("=" * 78)
    print("DEGENERATE FOLD-MODELS (predict one class for every subject at p>=0.5)")
    print("=" * 78)
    deg = df[df["degenerate"]]
    if deg.empty:
        print("  None found. The hypothesis cannot be tested on this data.")
        print(f"\nWrote {path}")
        return

    cols = ["cohort", "mode", "criterion", "fold", "direction", "n_pred_AD",
            "bal_acc", "oe", "ece", "brier"]
    print(deg[cols].to_string(index=False))

    print("\n" + "=" * 78)
    print("OE: DEGENERATE vs WORKING  (lower OE looks BETTER)")
    print("=" * 78)
    work = df[~df["degenerate"]]
    for label, sub in [("all_CN degenerate", deg[deg["direction"] == "all_CN"]),
                       ("all_AD degenerate", deg[deg["direction"] == "all_AD"]),
                       ("working models", work)]:
        if sub.empty:
            print(f"  {label:20s}  n=0")
            continue
        print(f"  {label:20s}  n={len(sub):3d}   "
              f"OE mean {sub['oe'].mean():.4f}  median {sub['oe'].median():.4f}  "
              f"range [{sub['oe'].min():.4f}, {sub['oe'].max():.4f}]")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    cn = deg[deg["direction"] == "all_CN"]
    if cn.empty:
        print("  No all-CN degenerate folds, so the specific blind spot")
        print("  predicted by OE's asymmetry cannot be demonstrated here.")
    elif work.empty:
        print("  No working folds to compare against.")
    else:
        cn_med, work_med = cn["oe"].median(), work["oe"].median()
        if cn_med < work_med:
            print(f"  SUPPORTED (descriptively): all-CN degenerate folds have")
            print(f"  LOWER median OE ({cn_med:.4f}) than working models "
                  f"({work_med:.4f}),")
            print(f"  i.e. OE scores broken one-class models as BETTER "
                  f"calibrated.")
            print(f"  n = {len(cn)} degenerate vs {len(work)} working -- "
                  f"descriptive, not a significance test.")
        else:
            print(f"  NOT SUPPORTED: all-CN degenerate folds have median OE "
                  f"{cn_med:.4f},")
            print(f"  not lower than working models ({work_med:.4f}). The "
                  f"hypothesis is wrong;")
            print(f"  do not claim OE has this blind spot.")

    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
