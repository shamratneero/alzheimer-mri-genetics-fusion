"""
5-fold cross-validated evaluation for the OASIS-3 Alzheimer's fusion model.

WHY THIS EXISTS
---------------
Single-split results on this cohort are not reportable. With 39 AD / 34 CN in
the test split, bootstrap simulation puts the standard deviation of test AUC
from sampling alone at ~0.060 - meaning two models whose true AUCs differ by
less than ~0.12 cannot be told apart. The observed spread across imaging-only
(0.796), clinical-only (0.781) and fusion (0.773) is 0.023, i.e. well inside
noise. Four single-split runs also showed val->test AUC drops of -0.110,
-0.119, -0.093 and -0.156, confirming that checkpoint selection on 37 val
subjects is unreliable.

WHAT THIS BUYS
--------------
Pooling out-of-fold predictions gives one AUC computed over all 365 subjects
rather than 73, cutting sampling noise ~2.3x (std ~0.060 -> ~0.027). It also
yields per-fold spread, so every reported number carries an interval.

BE CLEAR ABOUT WHAT IT DOES NOT BUY: even at n=365 the resolvable difference
is ~0.053. If the modes land within ~0.05 of each other, the correct
conclusion is that they are statistically indistinguishable on this cohort -
which is a legitimate, reportable finding, not a failed experiment.

DESIGN NOTES
------------
- StratifiedKFold on the outer loop: every subject is tested exactly once,
  so out-of-fold predictions cover the whole cohort with no reuse.
- An inner validation split is carved from each fold's training portion for
  early stopping and checkpoint selection. Test data never influences
  training or model selection.
- Age normalisation statistics come from each fold's own training split.
- Resume-safe: completed folds are written to disk and skipped on rerun,
  because full imaging runs take hours per fold and power cuts happen.

USAGE
-----
    python cross_validate.py --mode imaging_only  --tag cv_imaging
    python cross_validate.py --mode clinical_only --tag cv_clinical
    python cross_validate.py --mode fusion        --tag cv_fusion
    python cross_validate.py --compare cv_imaging cv_clinical cv_fusion
"""
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from dataset_3d import OASIS3Dataset
from model_3d import FusionModel
from train import evaluate, forward_pass, expected_calibration_error, set_seed

COHORT = "oasis3_cohort.csv"
PRE_DIR = r"D:\alhseimer\preprocessed"
CV_DIR = "outputs/cv"
os.makedirs(CV_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------
def build_cv_folds(cohort_csv, n_folds=5, val_frac=0.125, seed=42):
    """Stratified outer folds, each with an inner train/val split.

    val_frac is a fraction of the *training portion* of the fold, chosen so the
    inner validation set is about the same size as the old single-split one
    (~36 subjects) - this keeps early-stopping behaviour comparable to the
    single-split runs rather than changing two things at once.
    """
    df = pd.read_csv(cohort_csv)
    df["label"] = df["label"].astype(int)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in skf.split(df, df["label"]):
        trainval = df.iloc[train_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)
        train, val = train_test_split(
            trainval, test_size=val_frac, stratify=trainval["label"],
            random_state=seed)
        folds.append((train.reset_index(drop=True),
                      val.reset_index(drop=True),
                      test))
    return folds


# --------------------------------------------------------------------------
# one fold
# --------------------------------------------------------------------------
def run_fold(fold_idx, train_df, val_df, test_df, args, device):
    """Train on one fold and return test metrics plus out-of-fold predictions."""
    imaging_only = args.mode == "imaging_only"
    clinical_only = args.mode == "clinical_only"

    train_ds = OASIS3Dataset(train_df, PRE_DIR, target_size=args.target_size, augment=True)
    val_ds = OASIS3Dataset(val_df, PRE_DIR, target_size=args.target_size, augment=False)
    test_ds = OASIS3Dataset(test_df, PRE_DIR, target_size=args.target_size, augment=False)

    # fold-local train statistics only - no leakage from val/test
    val_ds.set_age_norm(train_ds.age_mean, train_ds.age_std)
    test_ds.set_age_norm(train_ds.age_mean, train_ds.age_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    n_clin = train_ds[0]["clinical"].shape[0]
    model = FusionModel(n_clinical_features=n_clin, base_ch=args.base_ch,
                        dropout=args.dropout).to(device)

    n_cn = (train_df["label"] == 0).sum()
    n_ad = (train_df["label"] == 1).sum()
    weights = torch.tensor([1.0, n_cn / n_ad], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)

    best_auc, best_epoch, no_improve = -1.0, -1, 0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, n_seen = 0.0, 0
        for batch in train_loader:
            vol = batch["volume"].to(device, non_blocking=True)
            clin = batch["clinical"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            out = forward_pass(model, vol, clin, imaging_only, clinical_only)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            n_seen += y.size(0)

        val_metrics, _ = evaluate(model, val_loader, device, criterion,
                                  imaging_only, clinical_only)
        scheduler.step(val_metrics["auc"])

        history.append({
            "epoch": epoch,
            "train_loss": round(running_loss / n_seen, 4),
            "val_auc": round(float(val_metrics["auc"]), 4),
            "val_bal_acc": round(float(val_metrics["balanced_accuracy"]), 4),
            "val_ece": round(float(val_metrics["ece"]), 4),
        })

        if val_metrics["auc"] > best_auc:
            best_auc, best_epoch, no_improve = val_metrics["auc"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        print(f"    fold {fold_idx} epoch {epoch:3d}  "
              f"train_loss {running_loss/n_seen:.4f}  "
              f"val_auc {val_metrics['auc']:.3f}  "
              f"val_bacc {val_metrics['balanced_accuracy']:.3f}  "
              f"val_ece {val_metrics['ece']:.3f}")

        if no_improve >= args.patience:
            print(f"    fold {fold_idx}: early stop at epoch {epoch} "
                  f"(best {best_auc:.3f} @ {best_epoch})")
            break

    model.load_state_dict(best_state)
    test_metrics, test_raw = evaluate(model, test_loader, device, criterion,
                                      imaging_only, clinical_only)

    print(f"    fold {fold_idx} TEST: auc {test_metrics['auc']:.3f}  "
          f"bacc {test_metrics['balanced_accuracy']:.3f}  "
          f"ece {test_metrics['ece']:.3f}  brier {test_metrics['brier']:.3f}")

    return {
        "fold": fold_idx,
        "best_val_auc": float(best_auc),
        "best_epoch": int(best_epoch),
        "test": {k: float(v) for k, v in test_metrics.items()},
        "history": history,
        "predictions": {
            "subject": list(test_raw["subjects"]),
            "label": [int(x) for x in test_raw["labels"]],
            "prob_ad": [float(x) for x in test_raw["probs"]],
        },
    }


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def summarise(results, tag):
    """Per-fold mean +/- std, plus pooled out-of-fold metrics over all subjects."""
    keys = ["auc", "balanced_accuracy", "accuracy", "ece", "brier",
            "sensitivity", "specificity"]

    print(f"\n{'='*68}")
    print(f"5-FOLD CROSS-VALIDATION SUMMARY  -  {tag}")
    print(f"{'='*68}")

    print(f"\n  Per-fold test AUC: " +
          "  ".join(f"{r['test']['auc']:.3f}" for r in results))

    print(f"\n  {'metric':<20} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    print(f"  {'-'*54}")
    summary = {}
    for k in keys:
        vals = np.array([r["test"][k] for r in results])
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
                      "min": float(vals.min()), "max": float(vals.max())}
        print(f"  {k:<20} {vals.mean():8.4f} {vals.std(ddof=1):8.4f} "
              f"{vals.min():8.4f} {vals.max():8.4f}")

    # pooled out-of-fold: every subject predicted exactly once
    all_probs = np.concatenate([r["predictions"]["prob_ad"] for r in results])
    all_labels = np.concatenate([r["predictions"]["label"] for r in results])
    pooled_auc = roc_auc_score(all_labels, all_probs)
    pooled_ece = expected_calibration_error(all_probs, all_labels)

    print(f"\n  Pooled out-of-fold (n={len(all_labels)}, every subject once):")
    print(f"    AUC {pooled_auc:.4f}    ECE {pooled_ece:.4f}")

    # bootstrap CI on the pooled estimate
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(all_labels), len(all_labels))
        if len(set(all_labels[idx])) > 1:
            boot.append(roc_auc_score(all_labels[idx], all_probs[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"    95% CI [{lo:.4f}, {hi:.4f}]  (bootstrap, 2000 resamples)")

    print("\n" + classification_report(
        all_labels, (all_probs >= 0.5).astype(int),
        target_names=["CN", "AD"], digits=4))

    summary["pooled"] = {
        "auc": float(pooled_auc), "ece": float(pooled_ece),
        "ci_low": float(lo), "ci_high": float(hi), "n": int(len(all_labels)),
    }
    return summary, all_probs, all_labels


# --------------------------------------------------------------------------
# comparison across modes
# --------------------------------------------------------------------------
def compare(tags):
    """Paired comparison of saved CV runs, with a DeLong-style bootstrap test.

    Uses the SAME bootstrap resample indices for every mode, so the difference
    is evaluated on identical subjects each time - a paired test, which is far
    more sensitive than comparing two independent confidence intervals.
    """
    loaded = {}
    for tag in tags:
        path = os.path.join(CV_DIR, f"{tag}_folds.json")
        if not os.path.exists(path):
            print(f"  ! missing {path} - skipping {tag}")
            continue
        with open(path) as fh:
            res = json.load(fh)
        probs = np.concatenate([r["predictions"]["prob_ad"] for r in res["folds"]])
        labels = np.concatenate([r["predictions"]["label"] for r in res["folds"]])
        subs = np.concatenate([r["predictions"]["subject"] for r in res["folds"]])
        order = np.argsort(subs)          # align modes by subject
        loaded[tag] = (probs[order], labels[order], subs[order])

    if len(loaded) < 2:
        print("  need at least two completed CV runs to compare")
        return

    tags = list(loaded)
    # sanity: all modes must cover the same subjects in the same order
    ref_subs = loaded[tags[0]][2]
    for t in tags[1:]:
        if not np.array_equal(loaded[t][2], ref_subs):
            print(f"  ! {t} covers different subjects than {tags[0]} - cannot pair")
            return

    print(f"\n{'='*68}")
    print("PAIRED COMPARISON  (pooled out-of-fold, identical subjects)")
    print(f"{'='*68}\n")

    labels = loaded[tags[0]][1]
    print(f"  {'mode':<20} {'AUC':>8}")
    print(f"  {'-'*30}")
    for t in tags:
        print(f"  {t:<20} {roc_auc_score(labels, loaded[t][0]):8.4f}")

    rng = np.random.default_rng(0)
    n = len(labels)
    idxs = [rng.integers(0, n, n) for _ in range(2000)]

    print(f"\n  Pairwise differences (positive = first mode better):\n")
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            a, b = tags[i], tags[j]
            diffs = []
            for idx in idxs:
                if len(set(labels[idx])) < 2:
                    continue
                diffs.append(roc_auc_score(labels[idx], loaded[a][0][idx]) -
                             roc_auc_score(labels[idx], loaded[b][0][idx]))
            diffs = np.array(diffs)
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            obs = (roc_auc_score(labels, loaded[a][0]) -
                   roc_auc_score(labels, loaded[b][0]))
            # two-sided bootstrap p-value
            p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
            verdict = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not distinguishable"
            print(f"    {a} vs {b}")
            print(f"      diff {obs:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
                  f"p={p:.3f}   -> {verdict}")

    print(f"\n  Note: with n={n}, differences below roughly 0.05 AUC are not")
    print("  resolvable. 'Not distinguishable' is a real finding, not a failure -")
    print("  it means the modalities carry largely redundant information here.")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["imaging_only", "clinical_only", "fusion"],
                    default="fusion")
    ap.add_argument("--tag", type=str, default=None,
                    help="output name; defaults to cv_<mode>")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--target_size", type=int, default=128)
    ap.add_argument("--base_ch", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                    help="skip folds already saved to disk")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="compare finished CV runs by tag, then exit")
    args = ap.parse_args()

    if args.compare:
        compare(args.compare)
        return

    if args.tag is None:
        args.tag = f"cv_{args.mode}"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"5-fold CV  |  mode={args.mode}  |  tag={args.tag}\n")

    folds = build_cv_folds(COHORT, n_folds=args.n_folds, seed=args.seed)
    for i, (tr, va, te) in enumerate(folds, 1):
        print(f"  fold {i}: train {len(tr):3d}  val {len(va):3d}  test {len(te):3d}"
              f"  (test {te['label'].sum()} AD / {(te['label']==0).sum()} CN)")
    print()

    out_path = os.path.join(CV_DIR, f"{args.tag}_folds.json")
    results = []
    if args.resume and os.path.exists(out_path):
        with open(out_path) as fh:
            results = json.load(fh)["folds"]
        print(f"Resuming: {len(results)} fold(s) already complete\n")

    t0 = time.time()
    for i, (tr, va, te) in enumerate(folds, 1):
        if any(r["fold"] == i for r in results):
            print(f"  fold {i}: already done, skipping")
            continue
        print(f"\n  --- fold {i}/{len(folds)} ---")
        set_seed(args.seed + i)          # different init per fold, reproducible
        res = run_fold(i, tr, va, te, args, device)
        results.append(res)
        results.sort(key=lambda r: r["fold"])
        with open(out_path, "w") as fh:
            json.dump({"tag": args.tag, "mode": args.mode,
                       "args": vars(args), "folds": results}, fh, indent=2)
        print(f"    fold {i} saved ({(time.time()-t0)/60:.1f} min elapsed)")

    summary, probs, labels = summarise(results, args.tag)

    with open(os.path.join(CV_DIR, f"{args.tag}_summary.json"), "w") as fh:
        json.dump({"tag": args.tag, "mode": args.mode,
                   "summary": summary, "args": vars(args)}, fh, indent=2)
    pd.DataFrame({"label": labels, "prob_ad": probs}).to_csv(
        os.path.join(CV_DIR, f"{args.tag}_oof_predictions.csv"), index=False)

    print(f"\nTotal time {(time.time()-t0)/60:.1f} min")
    print(f"Saved to {CV_DIR}/{args.tag}_*.json")
    print(f"\nWhen all three modes are done:")
    print(f"  python cross_validate.py --compare cv_imaging_only cv_clinical_only cv_fusion")


if __name__ == "__main__":
    main()
