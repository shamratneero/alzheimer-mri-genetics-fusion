"""
Training loop for the OASIS-3 Alzheimer's fusion model.

Evaluates every epoch on the validation set and tracks the best model by
validation AUC (not accuracy - AUC is threshold-independent and more stable
on small validation sets).

Records a full per-epoch history so training curves and overfitting can be
inspected afterwards. Saves checkpoints so power interruptions do not lose
progress.
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
from sklearn.metrics import (roc_auc_score, accuracy_score, balanced_accuracy_score,
                             confusion_matrix, classification_report, brier_score_loss)

from dataset_3d import OASIS3Dataset, build_splits
from model_3d import FusionModel
from tqdm import tqdm


def expected_calibration_error(probs, labels, n_bins=10):
    """ECE: weighted average gap between confidence and accuracy across
    equal-width probability bins. 0 = perfectly calibrated."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return ece

COHORT   = "oasis3_cohort.csv"
PRE_DIR  = r"D:\alhseimer\preprocessed"
CKPT_DIR = "outputs/checkpoints_3d"
LOG_DIR  = "outputs/logs"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def forward_pass(model, vol, clin, imaging_only=False, clinical_only=False):
    """Single place that decides which branch(es) of the model see data,
    so imaging_only / clinical_only / fusion all route through it identically."""
    if imaging_only:
        return model.forward_imaging_only(vol)
    if clinical_only:
        return model.forward_clinical_only(clin)
    return model(vol, clin)


@torch.no_grad()
def evaluate(model, loader, device, criterion, imaging_only=False, clinical_only=False):
    """Run the model over a loader and return metrics + raw predictions."""
    model.eval()
    losses, probs, labels, subjects = [], [], [], []

    for batch in tqdm(loader, desc="  eval", leave=False):
        vol = batch["volume"].to(device, non_blocking=True)
        clin = batch["clinical"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        out = forward_pass(model, vol, clin, imaging_only, clinical_only)
        loss = criterion(out, y)

        losses.append(loss.item() * y.size(0))
        probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
        labels.extend(y.cpu().numpy())
        subjects.extend(batch["subject"])

    probs, labels = np.array(probs), np.array(labels)
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "loss": sum(losses) / len(labels),
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "auc": roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan"),
        "brier": brier_score_loss(labels, probs),
        "ece": expected_calibration_error(probs, labels, n_bins=10),
    }
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) else 0.0   # recall on AD
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) else 0.0   # recall on CN

    return metrics, {"probs": probs, "labels": labels, "subjects": subjects}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--target_size", type=int, default=128)
    ap.add_argument("--base_ch", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=15, help="early stopping patience")
    ap.add_argument("--imaging_only", action="store_true",
                    help="ablation: ignore clinical features")
    ap.add_argument("--clinical_only", action="store_true",
                    help="ablation: ignore imaging, APOE+age only - trained/evaluated "
                         "through the same pipeline as imaging_only and fusion for a "
                         "true apples-to-apples comparison")
    ap.add_argument("--tag", type=str, default="fusion")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <tag>_latest.pt if it exists (power-cut recovery)")
    args = ap.parse_args()
    if args.imaging_only and args.clinical_only:
        raise ValueError("--imaging_only and --clinical_only are mutually exclusive")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "imaging_only" if args.imaging_only else ("clinical_only" if args.clinical_only else "fusion")
    print(f"Device: {device}")
    print(f"Run tag: {args.tag}  |  mode={mode}\n")

    # ---------------- data ----------------
    train_df, val_df, test_df = build_splits(COHORT, seed=args.seed)
    print(f"Train {len(train_df):3d}  ({train_df['label'].sum()} AD / {(train_df['label']==0).sum()} CN)")
    print(f"Val   {len(val_df):3d}  ({val_df['label'].sum()} AD / {(val_df['label']==0).sum()} CN)")
    print(f"Test  {len(test_df):3d}  ({test_df['label'].sum()} AD / {(test_df['label']==0).sum()} CN)\n")

    train_ds = OASIS3Dataset(train_df, PRE_DIR, target_size=args.target_size, augment=True)
    val_ds   = OASIS3Dataset(val_df,   PRE_DIR, target_size=args.target_size, augment=False)
    test_ds  = OASIS3Dataset(test_df,  PRE_DIR, target_size=args.target_size, augment=False)

    # normalise val/test ages with TRAIN statistics (no leakage)
    val_ds.set_age_norm(train_ds.age_mean, train_ds.age_std)
    test_ds.set_age_norm(train_ds.age_mean, train_ds.age_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---------------- model ----------------
    n_clin = train_ds[0]["clinical"].shape[0]
    model = FusionModel(n_clinical_features=n_clin, base_ch=args.base_ch,
                        dropout=args.dropout).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # class weighting from the training split
    n_cn, n_ad = (train_df["label"] == 0).sum(), (train_df["label"] == 1).sum()
    weights = torch.tensor([1.0, n_cn / n_ad], dtype=torch.float32, device=device)
    print(f"Class weights (CN, AD): {weights.tolist()}\n")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)

    # ---------------- training ----------------
    history = []
    best_auc, best_epoch, epochs_no_improve = -1.0, -1, 0
    start_epoch = 1

    latest_path = os.path.join(CKPT_DIR, f"{args.tag}_latest.pt")
    if args.resume and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        history = ckpt["history"]
        best_auc = ckpt["best_auc"]
        best_epoch = ckpt["best_epoch"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}  "
              f"(best val AUC so far {best_auc:.3f} @ epoch {best_epoch})\n")
    elif args.resume:
        print(f"--resume passed but no checkpoint found at {latest_path}; starting fresh.\n")

    t_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss, n_seen = 0.0, 0
        train_probs, train_labels = [], []

        loop = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs}", leave=False)
        for i, batch in enumerate(loop, 1):
            vol = batch["volume"].to(device, non_blocking=True)
            clin = batch["clinical"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            out = forward_pass(model, vol, clin, args.imaging_only, args.clinical_only)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            n_seen += y.size(0)
            train_probs.extend(torch.softmax(out.detach(), dim=1)[:, 1].cpu().numpy())
            train_labels.extend(y.cpu().numpy())

            loop.set_postfix(loss=f"{running_loss/n_seen:.4f}")

        train_auc = roc_auc_score(train_labels, train_probs) if len(set(train_labels)) > 1 else float("nan")
        train_loss = running_loss / n_seen

        val_metrics, _ = evaluate(model, val_loader, device, criterion, args.imaging_only, args.clinical_only)
        scheduler.step(val_metrics["auc"])

        rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_auc": round(float(train_auc), 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_auc": round(val_metrics["auc"], 4),
            "val_acc": round(val_metrics["accuracy"], 4),
            "val_bal_acc": round(val_metrics["balanced_accuracy"], 4),
            "val_brier": round(val_metrics["brier"], 4),
            "val_ece": round(val_metrics["ece"], 4),
            "val_sens": round(val_metrics["sensitivity"], 4),
            "val_spec": round(val_metrics["specificity"], 4),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(rec)

        gap = train_auc - val_metrics["auc"]
        print(f"  epoch {epoch:3d}  train_loss {train_loss:.4f}  train_auc {train_auc:.3f}  |  "
              f"val_loss {val_metrics['loss']:.4f}  val_auc {val_metrics['auc']:.3f}  "
              f"val_bacc {val_metrics['balanced_accuracy']:.3f}  val_ece {val_metrics['ece']:.3f}  "
              f"(gap {gap:+.3f})")

        if val_metrics["auc"] > best_auc:
            best_auc, best_epoch, epochs_no_improve = val_metrics["auc"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_auc": best_auc, "args": vars(args)},
                       os.path.join(CKPT_DIR, f"{args.tag}_best.pt"))
            print(f"           -> new best val AUC {best_auc:.3f}, checkpoint saved")
        else:
            epochs_no_improve += 1

        # always save a full-state "latest" checkpoint so an interrupted run
        # (e.g. power cut) can resume from here rather than restarting
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "history": history, "best_auc": best_auc, "best_epoch": best_epoch,
                    "epochs_no_improve": epochs_no_improve, "args": vars(args)},
                   os.path.join(CKPT_DIR, f"{args.tag}_latest.pt"))

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping: no val AUC improvement for {args.patience} epochs.")
            break

    mins = (time.time() - t_start) / 60
    print(f"\nTraining finished in {mins:.1f} min.  Best val AUC {best_auc:.3f} (epoch {best_epoch})")

    # ---------------- final test evaluation ----------------
    ckpt = torch.load(os.path.join(CKPT_DIR, f"{args.tag}_best.pt"))
    model.load_state_dict(ckpt["model"])
    test_metrics, test_raw = evaluate(model, test_loader, device, criterion, args.imaging_only, args.clinical_only)

    print(f"\n{'='*60}")
    print(f"TEST SET RESULTS  (best checkpoint, epoch {ckpt['epoch']})")
    print(f"{'='*60}")
    for k, v in test_metrics.items():
        print(f"  {k:<20} {v:.4f}")

    print(f"\n  Calibration  -  Brier: {test_metrics['brier']:.4f}   ECE: {test_metrics['ece']:.4f}"
          f"   (lower is better for both; ECE=0 is perfect calibration)")
    print(f"\n  EDA logistic-regression demographics baseline was 0.826 (different pipeline/split -")
    print(f"  for a true apples-to-apples number, compare against a --clinical_only run instead)")
    delta = test_metrics["auc"] - 0.826
    print(f"  This model's test AUC        : {test_metrics['auc']:.3f}  ({delta:+.3f})")
    if delta > 0:
        print(f"  -> imaging adds information beyond demographics + genetics")
    else:
        print(f"  -> imaging does NOT beat the non-imaging baseline")

    print("\n" + classification_report(test_raw["labels"],
                                       (test_raw["probs"] >= 0.5).astype(int),
                                       target_names=["CN", "AD"], digits=4))

    # ---------------- save artefacts ----------------
    pd.DataFrame(history).to_csv(os.path.join(LOG_DIR, f"{args.tag}_history.csv"), index=False)
    pd.DataFrame({"subject": test_raw["subjects"],
                  "label": test_raw["labels"],
                  "prob_ad": test_raw["probs"]}).to_csv(
        os.path.join(LOG_DIR, f"{args.tag}_test_predictions.csv"), index=False)
    with open(os.path.join(LOG_DIR, f"{args.tag}_test_metrics.json"), "w") as fh:
        json.dump({"test": test_metrics, "best_val_auc": best_auc,
                   "best_epoch": best_epoch, "args": vars(args)}, fh, indent=2)

    print(f"\nSaved history, predictions and metrics to {LOG_DIR}/")


if __name__ == "__main__":
    main()