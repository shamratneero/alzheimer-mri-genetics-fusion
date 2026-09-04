"""
Architecture generalisation test: does the checkpoint-selection gap reproduce on
a standard ImageNet-pretrained ResNet-18?

The question
------------
Phases 2-6 establish that AUC-based checkpoint selection conceals cross-model
probability incoherence, and that calibration-aware selection largely repairs
it. All of that evidence comes from ONE custom 3D CNN. A reviewer will ask
whether the phenomenon is a quirk of that architecture.

This runs the identical 5-fold protocol on a completely different backbone -
ImageNet-pretrained ResNet-18, imaging only - and reports whether the per-fold
vs pooled out-of-fold gap appears there too.

Both outcomes are reportable and neither is preferred:
  - gap reproduces      -> the effect is not architecture-specific
  - gap does not appear -> the effect is architecture-dependent, also useful

This is a CONTROLLED GENERALISATION TEST, not a model-development exercise.
No architecture-specific hyperparameter tuning is performed. If the ResNet
scores poorly that is a result, not a problem to optimise away.

Locked design decisions (fixed before running; see project correspondence)
-------------------------------------------------------------------------
1. **2.5D input.** Three adjacent axial slices become the three input channels,
   using the pretrained 3-channel stem as designed. Avoids inventing a
   slice-aggregation rule, which would be a new design choice that could itself
   affect calibration and confound the comparison.

2. **Fixed-fraction slice selection at 0.55**, NOT maximum brain area.
   preprocess.process_one reorients every volume to canonical RAS and crops to
   the brain bounding box before resizing, so a fixed index is the same
   anatomical level in every subject. Maximum-brain-area would be data-
   dependent and is label-correlated: AD subjects have enlarged ventricles, so
   the rule could systematically select different anatomy for AD than for CN -
   a selection artefact hidden inside the slice rule.
   SLICE_FRACTION is a module constant, deliberately NOT a CLI argument, so it
   cannot be quietly adjusted after seeing results.

3. **Same normalisation as the primary model.** Volumes keep their existing
   per-subject z-scoring; ImageNet mean/std is NOT applied afterwards. This is
   not a claim that ImageNet normalisation is wrong - it is a different valid
   choice - but applying it would mean the ResNet sees different inputs than
   the 3D CNN did, which would confound architecture with preprocessing. The
   point of the experiment is to vary exactly one thing.

4. **Fine-tune all layers**, matching common practice, rather than freezing the
   backbone as a feature extractor.

5. **Identical protocol otherwise**: same deterministic folds (build_cv_folds,
   seed 42), same class-weighted CrossEntropy, AdamW, ReduceLROnPlateau, early
   stopping on validation AUC, and the same four selection criteria evaluated
   simultaneously.

6. **Epoch budget is matched but early stopping runs normally.** A pretrained
   11M-parameter model converges faster than a 3.57M scratch model and may
   overfit 255 subjects quickly. Forcing symmetry would be false precision; the
   selected-epoch distribution is logged and is itself part of the diagnostic.

Output
  outputs/resnet2d/resnet2d_folds.json     per-fold, per-criterion, with the
                                           exact subject IDs of every split
  outputs/resnet2d/resnet2d_preds_{criterion}.csv   pooled OOF predictions
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss

from cross_validate import build_cv_folds, SELECTION_CRITERIA, PRIMARY_CRITERION
# ECE is imported, never reimplemented. train.py defines it as the gap between
# mean predicted probability and the POSITIVE RATE within each bin. An
# independent implementation using classification accuracy instead would be a
# different quantity, and every ECE reported in Phases 1-6 uses this one - a
# silent mismatch would make the ResNet numbers incomparable to the rest.
from train import expected_calibration_error

COHORT = "oasis3_cohort.csv"
PRE_DIR = r"D:\alhseimer\preprocessed"
OUT_DIR = "outputs/resnet2d"
CKPT_DIR = "outputs/resnet2d_checkpoints"   # default; overridden per run via --ckpt_dir

# --- LOCKED CONSTANTS: deliberately not CLI arguments ----------------------
SLICE_FRACTION = 0.55      # axial position through the cropped, RAS-oriented volume
SLICE_OFFSETS = (-1, 0, 1)  # three adjacent slices -> three input channels
TARGET_SIZE = 128           # matches the 3D CNN's input resolution
FOLD_SEED = 42              # fold construction ONLY - see note below
N_FOLDS = 5
VAL_FRAC = 0.125
# FOLD_SEED is locked because the entire point of this experiment is that the
# ResNet sees the SAME subject splits as the primary 3D CNN. Exposing it as a
# CLI argument would let `--seed 123` silently build a different cross-
# validation experiment while still being called an architecture-generalisation
# test, and the resulting numbers would not be comparable to Phases 2-6.
# Training-time randomness (fc head init, batch shuffling) is a separate knob
# and IS exposed as --train_seed, since varying it does not change which
# subjects are compared.
# --------------------------------------------------------------------------


def metrics_from(labels, probs, loss=None):
    labels, probs = np.asarray(labels), np.asarray(probs)
    preds = (probs >= 0.5).astype(int)
    out = {
        "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "auc": float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan"),
        "brier": float(brier_score_loss(labels, probs)),
        # note argument order: train.py takes (probs, labels)
        "ece": float(expected_calibration_error(probs, labels)),
        "sensitivity": float(((preds == 1) & (labels == 1)).sum() / max((labels == 1).sum(), 1)),
        "specificity": float(((preds == 0) & (labels == 0)).sum() / max((labels == 0).sum(), 1)),
    }
    if loss is not None:
        out["loss"] = float(loss)
    return out


class Slices25D(Dataset):
    """Three adjacent axial slices from a cached 3D volume, as RGB channels.

    The volume is resized 160 -> TARGET_SIZE exactly as dataset_3d does, THEN
    sliced, so the slice index refers to the same resolution the 3D CNN saw.
    Slicing before resizing would place the slice at a different anatomical
    level and quietly break the comparison.
    """

    def __init__(self, df, pre_dir=PRE_DIR, target_size=TARGET_SIZE):
        self.df = df.reset_index(drop=True)
        self.pre_dir = pre_dir
        self.target_size = target_size
        self.k = int(round(SLICE_FRACTION * target_size))
        idxs = [self.k + o for o in SLICE_OFFSETS]
        if min(idxs) < 0 or max(idxs) >= target_size:
            raise ValueError(f"slice indices {idxs} out of range for "
                             f"target_size {target_size}")
        self.slice_idxs = idxs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = os.path.join(self.pre_dir, f"{row['subject']}.npy")
        vol = np.load(path)
        if vol.ndim != 3 or not np.isfinite(vol).all():
            raise ValueError(f"{row['subject']}: bad volume {vol.shape}")

        t = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
        if t.shape[-1] != self.target_size:
            t = F.interpolate(t, size=(self.target_size,) * 3,
                              mode="trilinear", align_corners=False)
        t = t[0, 0]

        # axis 2 is superior-inferior after as_closest_canonical (RAS)
        chans = torch.stack([t[:, :, j] for j in self.slice_idxs], dim=0)
        return {"image": chans,
                "label": torch.tensor(int(row["label"]), dtype=torch.long),
                "subject": row["subject"]}


def build_resnet18(pretrained=True, dropout=0.3):
    """ImageNet-pretrained ResNet-18 with a 2-class head.

    The 3-channel stem is kept UNCHANGED so the pretrained first-layer filters
    are used as trained. This is the reason for the 2.5D representation: it
    matches the pretrained input shape without modifying or re-initialising
    conv1, which would discard part of what makes the model "pretrained".
    """
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feats, 2))
    return model


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    probs, labels, subjects, total, n = [], [], [], 0.0, 0
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        out = model(x)
        if out.shape[1] != 2 or not torch.isfinite(out).all():
            raise ValueError(f"bad model output {tuple(out.shape)}")
        total += criterion(out, y).item() * y.size(0)
        n += y.size(0)
        probs += torch.softmax(out, 1)[:, 1].cpu().tolist()
        labels += y.cpu().tolist()
        subjects += list(batch["subject"])
    return (metrics_from(labels, probs, total / max(n, 1)),
            {"subject": subjects, "label": labels, "prob_ad": probs})


def run_fold(fold_idx, train_df, val_df, test_df, args, device):
    # --- leakage guard: the three splits must be disjoint by subject --------
    s_tr, s_va, s_te = (set(train_df["subject"]), set(val_df["subject"]),
                        set(test_df["subject"]))
    for a, b, name in [(s_tr, s_va, "train/val"), (s_tr, s_te, "train/test"),
                       (s_va, s_te, "val/test")]:
        if a & b:
            raise ValueError(f"fold {fold_idx}: {len(a & b)} subjects appear in "
                             f"both {name} - aborting, this would invalidate "
                             f"the entire fold")

    train_ds = Slices25D(train_df)
    val_ds = Slices25D(val_df)
    test_ds = Slices25D(test_df)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0)

    model = build_resnet18(pretrained=not args.no_pretrained,
                           dropout=args.dropout).to(device)

    # identical class weighting to the primary model
    n_cn = (train_df["label"] == 0).sum()
    n_ad = (train_df["label"] == 1).sum()
    weights = torch.tensor([1.0, n_cn / n_ad], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)

    best_auc, best_epoch, no_improve = -1.0, -1, 0
    best = {name: {"score": -float("inf"), "epoch": -1, "state": None, "val": {}}
            for name in SELECTION_CRITERIA}
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)

        val_metrics, _ = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_metrics["auc"])
        history.append({
            "epoch": epoch,
            "train_loss": round(running / max(seen, 1), 4),
            "val_auc": round(val_metrics["auc"], 4),
            "val_bal_acc": round(val_metrics["balanced_accuracy"], 4),
            "val_ece": round(val_metrics["ece"], 4),
            "val_brier": round(val_metrics["brier"], 4),
        })

        # one snapshot serves every criterion that improved this epoch
        snapshot = None
        for name, score_fn in SELECTION_CRITERIA.items():
            score = score_fn(val_metrics)
            if score > best[name]["score"]:
                if snapshot is None:
                    snapshot = {k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()}
                best[name].update(score=score, epoch=epoch, state=snapshot,
                                  val=dict(val_metrics))

        # early stopping keys off AUC, exactly as the primary model does, so the
        # training trajectory is identical across criteria
        if val_metrics["auc"] > best_auc:
            best_auc, best_epoch, no_improve = val_metrics["auc"], epoch, 0
        else:
            no_improve += 1

        print(f"    fold {fold_idx} ep {epoch:3d}  loss {running/max(seen,1):.4f}  "
              f"val_auc {val_metrics['auc']:.3f}  "
              f"val_bacc {val_metrics['balanced_accuracy']:.3f}  "
              f"val_ece {val_metrics['ece']:.3f}")

        if no_improve >= args.patience:
            print(f"    fold {fold_idx}: early stop at epoch {epoch} "
                  f"(best val AUC {best_auc:.3f} @ ep {best_epoch})")
            break

    if not args.no_checkpoints:
        os.makedirs(args.ckpt_dir, exist_ok=True)
    by_criterion = {}
    print(f"    fold {fold_idx} TEST by criterion:")
    for name in SELECTION_CRITERIA:
        fell_back = False
        if best[name]["state"] is None:
            best[name] = dict(best[PRIMARY_CRITERION])
            fell_back = True

        model.load_state_dict(best[name]["state"])
        tm, traw = evaluate(model, test_loader, device, criterion)

        if args.no_checkpoints:
            ckpt_path = None
        else:
            ckpt_path = os.path.join(args.ckpt_dir,
                                     f"resnet2d_fold{fold_idx}_{name}.pt")
            torch.save({
                "model": best[name]["state"],
                "arch": "resnet18_2p5d",
                "pretrained": not args.no_pretrained,
                "fold": fold_idx,
                "criterion": name,
                "epoch": best[name]["epoch"],
                "fell_back": fell_back,
                "slice_fraction": SLICE_FRACTION,
                "slice_idxs": train_ds.slice_idxs,
                "target_size": TARGET_SIZE,
                "dropout": args.dropout,
                "val_at_selection": best[name]["val"],
                "test": tm,
            }, ckpt_path)

        by_criterion[name] = {
            "epoch": best[name]["epoch"],
            "fell_back": fell_back,
            "checkpoint": ckpt_path,
            "val_at_selection": best[name]["val"],
            "test": tm,
            "predictions": traw,
        }
        print(f"      {name:14s} ep{best[name]['epoch']:3d}  "
              f"auc {tm['auc']:.4f}  bacc {tm['balanced_accuracy']:.4f}  "
              f"ece {tm['ece']:.4f}  brier {tm['brier']:.4f}"
              f"{'  (fell back)' if fell_back else ''}")

    return {
        "fold": fold_idx,
        "best_val_auc": best_auc,
        "best_epoch": best_epoch,
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        # exact split membership, so the run is auditable like the primary model
        "subjects": {"train": sorted(s_tr), "val": sorted(s_va),
                     "test": sorted(s_te)},
        "slice_idxs": train_ds.slice_idxs,
        "history": history,
        "by_criterion": by_criterion,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--train_seed", type=int, default=42,
                    help="training randomness only (head init, shuffling). "
                         "Fold construction uses the locked FOLD_SEED and is "
                         "NOT affected by this.")
    ap.add_argument("--no_pretrained", action="store_true",
                    help="ablation: random init instead of ImageNet weights")
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--ckpt_dir", default=None,
                    help="where to write checkpoints. Defaults to "
                         "<out_dir>_checkpoints so repeated runs with "
                         "different --out_dir never overwrite each other.")
    ap.add_argument("--no_checkpoints", action="store_true",
                    help="skip writing .pt files entirely (~900MB per run). "
                         "Use for seed-repeat runs where only the summary "
                         "JSON is needed.")
    # NOTE: SLICE_FRACTION and FOLD_SEED are deliberately NOT exposed here.
    args = ap.parse_args()
    if args.ckpt_dir is None:
        args.ckpt_dir = args.out_dir.rstrip("/\\") + "_checkpoints"

    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k = int(round(SLICE_FRACTION * TARGET_SIZE))
    print(f"Device: {device}")
    print(f"ResNet-18 {'ImageNet-pretrained' if not args.no_pretrained else 'RANDOM INIT'}, "
          f"fine-tuning all layers")
    print(f"2.5D axial slices {[k + o for o in SLICE_OFFSETS]} "
          f"(SLICE_FRACTION={SLICE_FRACTION} of {TARGET_SIZE})")
    print(f"Folds: locked FOLD_SEED={FOLD_SEED} (train_seed={args.train_seed} "
          f"affects training randomness only)\n")

    folds = build_cv_folds(COHORT, n_folds=N_FOLDS, val_frac=VAL_FRAC,
                           seed=FOLD_SEED)

    # Hard guard: the reconstructed folds must match the splits the primary 3D
    # CNN was evaluated on. Locking FOLD_SEED is necessary but not sufficient -
    # a change to build_cv_folds, the cohort CSV, or sklearn's shuffling could
    # silently alter the splits while the seed stays 42. This checks the actual
    # subject IDs against the committed CV results rather than trusting that
    # nothing upstream moved.
    ref_path = "outputs/cv/sel_fusion_folds.json"
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path, encoding="utf-8"))
        for i, (_, _, test_df) in enumerate(folds, start=1):
            stored = set(ref["folds"][i - 1]["by_criterion"]["auc"]
                         ["predictions"]["subject"])
            if set(test_df["subject"]) != stored:
                raise ValueError(
                    f"fold {i} test split does NOT match the primary model's "
                    f"split in {ref_path}. This experiment would not be "
                    f"comparable to Phases 2-6. Aborting rather than producing "
                    f"numbers that look comparable but are not.")
        print(f"Fold splits verified identical to the primary model "
              f"({ref_path})\n")
    else:
        print(f"WARNING: {ref_path} not found - cannot verify that fold splits "
              f"match the primary model. Results may not be comparable.\n")
    results = []
    for i, (train_df, val_df, test_df) in enumerate(folds, start=1):
        print(f"=== fold {i}/5  train {len(train_df)} / val {len(val_df)} / "
              f"test {len(test_df)} ===")
        results.append(run_fold(i, train_df, val_df, test_df, args, device))

    # ---- pooled out-of-fold analysis, same as the primary model ------------
    summary = {}
    for name in SELECTION_CRITERIA:
        per_fold_auc = [r["by_criterion"][name]["test"]["auc"] for r in results]
        p, y, subj = [], [], []
        for r in results:
            pr = r["by_criterion"][name]["predictions"]
            p += pr["prob_ad"]; y += pr["label"]; subj += pr["subject"]
        if len(set(subj)) != len(subj):
            raise ValueError(f"{name}: a subject appears in more than one test "
                             f"fold - pooled OOF would be invalid")
        pooled = metrics_from(y, p)
        summary[name] = {
            "per_fold_mean_auc": float(np.mean(per_fold_auc)),
            "per_fold_sd_auc": float(np.std(per_fold_auc)),
            "pooled_oof_auc": pooled["auc"],
            "gap": float(np.mean(per_fold_auc)) - pooled["auc"],
            "pooled": pooled,
            "per_fold_mean_bacc": float(np.mean(
                [r["by_criterion"][name]["test"]["balanced_accuracy"] for r in results])),
            "n_degenerate_folds": int(sum(
                r["by_criterion"][name]["test"]["balanced_accuracy"] < 0.55
                for r in results)),
            "selected_epochs": [r["by_criterion"][name]["epoch"] for r in results],
        }
        pd.DataFrame({"subject": subj, "label": y, "prob_ad": p}).to_csv(
            os.path.join(args.out_dir, f"resnet2d_preds_{name}.csv"), index=False)

    out = {
        "arch": "resnet18_2p5d",
        "pretrained": not args.no_pretrained,
        "slice_fraction": SLICE_FRACTION,
        "slice_offsets": list(SLICE_OFFSETS),
        "fold_seed": FOLD_SEED,
        "train_seed": args.train_seed,
        "ckpt_dir": None if args.no_checkpoints else args.ckpt_dir,
        "n_folds": N_FOLDS,
        "val_frac": VAL_FRAC,
        "folds_verified_against": "outputs/cv/sel_fusion_folds.json",
        "args": vars(args),
        "folds": results,
        "summary": summary,
    }
    path = os.path.join(args.out_dir, "resnet2d_folds.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n" + "=" * 74)
    print("DOES THE PER-FOLD / POOLED GAP REPRODUCE ON RESNET-18?")
    print("=" * 74)
    for name, s in summary.items():
        print(f"  {name:14s} per-fold {s['per_fold_mean_auc']:.4f} "
              f"(sd {s['per_fold_sd_auc']:.4f})  pooled {s['pooled_oof_auc']:.4f}  "
              f"gap {s['gap']:.4f}  bacc {s['per_fold_mean_bacc']:.4f}  "
              f"degenerate {s['n_degenerate_folds']}/5")
        print(f"                 selected epochs {s['selected_epochs']}")
    a, n = summary.get("auc"), summary.get("neg_brier")
    if a and n:
        print(f"\n  auc gap {a['gap']:.4f}  ->  neg_brier gap {n['gap']:.4f}"
              f"   (3D CNN for reference: 0.1200 -> 0.0227 imaging-only)")
    print("\nEarly, clustered selected epochs would indicate fast overfitting of")
    print("a pretrained model on 255 subjects - read the gap with that in mind.")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()