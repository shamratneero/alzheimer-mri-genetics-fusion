"""
Temperature scaling vs calibration-aware checkpoint selection.

The question this answers
------------------------
A reviewer will ask: "temperature scaling is the standard post-hoc fix for
miscalibration - one parameter, no retraining. Why change checkpoint selection
instead?"

The hypothesis, stated before running anything:

  Temperature scaling divides every logit by the same positive scalar. That is
  a strictly monotonic transform of the logit, so it CANNOT change the ranking
  of predictions and therefore cannot change AUC at all. It also cannot rescue
  a checkpoint that has collapsed to predicting one class for every subject -
  it will make that model predict the same class less confidently, improving
  ECE and Brier while balanced accuracy stays at 0.500.

  If that holds, temperature scaling makes a degenerate model's calibration
  metrics look better while leaving it just as useless - i.e. it hides the
  failure that calibration-aware selection avoids.

The falsifier: if temperature scaling closes the per-fold/pooled gap as well as
neg_brier selection does, and repairs balanced accuracy on the degenerate folds,
then the contribution narrows to "selection matters only where it goes
degenerate". That would be a smaller claim, and it is better to find it here
than in review. Both outcomes are reported.

Design
------
- Temperature is fitted on each fold's VALIDATION split only, never on test.
  Fitting on test would be leakage and would trivially flatter the method.
- The validation split is reconstructed by importing build_cv_folds from
  cross_validate.py directly (not reimplemented). Verified: all five
  reconstructed test splits match the subject IDs stored in the committed CV
  JSON exactly, so the val splits are equally correct.
- Validation predictions are NOT stored in the CV JSON (only aggregate val
  metrics), so this script re-runs inference on the 37 validation subjects per
  fold to obtain the logits needed to fit T.
- T is fitted by minimising NLL on validation logits via LBFGS, the standard
  approach (Guo et al. 2017).

Four conditions compared per mode:
  1. auc selection, no scaling            (standard practice - the baseline)
  2. auc selection + temperature scaling  (the reviewer's proposed fix)
  3. neg_brier selection, no scaling      (this paper's method)
  4. neg_brier selection + temperature    (do they stack?)

Reported per condition: AUC, balanced accuracy, ECE, Brier, and the
per-fold-mean vs pooled out-of-fold gap.

Outputs
  outputs/temperature/temperature_results.json
  outputs/temperature/temperature_preds_{mode}_fold{k}_{criterion}.csv
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss

from model_3d import FusionModel
from cross_validate import build_cv_folds

COHORT = "oasis3_cohort.csv"
PRE_DIR = r"D:\alhseimer\preprocessed"
CKPT_DIR = "outputs/cv_checkpoints"
OUT_DIR = "outputs/temperature"
CRITERIA = ["auc", "neg_brier"]
MODES = {"clinical_only": "sel_clinical",
         "imaging_only": "sel_imaging",
         "fusion": "sel_fusion"}
N_BINS_ECE = 10
SEED = 42


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def expected_calibration_error(labels, probs, n_bins=N_BINS_ECE):
    """Equal-width binning ECE, matching the definition used during training."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if not m.any():
            continue
        acc = (labels[m] == (probs[m] >= 0.5)).mean()
        conf = probs[m].mean()
        ece += (m.sum() / len(probs)) * abs(acc - conf)
    return float(ece)


def all_metrics(labels, probs):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "ece": expected_calibration_error(labels, probs),
        "brier": float(brier_score_loss(labels, probs)),
        "sensitivity": float(((preds == 1) & (labels == 1)).sum() / max((labels == 1).sum(), 1)),
        "specificity": float(((preds == 0) & (labels == 0)).sum() / max((labels == 0).sum(), 1)),
        "mean_prob": float(probs.mean()),
    }


# --------------------------------------------------------------------------
# temperature fitting
# --------------------------------------------------------------------------
def fit_temperature(val_logits, val_labels, max_iter=200):
    """Fit a single scalar T minimising NLL on validation logits (Guo et al.).

    Optimises log_T rather than T directly so T stays strictly positive - a
    negative or zero temperature would invert or destroy the logits.
    """
    logits = torch.tensor(val_logits, dtype=torch.float64)
    labels = torch.tensor(val_labels, dtype=torch.long)
    log_T = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # T = exp(0) = 1
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / torch.exp(log_T), labels)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(torch.exp(log_T).item())
    with torch.no_grad():
        nll_before = float(F.cross_entropy(logits, labels).item())
        nll_after = float(F.cross_entropy(logits / T, labels).item())
    return T, nll_before, nll_after


def apply_temperature(logits, T):
    """Divide logits by T and return P(AD)."""
    lt = torch.tensor(logits, dtype=torch.float64) / T
    return torch.softmax(lt, dim=1)[:, 1].numpy()


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------
def load_volume(subject, target_size):
    path = os.path.join(PRE_DIR, f"{subject}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{subject}: no cached volume at {path}")
    vol = np.load(path)
    if vol.ndim != 3 or not np.isfinite(vol).all():
        raise ValueError(f"{subject}: bad volume shape/values {vol.shape}")
    t = torch.from_numpy(vol).unsqueeze(0).float()
    if t.shape[-1] != target_size:
        t = F.interpolate(t.unsqueeze(0), size=(target_size,) * 3,
                          mode="trilinear", align_corners=False).squeeze(0)
    return t


@torch.no_grad()
def collect_logits(model, df, mode, ckpt, device):
    """Run the model over a split and return raw logits (pre-softmax).

    Uses the checkpoint's own frozen clinical_norm, exactly as training and
    ADNI inference do - normalisation is never recomputed here.
    """
    norm = ckpt["clinical_norm"]
    target_size = ckpt["target_size"]
    n_clin = ckpt["n_clinical_features"]

    logits, labels, subjects = [], [], []
    for _, row in df.iterrows():
        age_n = (float(row["age_at_scan"]) - norm["age_mean"]) / (norm["age_std"] + 1e-6)
        clin_vals = [float(row["apoe_e4_count"]), age_n][:n_clin]
        clin = torch.tensor([clin_vals], dtype=torch.float32).to(device)

        if mode == "clinical_only":
            out = model.forward_clinical_only(clin)
        else:
            vol = load_volume(row["subject"], target_size).unsqueeze(0).to(device)
            out = (model.forward_imaging_only(vol) if mode == "imaging_only"
                   else model(vol, clin))

        if tuple(out.shape) != (1, 2) or not torch.isfinite(out).all():
            raise ValueError(f"{row['subject']}: bad output {out.shape}")

        logits.append(out[0].cpu().numpy())
        labels.append(int(row["label"]))
        subjects.append(row["subject"])

    return np.array(logits), np.array(labels), subjects


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = FusionModel(n_clinical_features=ckpt["n_clinical_features"],
                        base_ch=ckpt["base_ch"], dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=list(MODES),
                    choices=list(MODES))
    ap.add_argument("--ckpt_dir", default=CKPT_DIR)
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    folds = build_cv_folds(COHORT, n_folds=5, val_frac=0.125, seed=SEED)
    print(f"Reconstructed 5 folds "
          f"(train {len(folds[0][0])} / val {len(folds[0][1])} / "
          f"test {len(folds[0][2])})\n")

    results = []
    for mode in args.modes:
        stub = MODES[mode]
        for criterion in CRITERIA:
            per_fold = {"raw": [], "scaled": []}
            pooled = {"raw": {"p": [], "y": []}, "scaled": {"p": [], "y": []}}
            temps = []

            for k, (train_df, val_df, test_df) in enumerate(folds, start=1):
                ckpt_path = os.path.join(args.ckpt_dir,
                                         f"{stub}_fold{k}_{criterion}.pt")
                if not os.path.exists(ckpt_path):
                    print(f"  MISSING {ckpt_path} - skipping fold {k}")
                    continue

                model, ckpt = load_checkpoint(ckpt_path)
                model = model.to(device)

                # 1. fit T on VALIDATION only
                v_log, v_lab, _ = collect_logits(model, val_df, mode, ckpt, device)
                T, nll_b, nll_a = fit_temperature(v_log, v_lab)
                temps.append(T)

                # 2. apply to TEST
                t_log, t_lab, t_subj = collect_logits(model, test_df, mode, ckpt, device)
                p_raw = torch.softmax(torch.tensor(t_log, dtype=torch.float64),
                                      dim=1)[:, 1].numpy()
                p_scaled = apply_temperature(t_log, T)

                m_raw = all_metrics(t_lab, p_raw)
                m_scaled = all_metrics(t_lab, p_scaled)
                per_fold["raw"].append(m_raw)
                per_fold["scaled"].append(m_scaled)
                pooled["raw"]["p"] += list(p_raw)
                pooled["raw"]["y"] += list(t_lab)
                pooled["scaled"]["p"] += list(p_scaled)
                pooled["scaled"]["y"] += list(t_lab)

                pd.DataFrame({"subject": t_subj, "label": t_lab,
                              "prob_raw": p_raw, "prob_scaled": p_scaled,
                              "temperature": T}).to_csv(
                    os.path.join(args.out_dir,
                                 f"temperature_preds_{mode}_fold{k}_{criterion}.csv"),
                    index=False)

                print(f"{mode:14s} fold {k} [{criterion:9s}]  T={T:6.3f}  "
                      f"val NLL {nll_b:.3f}->{nll_a:.3f}  |  "
                      f"AUC {m_raw['auc']:.4f}->{m_scaled['auc']:.4f}  "
                      f"bacc {m_raw['balanced_accuracy']:.4f}->"
                      f"{m_scaled['balanced_accuracy']:.4f}  "
                      f"ECE {m_raw['ece']:.4f}->{m_scaled['ece']:.4f}")

            if not per_fold["raw"]:
                continue

            row = {"mode": mode, "criterion": criterion,
                   "temperatures": temps,
                   "mean_temperature": float(np.mean(temps))}
            for cond in ["raw", "scaled"]:
                pf_auc = float(np.mean([m["auc"] for m in per_fold[cond]]))
                pl_auc = float(roc_auc_score(pooled[cond]["y"], pooled[cond]["p"]))
                row[cond] = {
                    "per_fold_mean_auc": pf_auc,
                    "pooled_auc": pl_auc,
                    "gap": pf_auc - pl_auc,
                    "per_fold_mean_bacc": float(np.mean(
                        [m["balanced_accuracy"] for m in per_fold[cond]])),
                    "per_fold_mean_ece": float(np.mean(
                        [m["ece"] for m in per_fold[cond]])),
                    "per_fold_mean_brier": float(np.mean(
                        [m["brier"] for m in per_fold[cond]])),
                    "n_degenerate_folds": int(sum(
                        m["balanced_accuracy"] < 0.55 for m in per_fold[cond])),
                }
            results.append(row)

            print(f"\n  {mode} [{criterion}] SUMMARY")
            for cond in ["raw", "scaled"]:
                r = row[cond]
                print(f"    {cond:7s}: per-fold {r['per_fold_mean_auc']:.4f}  "
                      f"pooled {r['pooled_auc']:.4f}  gap {r['gap']:.4f}  "
                      f"bacc {r['per_fold_mean_bacc']:.4f}  "
                      f"ECE {r['per_fold_mean_ece']:.4f}  "
                      f"degenerate {r['n_degenerate_folds']}/5")
            print()

    out_path = os.path.join(args.out_dir, "temperature_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved to {out_path}")

    # ---- the headline comparison --------------------------------------
    print("\n" + "=" * 78)
    print("DOES TEMPERATURE SCALING SUBSTITUTE FOR CALIBRATION-AWARE SELECTION?")
    print("=" * 78)
    by = {(r["mode"], r["criterion"]): r for r in results}
    for mode in args.modes:
        a, n = by.get((mode, "auc")), by.get((mode, "neg_brier"))
        if not (a and n):
            continue
        print(f"\n{mode}")
        print(f"  auc + no scaling      gap {a['raw']['gap']:.4f}  "
              f"bacc {a['raw']['per_fold_mean_bacc']:.4f}  "
              f"ECE {a['raw']['per_fold_mean_ece']:.4f}  "
              f"degen {a['raw']['n_degenerate_folds']}/5")
        print(f"  auc + temperature     gap {a['scaled']['gap']:.4f}  "
              f"bacc {a['scaled']['per_fold_mean_bacc']:.4f}  "
              f"ECE {a['scaled']['per_fold_mean_ece']:.4f}  "
              f"degen {a['scaled']['n_degenerate_folds']}/5")
        print(f"  neg_brier (this work) gap {n['raw']['gap']:.4f}  "
              f"bacc {n['raw']['per_fold_mean_bacc']:.4f}  "
              f"ECE {n['raw']['per_fold_mean_ece']:.4f}  "
              f"degen {n['raw']['n_degenerate_folds']}/5")
        print(f"  neg_brier + temp      gap {n['scaled']['gap']:.4f}  "
              f"bacc {n['scaled']['per_fold_mean_bacc']:.4f}  "
              f"ECE {n['scaled']['per_fold_mean_ece']:.4f}  "
              f"degen {n['scaled']['n_degenerate_folds']}/5")
    print("\nKey check: if 'auc + temperature' improves ECE but leaves balanced")
    print("accuracy and the degenerate-fold count unchanged, temperature scaling")
    print("is masking the failure rather than fixing it.")


if __name__ == "__main__":
    main()
