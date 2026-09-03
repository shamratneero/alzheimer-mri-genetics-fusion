"""
External validation: apply each OASIS-3 CV fold-model to ADNI.

Design decisions locked with Prof. Krapunski (see project correspondence):
  - Apply each of the 5 OASIS-3 fold-models to ADNI separately (not
    ensembled, not retrained) - preserves the per-fold-vs-pooled comparison
    that is this paper's central diagnostic.
  - Feature normalization is frozen from OASIS-3 training folds and applied
    UNCHANGED to ADNI. Age is z-scored with each fold's stored
    (age_mean, age_std); APOE e4 count is passed raw (0/1/2), matching
    OASIS3Dataset exactly. No ADNI-derived preprocessing parameters enter
    inference.
  - Both auc-selected and neg_brier-selected checkpoints are run per fold,
    to test whether calibration-aware selection transfers cross-cohort.
  - Only the 2-feature clinical branch runs on ADNI (extended_clinical
    checkpoints are skipped) - ADNI has no SES equivalent.

Checkpoint discovery does NOT rely on filename/tag conventions. Every
checkpoint written by cross_validate.py is self-contained (mode, fold,
criterion, extended_clinical, clinical_norm, and all architecture args are
stored inside the .pt file - see run_fold() in cross_validate.py), so this
script scans every .pt file in CKPT_DIR, reads each one's own metadata, and
groups by (mode, fold, criterion). This sidesteps the cv_*/sel_* tag-naming
ambiguity entirely - it does not matter what the files are named.

Per-fold auditable logging (per Prof. Krapunski's request): for every
checkpoint run, this script records the checkpoint path, the age mean/std
actually used, subject/prediction counts, AUC, bootstrap CI, and the ADNI
CN/AD age distribution alongside the prediction results - written to
outputs/adni_external/adni_external_results.json plus one predictions CSV
per (mode, fold, criterion).

Also computes, per (mode, criterion), the ADNI analogue of the OASIS-3
per-fold-vs-pooled comparison:
  - per-fold mean AUC     : average of the 5 folds' individual AUCs on ADNI
  - "cross-fold pooled-style AUC": all 5 folds' predictions on ADNI stacked
                          into one array and scored as a single AUC - tests
                          whether the 5 fold-models place probabilities on a
                          comparable scale, the same question pooled OOF AUC
                          answered on OASIS-3, now asked of an external
                          cohort instead of a held-out split. NOTE: unlike
                          OASIS-3's pooled OOF AUC, this is NOT a pooled AUC
                          over independent observations - each ADNI subject
                          contributes 5 (correlated) rows, one per fold-model.
                          Report it under this name specifically, not as a
                          conventional pooled AUC, to avoid overstating what
                          it measures.

Every checkpoint is validated against an explicit schema before use, every
loaded volume is independently sanity-checked (shape/finite/non-constant),
every model output is checked for shape/NaN before scoring, and the 5-fold
pooling step asserts identical subject sets across folds before stacking -
this script is written to fail loudly on any of these rather than silently
producing a number from malformed input.
"""
import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from model_3d import FusionModel

ADNI_COHORT = "adni_scans_selected.csv"
ADNI_NPY_DIR = r"D:\alhseimer\preprocessed_adni"
CKPT_DIR = "outputs/cv_checkpoints"
OUT_DIR = "outputs/adni_external"
CRITERIA_TO_RUN = ["auc", "neg_brier"]
N_BOOTSTRAP = 2000
SEED = 42

# Subjects excluded from the ADNI analysis set: PET series returned instead
# of T1 at sites 057, 098, 126 - caught by a dimension/voxel-size audit
# before preprocessing (64x64x64x2 volumes at ~4.7mm, vs proper T1 at
# 240x256x176 / ~1mm). See project record / RESULTS.md.
ADNI_EXCLUDED_PET = [
    "057_S_6746", "057_S_6869",
    "098_S_6343", "098_S_6601", "098_S_6655", "098_S_6658",
    "126_S_6683",
]


REQUIRED_CKPT_KEYS = ["mode", "fold", "criterion", "extended_clinical",
                      "clinical_norm", "n_clinical_features", "base_ch",
                      "dropout", "target_size", "model"]
VALID_MODES = {"clinical_only", "imaging_only", "fusion"}


def validate_checkpoint(p, ckpt):
    """Fail loudly on any checkpoint that doesn't match the expected
    contract, rather than silently trusting fields that happen to exist."""
    missing = [k for k in REQUIRED_CKPT_KEYS if k not in ckpt]
    if missing:
        raise ValueError(f"{p}: checkpoint missing expected keys {missing}")

    if ckpt["mode"] not in VALID_MODES:
        raise ValueError(f"{p}: unexpected mode '{ckpt['mode']}'")
    if not (1 <= ckpt["fold"] <= 5):
        raise ValueError(f"{p}: fold {ckpt['fold']} out of expected range 1-5")
    if ckpt["n_clinical_features"] not in (1, 2, 5):
        raise ValueError(f"{p}: unexpected n_clinical_features "
                          f"{ckpt['n_clinical_features']}")
    if not (32 <= ckpt["target_size"] <= 256):
        raise ValueError(f"{p}: implausible target_size {ckpt['target_size']}")
    norm = ckpt["clinical_norm"]
    if "age_mean" not in norm or "age_std" not in norm:
        raise ValueError(f"{p}: clinical_norm missing age_mean/age_std")
    if norm["age_std"] <= 0:
        raise ValueError(f"{p}: non-positive age_std {norm['age_std']} - "
                          f"would blow up normalization")
    if not (30 <= norm["age_mean"] <= 110):
        raise ValueError(f"{p}: implausible age_mean {norm['age_mean']}")


def discover_checkpoints(ckpt_dir):
    """Scan every .pt file and group by (mode, fold, criterion) using each
    checkpoint's own saved metadata - filenames are never parsed.
    Every checkpoint is validated against an explicit contract before use;
    anything malformed raises immediately rather than being trusted."""
    found = {}
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    if not paths:
        raise FileNotFoundError(f"No .pt files found in {ckpt_dir}")

    for p in paths:
        # weights_only=False: these checkpoints are self-generated by
        # cross_validate.py on this same machine (not downloaded from a
        # third party), and contain plain Python/numpy scalars in
        # clinical_norm/val_at_selection/test beyond raw tensors, which
        # PyTorch's default weights_only=True (as of 2.6) refuses to
        # unpickle. Safe here because the source is trusted.
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        validate_checkpoint(p, ckpt)
        if ckpt.get("extended_clinical", False):
            continue  # 5-feature branch: not usable on ADNI (no SES)
        if ckpt["criterion"] not in CRITERIA_TO_RUN:
            continue
        key = (ckpt["mode"], ckpt["fold"], ckpt["criterion"])
        if key in found:
            raise ValueError(
                f"Duplicate checkpoint for {key}: {found[key]['path']} "
                f"vs {p}. Both claim the same (mode, fold, criterion) - "
                f"resolve which is authoritative before running inference; "
                f"silently picking one could mean using stale weights.")
        found[key] = {"path": p, "ckpt": ckpt}

    return found


def load_adni_cohort(out_dir):
    df = pd.read_csv(ADNI_COHORT)
    df["label"] = df["label"].astype(int)
    df = df[df["within_365d"] == True].reset_index(drop=True)
    df = df[~df["subject"].isin(ADNI_EXCLUDED_PET)].reset_index(drop=True)

    has_npy = df["subject"].apply(
        lambda s: os.path.exists(os.path.join(ADNI_NPY_DIR, f"{s}.npy")))
    missing = df[~has_npy]
    if len(missing):
        excl_path = os.path.join(out_dir, "adni_excluded_no_npy.csv")
        missing[["subject", "label"]].to_csv(excl_path, index=False)
        print(f"  {len(missing)} cohort subjects have no cached .npy yet, "
              f"skipping - full list written to {excl_path}")
    df = df[has_npy].reset_index(drop=True)
    return df


def load_volume(subject, target_size):
    """Load one preprocessed ADNI volume, independently re-verifying it is a
    plausible 3D structural volume rather than trusting the cohort CSV /
    prior QC pass alone (audit-grade check, not a repeat of the earlier
    dimension/voxel-size audit that caught the PET contamination)."""
    path = os.path.join(ADNI_NPY_DIR, f"{subject}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{subject}: expected volume not found at {path}")
    vol = np.load(path)

    if vol.ndim != 3:
        raise ValueError(f"{subject}: expected 3D volume, got shape {vol.shape}")
    if any(d < 32 for d in vol.shape):
        raise ValueError(f"{subject}: implausibly small volume {vol.shape}")
    if not np.isfinite(vol).all():
        raise ValueError(f"{subject}: volume contains NaN/Inf")
    if vol.max() == vol.min():
        raise ValueError(f"{subject}: volume is constant (empty/blank scan)")

    t = torch.from_numpy(vol).unsqueeze(0).float()
    if t.shape[-1] != target_size:
        t = F.interpolate(t.unsqueeze(0), size=(target_size,) * 3,
                           mode="trilinear", align_corners=False).squeeze(0)
    return t


def build_model(ckpt):
    model = FusionModel(n_clinical_features=ckpt["n_clinical_features"],
                        base_ch=ckpt["base_ch"], dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def forward_pass(model, vol, clin, mode):
    if mode == "imaging_only":
        return model.forward_imaging_only(vol)
    if mode == "clinical_only":
        return model.forward_clinical_only(clin)
    return model(vol, clin)


def bootstrap_ci(labels, probs, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(labels)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(labels[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(labels[idx], probs[idx]))
    if not boot_aucs:
        return None, None
    return (float(np.percentile(boot_aucs, 2.5)),
            float(np.percentile(boot_aucs, 97.5)))


@torch.no_grad()
def run_one_checkpoint(mode, fold, criterion, path, ckpt, adni_df, device):
    norm = ckpt["clinical_norm"]
    age_mean, age_std = norm["age_mean"], norm["age_std"]
    target_size = ckpt["target_size"]
    n_clin = ckpt["n_clinical_features"]

    model = build_model(ckpt).to(device)

    probs, labels, subjects, ages = [], [], [], []
    for _, row in adni_df.iterrows():
        vol = load_volume(row["subject"], target_size).unsqueeze(0).to(device)
        age_norm = (float(row["age_at_scan"]) - age_mean) / (age_std + 1e-6)
        apoe = float(row["apoe_e4_count"])
        # n_clin==2 is the expected case for every checkpoint used here
        # (2-feature branch); guard for 1 defensively rather than assume.
        clin_vals = [apoe, age_norm] if n_clin == 2 else [apoe]
        clin = torch.tensor([clin_vals], dtype=torch.float32).to(device)

        out = forward_pass(model, vol, clin, mode)
        if tuple(out.shape) != (1, 2):
            raise ValueError(f"{row['subject']}: unexpected model output "
                              f"shape {tuple(out.shape)}, expected (1, 2)")
        if not torch.isfinite(out).all():
            raise ValueError(f"{row['subject']}: model output contains NaN/Inf "
                              f"(mode={mode}, fold={fold}, criterion={criterion})")
        prob = torch.softmax(out, dim=1)[0, 1].item()
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"{row['subject']}: probability {prob} outside [0,1]")

        probs.append(prob)
        labels.append(int(row["label"]))
        subjects.append(row["subject"])
        ages.append(float(row["age_at_scan"]))

    probs = np.array(probs)
    labels = np.array(labels)

    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan")
    ci_low, ci_high = bootstrap_ci(labels, probs)

    cn_age = [a for a, l in zip(ages, labels) if l == 0]
    ad_age = [a for a, l in zip(ages, labels) if l == 1]

    result = {
        "mode": mode,
        "fold": fold,
        "criterion": criterion,
        "checkpoint_path": path,
        "age_mean_used": age_mean,
        "age_std_used": age_std,
        "n_subjects": int(len(labels)),
        "n_predictions": int(len(probs)),
        "auc": float(auc),
        "auc_ci_low": ci_low,
        "auc_ci_high": ci_high,
        "bootstrap_n": N_BOOTSTRAP,
        "bootstrap_seed": SEED,
        "adni_cn_age_mean": float(np.mean(cn_age)) if cn_age else None,
        "adni_ad_age_mean": float(np.mean(ad_age)) if ad_age else None,
    }

    preds_df = pd.DataFrame({
        "subject": subjects, "label": labels, "prob": probs, "age_at_scan": ages,
    })

    return result, preds_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default=CKPT_DIR)
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Discovering checkpoints (by embedded metadata, not filename)...")
    found = discover_checkpoints(args.ckpt_dir)
    modes = sorted(set(k[0] for k in found))
    print(f"  found {len(found)} usable (2-feature) checkpoints across "
          f"modes {modes}\n")
    expected = len(modes) * 5 * len(CRITERIA_TO_RUN)
    if len(found) != expected:
        print(f"  NOTE: expected {expected} ({len(modes)} modes x 5 folds x "
              f"{len(CRITERIA_TO_RUN)} criteria), found {len(found)} - "
              f"some (mode, fold, criterion) combinations will be skipped "
              f"below if missing.\n")

    print("Loading ADNI cohort...")
    adni_df = load_adni_cohort(args.out_dir)
    print(f"  {len(adni_df)} ADNI subjects with cached volumes "
          f"({(adni_df['label']==1).sum()} AD / "
          f"{(adni_df['label']==0).sum()} CN)\n")

    all_results = []
    for mode in modes:
        for criterion in CRITERIA_TO_RUN:
            per_fold_probs = []
            for fold in range(1, 6):
                key = (mode, fold, criterion)
                if key not in found:
                    print(f"  MISSING checkpoint for {key} - skipping")
                    continue
                entry = found[key]
                print(f"Running {mode} fold {fold} [{criterion}] "
                      f"({entry['path']})...")
                result, preds_df = run_one_checkpoint(
                    mode, fold, criterion, entry["path"], entry["ckpt"],
                    adni_df, device)
                all_results.append(result)
                print(f"    AUC={result['auc']:.4f}  "
                      f"CI=[{result['auc_ci_low']:.3f}, "
                      f"{result['auc_ci_high']:.3f}]  n={result['n_subjects']}  "
                      f"age_mean_used={result['age_mean_used']:.2f}")

                preds_path = os.path.join(
                    args.out_dir, f"adni_preds_{mode}_fold{fold}_{criterion}.csv")
                preds_df.to_csv(preds_path, index=False)
                per_fold_probs.append(preds_df.set_index("subject")["prob"])

            if len(per_fold_probs) == 5:
                # Hard check: every fold must have predicted on the exact
                # same subject set, or the pooled-style AUC below would be
                # silently comparing mismatched subjects/labels across folds.
                subject_sets = [set(fp.index) for fp in per_fold_probs]
                ref_set = subject_sets[0]
                for i, s in enumerate(subject_sets[1:], start=2):
                    if s != ref_set:
                        sym_diff = s.symmetric_difference(ref_set)
                        raise ValueError(
                            f"{mode} [{criterion}]: fold 1 and fold {i} "
                            f"predicted on different subject sets "
                            f"({len(sym_diff)} subjects differ) - refusing "
                            f"to pool, this would silently misalign "
                            f"predictions and labels.")
                for fp in per_fold_probs:
                    if fp.index.duplicated().any():
                        raise ValueError(
                            f"{mode} [{criterion}]: duplicate subject IDs "
                            f"within a single fold's predictions - check "
                            f"the ADNI cohort CSV for repeated subject rows.")

                stacked_probs = pd.concat(per_fold_probs).values
                stacked_labels = np.concatenate([
                    adni_df.set_index("subject").loc[fp.index, "label"].values
                    for fp in per_fold_probs
                ])
                assert len(stacked_probs) == len(stacked_labels) == 5 * len(ref_set), (
                    "pooled array length mismatch - aborting rather than "
                    "reporting a possibly-corrupted pooled AUC")
                pooled_auc = roc_auc_score(stacked_labels, stacked_probs)
                per_fold_mean_auc = float(np.mean(
                    [r["auc"] for r in all_results
                     if r["mode"] == mode and r["criterion"] == criterion]))
                gap = per_fold_mean_auc - pooled_auc
                print(f"\n  {mode} [{criterion}] SUMMARY: "
                      f"per-fold mean AUC = {per_fold_mean_auc:.4f}, "
                      f"pooled-style AUC = {pooled_auc:.4f}, gap = {gap:.4f}\n")
                all_results.append({
                    "mode": mode, "fold": "pooled_summary", "criterion": criterion,
                    "checkpoint_path": None,
                    "pooled_style_auc": float(pooled_auc),
                    "per_fold_mean_auc": per_fold_mean_auc,
                    "gap": float(gap),
                })
            else:
                print(f"\n  {mode} [{criterion}]: only {len(per_fold_probs)}/5 "
                      f"folds ran - skipping pooled-style summary\n")

    out_path = os.path.join(args.out_dir, "adni_external_results.json")
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"Saved full results to {out_path}")


if __name__ == "__main__":
    main()