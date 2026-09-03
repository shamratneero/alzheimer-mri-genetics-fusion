"""
Grad-CAM for the 3D imaging branch: what does the model actually look at?

Motivation
----------
Phases 2-3 establish that AUC-based checkpoint selection can pick models that
rank well but are degenerate at the decision threshold, and that calibration-
aware selection avoids this. All of that evidence is scalar. This script asks
whether the difference is visible in the model's spatial attention.

The centrepiece comparison is fusion fold 4 on OASIS-3:
  auc       : epoch 10, test AUC 0.8974, balanced acc 0.5000,
              sensitivity 0.000 / specificity 1.000  (predicts CN for everyone)
  neg_brier : epoch 17, test AUC 0.8597, balanced acc 0.7285,
              sensitivity 0.692 / specificity 0.765
Same fold, same data, same run - two different epochs. If the degenerate
checkpoint's attention is diffuse or non-anatomical while the calibration-aware
pick concentrates on medial temporal structures, that is visual evidence for the
paper's central claim.

IMPORTANT - this may not show a clean difference. A model that outputs one class
for every subject can still have anatomically plausible internal features;
degenerate *output* does not guarantee degenerate *attention*. If the result is
ambiguous, it is reported as ambiguous. The pre-registered subject-selection
rule below exists so that "look at this striking example" cannot be reached by
browsing until something striking appears.

Spatial resolution caveat
-------------------------
block5 outputs 4x4x4 = 64 spatial cells for the whole brain. Upsampled to 128^3
each cell covers ~32^3 voxels - roughly lobe-scale. Claims must stay at the
level of "attention concentrates in the medial temporal region", never
"attention localises to the hippocampus". block4 (8x8x8, ~16^3 per cell) is
also computed as finer supporting evidence, but its features are less semantic.

Group means show spatial PATTERN, not magnitude
-----------------------------------------------
Each subject's CAM is min-max normalised to [0,1] before being averaged into the
AD/CN group means. This is necessary because raw CAM magnitudes differ by orders
of magnitude between subjects and checkpoints (they scale with the logit
gradient), so an unnormalised mean would be dominated by whichever subjects
happened to produce the largest gradients.

The consequence is that the group-mean maps and the AD-CN difference map answer
"WHERE does attention concentrate, on average?" and NOT "how MUCH more attention
does one group receive?". A brighter region in the AD mean means attention more
consistently lands there across AD subjects - it does not mean the model attends
more strongly to AD scans. Any magnitude claim must come from the per-subject
statistics (brain_selectivity, concentration, entropy), which are computed
before averaging, never from the mean image intensity.

Resampling
----------
Both cohorts cache volumes at 160^3 (preprocess.CACHE_SIZE) and the model
consumes 128^3 (dataset_3d interpolates on load). Grad-CAM therefore applies the
identical 160->128 interpolation before the forward pass and overlays the CAM on
the 128^3 volume. Overlaying on the raw 160^3 cache would misalign the heatmap
against the anatomy by ~20%.

Outputs (figures_gradcam/)
  {cohort}_{mode}_fold{k}_{criterion}_examples.png   individual subjects
  {cohort}_{mode}_fold{k}_{criterion}_groupmean.png  AD mean / CN mean / diff
  gradcam_summary.json                               quantitative attention stats
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_3d import FusionModel

OASIS_COHORT = "oasis3_cohort.csv"
OASIS_NPY_DIR = r"D:\alhseimer\preprocessed"
ADNI_COHORT = "adni_scans_selected.csv"
ADNI_NPY_DIR = r"D:\alhseimer\preprocessed_adni"
CKPT_DIR = "outputs/cv_checkpoints"
OUT_DIR = "figures_gradcam"
SEED = 42

ADNI_EXCLUDED_PET = [
    "057_S_6746", "057_S_6869",
    "098_S_6343", "098_S_6601", "098_S_6655", "098_S_6658",
    "126_S_6683",
]

# Pre-registered: fixed before any output was inspected.
N_CONFIDENT_CORRECT_PER_CLASS = 2
N_CONFIDENT_WRONG = 2


class GradCAM3D:
    """Grad-CAM on a 3D conv feature map.

    Hooks the chosen block's output, backprops the AD logit, and weights each
    channel by its globally-average-pooled gradient (the standard Grad-CAM
    channel weight), then ReLUs the weighted sum.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fh = target_layer.register_forward_hook(self._save_activation)
        self._bh = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def __call__(self, volume, clinical, mode, class_idx=1):
        self.model.zero_grad(set_to_none=True)
        if mode == "imaging_only":
            logits = self.model.forward_imaging_only(volume)
        elif mode == "fusion":
            logits = self.model(volume, clinical)
        else:
            raise ValueError(f"{mode} has no imaging pathway to visualise")

        score = logits[0, class_idx]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("hook did not fire - wrong target layer?")

        self.feature_shape = tuple(self.activations.shape[2:])
        if int(np.prod(self.feature_shape)) <= 1:
            raise RuntimeError(
                f"feature map is {self.feature_shape} - only one spatial cell, "
                f"so Grad-CAM carries no localisation information here. Use an "
                f"earlier block or a larger input size.")

        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)   # (1,C,1,1,1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=volume.shape[2:], mode="trilinear",
                            align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        prob = torch.softmax(logits, dim=1)[0, class_idx].item()
        return cam, prob


def load_volume(npy_dir, subject, target_size):
    """Load a cached volume and apply the SAME resize the training/inference
    dataloader applies, so CAM overlays align with the anatomy the model saw."""
    path = os.path.join(npy_dir, f"{subject}.npy")
    vol = np.load(path)
    if vol.ndim != 3:
        raise ValueError(f"{subject}: expected 3D volume, got {vol.shape}")
    if not np.isfinite(vol).all():
        raise ValueError(f"{subject}: volume contains NaN/Inf")
    t = torch.from_numpy(vol).unsqueeze(0).float()
    if t.shape[-1] != target_size:
        t = F.interpolate(t.unsqueeze(0), size=(target_size,) * 3,
                          mode="trilinear", align_corners=False).squeeze(0)
    return t


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = FusionModel(n_clinical_features=ckpt["n_clinical_features"],
                        base_ch=ckpt["base_ch"], dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def normalise_cam(cam, context="", warn=True):
    """Min-max normalise a CAM to [0,1].

    A flat CAM (min == max) means the layer produced no spatial variation -
    either the feature map is 1x1x1 at this input size, or every channel
    gradient cancelled. Silently returning zeros would render as a blank
    overlay indistinguishable from 'the model looked nowhere', so this warns.
    """
    lo, hi = cam.min(), cam.max()
    if hi - lo < 1e-8:
        if warn:
            print(f"    WARNING: flat CAM (min==max=={lo:.6f}) {context} - "
                  f"no spatial variation at this layer/input size")
        return np.zeros_like(cam)
    return (cam - lo) / (hi - lo)


def select_subjects(preds_df):
    """Pre-registered selection: most-confident-correct per class, plus the
    most-confident-wrong. Fully deterministic - no manual browsing.

    Ties in confidence are broken by subject ID (ascending). Without this,
    `nlargest` would fall back to DataFrame row order, so two subjects with
    identical confidence could swap places depending on how the predictions
    were loaded. There is no random component here, which is why no seed is
    taken - the rule is a total order, not a sample.
    """
    df = preds_df.copy()
    df["pred"] = (df["prob"] >= 0.5).astype(int)
    df["correct"] = df["pred"] == df["label"]
    df["confidence"] = np.where(df["prob"] >= 0.5, df["prob"], 1 - df["prob"])
    # Sort by confidence desc, then subject asc: total order, ties resolved.
    df = df.sort_values(["confidence", "subject"],
                        ascending=[False, True]).reset_index(drop=True)

    picks = []
    for lbl, name in [(1, "AD"), (0, "CN")]:
        sub = df[(df["label"] == lbl) & df["correct"]].head(
            N_CONFIDENT_CORRECT_PER_CLASS)
        for _, r in sub.iterrows():
            picks.append((r["subject"], int(r["label"]), float(r["prob"]),
                          f"confident correct {name}"))

    wrong = df[~df["correct"]].head(N_CONFIDENT_WRONG)
    for _, r in wrong.iterrows():
        picks.append((r["subject"], int(r["label"]), float(r["prob"]),
                      f"confident WRONG (true {'AD' if r['label'] else 'CN'})"))
    return picks


def plot_examples(vol_cams, title, out_path):
    """One row per subject: sagittal / coronal / axial mid-slices with CAM."""
    n = len(vol_cams)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i, (subj, label, prob, note, vol, cam) in enumerate(vol_cams):
        mid = [s // 2 for s in vol.shape]
        views = [(vol[mid[0], :, :], cam[mid[0], :, :], "sagittal"),
                 (vol[:, mid[1], :], cam[:, mid[1], :], "coronal"),
                 (vol[:, :, mid[2]], cam[:, :, mid[2]], "axial")]
        for j, (v, c, vname) in enumerate(views):
            ax = axes[i, j]
            ax.imshow(np.rot90(v), cmap="gray")
            ax.imshow(np.rot90(c), cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ax.axis("off")
            if j == 0:
                ax.set_title(f"{subj} [{'AD' if label else 'CN'}] "
                             f"p(AD)={prob:.2f}\n{note}", fontsize=7, loc="left")
            else:
                ax.set_title(vname, fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_group_means(ad_mean, cn_mean, ref_vol, title, out_path):
    """AD mean CAM, CN mean CAM, and their difference. Group means are the
    evidence; individual examples are illustration."""
    diff = ad_mean - cn_mean
    mid = [s // 2 for s in ref_vol.shape]
    rows = [("AD mean", ad_mean, "jet", 0, 1),
            ("CN mean", cn_mean, "jet", 0, 1),
            ("AD - CN", diff, "bwr", -np.abs(diff).max(), np.abs(diff).max())]

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i, (rname, arr, cmap, vmin, vmax) in enumerate(rows):
        views = [(ref_vol[mid[0], :, :], arr[mid[0], :, :], "sagittal"),
                 (ref_vol[:, mid[1], :], arr[:, mid[1], :], "coronal"),
                 (ref_vol[:, :, mid[2]], arr[:, :, mid[2]], "axial")]
        for j, (v, c, vname) in enumerate(views):
            ax = axes[i, j]
            ax.imshow(np.rot90(v), cmap="gray")
            im = ax.imshow(np.rot90(c), cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)
            ax.axis("off")
            ax.set_title(f"{rname} - {vname}" if j == 0 else vname, fontsize=7)
        fig.colorbar(im, ax=axes[i, :].tolist(), fraction=0.02)
    fig.suptitle(title + "\nPer-subject min-max normalised: shows WHERE attention "
                         "lands, not HOW MUCH", fontsize=9)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def make_brain_mask(vol_np):
    """Brain mask from the skull-stripped volume.

    Volumes are already skull-stripped (deepbet) and background is exactly
    zero, so the mask is simply the non-zero region. An intensity-percentile
    threshold would produce a DIFFERENT mask per subject, which makes
    attention statistics incomparable across subjects and checkpoints.
    """
    return vol_np > 0


def attention_stats(cam, brain_mask):
    """Quantitative descriptors so the comparison is not purely visual.

    concentration   : fraction of total attention inside the top 10% of voxels
                      (high = focal, low = diffuse)
    brain_frac      : fraction of total attention inside the brain mask
    brain_vol_frac  : fraction of the VOLUME that is brain - the chance baseline
    brain_selectivity : brain_frac / brain_vol_frac. This is the interpretable
                      quantity. 1.0 = attention no better than uniform;
                      >1 = preferentially attending to brain tissue.
                      Reporting brain_frac alone is meaningless because the
                      brain occupies only ~10% of a skull-stripped volume.
    entropy         : normalised Shannon entropy (high = spread, low = peaked)

    Returns None for a dead CAM. Callers must EXCLUDE these from averages -
    substituting 0.0 would silently drag every mean toward zero in proportion
    to how often the model failed to produce attention at all.
    """
    flat = cam.flatten()
    total = flat.sum()
    if total < 1e-8:
        return None
    k = max(1, int(0.10 * flat.size))
    top = np.partition(flat, -k)[-k:]
    p = flat / total
    nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum() / np.log(flat.size))
    brain_frac = float(cam[brain_mask].sum() / total)
    brain_vol_frac = float(brain_mask.sum() / brain_mask.size)
    return {
        "concentration": float(top.sum() / total),
        "brain_frac": brain_frac,
        "brain_vol_frac": brain_vol_frac,
        "brain_selectivity": brain_frac / max(brain_vol_frac, 1e-8),
        "entropy": ent,
    }


def run_one(cohort, npy_dir, cohort_df, preds_df, mode, fold, criterion,
            ckpt_path, block_name, device, out_dir, max_group=None):
    model, ckpt = load_checkpoint(ckpt_path)
    model = model.to(device)
    target_size = ckpt["target_size"]
    norm = ckpt["clinical_norm"]
    target_layer = getattr(model.imaging, block_name)
    cam_engine = GradCAM3D(model, target_layer)

    merged = preds_df.merge(
        cohort_df[["subject", "apoe_e4_count", "age_at_scan"]],
        on="subject", how="left")

    picks = select_subjects(merged)
    example_rows = []
    for subj, label, prob, note in picks:
        vol_t = load_volume(npy_dir, subj, target_size).unsqueeze(0).to(device)
        row = merged[merged["subject"] == subj].iloc[0]
        age_n = (float(row["age_at_scan"]) - norm["age_mean"]) / (norm["age_std"] + 1e-6)
        clin = torch.tensor([[float(row["apoe_e4_count"]), age_n]],
                            dtype=torch.float32).to(device)
        cam, _ = cam_engine(vol_t, clin, mode)
        example_rows.append((subj, label, prob, note,
                             vol_t[0, 0].cpu().numpy(),
                             normalise_cam(cam, f"[{subj}]")))

    tag = f"{cohort}_{mode}_fold{fold}_{criterion}_{block_name}"
    plot_examples(example_rows,
                  f"{cohort.upper()} {mode} fold {fold} [{criterion}] "
                  f"({block_name}) - individual examples",
                  os.path.join(out_dir, f"{tag}_examples.png"))

    # group means over the full set (or a capped subsample for speed)
    grp = merged if max_group is None else merged.sample(
        n=min(max_group, len(merged)), random_state=SEED)
    ad_sum = cn_sum = None
    ad_n = cn_n = 0
    dead_ad = dead_cn = 0
    ref_vol = None
    stats_ad, stats_cn = [], []

    for _, row in grp.iterrows():
        vol_t = load_volume(npy_dir, row["subject"], target_size).unsqueeze(0).to(device)
        age_n = (float(row["age_at_scan"]) - norm["age_mean"]) / (norm["age_std"] + 1e-6)
        clin = torch.tensor([[float(row["apoe_e4_count"]), age_n]],
                            dtype=torch.float32).to(device)
        cam, _ = cam_engine(vol_t, clin, mode)
        camn = normalise_cam(cam, f"[{row['subject']}]", warn=False)
        vol_np = vol_t[0, 0].cpu().numpy()
        if ref_vol is None:
            ref_vol = vol_np
        st = attention_stats(camn, make_brain_mask(vol_np))
        is_ad = int(row["label"]) == 1
        if st is None:
            # Dead CAM: contributes nothing. Excluded from BOTH the group mean
            # and the statistics - averaging in a zero array would bias every
            # number toward zero in proportion to the dead-CAM rate.
            if is_ad:
                dead_ad += 1
            else:
                dead_cn += 1
            continue
        if is_ad:
            ad_sum = camn if ad_sum is None else ad_sum + camn
            ad_n += 1
            stats_ad.append(st)
        else:
            cn_sum = camn if cn_sum is None else cn_sum + camn
            cn_n += 1
            stats_cn.append(st)

    cam_engine.remove()

    if ad_n == 0 or cn_n == 0:
        print(f"    SKIPPING group-mean figure: only {ad_n} live AD / {cn_n} "
              f"live CN CAMs ({dead_ad + dead_cn} dead) - not enough to average")
        ad_mean = np.zeros_like(ref_vol) if ad_sum is None else ad_sum / max(ad_n, 1)
        cn_mean = np.zeros_like(ref_vol) if cn_sum is None else cn_sum / max(cn_n, 1)
    else:
        ad_mean = ad_sum / ad_n
        cn_mean = cn_sum / cn_n
        plot_group_means(ad_mean, cn_mean, ref_vol,
                         f"{cohort.upper()} {mode} fold {fold} [{criterion}] "
                         f"({block_name}) - group mean attention "
                         f"(AD n={ad_n}, CN n={cn_n}; "
                         f"{dead_ad + dead_cn} dead CAMs excluded)",
                         os.path.join(out_dir, f"{tag}_groupmean.png"))

    def agg(sts, key):
        vals = [s[key] for s in sts]
        return float(np.mean(vals)) if vals else None

    return {
        "cohort": cohort, "mode": mode, "fold": fold, "criterion": criterion,
        "block": block_name, "checkpoint": ckpt_path,
        "selected_epoch": ckpt.get("epoch"),
        "n_ad_live": ad_n, "n_cn_live": cn_n,
        "n_ad_dead": dead_ad, "n_cn_dead": dead_cn,
        "dead_cam_rate": float((dead_ad + dead_cn) / max(len(grp), 1)),
        "ad_concentration": agg(stats_ad, "concentration"),
        "cn_concentration": agg(stats_cn, "concentration"),
        "ad_brain_selectivity": agg(stats_ad, "brain_selectivity"),
        "cn_brain_selectivity": agg(stats_cn, "brain_selectivity"),
        "brain_vol_frac": agg(stats_ad + stats_cn, "brain_vol_frac"),
        "ad_entropy": agg(stats_ad, "entropy"),
        "cn_entropy": agg(stats_cn, "entropy"),
        "_note_group_mean": (
            "Group-mean images are per-subject min-max normalised before "
            "averaging: they show spatial pattern (WHERE attention lands), "
            "not comparable attention magnitude. Magnitude claims must come "
            "from brain_selectivity / concentration / entropy, which are "
            "computed per subject before any averaging."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["oasis", "adni", "both"], default="oasis")
    ap.add_argument("--mode", choices=["fusion", "imaging_only"], default="fusion")
    ap.add_argument("--fold", type=int, default=4,
                    help="fold 4 is the documented degenerate-vs-calibrated case")
    ap.add_argument("--blocks", nargs="+", default=["block5", "block4"])
    ap.add_argument("--max_group", type=int, default=None,
                    help="cap subjects per group-mean map (speed); default all")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_stub = {"fusion": "sel_fusion", "imaging_only": "sel_imaging"}[args.mode]
    results = []

    cohorts = ["oasis", "adni"] if args.cohort == "both" else [args.cohort]
    for cohort in cohorts:
        if cohort == "oasis":
            cohort_df = pd.read_csv(OASIS_COHORT)
            npy_dir = OASIS_NPY_DIR
        else:
            cohort_df = pd.read_csv(ADNI_COHORT)
            cohort_df = cohort_df[cohort_df["within_365d"] == True]
            cohort_df = cohort_df[~cohort_df["subject"].isin(ADNI_EXCLUDED_PET)]
            npy_dir = ADNI_NPY_DIR
        cohort_df = cohort_df.reset_index(drop=True)

        for criterion in ["auc", "neg_brier"]:
            ckpt_path = os.path.join(
                CKPT_DIR, f"{ckpt_stub}_fold{args.fold}_{criterion}.pt")
            if not os.path.exists(ckpt_path):
                print(f"  MISSING {ckpt_path} - skipping")
                continue

            # predictions: OASIS-3 from the CV JSON, ADNI from the external CSVs
            if cohort == "oasis":
                folds = json.load(open(
                    f"outputs/cv/{ckpt_stub}_folds.json", encoding="utf-8"))
                fd = next(f for f in folds["folds"] if f["fold"] == args.fold)
                pr = fd["by_criterion"][criterion]["predictions"]
                preds_df = pd.DataFrame({"subject": pr["subject"],
                                         "label": pr["label"],
                                         "prob": pr["prob_ad"]})
            else:
                csv_path = (f"outputs/adni_external/adni_preds_{args.mode}"
                            f"_fold{args.fold}_{criterion}.csv")
                if not os.path.exists(csv_path):
                    print(f"  MISSING {csv_path} - skipping")
                    continue
                preds_df = pd.read_csv(csv_path)[["subject", "label", "prob"]]

            for block in args.blocks:
                print(f"Grad-CAM: {cohort} {args.mode} fold {args.fold} "
                      f"[{criterion}] {block} (n={len(preds_df)})...")
                r = run_one(cohort, npy_dir, cohort_df, preds_df, args.mode,
                            args.fold, criterion, ckpt_path, block, device,
                            args.out_dir, args.max_group)
                results.append(r)
                def _f(v):
                    return "n/a" if v is None else f"{v:.3f}"
                print(f"    epoch={r['selected_epoch']}  "
                      f"live AD={r['n_ad_live']} CN={r['n_cn_live']}  "
                      f"dead={r['n_ad_dead'] + r['n_cn_dead']} "
                      f"({r['dead_cam_rate']*100:.0f}%)")
                print(f"      brain selectivity (1.0 = chance): "
                      f"AD={_f(r['ad_brain_selectivity'])} "
                      f"CN={_f(r['cn_brain_selectivity'])}   "
                      f"[brain is {_f(r['brain_vol_frac'])} of volume]")
                print(f"      concentration: AD={_f(r['ad_concentration'])} "
                      f"CN={_f(r['cn_concentration'])}   "
                      f"entropy: AD={_f(r['ad_entropy'])} "
                      f"CN={_f(r['cn_entropy'])}")

    out_json = os.path.join(args.out_dir, "gradcam_summary.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved {len(results)} runs to {out_json}")
    print(f"Figures in {args.out_dir}/")


if __name__ == "__main__":
    main()