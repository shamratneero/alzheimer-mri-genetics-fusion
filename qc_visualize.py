"""
Visual QC for preprocessed volumes.

Renders axial/coronal/sagittal mid-slices for:
  - the subjects with the lowest brain fractions (most likely failures)
  - the subjects with the highest brain fractions (possible skull retention)
  - a random sample of typical subjects
  - a few AD and CN examples side by side

Look for: complete brain, symmetric hemispheres, intact temporal lobes
(the lower-side regions), no obvious chunks missing.

Numbers alone are not sufficient here. The earlier Otsu-based skull-strip
produced plausible-looking brain fractions while visibly failing - retaining
skull on some subjects and removing brain tissue on others. It was caught by
looking at the images, not by the QC statistics.

USAGE
-----
    python qc_visualize.py                      # OASIS-3 (defaults)
    python qc_visualize.py --cohort_name adni   # ADNI
    python qc_visualize.py --pre_dir ... --qc_csv ... --cohort_csv ...
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRESETS = {
    "oasis3": dict(pre_dir=r"D:\alhseimer\preprocessed",
                   qc_csv="preprocess_qc.csv",
                   cohort_csv="oasis3_cohort.csv"),
    "adni": dict(pre_dir=r"D:\alhseimer\preprocessed_adni",
                 qc_csv="preprocess_qc_adni.csv",
                 cohort_csv="adni_scans_selected.csv"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--cohort_name", choices=list(PRESETS), default="oasis3")
ap.add_argument("--pre_dir", default=None)
ap.add_argument("--qc_csv", default=None)
ap.add_argument("--cohort_csv", default=None)
ap.add_argument("--figdir", default=None)
args = ap.parse_args()

cfg = PRESETS[args.cohort_name]
PRE_DIR = args.pre_dir or cfg["pre_dir"]
QC_CSV = args.qc_csv or cfg["qc_csv"]
COHORT_CSV = args.cohort_csv or cfg["cohort_csv"]
FIGDIR = args.figdir or f"figures_{args.cohort_name}"
os.makedirs(FIGDIR, exist_ok=True)

print(f"cohort   : {args.cohort_name}")
print(f"volumes  : {PRE_DIR}")
print(f"qc       : {QC_CSV}")
print(f"figures  : {FIGDIR}\n")

qc = pd.read_csv(QC_CSV)
cohort = pd.read_csv(COHORT_CSV)
qc = qc.merge(cohort[["subject", "label_name"]], on="subject", how="left")

# QC csv is appended to across batches, so a subject can appear more than once
qc = qc.drop_duplicates("subject", keep="last")
print(f"{len(qc)} subjects with QC records\n")


def show_grid(subjects, title, outfile, ncols=3):
    """One row per subject: sagittal, coronal, axial mid-slices."""
    n = len(subjects)
    fig, axes = plt.subplots(n, ncols, figsize=(3.2 * ncols, 3.2 * n))
    if n == 1:
        axes = axes[None, :]

    for r, sid in enumerate(subjects):
        v = np.load(os.path.join(PRE_DIR, f"{sid}.npy"))
        mid = [s // 2 for s in v.shape]
        planes = [v[mid[0], :, :], v[:, mid[1], :], v[:, :, mid[2]]]
        names = ["sagittal", "coronal", "axial"]
        info = qc[qc.subject == sid]
        bf = info["brain_fraction"].values[0] if len(info) else float("nan")
        lab = info["label_name"].values[0] if len(info) else "?"

        for c in range(ncols):
            ax = axes[r, c]
            ax.imshow(np.rot90(planes[c]), cmap="gray")
            ax.axis("off")
            if c == 0:
                ax.set_title(f"{sid} [{lab}]  bf={bf:.3f}", fontsize=9, loc="left")
            else:
                ax.set_title(names[c], fontsize=8)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    path = os.path.join(FIGDIR, outfile)
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"  wrote {path}")


print("Generating QC figures...")

n = len(qc)

# 1. worst brain fractions - most likely to have lost tissue
worst = qc.nsmallest(min(6, n), "brain_fraction")["subject"].tolist()
show_grid(worst, "LOWEST brain fraction - check for missing tissue",
          "qc_worst_brainfrac.png")

# 2. highest brain fractions - most likely to have retained skull
best = qc.nlargest(min(4, n), "brain_fraction")["subject"].tolist()
show_grid(best, "HIGHEST brain fraction - check for retained skull",
          "qc_highest_brainfrac.png")

# 3. random typical subjects.
# Selected by quantile rather than fixed thresholds: the two cohorts have
# different brain-fraction distributions (OASIS-3 mean 0.102 over a wide
# 0.052-0.323 range; ADNI mean 0.124 over a much tighter 0.099-0.171), so a
# hardcoded 0.13-0.25 window that works for one can select nothing for the
# other.
lo, hi = qc["brain_fraction"].quantile([0.25, 0.75])
typical = qc[(qc.brain_fraction >= lo) & (qc.brain_fraction <= hi)]
if len(typical):
    rng = np.random.default_rng(42)
    k = min(6, len(typical))
    sample = rng.choice(typical["subject"].values, size=k, replace=False)
    show_grid(list(sample),
              f"Random typical subjects (IQR {lo:.3f}-{hi:.3f})", "qc_typical.png")

# 4. AD vs CN comparison
ad_pool = qc[qc.label_name == "AD"]
cn_pool = qc[qc.label_name == "CN"]
k = min(3, len(ad_pool), len(cn_pool))
if k:
    ad = ad_pool.sample(k, random_state=1)["subject"].tolist()
    cn = cn_pool.sample(k, random_state=1)["subject"].tolist()
    show_grid(ad + cn, f"AD (top {k}) vs CN (bottom {k})", "qc_ad_vs_cn.png")
else:
    print("  skipping AD/CN grid - not enough of one group in this batch")

print("\nBrain fraction summary:")
print(qc["brain_fraction"].describe().round(4).to_string())
if qc["label_name"].notna().any():
    print("\nBy group:")
    print(qc.groupby("label_name")["brain_fraction"]
            .describe()[["count", "mean", "std", "min", "max"]].round(4).to_string())

print(f"\nOpen the PNGs in {FIGDIR}/ and check each brain is complete, "
      "hemispheres symmetric, temporal lobes intact, no skull retained.")