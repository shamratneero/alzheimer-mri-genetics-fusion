"""
Visual QC for preprocessed volumes.

Renders axial/coronal/sagittal mid-slices for:
  - the subjects with the lowest brain fractions (most likely failures)
  - a random sample of typical subjects
  - a few AD and CN examples side by side

Look for: complete brain, symmetric hemispheres, intact temporal lobes
(the lower-side regions), no obvious chunks missing.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRE_DIR = r"D:\alhseimer\preprocessed"
FIGDIR  = "figures"
os.makedirs(FIGDIR, exist_ok=True)

qc = pd.read_csv("preprocess_qc.csv")
cohort = pd.read_csv("oasis3_cohort.csv")
qc = qc.merge(cohort[["subject", "label_name"]], on="subject", how="left")


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

# 1. worst brain fractions - most likely failures
worst = qc.nsmallest(6, "brain_fraction")["subject"].tolist()
show_grid(worst, "LOWEST brain fraction - check for missing tissue",
          "qc_worst_brainfrac.png")

# 2. highest brain fractions - possible skull retention
best = qc.nlargest(4, "brain_fraction")["subject"].tolist()
show_grid(best, "HIGHEST brain fraction - check for retained skull",
          "qc_highest_brainfrac.png")

# 3. random typical subjects
rng = np.random.default_rng(42)
typical = qc[(qc.brain_fraction > 0.13) & (qc.brain_fraction < 0.25)]
sample = rng.choice(typical["subject"].values, size=min(6, len(typical)), replace=False)
show_grid(list(sample), "Random typical subjects", "qc_typical.png")

# 4. AD vs CN comparison
ad = qc[qc.label_name == "AD"].sample(3, random_state=1)["subject"].tolist()
cn = qc[qc.label_name == "CN"].sample(3, random_state=1)["subject"].tolist()
show_grid(ad + cn, "AD (top 3) vs CN (bottom 3)", "qc_ad_vs_cn.png")

print("\nBrain fraction summary:")
print(qc["brain_fraction"].describe().round(4).to_string())
print("\nBy group:")
print(qc.groupby("label_name")["brain_fraction"].describe()[["count","mean","std","min","max"]].round(4).to_string())