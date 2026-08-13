"""
Test deepbet brain extraction on a single subject before committing to
a full re-preprocessing run.
"""
import csv
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- pick a subject that failed badly with the old method ---
TEST_SUBJECT = "OAS31029"   # brain_frac was 0.056, images showed only skull

with open("oasis3_cohort.csv", newline="") as fh:
    rows = list(csv.DictReader(fh))
row = next(r for r in rows if r["subject"] == TEST_SUBJECT)
path = row["filepath"]
print(f"Testing on {TEST_SUBJECT} ({row['label_name']})")
print(f"  {path}\n")

# --- run deepbet ---
from deepbet import run_bet

out_brain = "test_brain.nii.gz"
out_mask  = "test_mask.nii.gz"

print("Running deepbet...")
run_bet([path], [out_brain], [out_mask], threshold=0.5, n_dilate=0, no_gpu=False)
print("done.\n")

# --- inspect ---
orig = np.asarray(nib.load(path).dataobj, dtype=np.float32)
brain = np.asarray(nib.load(out_brain).dataobj, dtype=np.float32)
mask = np.asarray(nib.load(out_mask).dataobj) > 0

print(f"Original shape     : {orig.shape}")
print(f"Brain mask voxels  : {mask.sum():,}")
print(f"Brain fraction     : {mask.mean():.4f}")

# --- visualise: original vs extracted, three planes ---
mid = [s // 2 for s in orig.shape]
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
planes_o = [orig[mid[0], :, :], orig[:, mid[1], :], orig[:, :, mid[2]]]
planes_b = [brain[mid[0], :, :], brain[:, mid[1], :], brain[:, :, mid[2]]]
names = ["sagittal", "coronal", "axial"]

for c in range(3):
    axes[0, c].imshow(np.rot90(planes_o[c]), cmap="gray")
    axes[0, c].set_title(f"original - {names[c]}", fontsize=9)
    axes[0, c].axis("off")
    axes[1, c].imshow(np.rot90(planes_b[c]), cmap="gray")
    axes[1, c].set_title(f"deepbet - {names[c]}", fontsize=9)
    axes[1, c].axis("off")

plt.suptitle(f"{TEST_SUBJECT}: original (top) vs deepbet extraction (bottom)")
plt.tight_layout()
plt.savefig("figures/test_deepbet.png", dpi=140)
plt.close()
print("Wrote figures/test_deepbet.png")