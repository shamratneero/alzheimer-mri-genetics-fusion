"""
Pipeline (per subject):
  1. Skull-strip with deepbet (deep-learning brain extraction)
  2. Apply mask, reorient to canonical (RAS)
  3. Resample to 1mm isotropic voxels          <- removes protocol confound
  4. Crop to the brain bounding box
  5. Resize to a fixed cube (CACHE_SIZE)
  6. Z-score normalise intensities within the brain mask
  7. Save as float32 .npy

Also writes a QC CSV recording brain-volume fraction and intensity stats so
failed skull-strips can be detected without opening every image.
"""
import os
import csv
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.transform import resize

COHORT     = "oasis3_cohort.csv"
OUT_DIR    = r"D:\alhseimer\preprocessed"
QC_CSV     = "preprocess_qc.csv"
CACHE_SIZE = 160          # cached cube side; loader can downsample further
TARGET_MM  = 1.0          # isotropic voxel size

os.makedirs(OUT_DIR, exist_ok=True)


import tempfile
from deepbet import run_bet

def skull_strip_deepbet(nifti_path):
    """Brain extraction using deepbet (deep-learning based).
    Returns the boolean mask in the ORIGINAL image space."""
    with tempfile.TemporaryDirectory() as td:
        mask_path = os.path.join(td, "mask.nii.gz")
        brain_path = os.path.join(td, "brain.nii.gz")
        run_bet([nifti_path], [brain_path], [mask_path],
                threshold=0.5, n_dilate=0, no_gpu=False)
        mask = np.asarray(nib.load(mask_path).dataobj) > 0
    return mask


def crop_to_mask(vol, mask, pad=4):
    """Crop volume to the bounding box of the mask, with a small margin."""
    idx = np.where(mask)
    if idx[0].size == 0:
        return vol, mask
    lo = [max(int(i.min()) - pad, 0) for i in idx]
    hi = [min(int(i.max()) + pad + 1, s) for i, s in zip(idx, vol.shape)]
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    return vol[sl], mask[sl]


def process_one(path):
    """Full preprocessing for a single scan. Returns (volume, qc_dict)."""
    # 1. brain extraction in native space
    mask_native = skull_strip_deepbet(path)

    # 2. load, apply mask, then reorient + resample
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32) * mask_native
    masked_img = nib.Nifti1Image(data, img.affine, img.header)

    masked_img = nib.as_closest_canonical(masked_img)
    masked_img = resample_to_output(masked_img, voxel_sizes=(TARGET_MM,) * 3, order=1)

    vol = np.asarray(masked_img.dataobj, dtype=np.float32)
    orig_shape = vol.shape
    mask = vol > 0
    brain_frac = float(mask.sum()) / mask.size

    vol, mask = crop_to_mask(vol, mask)
    cropped_shape = vol.shape

    vol = resize(vol, (CACHE_SIZE,) * 3, order=1,
                 preserve_range=True, anti_aliasing=True).astype(np.float32)
    mask_r = resize(mask.astype(np.float32), (CACHE_SIZE,) * 3, order=0,
                    preserve_range=True) > 0.5

    brain = vol[mask_r]
    mu, sd = (float(brain.mean()), float(brain.std())) if brain.size else (0.0, 1.0)
    vol = (vol - mu) / sd if sd > 0 else vol - mu
    vol[~mask_r] = 0.0

    qc = {
        "orig_shape": "x".join(map(str, orig_shape)),
        "cropped_shape": "x".join(map(str, cropped_shape)),
        "brain_fraction": round(brain_frac, 4),
        "brain_mean_raw": round(mu, 2),
        "brain_std_raw": round(sd, 2),
        "out_min": round(float(vol.min()), 3),
        "out_max": round(float(vol.max()), 3),
        "nonzero_frac": round(float((vol != 0).mean()), 4),
    }
    return vol, qc


def main():
    with open(COHORT, newline="") as fh:
        subjects = list(csv.DictReader(fh))

    print(f"Preprocessing {len(subjects)} subjects -> {OUT_DIR}")
    print(f"Cache size {CACHE_SIZE}^3 at {TARGET_MM}mm isotropic\n")

    qc_rows, failed = [], []

    for i, s in enumerate(subjects, 1):
        sid = s["subject"]
        out_path = os.path.join(OUT_DIR, f"{sid}.npy")

        if os.path.exists(out_path):
            print(f"[{i}/{len(subjects)}] {sid} - cached, skipping")
            continue

        try:
            vol, qc = process_one(s["filepath"])
            np.save(out_path, vol)
            qc_rows.append({"subject": sid, "label": s["label"], **qc})
            flag = ""
            if qc["brain_fraction"] < 0.05 or qc["brain_fraction"] > 0.60:
                flag = "  <-- CHECK brain fraction"
            print(f"[{i}/{len(subjects)}] {sid} - ok  "
                  f"brain_frac={qc['brain_fraction']:.3f}  "
                  f"crop={qc['cropped_shape']}{flag}")
        except Exception as e:
            failed.append((sid, f"{type(e).__name__}: {e}"))
            print(f"[{i}/{len(subjects)}] {sid} - FAILED: {e}")

    if qc_rows:
        write_header = not os.path.exists(QC_CSV)
        with open(QC_CSV, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(qc_rows[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(qc_rows)

    print(f"\n{'='*60}")
    print(f"Processed this run : {len(qc_rows)}")
    print(f"Failed             : {len(failed)}")
    for sid, err in failed:
        print(f"   {sid}: {err}")

    if qc_rows:
        fr = np.array([r["brain_fraction"] for r in qc_rows])
        print(f"\nBrain fraction: mean {fr.mean():.3f}  "
              f"min {fr.min():.3f}  max {fr.max():.3f}")
        odd = [r["subject"] for r in qc_rows
               if r["brain_fraction"] < 0.05 or r["brain_fraction"] > 0.60]
        if odd:
            print(f"Subjects to inspect visually ({len(odd)}): {odd[:15]}")

    print(f"\nQC written to {QC_CSV}")


if __name__ == "__main__":
    main()