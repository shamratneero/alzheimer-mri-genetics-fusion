"""
Preprocess the ADNI cohort using EXACTLY the same pipeline as OASIS-3.

WHY THIS FILE IMPORTS RATHER THAN COPIES
----------------------------------------
The cross-cohort claim in this project is that ADNI and OASIS-3 are processed
identically, so that any performance difference is attributable to the cohort
and not to preprocessing. If this file contained its own copy of the
processing steps, the two copies could drift apart later - a changed
resampling order, a different crop padding, a tweaked normalisation - and
nothing would error. The comparison would quietly become invalid.

So `process_one`, `skull_strip_deepbet` and `crop_to_mask` are imported from
`preprocess.py`. There is one implementation, used by both cohorts. The only
things that differ here are:

  * which cohort CSV is read              (adni_scans_selected.csv)
  * where the cached .npy files are written (a separate ADNI directory, so the
    OASIS-3 cache is never overwritten)
  * how each subject's scan file is located

CACHE_SIZE (160) and TARGET_MM (1.0) are also imported, so they cannot be set
to different values for the two cohorts by accident.

INPUT
-----
Expects NIfTI files produced by dcm2niix, one per subject, named by subject ID:

    D:\\alhseimer\\adni_nifti\\002_S_0295.nii.gz

produced by:

    dcm2niix.exe -z y -f %i -o D:\\alhseimer\\adni_nifti -d 9 -i y <dicom_root>

The `-f %i` flag names output by DICOM PatientID, which for ADNI is the subject
ID, so files map directly onto adni_scans_selected.csv without a lookup table.

USAGE
-----
    python preprocess_adni.py                 # process everything not yet cached
    python preprocess_adni.py --limit 20      # pilot: first 20 only
    python preprocess_adni.py --nifti_dir ... --out_dir ...
"""
import os
import csv
import argparse
import numpy as np

# Single source of truth for the processing steps and their parameters.
# Importing (rather than reimplementing) is what guarantees the two cohorts
# are treated identically.
from preprocess import process_one, CACHE_SIZE, TARGET_MM

COHORT_CSV = "adni_scans_selected.csv"
NIFTI_DIR = r"D:\alhseimer\adni_nifti"
OUT_DIR = r"D:\alhseimer\preprocessed_adni"
QC_CSV = "preprocess_qc_adni.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=COHORT_CSV)
    ap.add_argument("--nifti_dir", default=NIFTI_DIR)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--qc_csv", default=QC_CSV)
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N subjects (for a pilot run)")
    ap.add_argument("--strict_only", action="store_true",
                    help="only subjects whose scan is within 365 days of the "
                         "diagnosis visit, matching the OASIS-3 rule")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cohort, newline="") as fh:
        subjects = list(csv.DictReader(fh))

    if args.strict_only:
        before = len(subjects)
        subjects = [s for s in subjects
                    if str(s.get("within_365d", "")).strip().lower() == "true"]
        print(f"strict filter: {len(subjects)} of {before} subjects "
              f"within 365 days of diagnosis")

    # only those whose converted NIfTI actually exists on disk - the download
    # and conversion happen in batches, so this will normally be a subset
    present, missing = [], []
    for s in subjects:
        p = os.path.join(args.nifti_dir, f"{s['subject']}.nii.gz")
        (present if os.path.exists(p) else missing).append(s)

    if args.limit:
        present = present[:args.limit]

    print(f"Preprocessing {len(present)} subjects -> {args.out_dir}")
    print(f"Cache size {CACHE_SIZE}^3 at {TARGET_MM}mm isotropic "
          f"(imported from preprocess.py - identical to OASIS-3)")
    if missing:
        print(f"{len(missing)} cohort subjects have no NIfTI yet "
              f"(not downloaded/converted)")
    print()

    qc_rows, failed = [], []

    for i, s in enumerate(present, 1):
        sid = s["subject"]
        out_path = os.path.join(args.out_dir, f"{sid}.npy")

        if os.path.exists(out_path):
            print(f"[{i}/{len(present)}] {sid} - cached, skipping")
            continue

        nifti_path = os.path.join(args.nifti_dir, f"{sid}.nii.gz")

        try:
            vol, qc = process_one(nifti_path)
            np.save(out_path, vol)
            qc_rows.append({"subject": sid, "label": s.get("label", ""), **qc})
            flag = ""
            if qc["brain_fraction"] < 0.05 or qc["brain_fraction"] > 0.60:
                flag = "  <-- CHECK brain fraction"
            print(f"[{i}/{len(present)}] {sid} - ok  "
                  f"brain_frac={qc['brain_fraction']:.3f}  "
                  f"orig={qc['orig_shape']}  crop={qc['cropped_shape']}{flag}")
        except Exception as e:
            failed.append((sid, f"{type(e).__name__}: {e}"))
            print(f"[{i}/{len(present)}] {sid} - FAILED: {e}")

    if qc_rows:
        write_header = not os.path.exists(args.qc_csv)
        with open(args.qc_csv, "a", newline="") as fh:
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
        print("  (OASIS-3 reference: mean 0.102, range 0.052-0.323 - a large "
              "departure here would suggest deepbet is behaving differently "
              "on ADNI and warrants visual inspection)")
        odd = [r["subject"] for r in qc_rows
               if r["brain_fraction"] < 0.05 or r["brain_fraction"] > 0.60]
        if odd:
            print(f"Subjects to inspect visually ({len(odd)}): {odd[:15]}")

    print(f"\nQC written to {args.qc_csv}")
    print("\nNEXT: inspect a few outputs visually (qc_visualize.py) before "
          "trusting the full run - ADNI spans different scanners, field "
          "strengths and orientations than OASIS-3.")


if __name__ == "__main__":
    main()
