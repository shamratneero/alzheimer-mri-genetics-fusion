"""Reconcile converted NIfTI files against the target ADNI cohort.

Checks BOTH nifti folders (the pilot batch and the main batch), because
preprocess_adni.py only reads one directory at a time and any subject that
lives in the other one would silently be treated as 'not downloaded'.
"""
import os
import pandas as pd

NIFTI_DIRS = [r"E:\adni_nifti", r"D:\alhseimer\adni_nifti"]
COHORT = "adni_scans_selected.csv"


def scan(d):
    if not os.path.isdir(d):
        print(f"  {d}  -- does not exist")
        return set()
    s = {f[:-7] for f in os.listdir(d) if f.endswith(".nii.gz")}
    print(f"  {d}  -- {len(s)} .nii.gz")
    return s


def main():
    print("NIfTI folders:")
    found = set()
    per_dir = {}
    for d in NIFTI_DIRS:
        s = scan(d)
        per_dir[d] = s
        found |= s

    df = pd.read_csv(COHORT)
    strict = set(df.loc[df.within_365d == True, "subject"])
    loose = set(df.loc[df.within_365d == False, "subject"])

    print(f"\ncohort file        : {len(df)} rows")
    print(f"  strict (<=365d)  : {len(strict)}")
    print(f"  loose  (>365d)   : {len(loose)}")

    print(f"\nconverted, unique across both folders : {len(found)}")
    print(f"  in strict target                    : {len(found & strict)}")
    print(f"  in loose (>365d, will be skipped)   : {len(found & loose)}")
    print(f"  in neither (unexpected)             : {len(found - strict - loose)}")

    missing = strict - found
    print(f"\nstrict target still MISSING a NIfTI   : {len(missing)}")
    if missing:
        ex = sorted(missing)[:10]
        print(f"  examples: {ex}")

    overlap = per_dir.get(NIFTI_DIRS[0], set()) & per_dir.get(NIFTI_DIRS[1], set())
    if overlap:
        print(f"\n{len(overlap)} subject(s) present in BOTH folders (harmless duplicates)")

    pct = len(found & strict) / len(strict) * 100 if strict else 0
    print(f"\ncoverage of strict target: {pct:.1f}%")
    if missing:
        print("\nNOTE: preprocess_adni.py reads ONE --nifti_dir. If subjects are split")
        print("across two folders, either copy them into one folder or run the script")
        print("once per folder - it skips already-cached subjects, so that is safe.")


if __name__ == "__main__":
    main()
