"""Write the IDA Image IDs for cohort subjects that still lack a NIfTI."""
import os, pandas as pd

NIFTI_DIRS = [r"E:\adni_nifti", r"D:\alhseimer\adni_nifti"]
CHUNK = 250

found = set()
for d in NIFTI_DIRS:
    if os.path.isdir(d):
        found |= {f[:-7] for f in os.listdir(d) if f.endswith(".nii.gz")}

df = pd.read_csv("adni_scans_selected.csv")
strict = df[df.within_365d == True]
missing = strict[~strict.subject.isin(found)]

print(f"{len(missing)} subjects still need downloading")
ids = [str(i) for i in missing["Image ID"]]
with open("adni_missing_image_ids.txt", "w") as fh:
    for i in range(0, len(ids), CHUNK):
        batch = ids[i:i+CHUNK]
        fh.write(f"--- batch {i//CHUNK+1}  ({len(batch)} ids) ---\n")
        fh.write(",".join(batch) + "\n\n")
print(f"wrote adni_missing_image_ids.txt in {(len(ids)+CHUNK-1)//CHUNK} batch(es)")
missing[["subject","label_name","Image ID","Phase"]].to_csv("adni_missing_subjects.csv", index=False)
print("wrote adni_missing_subjects.csv")
