"""
OASIS-3 data integrity check and scan inventory.

Reads NIfTI headers (lazy - does not load voxel data) for every T1w file
across all data locations, records shape/voxel-size/datatype, flags
anomalies, and writes an inventory CSV.
"""
import os
import csv
import nibabel as nib

# All locations holding downloaded OASIS-3 subjects
DATA_ROOTS = [
    r"F:\oasis_T1w",        # original 300-subject batch
    r"F:\oasis_T1w_new",    # additional 150 AD subjects
]
OUT_CSV = "oasis3_scan_inventory.csv"

rows = []
errors = []

# --- Collect subject folders from every root ---
subject_paths = []
for root_dir in DATA_ROOTS:
    if not os.path.isdir(root_dir):
        print(f"WARNING: {root_dir} does not exist, skipping")
        continue
    for d in sorted(os.listdir(root_dir)):
        if d.startswith("sub-"):
            subject_paths.append((d, os.path.join(root_dir, d)))

print(f"Found {len(subject_paths)} subject folders across {len(DATA_ROOTS)} locations\n")

# --- Inventory every NIfTI ---
for i, (subj, subj_path) in enumerate(subject_paths, 1):
    for root, _, files in os.walk(subj_path):
        for f in files:
            if not f.endswith(".nii.gz"):
                continue
            fpath = os.path.join(root, f)
            try:
                img = nib.load(fpath)          # lazy - header only
                hdr = img.header
                shape = img.shape
                zooms = hdr.get_zooms()
                rows.append({
                    "subject": subj.replace("sub-", ""),
                    "filename": f,
                    "filepath": fpath,
                    "size_mb": round(os.path.getsize(fpath) / 1e6, 2),
                    "ndim": len(shape),
                    "shape_x": shape[0] if len(shape) > 0 else None,
                    "shape_y": shape[1] if len(shape) > 1 else None,
                    "shape_z": shape[2] if len(shape) > 2 else None,
                    "vox_x": round(float(zooms[0]), 3) if len(zooms) > 0 else None,
                    "vox_y": round(float(zooms[1]), 3) if len(zooms) > 1 else None,
                    "vox_z": round(float(zooms[2]), 3) if len(zooms) > 2 else None,
                    "dtype": str(hdr.get_data_dtype()),
                    "has_run": "run-" in f,
                    "has_echo": "echo-" in f,
                })
            except Exception as e:
                errors.append((fpath, str(e)))
    if i % 50 == 0:
        print(f"  processed {i}/{len(subject_paths)} subjects")

if not rows:
    print("No NIfTI files found - check DATA_ROOTS paths.")
    raise SystemExit(1)

# --- Write inventory ---
with open(OUT_CSV, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# --- Summary ---
print(f"\n{'='*55}")
print(f"Total NIfTI files inventoried : {len(rows)}")
print(f"Files that failed to load     : {len(errors)}")
print(f"Unique subjects               : {len(set(r['subject'] for r in rows))}")

print(f"\n--- Dimensionality ---")
ndims = {}
for r in rows:
    ndims[r["ndim"]] = ndims.get(r["ndim"], 0) + 1
for k, v in sorted(ndims.items()):
    print(f"  {k}D volumes: {v}")

print(f"\n--- Volume shapes (top 10) ---")
shapes = {}
for r in rows:
    key = f"{r['shape_x']}x{r['shape_y']}x{r['shape_z']}"
    shapes[key] = shapes.get(key, 0) + 1
for k, v in sorted(shapes.items(), key=lambda x: -x[1])[:10]:
    print(f"  {k:20s} : {v}")

print(f"\n--- Voxel sizes (top 10) ---")
vox = {}
for r in rows:
    key = f"{r['vox_x']}x{r['vox_y']}x{r['vox_z']}"
    vox[key] = vox.get(key, 0) + 1
for k, v in sorted(vox.items(), key=lambda x: -x[1])[:10]:
    print(f"  {k:20s} : {v}")

print(f"\n--- Scans per subject ---")
per_subj = {}
for r in rows:
    per_subj[r["subject"]] = per_subj.get(r["subject"], 0) + 1
dist = {}
for n in per_subj.values():
    dist[n] = dist.get(n, 0) + 1
for k, v in sorted(dist.items()):
    print(f"  {k} scan(s): {v} subjects")

print(f"\n--- Anomalies ---")
small = [r for r in rows if r["size_mb"] < 3]
print(f"  Files under 3MB: {len(small)}")
for r in small:
    print(f"    {r['subject']}: {r['filename']}  "
          f"({r['size_mb']}MB, shape {r['shape_x']}x{r['shape_y']}x{r['shape_z']})")

multiecho = [r for r in rows if r["has_echo"]]
print(f"  Multi-echo files: {len(multiecho)}")

if errors:
    print(f"\n--- LOAD ERRORS ---")
    for p, e in errors[:10]:
        print(f"  {p}: {e}")

print(f"\nInventory written to {OUT_CSV}")