"""
Select one canonical T1w scan per subject from the OASIS-3 inventory.

Selection rule (documented for Methods section):
  1. Exclude scans with <120 slices in any dimension (incomplete brain coverage,
     localizers, single-slice images, targeted regional acquisitions).
  2. Among remaining scans, prefer isotropic voxels (1.0 x 1.0 x 1.0 mm).
  3. Tie-break by first run (alphabetical filename order) for reproducibility.

Outputs:
  - oasis3_selected_scans.csv : one row per subject (the chosen scan)
  - oasis3_excluded_scans.csv : every excluded scan with the reason
"""
import csv
from collections import defaultdict

INVENTORY = "oasis3_scan_inventory.csv"
SELECTED_OUT = "oasis3_selected_scans.csv"
EXCLUDED_OUT = "oasis3_excluded_scans.csv"

MIN_SLICES = 120  # minimum extent in every dimension for whole-brain coverage

# --- Load inventory ---
with open(INVENTORY, newline="") as fh:
    scans = list(csv.DictReader(fh))

print(f"Loaded {len(scans)} scans from inventory\n")

# --- Step 1: filter on coverage ---
kept, excluded = [], []
for s in scans:
    dims = [int(s["shape_x"]), int(s["shape_y"]), int(s["shape_z"])]
    if min(dims) < MIN_SLICES:
        excluded.append({**s, "exclusion_reason":
                         f"incomplete coverage (min dim {min(dims)} < {MIN_SLICES})"})
    else:
        kept.append(s)

print(f"After coverage filter: {len(kept)} kept, {len(excluded)} excluded")

# --- Step 2 & 3: one scan per subject ---
by_subject = defaultdict(list)
for s in kept:
    by_subject[s["subject"]].append(s)

def is_isotropic(s, tol=0.01):
    v = [float(s["vox_x"]), float(s["vox_y"]), float(s["vox_z"])]
    return max(v) - min(v) < tol

selected = []
for subj, subj_scans in by_subject.items():
    iso = [s for s in subj_scans if is_isotropic(s)]
    pool = iso if iso else subj_scans          # prefer isotropic if available
    pool_sorted = sorted(pool, key=lambda s: s["filename"])
    chosen = pool_sorted[0]
    chosen = {**chosen,
              "n_scans_available": len(subj_scans),
              "isotropic_available": len(iso) > 0,
              "selection_note": "isotropic preferred" if iso else "no isotropic option"}
    selected.append(chosen)
    for s in subj_scans:
        if s["filepath"] != chosen["filepath"]:
            reason = ("not selected: isotropic preferred"
                      if iso and not is_isotropic(s)
                      else "not selected: duplicate run")
            excluded.append({**s, "exclusion_reason": reason})

selected.sort(key=lambda s: s["subject"])

# --- Write outputs ---
with open(SELECTED_OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(selected[0].keys()))
    w.writeheader(); w.writerows(selected)

with open(EXCLUDED_OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(excluded[0].keys()))
    w.writeheader(); w.writerows(excluded)

# --- Report ---
print(f"\n{'='*55}")
print(f"Subjects with a selected scan : {len(selected)}")
print(f"Total scans excluded          : {len(excluded)}")

print(f"\n--- Selected scan shapes ---")
shapes = defaultdict(int)
for s in selected:
    shapes[f"{s['shape_x']}x{s['shape_y']}x{s['shape_z']}"] += 1
for k, v in sorted(shapes.items(), key=lambda x: -x[1]):
    print(f"  {k:20s} : {v}")

print(f"\n--- Selected voxel sizes ---")
vox = defaultdict(int)
for s in selected:
    vox[f"{s['vox_x']}x{s['vox_y']}x{s['vox_z']}"] += 1
for k, v in sorted(vox.items(), key=lambda x: -x[1]):
    print(f"  {k:20s} : {v}")

n_iso = sum(1 for s in selected if s["isotropic_available"])
print(f"\nSubjects where an isotropic scan was available: {n_iso}/{len(selected)}")

print(f"\n--- Exclusion reasons ---")
reasons = defaultdict(int)
for e in excluded:
    reasons[e["exclusion_reason"]] += 1
for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
    print(f"  {v:4d}  {k}")

# Sanity: any subject lost entirely?
lost = set(s["subject"] for s in scans) - set(s["subject"] for s in selected)
if lost:
    print(f"\n!! WARNING: {len(lost)} subject(s) have NO usable scan: {sorted(lost)}")
else:
    print(f"\nAll 300 subjects retained a usable scan.")

print(f"\nWrote {SELECTED_OUT} and {EXCLUDED_OUT}")