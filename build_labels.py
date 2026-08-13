"""
Build the labeled analysis cohort for OASIS-3.

Links each selected T1w scan to:
  - Alzheimer's diagnosis (PROBAD) vs cognitively normal (NORMCOG),
    taken from the D1 clinical visit CLOSEST IN TIME to the scan date
  - APOE genotype (and derived e4 allele count)
  - Demographics: age at scan, sex, education, SES
  - CDR and MMSE from the nearest visit (for characterisation / Table 1)

Subjects are EXCLUDED if:
  - the nearest visit is neither PROBAD nor NORMCOG (e.g. MCI, non-AD dementia)
  - APOE genotype is missing/invalid
  - no clinical visit exists within MAX_DAYS of the scan

Outputs:
  oasis3_cohort.csv           final labelled cohort
  oasis3_cohort_excluded.csv  every dropped subject with the reason
"""
import csv, os
from collections import defaultdict

BASE = r"D:\alhseimer\OASIS3_data_files\OASIS3_data_files\scans"
DEMO = os.path.join(BASE, r"demo-demographics\resources\csv\files\OASIS3_demographics.csv")
DX   = os.path.join(BASE, r"UDSd1-Form_D1__Clinician_Diagnosis___Cognitive_Status_and_Dementia\resources\csv\files\OASIS3_UDSd1_diagnoses.csv")
CDR  = os.path.join(BASE, r"UDSb4-Form_B4__Global_Staging__CDR__Standard_and_Supplemental\resources\csv\files\OASIS3_UDSb4_cdr.csv")
COG  = os.path.join(BASE, r"pychometrics-Form_C1__Cognitive_Assessments\resources\csv\files\OASIS3_UDSc1_cognitive_assessments.csv")

SELECTED = "oasis3_selected_scans.csv"
OUT      = "oasis3_cohort.csv"
OUT_EXCL = "oasis3_cohort_excluded.csv"

MAX_DAYS = 365   # nearest clinical visit must be within 1 year of the scan


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def day_from_label(label):
    """'OAS30001_MR_d0129' or 'OAS30001_ClinicalData_d0339' -> 129 / 339"""
    try:
        return int(label.rsplit("_d", 1)[1])
    except Exception:
        return None


def find_day_column(rows):
    """OASIS uses different day-offset column names across forms."""
    if not rows:
        return None
    cands = ["days_to_visit", "ADRC_ADRCCLINICALDATA ID", "OASIS_session_label",
             "UDS_D1DXDATA ID", "UDS_B4CDRDATA ID", "UDS_C1CSDATA ID"]
    for c in cands:
        if c in rows[0]:
            return c
    for k in rows[0]:
        if "day" in k.lower():
            return k
    return None


def visit_day(row, daycol):
    v = str(row.get(daycol, "")).strip()
    if v.isdigit():
        return int(v)
    return day_from_label(v)


# ---------------- load sources ----------------
selected = load_csv(SELECTED)
demo_rows = load_csv(DEMO)
dx_rows   = load_csv(DX)
cdr_rows  = load_csv(CDR)
cog_rows  = load_csv(COG)

print(f"Selected scans : {len(selected)}")
print(f"Demographics   : {len(demo_rows)}")
print(f"D1 diagnoses   : {len(dx_rows)}")
print(f"CDR records    : {len(cdr_rows)}")
print(f"Cognitive recs : {len(cog_rows)}\n")

dx_daycol  = find_day_column(dx_rows)
cdr_daycol = find_day_column(cdr_rows)
cog_daycol = find_day_column(cog_rows)
print(f"Day columns -> D1: {dx_daycol} | CDR: {cdr_daycol} | COG: {cog_daycol}\n")

# demographics keyed by subject
demo = {r["OASISID"].strip(): r for r in demo_rows}

# visits keyed by subject
def index_visits(rows, daycol):
    idx = defaultdict(list)
    for r in rows:
        sid = (r.get("OASISID") or r.get("OASIS_id") or "").strip()
        d = visit_day(r, daycol) if daycol else None
        if sid and d is not None:
            idx[sid].append((d, r))
    for sid in idx:
        idx[sid].sort(key=lambda t: t[0])
    return idx

dx_by_subj  = index_visits(dx_rows, dx_daycol)
cdr_by_subj = index_visits(cdr_rows, cdr_daycol)
cog_by_subj = index_visits(cog_rows, cog_daycol)


def nearest(visits, target_day):
    if not visits:
        return None, None
    best = min(visits, key=lambda t: abs(t[0] - target_day))
    return best[1], abs(best[0] - target_day)


def apoe_e4_count(apoe):
    a = str(apoe).strip()
    if a in ("", "#N/A", "0", "nan", "NA"):
        return None
    return a.count("4")


# ---------------- build cohort ----------------
cohort, excluded = [], []

for s in selected:
    sid = s["subject"]
    scan_day = day_from_label(os.path.basename(s["filename"]).split("_ses-")[1].split("_")[0]
                              .replace("d", "_d")) if False else None
    # scan day comes from the session folder in the path: ses-dXXXX
    ses = [p for p in s["filepath"].split(os.sep) if p.startswith("ses-d")]
    scan_day = int(ses[0].replace("ses-d", "")) if ses else None

    if scan_day is None:
        excluded.append({"subject": sid, "reason": "could not parse scan day"})
        continue

    d = demo.get(sid)
    if d is None:
        excluded.append({"subject": sid, "reason": "no demographics record"})
        continue

    e4 = apoe_e4_count(d.get("APOE"))
    if e4 is None:
        excluded.append({"subject": sid, "reason": f"invalid APOE ({d.get('APOE')})"})
        continue

    dxrow, gap = nearest(dx_by_subj.get(sid, []), scan_day)
    if dxrow is None:
        excluded.append({"subject": sid, "reason": "no D1 diagnosis visit"})
        continue
    if gap > MAX_DAYS:
        excluded.append({"subject": sid,
                         "reason": f"nearest diagnosis {gap} days from scan (> {MAX_DAYS})"})
        continue

    probad  = str(dxrow.get("PROBAD", "")).strip()
    normcog = str(dxrow.get("NORMCOG", "")).strip()

    if probad == "1" and normcog != "1":
        label, label_name = 1, "AD"
    elif normcog == "1" and probad != "1":
        label, label_name = 0, "CN"
    else:
        excluded.append({"subject": sid,
                         "reason": f"not AD or CN at nearest visit "
                                   f"(PROBAD={probad or '-'}, NORMCOG={normcog or '-'}), gap {gap}d"})
        continue

    cdrrow, cdr_gap = nearest(cdr_by_subj.get(sid, []), scan_day)
    cogrow, cog_gap = nearest(cog_by_subj.get(sid, []), scan_day)

    age_entry = d.get("AgeatEntry", "")
    try:
        age_at_scan = round(float(age_entry) + scan_day / 365.25, 1)
    except Exception:
        age_at_scan = ""

    cohort.append({
        "subject": sid,
        "label": label,
        "label_name": label_name,
        "scan_day": scan_day,
        "dx_gap_days": gap,
        "filepath": s["filepath"],
        "filename": s["filename"],
        "shape": f"{s['shape_x']}x{s['shape_y']}x{s['shape_z']}",
        "voxel": f"{s['vox_x']}x{s['vox_y']}x{s['vox_z']}",
        "isotropic": s["isotropic_available"],
        "n_scans_available": s["n_scans_available"],
        "apoe": d.get("APOE", "").strip(),
        "apoe_e4_count": e4,
        "age_at_entry": age_entry,
        "age_at_scan": age_at_scan,
        "sex": d.get("GENDER", "").strip(),      # 1=male, 2=female in OASIS
        "education": d.get("EDUC", "").strip(),
        "ses": d.get("SES", "").strip(),
        "race": d.get("race", "").strip(),
        "handedness": d.get("HAND", "").strip(),
        "cdr": (cdrrow or {}).get("CDRTOT", (cdrrow or {}).get("cdr", "")),
        "cdr_gap_days": cdr_gap if cdrrow else "",
        "mmse": (cdrrow or {}).get("MMSE", ""),
        "mmse_gap_days": cdr_gap if cdrrow else "",
    })

cohort.sort(key=lambda r: r["subject"])

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(cohort[0].keys())); w.writeheader(); w.writerows(cohort)
if excluded:
    with open(OUT_EXCL, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject", "reason"]); w.writeheader(); w.writerows(excluded)

# ---------------- report ----------------
print("=" * 60)
print(f"Final cohort : {len(cohort)} subjects")
print(f"Excluded     : {len(excluded)} subjects\n")

n_ad = sum(1 for r in cohort if r["label"] == 1)
n_cn = len(cohort) - n_ad
print(f"  Alzheimer's (PROBAD) : {n_ad}")
print(f"  Cognitively normal   : {n_cn}")

if excluded:
    print("\n--- Exclusion reasons ---")
    agg = defaultdict(int)
    for e in excluded:
        key = e["reason"].split("(")[0].split(",")[0].strip()
        agg[key] += 1
    for k, v in sorted(agg.items(), key=lambda x: -x[1]):
        print(f"  {v:4d}  {k}")

gaps = sorted(r["dx_gap_days"] for r in cohort)
if gaps:
    print(f"\nScan-to-diagnosis gap (days): median {gaps[len(gaps)//2]}, max {gaps[-1]}")

print("\n--- APOE e4 allele count by group ---")
tab = defaultdict(lambda: [0, 0])
for r in cohort:
    tab[r["apoe_e4_count"]][r["label"]] += 1
print(f"  {'e4 copies':<10}{'CN':>6}{'AD':>6}")
for k in sorted(tab):
    print(f"  {k:<10}{tab[k][0]:>6}{tab[k][1]:>6}")

print(f"\nWrote {OUT}" + (f" and {OUT_EXCL}" if excluded else ""))