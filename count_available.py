"""
Count how many OASIS-3 subjects have an MRI scan taken at a time when they
were diagnosed with probable Alzheimer's (or were cognitively normal),
using the same nearest-visit rule as the cohort builder.

This tells us the true size of the available pool before downloading more.
"""
import csv, os
from collections import defaultdict

BASE = r"D:\alhseimer\OASIS3_data_files\OASIS3_data_files\scans"
DEMO = os.path.join(BASE, r"demo-demographics\resources\csv\files\OASIS3_demographics.csv")
DX   = os.path.join(BASE, r"UDSd1-Form_D1__Clinician_Diagnosis___Cognitive_Status_and_Dementia\resources\csv\files\OASIS3_UDSd1_diagnoses.csv")
MRI  = os.path.join(BASE, r"MRI-json-MRI_json_information\resources\csv\files\OASIS3_MR_json.csv")

MAX_DAYS = 365

def load(p):
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))

demo_rows, dx_rows, mri_rows = load(DEMO), load(DX), load(MRI)

# APOE lookup
def e4(a):
    a = str(a).strip()
    return None if a in ("", "#N/A", "0", "nan", "NA") else a.count("4")

demo = {r["OASISID"].strip(): r for r in demo_rows}

# diagnosis visits per subject
dx_by_subj = defaultdict(list)
for r in dx_rows:
    sid = r["OASISID"].strip()
    d = str(r.get("days_to_visit", "")).strip()
    if sid and d.isdigit():
        dx_by_subj[sid].append((int(d), r))
for s in dx_by_subj: dx_by_subj[s].sort()

# T1w MR sessions per subject
def sess_day(label):
    try: return int(label.rsplit("_d", 1)[1])
    except Exception: return None

t1w_by_subj = defaultdict(list)
for r in mri_rows:
    if r.get("scan category", "").strip().lower() != "t1w":
        continue
    sid = r["subject_id"].strip()
    lbl = r["label"].strip()
    d = sess_day(lbl)
    if sid and d is not None:
        t1w_by_subj[sid].append((d, lbl))
for s in t1w_by_subj: t1w_by_subj[s] = sorted(set(t1w_by_subj[s]))

# For each subject, find scan sessions whose nearest diagnosis is AD or CN
ad_subjects, cn_subjects = {}, {}
no_apoe = mci_or_other = no_dx = 0

for sid, sessions in t1w_by_subj.items():
    d = demo.get(sid)
    if not d or e4(d.get("APOE")) is None:
        no_apoe += 1
        continue
    visits = dx_by_subj.get(sid, [])
    if not visits:
        no_dx += 1
        continue

    best_ad = best_cn = None
    for day, lbl in sessions:
        vday, vrow = min(visits, key=lambda t: abs(t[0] - day))
        if abs(vday - day) > MAX_DAYS:
            continue
        probad  = str(vrow.get("PROBAD", "")).strip()
        normcog = str(vrow.get("NORMCOG", "")).strip()
        if probad == "1" and normcog != "1":
            if best_ad is None or abs(vday - day) < best_ad[1]:
                best_ad = (lbl, abs(vday - day))
        elif normcog == "1" and probad != "1":
            if best_cn is None or abs(vday - day) < best_cn[1]:
                best_cn = (lbl, abs(vday - day))

    if best_ad: ad_subjects[sid] = best_ad[0]
    elif best_cn: cn_subjects[sid] = best_cn[0]
    else: mci_or_other += 1

print("=" * 60)
print("AVAILABLE POOL (scan-time diagnosis, valid APOE, T1w exists)")
print("=" * 60)
print(f"  Probable Alzheimer's at scan time : {len(ad_subjects)}")
print(f"  Cognitively normal at scan time   : {len(cn_subjects)}")
print()
print(f"  Skipped - no valid APOE           : {no_apoe}")
print(f"  Skipped - no diagnosis record     : {no_dx}")
print(f"  Skipped - MCI/other at all scans  : {mci_or_other}")

# what we already have
have = set()
if os.path.exists("oasis3_cohort.csv"):
    for r in load("oasis3_cohort.csv"):
        have.add(r["subject"])

new_ad = [s for s in ad_subjects if s not in have]
new_cn = [s for s in cn_subjects if s not in have]
print()
print(f"  Already downloaded & usable       : {len(have)}")
print(f"  NEW AD subjects available         : {len(new_ad)}")
print(f"  NEW CN subjects available         : {len(new_cn)}")

# write a download list for the new AD subjects
with open("oasis3_additional_AD.csv", "w", newline="\n") as fh:
    for s in sorted(new_ad):
        fh.write(ad_subjects[s] + "\n")
print(f"\nWrote oasis3_additional_AD.csv ({len(new_ad)} sessions)")