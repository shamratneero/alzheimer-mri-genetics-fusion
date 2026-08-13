"""
Count how many OASIS-3 MR sessions (not subjects) qualify as AD or CN
at scan time, to gauge whether a session-level dataset would be worthwhile.
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

def e4(a):
    a = str(a).strip()
    return None if a in ("", "#N/A", "0", "nan", "NA") else a.count("4")

demo = {r["OASISID"].strip(): r for r in load(DEMO)}

dx_by_subj = defaultdict(list)
for r in load(DX):
    sid, d = r["OASISID"].strip(), str(r.get("days_to_visit", "")).strip()
    if sid and d.isdigit():
        dx_by_subj[sid].append((int(d), r))

def sess_day(lbl):
    try: return int(lbl.rsplit("_d", 1)[1])
    except Exception: return None

t1w = defaultdict(set)
for r in load(MRI):
    if r.get("scan category", "").strip().lower() != "t1w":
        continue
    sid, lbl = r["subject_id"].strip(), r["label"].strip()
    if sess_day(lbl) is not None:
        t1w[sid].add(lbl)

ad_sessions, cn_sessions = [], []
ad_subj, cn_subj = set(), set()

for sid, sessions in t1w.items():
    d = demo.get(sid)
    if not d or e4(d.get("APOE")) is None: continue
    visits = dx_by_subj.get(sid, [])
    if not visits: continue
    for lbl in sessions:
        day = sess_day(lbl)
        vday, vrow = min(visits, key=lambda t: abs(t[0] - day))
        if abs(vday - day) > MAX_DAYS: continue
        probad  = str(vrow.get("PROBAD", "")).strip()
        normcog = str(vrow.get("NORMCOG", "")).strip()
        if probad == "1" and normcog != "1":
            ad_sessions.append((sid, lbl)); ad_subj.add(sid)
        elif normcog == "1" and probad != "1":
            cn_sessions.append((sid, lbl)); cn_subj.add(sid)

have = set()
if os.path.exists("oasis3_cohort.csv"):
    have = {r["subject"] for r in load("oasis3_cohort.csv")}

print("="*60)
print("SESSION-LEVEL POOL (scan-time dx, valid APOE, T1w exists)")
print("="*60)
print(f"  AD sessions : {len(ad_sessions):5d}  across {len(ad_subj)} subjects")
print(f"  CN sessions : {len(cn_sessions):5d}  across {len(cn_subj)} subjects")
print()
print(f"  Mean sessions per AD subject : {len(ad_sessions)/max(len(ad_subj),1):.2f}")
print(f"  Mean sessions per CN subject : {len(cn_sessions)/max(len(cn_subj),1):.2f}")
print()
print(f"  You currently use 1 session for each of {len(have)} subjects")

new_ad = sum(1 for s, _ in ad_sessions if s not in have)
new_cn = sum(1 for s, _ in cn_sessions if s not in have)
print(f"\n  Sessions from subjects NOT in your cohort:")
print(f"    AD : {new_ad}")
print(f"    CN : {new_cn}")

extra_ad = sum(1 for s, _ in ad_sessions if s in have)
extra_cn = sum(1 for s, _ in cn_sessions if s in have)
print(f"\n  Sessions from subjects ALREADY in your cohort:")
print(f"    AD : {extra_ad}  (you use {len([s for s in have])} total, 1 each)")
print(f"    CN : {extra_cn}")