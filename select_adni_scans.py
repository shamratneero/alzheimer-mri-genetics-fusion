"""
Pick exactly ONE T1w scan per ADNI cohort subject.

WHY THIS IS NEEDED
------------------
The IDA search returns 14,644 scans over 1,699 subjects - about 9 scans each,
because ADNI is longitudinal and most sessions contain several acquisitions.
Downloading all of them would waste ~90% of the bandwidth and would then need
de-duplicating anyway.

The OASIS-3 cohort used ONE scan per subject, chosen nearest the clinical
visit that supplied the label. This script applies the same rule to ADNI so
the two cohorts are directly comparable.

TWO TRAPS IN THE RAW SEARCH RESULTS
-----------------------------------
1. Non-brain scans. "B1-Calibration Body", "B1-Calibration PA",
   "B1-calibration Head", "Field Mapping" and "Calibration Scan" are scanner
   calibration acquisitions, not anatomy. They pass a T1/3D filter but are
   useless here, and there are ~3,700 of them.

2. Sequence names are not uniform. An `MPRAGE*` filter misses most of the
   cohort, because ADNI also uses "MP-RAGE" (hyphen), "Accelerated Sagittal
   MPRAGE", "Sag IR-FSPGR", "Sag IR-SPGR", "MPRAGE GRAPPA2", "MPRAGE SENSE2"
   and more, varying by vendor and phase. Matching is therefore done by
   excluding known non-anatomical scans rather than by listing every valid
   sequence name.

SELECTION RULE (in order)
-------------------------
  1. drop calibration / field-mapping / localiser acquisitions
  2. keep only subjects present in adni_cohort.csv
  3. prefer a plain acquisition over a "REPEAT" re-run of the same session
     (the repeat exists because the first was usually judged adequate; where
     only a repeat exists it is kept)
  4. among what remains, take the scan whose Study Date is closest to the
     subject's diagnosis EXAMDATE
  5. record the gap in days so a max-gap threshold can be applied later,
     mirroring the 365-day rule used for OASIS-3
"""
import pandas as pd
import numpy as np

IDA = "ida4.csv"
COHORT = "adni_cohort.csv"
OUT = "adni_scans_selected.csv"
OUT_IDS = "adni_image_ids.txt"

# acquisitions that are not brain anatomy
NON_ANATOMICAL = [
    "calibration", "field mapping", "localizer", "localiser", "scout",
    "b1-", "b1_", "phantom",
]


def main():
    ida = pd.read_csv(IDA)
    coh = pd.read_csv(COHORT)

    print(f"IDA search results : {len(ida):,} scans, "
          f"{ida['Subject ID'].nunique():,} subjects")
    print(f"cohort             : {len(coh):,} subjects")

    # ---- 1. drop non-anatomical acquisitions --------------------------------
    desc = ida["Description"].str.lower()
    bad = pd.Series(False, index=ida.index)
    for pat in NON_ANATOMICAL:
        bad |= desc.str.contains(pat, regex=False, na=False)
    print(f"\ndropping {bad.sum():,} calibration/field-map/localiser scans")
    ida = ida[~bad].copy()
    print(f"  remaining: {len(ida):,} scans, {ida['Subject ID'].nunique():,} subjects")

    # ---- 2. restrict to cohort subjects -------------------------------------
    ida = ida[ida["Subject ID"].isin(set(coh["subject"]))].copy()
    print(f"\nrestricted to cohort: {len(ida):,} scans, "
          f"{ida['Subject ID'].nunique():,} subjects")

    # ---- 3. flag repeats ----------------------------------------------------
    ida["is_repeat"] = ida["Description"].str.lower().str.contains("repeat", na=False)

    # ---- 4. match to the diagnosis date -------------------------------------
    ida["Study Date"] = pd.to_datetime(ida["Study Date"], errors="coerce")
    coh["EXAMDATE"] = pd.to_datetime(coh["EXAMDATE"], errors="coerce")

    m = ida.merge(coh[["subject", "label", "label_name", "EXAMDATE", "PHASE"]],
                  left_on="Subject ID", right_on="subject", how="inner")
    m["gap_days"] = (m["Study Date"] - m["EXAMDATE"]).dt.days.abs()

    # prefer non-repeat, then smallest gap
    m = m.sort_values(["subject", "is_repeat", "gap_days"])
    sel = m.groupby("subject", as_index=False).first()

    print(f"\nselected one scan for {len(sel):,} subjects")
    print(f"  AD {(sel.label==1).sum():,} / CN {(sel.label==0).sum():,}")

    # ---- 5. report the date gap --------------------------------------------
    print(f"\nscan-to-diagnosis gap (days):")
    for q in [0.5, 0.75, 0.9, 0.95, 1.0]:
        print(f"    {int(q*100):3d}th percentile: {sel.gap_days.quantile(q):,.0f}")
    within = (sel.gap_days <= 365).sum()
    print(f"  within 365 days (the OASIS-3 rule): {within:,} "
          f"({within/len(sel)*100:.1f}%)")

    sel["within_365d"] = sel.gap_days <= 365
    strict = sel[sel.within_365d]
    print(f"    -> strict subset: AD {(strict.label==1).sum():,} / "
          f"CN {(strict.label==0).sum():,}  (total {len(strict):,})")

    print(f"\nfield strength: {sel['Imaging Protocol'].value_counts().to_dict()}")
    print(f"phase: {sel['Phase'].value_counts().to_dict()}")
    print(f"\ntop sequences chosen:")
    for k, v in sel["Description"].value_counts().head(8).items():
        print(f"    {k:40s} {v:,}")

    out = sel[["subject", "label", "label_name", "Image ID", "Study Date",
               "EXAMDATE", "gap_days", "within_365d", "Description",
               "Imaging Protocol", "Phase"]].sort_values("subject")
    out.to_csv(OUT, index=False)

    # image IDs for the IDA download step
    with open(OUT_IDS, "w") as fh:
        fh.write(",".join(str(i) for i in out["Image ID"]))

    print(f"\nWrote {OUT} ({len(out):,} rows)")
    print(f"Wrote {OUT_IDS} - paste into the IDA 'Image ID' search box to")
    print("fetch exactly these scans (in batches if the box has a length limit).")


if __name__ == "__main__":
    main()
