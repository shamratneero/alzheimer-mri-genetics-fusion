"""
Build an ADNI cohort table matching the OASIS-3 AD/CN definition.

WHY THIS IS NOT A ONE-LINER
---------------------------
ADNI's diagnosis coding is not uniform across phases, and the naive mapping
loses most of the AD cases:

  DIAGNOSIS  (all phases)  1=CN, 2=MCI, 3=Dementia
                           -> note "3" is dementia of ANY cause, not AD.

  Distinguishing Alzheimer's dementia from other dementias uses DIFFERENT
  columns depending on phase:
    ADNI1                : DXAD    (1=Alzheimer's Disease)
                           DXOTHDEM(1=Other dementia, not AD)
    ADNI2/GO/3/4         : DXDDUE  (1=Dementia due to AD, 2=other etiology)

  DXAD is populated ONLY in ADNI1 (1,132 records); it is 100% null for the
  1,906 dementia records in ADNI2/GO/3/4. Filtering on DXAD alone would
  silently discard ~63% of the AD cohort. DXDDUE covers the rest.

This matters because the project's task is strictly Alzheimer's disease vs
cognitively normal (matching OASIS-3 PROBAD vs NORMCOG). Non-AD dementias are
excluded, since APOE is an Alzheimer's-specific risk gene.

SUBJECT-LEVEL RULE
------------------
ADNI is longitudinal and subjects convert between states. Counting visits
instead of subjects would both inflate n and leak the same person across
train/test splits. The rule here mirrors the OASIS-3 cohort (one scan per
subject, stable diagnosis):

  AD  : subject has >=1 Alzheimer's-dementia visit and NEVER a non-AD dementia
        visit. Label date = FIRST AD visit (earliest confirmed diagnosis).
  CN  : subject is cognitively normal at EVERY visit - never MCI, never any
        dementia. Stable normals only.
  Excluded: MCI-only subjects, any subject with non-AD dementia, and CN->AD
        converters are handled by the above (they qualify as AD, dated from
        their first AD visit).

APOE
----
APOERES gives two alleles (APGEN1/APGEN2). e4 count = number of alleles equal
to 4, giving 0/1/2 exactly as in the OASIS-3 cohort.

OUTPUT
------
adni_cohort.csv with the same columns the existing pipeline expects:
  subject, label, label_name, age_at_scan, sex, education, apoe_e4_count, apoe
plus ADNI-specific provenance columns (RID, PHASE, EXAMDATE, dx_source).

Imaging is NOT downloaded by this script. Build the cohort first, then fetch
T1w scans only for the subjects listed here.
"""
import pandas as pd
import numpy as np

DXSUM = "DXSUM_18Aug2026.csv"
APOE = "APOERES_18Aug2026.csv"
DEMOG = "PTDEMOG_18Aug2026.csv"
OUT = "adni_cohort.csv"


# ---------------------------------------------------------------- diagnosis
def classify_visit(row):
    """Return 'AD', 'OTHER_DEM', 'CN', 'MCI' or None for a single visit."""
    dx = row["DIAGNOSIS"]
    if dx == 1:
        return "CN"
    if dx == 2:
        return "MCI"
    if dx == 3:
        # dementia - but of what cause? column differs by phase
        if row["PHASE"] == "ADNI1":
            if row["DXAD"] == 1:
                return "AD"
            if row["DXOTHDEM"] == 1:
                return "OTHER_DEM"
        else:
            if row["DXDDUE"] == 1:
                return "AD"
            if row["DXDDUE"] == 2:
                return "OTHER_DEM"
        return None          # dementia of unspecified cause - excluded
    return None


def build_diagnosis_table():
    dx = pd.read_csv(DXSUM, low_memory=False)
    dx["visit_dx"] = dx.apply(classify_visit, axis=1)
    dx["EXAMDATE"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")

    print(f"DXSUM: {len(dx):,} visit records over {dx.PTID.nunique():,} subjects")
    print("  visit-level classification:")
    for k, v in dx.visit_dx.value_counts(dropna=False).items():
        print(f"    {str(k):12s} {v:6,d}")

    rows = []
    for ptid, g in dx.groupby("PTID"):
        states = set(g.visit_dx.dropna())

        if "OTHER_DEM" in states:
            continue                       # non-AD dementia anywhere -> exclude

        if "AD" in states:
            ad = g[g.visit_dx == "AD"].sort_values("EXAMDATE")
            first = ad.iloc[0]
            rows.append({
                "subject": ptid, "RID": first["RID"], "label": 1,
                "label_name": "AD", "PHASE": first["PHASE"],
                "EXAMDATE": first["EXAMDATE"], "VISCODE": first["VISCODE"],
                "dx_source": "DXAD" if first["PHASE"] == "ADNI1" else "DXDDUE",
                "n_visits": len(g),
            })
        elif states == {"CN"}:
            # stable normal at every visit - no MCI, no dementia, ever
            cn = g[g.visit_dx == "CN"].sort_values("EXAMDATE")
            first = cn.iloc[0]
            rows.append({
                "subject": ptid, "RID": first["RID"], "label": 0,
                "label_name": "CN", "PHASE": first["PHASE"],
                "EXAMDATE": first["EXAMDATE"], "VISCODE": first["VISCODE"],
                "dx_source": "DIAGNOSIS", "n_visits": len(g),
            })

    coh = pd.DataFrame(rows)
    print(f"\n  subject-level: {(coh.label==1).sum():,} AD / {(coh.label==0).sum():,} CN"
          f"  (total {len(coh):,})")
    return coh


# --------------------------------------------------------------------- APOE
def add_apoe(coh):
    """ADNI stores APOE as a single GENOTYPE string like '3/4' (one row per
    subject), unlike some older ADNI exports that split it into APGEN1/APGEN2.
    e4 count = number of '4' alleles, matching the OASIS-3 apoe_e4_count."""
    ap = pd.read_csv(APOE, low_memory=False)
    if "GENOTYPE" not in ap.columns:
        raise SystemExit(f"GENOTYPE column not found; have {list(ap.columns)}")

    ap = ap.dropna(subset=["GENOTYPE"]).drop_duplicates("RID", keep="first").copy()
    alleles = ap["GENOTYPE"].str.split("/", expand=True)
    ap["apoe_e4_count"] = ((alleles[0] == "4").astype(int) +
                           (alleles[1] == "4").astype(int))
    ap["apoe"] = ap["GENOTYPE"]

    before = len(coh)
    coh = coh.merge(ap[["RID", "apoe_e4_count", "apoe"]], on="RID", how="inner")
    print(f"\nAPOE: {len(coh):,} of {before:,} subjects have genotype "
          f"({before-len(coh):,} dropped)")
    return coh


# ------------------------------------------------------------- demographics
def add_demographics(coh):
    dm = pd.read_csv(DEMOG, low_memory=False)

    keep = ["RID"]
    ren = {}
    if "PTGENDER" in dm.columns:
        keep.append("PTGENDER"); ren["PTGENDER"] = "sex"
    if "PTEDUCAT" in dm.columns:
        keep.append("PTEDUCAT"); ren["PTEDUCAT"] = "education"
    if "PTDOB" in dm.columns:
        keep.append("PTDOB")
    elif "PTDOBYY" in dm.columns:
        keep.append("PTDOBYY")

    dm = dm[keep].drop_duplicates("RID", keep="first").rename(columns=ren)
    before = len(coh)
    coh = coh.merge(dm, on="RID", how="left")

    # age at the diagnosis visit, from year of birth
    if "PTDOBYY" in coh.columns:
        coh["age_at_scan"] = coh["EXAMDATE"].dt.year - coh["PTDOBYY"]
    elif "PTDOB" in coh.columns:
        dob = pd.to_datetime(coh["PTDOB"], errors="coerce")
        coh["age_at_scan"] = (coh["EXAMDATE"] - dob).dt.days / 365.25

    print(f"demographics: merged for {coh['sex'].notna().sum():,} of {before:,}")
    return coh


# --------------------------------------------------------------------- main
def main():
    print("=" * 66)
    print("Building ADNI cohort (AD vs CN, matching the OASIS-3 definition)")
    print("=" * 66 + "\n")

    coh = build_diagnosis_table()
    coh = add_apoe(coh)
    coh = add_demographics(coh)

    cols = ["subject", "RID", "label", "label_name", "age_at_scan", "sex",
            "education", "apoe_e4_count", "apoe", "PHASE", "EXAMDATE",
            "VISCODE", "dx_source", "n_visits"]
    coh = coh[[c for c in cols if c in coh.columns]]
    coh = coh.sort_values("subject").reset_index(drop=True)

    print("\n" + "=" * 66)
    print("FINAL COHORT")
    print("=" * 66)
    print(f"  subjects      : {len(coh):,}  ({(coh.label==1).sum():,} AD / "
          f"{(coh.label==0).sum():,} CN)")
    print(f"  unique?       : {coh.subject.nunique() == len(coh)}")

    if "age_at_scan" in coh:
        for lab, name in [(1, "AD"), (0, "CN")]:
            s = coh[coh.label == lab]["age_at_scan"]
            print(f"  age {name}        : {s.mean():.1f} +/- {s.std():.1f}")

    print("\n  APOE e4 carrier rate (validates label linkage; "
          "expect AD >> CN):")
    for lab, name in [(1, "AD"), (0, "CN")]:
        s = coh[coh.label == lab]["apoe_e4_count"]
        print(f"    {name}: {(s > 0).mean()*100:.1f}%  "
              f"(e4 count mean {s.mean():.2f})")

    print("\n  missing values:")
    miss = coh.isna().sum()
    print("   ", (miss[miss > 0].to_dict() or "none"))

    print(f"\n  phases: {coh.PHASE.value_counts().to_dict()}")

    coh.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")
    print("\nNEXT: download T1w MRI for these subject IDs only, then preprocess")
    print("with the same pipeline as OASIS-3 (deepbet, 1mm iso, 160^3 cache).")


if __name__ == "__main__":
    main()
