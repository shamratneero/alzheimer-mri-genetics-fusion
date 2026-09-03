"""
Age-shift diagnostic: OASIS-3 (5 CV folds) vs ADNI.

Purpose
-------
Before running external validation, quantify (not correct) any age-distribution
shift between OASIS-3 (training) and ADNI (external test), and show what that
shift looks like once each fold's frozen OASIS-3 age-normalization statistics
are applied to ADNI. This produces the numbers for the methods paragraph:
"we retained fold-specific OASIS-3 normalization parameters during external
validation rather than recalibrating on ADNI."

Reproduces the EXACT fold splits used in cross_validate.py:
  StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  -> train_test_split(test_size=0.125, stratify=label, random_state=42)
The age_mean/age_std frozen into each checkpoint's `clinical_norm` are computed
on the resulting inner TRAIN split only (not train+val), matching
OASIS3Dataset.get_clinical_norm() called on train_ds in run_fold().

No dependency on outputs/cv_checkpoints/ (whose exact filenames are still
unresolved) - the split is deterministic from the seed, so this is
reproducible from the two cohort CSVs alone.
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

OASIS_COHORT = "oasis3_cohort.csv"
ADNI_COHORT = "adni_scans_selected.csv"
SEED = 42
N_FOLDS = 5
VAL_FRAC = 0.125

# Subjects excluded from the final ADNI analysis set: PET series returned
# instead of T1 (sites 057, 098, 126), caught by dimension/voxel-size audit
# before preprocessing. See conversation record / RESULTS.md.
ADNI_EXCLUDED_PET = [
    "057_S_6746", "057_S_6869",
    "098_S_6343", "098_S_6601", "098_S_6655", "098_S_6658",
    "126_S_6683",
]


def describe(age_series):
    return {
        "n": int(age_series.shape[0]),
        "mean_age": round(float(age_series.mean()), 3),
        "sd_age": round(float(age_series.std()), 3),
    }


def main():
    oasis = pd.read_csv(OASIS_COHORT)
    oasis["label"] = oasis["label"].astype(int)

    adni = pd.read_csv(ADNI_COHORT)
    adni["label"] = adni["label"].astype(int)
    # strict cohort (<=365d), the target analysis set
    adni = adni[adni["within_365d"] == True].reset_index(drop=True)
    n_before_pet = len(adni)
    adni = adni[~adni["subject"].isin(ADNI_EXCLUDED_PET)].reset_index(drop=True)
    n_after_pet = len(adni)

    print("=" * 78)
    print("PART 1: overall cohort age distributions")
    print("=" * 78)
    header = f"{'Cohort':<10}{'Group':<8}{'N':>6}{'Mean age':>12}{'SD age':>10}"
    print(header)
    print("-" * len(header))

    rows = []
    for name, df in [("OASIS-3", oasis), ("ADNI", adni)]:
        for grp_label, grp_name in [(None, "All"), (0, "CN"), (1, "AD")]:
            sub = df if grp_label is None else df[df["label"] == grp_label]
            d = describe(sub["age_at_scan"].astype(float))
            rows.append((name, grp_name, d))
            print(f"{name:<10}{grp_name:<8}{d['n']:>6}{d['mean_age']:>12.2f}{d['sd_age']:>10.2f}")

    print(f"\nADNI strict cohort: {n_before_pet} subjects before PET exclusion, "
          f"{n_after_pet} after ({n_before_pet - n_after_pet} excluded).")

    print()
    print("=" * 78)
    print("PART 2: per-fold OASIS-3 training-set stats, applied to ADNI")
    print("=" * 78)
    print("(Reproduces cross_validate.py's build_cv_folds exactly - seed=42.")
    print(" age_mean/age_std below are computed on the INNER TRAIN split only,")
    print(" matching what OASIS3Dataset.get_clinical_norm() stores per fold.)\n")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    adni_age = adni["age_at_scan"].astype(float)
    adni_cn_age = adni[adni["label"] == 0]["age_at_scan"].astype(float)
    adni_ad_age = adni[adni["label"] == 1]["age_at_scan"].astype(float)

    fold_stats = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(oasis, oasis["label"]), 1):
        trainval = oasis.iloc[train_idx].reset_index(drop=True)
        train_df, val_df = train_test_split(
            trainval, test_size=VAL_FRAC, stratify=trainval["label"], random_state=SEED)

        age_mean = float(train_df["age_at_scan"].astype(float).mean())
        age_std = float(train_df["age_at_scan"].astype(float).std())

        def z(series):
            return (series - age_mean) / (age_std + 1e-6)

        z_all = z(adni_age)
        z_cn = z(adni_cn_age)
        z_ad = z(adni_ad_age)

        fold_stats.append({
            "fold": fold_idx,
            "train_n": len(train_df),
            "age_mean": round(age_mean, 3),
            "age_std": round(age_std, 3),
            "adni_z_all_mean": round(float(z_all.mean()), 3),
            "adni_z_cn_mean": round(float(z_cn.mean()), 3),
            "adni_z_ad_mean": round(float(z_ad.mean()), 3),
        })

    hdr = (f"{'Fold':<6}{'TrainN':>8}{'age_mean':>10}{'age_std':>10}"
           f"{'z(ADNI all)':>14}{'z(ADNI CN)':>13}{'z(ADNI AD)':>13}")
    print(hdr)
    print("-" * len(hdr))
    for fs in fold_stats:
        print(f"{fs['fold']:<6}{fs['train_n']:>8}{fs['age_mean']:>10.2f}"
              f"{fs['age_std']:>10.2f}{fs['adni_z_all_mean']:>14.3f}"
              f"{fs['adni_z_cn_mean']:>13.3f}{fs['adni_z_ad_mean']:>13.3f}")

    mean_z_all = sum(f["adni_z_all_mean"] for f in fold_stats) / len(fold_stats)
    print(f"\nMean z(ADNI, all subjects) across 5 folds: {mean_z_all:.3f}")
    print("(0.0 would mean no shift - i.e. ADNI's average age happens to land")
    print(" exactly on OASIS-3's training-fold average age.)")

    return rows, fold_stats


if __name__ == "__main__":
    main()
