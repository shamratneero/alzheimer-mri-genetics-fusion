# Alzheimer's Disease Classification: MRI + Genetics Fusion

Multimodal Alzheimer's disease classification from structural MRI and APOE
genotype, with an emphasis on **whether reported benchmark performance
corresponds to deployable reliability**.

The prediction task is strictly Alzheimer's disease (`PROBAD` vs `NORMCOG`),
not general dementia. Non-AD dementias are excluded, since APOE is an
Alzheimer's-specific risk gene and mixing aetiologies would confound both the
genetic and structural signal.

---

## Headline finding

On OASIS-3 (365 subjects, 5-fold cross-validation), **the way results are
conventionally reported reverses the conclusion**:

| mode | per-fold mean AUC | pooled out-of-fold AUC | mean ECE |
|---|---|---|---|
| clinical only (APOE + age) | 0.786 | 0.7638 | 0.0955 |
| clinical only (5 features) | 0.770 | **0.7675** | 0.126 |
| imaging only (3D CNN) | 0.849 | 0.729 | 0.225 |
| fusion | **0.869** | **0.670** | 0.309 |

Ranked by mean per-fold AUC — the near-universal convention in this literature
— fusion wins. Ranked by pooled out-of-fold AUC over the same predictions,
fusion is **significantly worst** (vs clinical: ΔAUC 0.094, p=0.003; vs imaging:
ΔAUC 0.059, p=0.039; paired bootstrap on identical subjects).

The discrepancy is diagnostic rather than cosmetic. Per-fold AUC measures
ranking *within* a fold; pooled AUC additionally requires the fold-models to put
probabilities on a *comparable scale*. The imaging-based models fail that second
requirement, and per-fold reporting cannot see it.

Two APOE/age scalars outperform a 3.57M-parameter 3D CNN over 128³ volumes —
and so does the 5-feature clinical model, which beats fusion by a slightly wider
margin (+0.097, p=0.004). Adding sex, education and SES to APOE and age moves
pooled AUC by +0.004 (p=0.856), so the non-imaging signal here is APOE and age
and essentially nothing else.

See [`RESULTS.md`](RESULTS.md) for the full experimental record.

---

## Why this matters

A model can rank patients well (high AUC) while being unusable at any fixed
decision threshold. Concretely, in imaging fold 2:

- test AUC **0.846** — the highest of any imaging fold
- balanced accuracy **0.500** — chance
- sensitivity **0.000** — it detected none of the 39 AD cases
- ECE **0.480**

That checkpoint was selected on validation AUC (0.947, the run's highest) at an
epoch where validation balanced accuracy had **already collapsed to 0.500**. The
selection criterion actively preferred a model that was visibly broken on the
data it was selecting from.

Reported as most papers report — AUC and accuracy on a single split — this would
have looked like the best model in the study.

---

## Repository layout

### Pipeline (in order of use)

| script | purpose |
|---|---|
| `download_oasis.py` | fetch OASIS-3 T1w scans from NITRC-IR (BIDS) |
| `select_scans.py` | choose one scan per subject; nearest diagnosis within 365 days |
| `build_labels.py` / `prepare_labels.py` | derive AD/CN labels, join APOE + demographics |
| `check_integrity.py`, `count_sessions.py`, `count_available.py` | download/inventory sanity checks |
| `preprocess.py` | deepbet skull-strip, 1mm isotropic resample, z-score in mask, cache 160³ `.npy` |
| `qc_visualize.py`, `test_deepbet.py` | visual QC of skull-stripping |
| `dataset_3d.py` | dataset, augmentation, train-only age normalisation, split construction |
| `model_3d.py` | 3D CNN + clinical MLP, per-mode classifier heads |
| `train.py` | single-split training; `--imaging_only` / `--clinical_only` / fusion |
| `cross_validate.py` | 5-fold CV, pooled out-of-fold metrics, paired bootstrap comparison |

### OASIS-1 baseline (earlier phase)

`dataset.py`, `train_baseline.py`, `evaluate.py` — 2D ResNet-18 experiments on
pre-sliced OASIS-1 JPEGs. Retained as a documented reference point
(best: 80.5% test accuracy), not as a headline result.

### Data and results

- `oasis3_cohort.csv` — final 365-subject cohort with labels, APOE, demographics
- `preprocess_qc.csv` — per-subject preprocessing QC
- `baseline_demographics_results.csv` — non-imaging logistic regression baselines
- `table1.csv` / `table1.txt` — cohort characteristics with confound checks
- `outputs/cv/` — cross-validation results (per-fold, summary, out-of-fold predictions)
- `eda.ipynb`, `exploration.ipynb` — exploratory analysis

---

## Reproducing

```bash
# single-split ablations
python train.py --imaging_only  --tag imaging_ablation
python train.py --clinical_only --tag clinical_ablation
python train.py                 --tag fusion

# 5-fold cross-validation (the primary result)
python cross_validate.py --mode clinical_only --tag cv_clinical_only
python cross_validate.py --mode clinical_only --tag cv_clinical_5feat --extended_clinical
python cross_validate.py --mode imaging_only  --tag cv_imaging_only  --resume
python cross_validate.py --mode fusion        --tag cv_fusion        --resume

# paired comparison across modes
python cross_validate.py --compare cv_clinical_only cv_clinical_5feat cv_imaging_only cv_fusion
```

All training entry points are resume-safe (`--resume`); `train.py` checkpoints
every epoch and `cross_validate.py` every fold.

Hardware used: RTX 4060 (8GB), Ryzen 5 3600, 24GB RAM. Imaging folds run roughly
5 hours each; clinical folds a few minutes.

---

## Method notes

- **Splits** are stratified and subject-level; no subject appears in more than
  one split, and every subject is tested exactly once across CV folds.
- **Age normalisation** uses training-split statistics only.
- **Ablations share one pipeline.** Each mode has its own correctly-sized
  classifier head rather than zero-padding into the fusion head — an earlier
  version of this code did the latter, which handicapped the clinical branch by
  ~4.4x at initialisation (`nn.Linear` scales weights by 1/sqrt(fan_in)).
  Gradient isolation between branches is verified.
- **Both branch outputs are LayerNorm'd.** Global average pooling left imaging
  features at std ~0.005 against the clinical branch's ~0.21 — a ~45x imbalance
  that, combined with the head defect above, partially cancelled and produced
  misleading early fusion results.
- **Calibration is measured throughout**, not post hoc: ECE and Brier score are
  logged every epoch alongside AUC and balanced accuracy.

---

## Status and limitations

Current results are **internal to OASIS-3** and use **one architecture**. The
open question is whether the instability reported here is a small-cohort
artefact or a property of the approach; that cannot be settled from a single
cohort.

Planned:

1. ADNI external validation (access approved), then ADNI internal CV at
   n≈1,200 — the direct test of the sample-size explanation
2. Reverse direction (train ADNI → test OASIS-3) and pooled-cohort CV
3. Additional architectures: MedicalNet-pretrained 3D ResNet, 2D-slice
   ImageNet-pretrained ResNet-18
4. Calibration-aware checkpoint selection and temperature scaling as candidate
   remedies
5. GradCAM — whether attention falls on hippocampal / medial temporal regions
   or on artefacts

Known limitations: n=365 caps absolute performance; OASIS-3 labels derive from
CDR whereas ADNI/AIBL use physician-assigned NINCDS-ADRDA criteria, so
cross-cohort label equivalence is not exact.

---

## Data access

OASIS-3 is available from [oasis-brains.org](https://www.oasis-brains.org/) via
NITRC-IR. ADNI and AIBL are available through
[LONI/IDA](https://ida.loni.usc.edu/). All require separate data use agreements;
no imaging data is redistributed in this repository.