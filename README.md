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

### The failure is a selection artefact, and it has a remedy

Selecting checkpoints on validation **Brier score** rather than validation AUC —
same runs, same folds, same epochs available — largely repairs the incoherence:

| mode | gap under AUC selection | gap under calibration-aware selection |
|---|---|---|
| clinical | 0.0222 | -0.0053 |
| imaging | 0.1200 | 0.0227 |
| fusion | **0.1989** | **0.0408** |

Degenerate folds (test balanced accuracy exactly 0.500, one class predicted for
every subject) go from one per mode to **zero** under every calibration-aware
criterion. The apparent inferiority of multimodal fusion is therefore
substantially an artefact of checkpoint selection, not a property of fusion.

### It reproduces on an independent cohort

Applying the frozen OASIS-3 fold-models to **ADNI (n=1,287; 433 AD / 854 CN)**,
with normalisation statistics frozen from the OASIS-3 training splits and no
adaptation of any kind:

| mode | selection | per-fold mean AUC | cross-fold pooled-style AUC | gap |
|---|---|---|---|---|
| clinical | auc | 0.7795 | 0.7407 | 0.0388 |
| clinical | neg_brier | 0.7877 | 0.7759 | **0.0118** |
| imaging | auc | 0.7699 | 0.6661 | 0.1039 |
| imaging | neg_brier | 0.7711 | 0.7535 | **0.0176** |
| fusion | auc | 0.8062 | 0.6330 | **0.1732** |
| fusion | neg_brier | 0.8221 | **0.8014** | **0.0208** |

Both halves of the finding transfer. The failure reproduces at nearly the same
magnitude on a cohort 3.5× larger from different sites and scanners (fusion gap
0.1989 → 0.1732), so it is **not a small-cohort artefact**. The remedy transfers
too, reducing the ADNI fusion gap by **88%** — and under calibration-aware
selection fusion becomes the *best* mode externally (pooled-style AUC 0.8014),
inverting the ranking that standard practice produces.

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

### ADNI cross-cohort pipeline

| script | purpose |
|---|---|
| `build_adni_cohort.py` | phase-aware AD/CN labels from DXSUM (`DXAD` for ADNI1, `DXDDUE` for ADNI2/GO/3/4), joined to APOE + demographics |
| `select_adni_scans.py` | one scan per subject, nearest diagnosis within 365 days; carries clinical columns through the merge |
| `check_adni_coverage.py` | conversion coverage against the strict-target cohort |
| `preprocess_adni.py` | imports the processing functions from `preprocess.py` — one implementation, both cohorts |
| `age_shift_diagnostic.py` | quantifies the OASIS-3 → ADNI age shift under each fold's frozen normalisation (reported, never corrected) |
| `inference_adni.py` | applies the five frozen OASIS-3 fold-models to ADNI under both selection criteria |
| `tools/verify_phase3_numbers.py` | cross-checks the Phase 3 numbers in `RESULTS.md` against the results JSON |

### Analysis (Phases 4-5)

| script | purpose |
|---|---|
| `gradcam_analysis.py` | Grad-CAM on the imaging branch, comparing `auc`- and `neg_brier`-selected checkpoints; reports brain selectivity against a chance baseline (Phase 4, null result) |
| `temperature_scaling.py` | fits temperature on each fold's validation split and tests whether post-hoc calibration substitutes for calibration-aware selection (Phase 5) |
| `tools/verify_results_numbers.py` | cross-checks Phase 3, 4 and 5 numbers in `RESULTS.md` against their committed JSON |

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
- `adni_cohort.csv`, `adni_scans_selected.csv` — ADNI cohort and selected scans
- `preprocess_qc_adni.csv` — per-subject ADNI preprocessing QC
- `outputs/adni_external/` — external validation results JSON and per-fold prediction CSVs
- `figures_gradcam/`, `figures_gradcam_fusion/` — Grad-CAM figures and summary JSON
- `outputs/temperature/` — temperature scaling results and per-fold predictions
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

# ADNI external validation (no training; uses the saved CV checkpoints)
python age_shift_diagnostic.py              # cohort shift, run before inference
python inference_adni.py                    # 3 modes x 5 folds x 2 criteria
python tools/verify_phase3_numbers.py       # RESULTS.md vs results JSON

# interpretability and calibration remedies (inference only)
python gradcam_analysis.py --cohort oasis --mode fusion --fold 4
python temperature_scaling.py
python tools/verify_results_numbers.py      # checks Phases 3, 4 and 5
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
- **External validation freezes preprocessing.** Age is standardised on ADNI
  using each fold's stored OASIS-3 *training-split* statistics; they are never
  re-estimated on ADNI. The normalisation parameters are learned preprocessing
  parameters of the trained model — re-estimating them externally would be
  test-time distribution adaptation, and would centre ADNI's age distribution at
  zero by construction, erasing the cohort shift external validation exists to
  expose. APOE ε4 count is passed raw (0/1/2), exactly as in training.
- **Fold-models are applied to ADNI separately**, not ensembled and not replaced
  by a single model retrained on all 365 subjects. Ensembling would average away
  the between-fold variance this work measures; retraining would confound
  training-set size with cohort and require an ad-hoc validation split for
  checkpoint selection.
- **Cross-cohort preprocessing is one implementation.** `preprocess_adni.py`
  imports from `preprocess.py` rather than duplicating it, and ADNI uses
  original DICOM with local dcm2niix conversion rather than LONI's pre-processed
  NIfTI (which carries corrections OASIS-3 never had).

---

## Status and limitations

The core finding has been **validated across two independent cohorts**: the
per-fold/pooled incoherence and the calibration-aware remedy both reproduce on
ADNI (n=1,287) with frozen OASIS-3 preprocessing and no adaptation. The
sample-size explanation is therefore ruled out — the instability persists on a
cohort 3.5× larger.

**Temperature scaling has been tested as the obvious alternative remedy and is
insufficient** (Phase 5). It improves calibration metrics without changing
behaviour: on fusion it cut mean ECE from 0.3905 to 0.2219 while balanced
accuracy stayed at 0.6231 to four decimal places and the degenerate fold count
stayed at 1/5. On fusion fold 4 — which predicts CN for all 73 test subjects —
the fitted temperature reached 145,474, flattening every probability toward 0.5
and improving ECE by 92% while balanced accuracy remained exactly 0.5000.
Temperature scaling can make a model that predicts one class for every subject
appear well calibrated; it acts after selection has already failed.

**Grad-CAM did not yield interpretable localisation** (Phase 4, reported as a
null result). Attention was diffuse (entropy ≈ 0.96) and *below chance* with
respect to brain tissue (selectivity 0.49–0.78, where 1.0 is chance), for both
modes and both selection criteria. The likely mechanism is the global average
pooling stage, which discards spatial information before the classifier. No
anatomical claim is made from it.

Remaining:

1. Additional architectures: 2D-slice ImageNet-pretrained ResNet-18, then
   MedicalNet-pretrained 3D ResNet — the open question is whether the effect is
   specific to this custom CNN or holds for standard backbones
2. Sample-size sweep: subsample ADNI at increasing n to locate where the
   per-fold/pooled gap closes
3. Reverse direction (train ADNI → test OASIS-3) and pooled-cohort CV. Note that
   once ADNI enters training it is permanently unusable as a clean external test
   set, so this comes last

Known limitations: n=365 for training caps absolute performance; OASIS-3 labels
derive from CDR whereas ADNI uses physician-assigned criteria (`DXAD` for ADNI1,
`DXDDUE` for ADNI2/GO/3/4), so cross-cohort label equivalence is not exact; ADNI
has no SES equivalent, so all cross-cohort work uses the 2-feature clinical
branch; ADNI is CN-heavy (1:1.97) against OASIS-3's near balance (1:0.89), which
affects threshold-dependent metrics; "cross-fold pooled-style AUC" on ADNI is
a coherence diagnostic over five correlated predictions per subject, not a
pooled AUC over independent observations; and temperature is fitted on only 37
validation subjects per fold, which is marginal for stable estimation of even a
single parameter.

---

## Data access

OASIS-3 is available from [oasis-brains.org](https://www.oasis-brains.org/) via
NITRC-IR. ADNI and AIBL are available through
[LONI/IDA](https://ida.loni.usc.edu/). All require separate data use agreements;
no imaging data is redistributed in this repository.