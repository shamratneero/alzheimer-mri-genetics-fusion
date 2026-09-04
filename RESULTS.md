# Results Log

This document records experiments, their configurations, results, and interpretation.
It is working documentation intended to feed into the eventual paper's methods, baseline,
and ablation sections — not a standalone manuscript.

---

## Phase 1 — OASIS-1 Baseline (2D, imaging-only)

### Purpose

Establish a working end-to-end pipeline (data loading → training → evaluation) and
a baseline imaging classifier before scaling to richer data (OASIS-3, ADNI) and
adding the genetics/fusion branch. OASIS-1 is used here as a fast, no-gate,
proof-of-concept dataset — not as a source of publishable performance claims.

### Dataset

- **Source:** OASIS-1, obtained via a Kaggle redistribution (Roboflow-exported,
  pre-sliced 2D axial JPEG slices, resized to 224×224 with black-edge padding).
- **Task:** Binary classification — `NonDemented` vs `Demented`
  (Demented = collapse of VeryMild + Mild + Moderate).
- **Real subject counts (from clinical metadata CSVs):**
  - Train: 192 patients (121 NonDemented, 71 Demented)
  - Test: 244 patients (215 NonDemented, 29 Demented)
- **Image counts (after class-specific augmentation applied by the dataset creator):**
  - Train: 55,443 images
  - Test: 46,184 images

### Data integrity checks

- **Patient-level train/test overlap: 0** — confirmed no subject appears in both
  splits (checked via `ID` field in the clinical metadata CSVs). The split is clean
  at the patient level.
- **Severe class imbalance** identified, unevenly distributed across splits:
  - `ModerateDemented`: only 1 patient in train, 1 in test — statistically
    unusable as its own class.
  - `MildDemented`: 24 train, 4 test — too few in test for reliable per-class metrics.
  - This is the primary reason the task was collapsed to binary rather than kept
    as 4-class severity: two of the four classes have too few real patients behind
    them for any 4-class metric to be meaningful.
- **Augmentation note:** the dataset creator generated up to 5× rotated/blurred
  copies of rarer classes (e.g., ModerateDemented inflated from 1 real patient to
  ~1000+ images). These are near-duplicates of the same underlying scans and do not
  add independent information — a key caveat when interpreting any accuracy on
  the minority class.

### Model & training setup

- **Architecture:** ResNet18 (ImageNet-pretrained), final layer replaced for
  binary output.
- **Loss:** Cross-entropy with class weighting (`[1.0, 121/71]`) to counter train
  imbalance.
- **Optimizer:** Adam, lr = 1e-4.
- **Hardware:** NVIDIA RTX 4060 (~8GB VRAM), CUDA-enabled PyTorch, mixed use of
  `num_workers` for loading.

### Experiments — capacity ablation

Three configurations were run to probe the capacity/generalization tradeoff on a
small (192-patient) single-cohort dataset.

| # | Configuration                     | Trainable params | Train Acc | Best Test Acc | Diagnosis                         |
|---|-----------------------------------|------------------|-----------|---------------|-----------------------------------|
| 1 | Full fine-tune                    | 100% (~11.2M)    | 99.5%     | 71.2%         | Overfits badly                    |
| 2 | Frozen backbone except layer4 + dropout(0.4) + weight decay + LR scheduler | 75.1% (~8.4M) | 99.7% | **80.5%** | Overfits, but best result         |
| 3 | Full freeze (linear probe; only classifier head trains) | 0.0% (~1,026) | ~84.8% | 58.75% | Underfits — features too generic  |

### Detailed evaluation (best model, Experiment 2 — `frozen_resnet18_best.pth`)

Test set (0 = NonDemented, 1 = Demented):

| Class          | Precision | Recall  | F1     | Support |
|----------------|-----------|---------|--------|---------|
| 0 NonDemented  | 0.9545    | 0.8183  | 0.8812 | 40,852  |
| 1 Demented     | 0.3350    | 0.7012  | 0.4534 | 5,332   |
| **Accuracy**   |           |         | 0.8048 | 46,184  |

Confusion matrix:
```
[[33429  7423]
 [ 1593  3739]]
```

For contrast, Experiment 1 (full fine-tune) at 71.2% accuracy showed higher
Demented recall (0.884) but far worse precision (0.271) and more false positives
(12,670), i.e., a different point on the precision/recall tradeoff, not a strictly
better or worse model.

### Interpretation

- **The pipeline works end-to-end** — data loading, GPU training, checkpointing,
  and evaluation are all functional and produce consistent, honest results.
- **Overfitting is driven by data quantity, not a fixable modeling error.** With
  only 192 real training patients (and heavy near-duplicate augmentation),
  ResNet18-scale models rapidly memorize. Regularization (dropout, weight decay,
  freezing) shifts the tradeoff but does not resolve it.
- **The capacity ablation cleanly brackets the failure modes:** full capacity
  overfits (Exp 1), zero adaptation underfits (Exp 3), and partial fine-tuning of
  the last block is the sweet spot (Exp 2). This is a concrete, empirical
  illustration of the project's central thesis — that single-cohort benchmark
  accuracy is a poor proxy for genuine, deployable reliability.
- **80.5% is not a headline result and is not treated as one.** It is a baseline
  and a diagnostic, establishing the "insufficient data" reference point against
  which the OASIS-3 (larger, real 3D volumes, with APOE) results will be contrasted.

### Limitations (OASIS-1 phase)

- 2D pre-sliced JPEGs, not volumetric MRI — spatial structure is partially lost,
  and this is not the intended final architecture.
- No genetics/clinical fusion — imaging-only.
- No cross-cohort validation yet — trained and tested only within OASIS-1.
- Small N and augmentation-driven near-duplicates limit the trustworthiness of any
  single train/test split; patient-level k-fold cross-validation is planned to
  produce a variance-aware estimate.

### Artifacts

- Code: `dataset.py`, `train_baseline.py`, `evaluate.py`
- Checkpoints: `outputs/checkpoints/` (best model: `frozen_resnet18_best.pth`)
- Derived labels: `train_binary.csv`, `test_binary.csv`

---

## Phase 2 — OASIS-3 (3D, imaging + genetics)

Access granted. Primary training cohort:
- 365 usable subjects (vs 192 in OASIS-1), full 3D T1w NIfTI volumes.
- Includes APOE genotype → enables the genetics branch and fusion architecture.
- 3D pipeline complete: `dataset_3d.py`, `model_3d.py`, `train.py`,
  `cross_validate.py`.

### Cohort and preprocessing (final)

365 subjects (193 probable AD / 172 cognitively normal), one T1w scan each.
Task is strictly AD vs CN: `PROBAD=1` vs `NORMCOG=1`, non-AD dementias excluded.
Scan-time diagnosis matched to the nearest clinical visit within 365 days.

Preprocessing: deepbet skull-stripping (Otsu thresholding was tried first and
rejected — it retained skull/face while sometimes deleting brain tissue),
resampling to 1mm isotropic, z-scoring within the brain mask, cached as 160³
float32 `.npy`, resized to 128³ at load time.

Clinical inputs: APOE e4 allele count (0/1/2) and age at scan, normalised using
training-split statistics only. APOE e4 carrier rates were 62% in AD vs 30% in
CN, confirming correct label linkage.

Known confounds (from EDA): age differs substantially between groups
(AD 76.8 vs CN 69.6, Cohen's d = 0.87); scanner protocol is mildly predictive
(AUC 0.589). A demographics-only logistic regression on five features
(age, sex, education, SES, APOE) reached AUC 0.826 — recorded here as the
reference the imaging model must clear.

---

## Phase 2a — Architecture bugs found and fixed

Two initialisation defects were found during the ablation work. Both are
recorded because they materially changed the results, and because the second
was only visible after fixing the first.

**Bug 1 — shared classifier head penalised single-branch ablations.**
The fusion classifier took a 272-d input (256 imaging + 16 clinical). The
single-branch ablations reused that head with the absent branch zero-padded.
Because `nn.Linear` initialises weights with std ~ 1/sqrt(fan_in), a 16-d
clinical vector entering a 272-wide head starts ~4.4x weaker than it would in a
correctly-sized 16-d head (measured: pre-activation std 0.098 vs 0.432). This
pinned `clinical_only` balanced accuracy at exactly 0.500 with training loss
stuck near ln(2) — the model was not committing to predictions at all.

**Bug 2 — branch outputs were on wildly different scales.**
Fixing Bug 1 exposed a larger problem upstream. Global average pooling over the
final 4×4×4 feature grid left imaging features at std **0.0047**, against the
clinical branch's **0.212** — a ~45x imbalance, before any classifier is
involved. In the original fusion model the two defects partially cancelled
(weak imaging features got the full-width head; strong clinical features got the
diluted one), which is an accident rather than a design.

**Fix:** a dedicated correctly-sized classifier head per mode
(`_make_head()` → `classifier` / `classifier_imaging` / `classifier_clinical`),
plus `LayerNorm` on both branch outputs. Signal ratio entering the heads went
from 4.4x → 0.69x; branch feature ratio from 45x → 0.83x. Verified: correct
output shapes for all three modes, and bidirectional gradient isolation
(clinical-only training produces zero gradient in the imaging branch and vice
versa).

**Also fixed:** `torch.load` required `weights_only=False` after PyTorch 2.6
changed the default. The per-epoch `history` list contains numpy scalars from
sklearn, which the strict loader rejects. This only surfaced when `--resume` was
first exercised after a power cut — the code path had been writing checkpoints
correctly for weeks but had never been read.

**Repo audit (post-fix).** Split geometry reproduces exactly (255/37/73) and is
stratified; no subject appears in more than one split; age normalisation uses
train-split statistics only; rotation augmentation verified to rotate correctly
and preserve signal; class weights are in the correct direction (they mildly
favour CN, i.e. they oppose rather than cause the observed AD bias). No
remaining bugs found.

---

## Phase 2b — Single-split ablations (255 / 37 / 73)

All three modes through an identical pipeline: same split, same architecture,
same seed, same evaluation.

| mode | test AUC | bal. acc | ECE | Brier | sens | spec |
|---|---|---|---|---|---|---|
| imaging only  | 0.796 | 0.715 | 0.181 | 0.222 | 0.872 | 0.559 |
| clinical only | 0.781 | 0.714 | 0.099 | 0.192 | 0.692 | 0.735 |
| fusion        | 0.773 | 0.546 | 0.360 | 0.353 | 0.974 | 0.118 |

Fusion recorded the **highest validation AUC of any run (0.929)** and the worst
deployment behaviour: specificity 0.118 means it correctly identified 4 of 34
CN subjects, i.e. it is close to predicting AD for everyone. Reported by AUC and
accuracy alone — the convention in this literature — it would have looked like
the best of the three.

**Validation consistently overstated test performance.** Val→test AUC drops
across four runs: −0.110, −0.119, −0.093, −0.156.

**These differences are not resolvable at this sample size.** A bootstrap of a
model with true AUC ≈ 0.714 evaluated on 39 AD / 34 CN gives a test-AUC standard
deviation of **0.060** (95% range 0.592–0.828). The observed spread across the
three modes is 0.023 — less than half of single-run noise. Single-split results
on n=73 cannot rank these models, which is what motivated the move to
cross-validation.

---

## Phase 2c — 5-fold cross-validation (primary result)

Protocol: `StratifiedKFold` (5 folds, seed 42) on the outer loop, with an inner
validation split carved from each fold's training portion. Fold geometry is
identical to the single-split runs (255/37/73), so results are directly
comparable. Every subject is tested exactly once, so out-of-fold predictions
cover all 365 subjects with no reuse. Age normalisation uses each fold's own
training statistics. Test data never influences training or checkpoint
selection.

### Per-fold results

| mode | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | mean ± std |
|---|---|---|---|---|---|---|
| clinical | 0.782 | 0.765 | 0.703 | 0.825 | 0.855 | 0.786 ± 0.058 |
| imaging  | 0.814 | 0.846 | 0.837 | 0.859 | 0.890 | 0.849 ± 0.028 |
| fusion   | 0.835 | 0.891 | 0.844 | 0.897 | 0.878 | 0.869 ± 0.028 |

### Per-fold means vs pooled out-of-fold

| mode | per-fold mean AUC | pooled OOF AUC | gap | pooled 95% CI | mean ECE |
|---|---|---|---|---|---|
| clinical | 0.786 | **0.764** | 0.022 | [0.715, 0.810] | 0.0955 |
| imaging  | 0.849 | 0.729 | 0.120 | [0.677, 0.780] | 0.225 |
| fusion   | **0.869** | **0.670** | **0.199** | [0.614, 0.722] | 0.309 |

**The ranking inverts.** By per-fold mean AUC — the standard way these results
are reported — fusion is the clear winner and clinical the clear loser. By
pooled out-of-fold AUC, the order is exactly reversed.

The gap between the two measures is diagnostic. Per-fold AUC measures ranking
*within* a fold. Pooled AUC additionally requires that the five fold-models
place their probabilities on a *comparable scale*. Clinical passes this test
(gap 0.022); imaging fails it (0.120); fusion fails it badly (0.199). The
imaging-based models produce fold-specific probability scales that do not
transfer — a failure invisible to per-fold reporting.

### Threshold-level instability

| mode | bal. acc range | sensitivity range | specificity range |
|---|---|---|---|
| clinical | 0.529 – 0.773 | 0.421 – 1.000 | 0.059 – 0.886 |
| imaging  | 0.500 – 0.785 | **0.000** – 0.974 | 0.314 – 1.000 |
| fusion   | 0.500 – 0.750 | **0.000** – 1.000 | 0.206 – 1.000 |

Fusion sensitivity has std **0.430** across folds: fold 4 detected **zero** of
39 AD cases, fold 5 detected **all** of them. Same architecture, same
hyperparameters, same procedure — only the subject split differs. Per-fold AUC
across those same two folds was 0.897 and 0.878, i.e. essentially identical and
uniformly excellent.

### Extended clinical baseline (5 features)

The two-feature clinical branch (APOE e4 count + age) invites the objection that
the imaging model was compared against a deliberately weakened baseline — the
EDA logistic regression on five features reached AUC 0.826. The clinical branch
was therefore re-run with sex, education and socioeconomic status added, through
the same pipeline and the same folds (`--extended_clinical`).

| clinical branch | per-fold mean AUC | pooled OOF AUC | gap | mean ECE |
|---|---|---|---|---|
| 2 features (APOE, age) | 0.786 | 0.7638 | 0.022 | 0.0955 |
| 5 features (+ sex, educ, SES) | 0.770 | **0.7675** | **0.003** | 0.126 |

Per-fold AUCs for the 5-feature model: 0.717, 0.786, 0.670, 0.836, 0.841
(mean 0.770 ± 0.075); pooled 95% CI [0.721, 0.814].

Two things follow. First, the three extra demographics are worth +0.004 pooled
AUC — statistically nothing (see the paired test below, p=0.856). **APOE e4
count and age carry essentially all of the non-imaging signal in this cohort.**
Second, the 5-feature model has a per-fold-to-pooled gap of **0.003**, the
smallest of any model run here, against 0.120 for imaging and 0.199 for fusion.
The clinical models are not merely competitive on discrimination; their
fold-models agree on what a given probability means, and the imaging-based ones
do not.

### Paired comparison (identical subjects, identical bootstrap resamples)

| mode | pooled OOF AUC |
|---|---|
| clinical, 5 features | **0.7675** |
| clinical, 2 features | 0.7638 |
| imaging only | 0.7292 |
| fusion | 0.6702 |

| comparison | ΔAUC | 95% CI | p | verdict |
|---|---|---|---|---|
| clinical 2-feat vs clinical 5-feat | −0.004 | [−0.049, +0.040] | 0.856 | not distinguishable |
| clinical 2-feat vs imaging | +0.035 | [−0.024, +0.093] | 0.253 | not distinguishable |
| clinical 5-feat vs imaging | +0.038 | [−0.028, +0.107] | 0.253 | not distinguishable |
| clinical 2-feat vs fusion | **+0.094** | [+0.033, +0.158] | **0.003** | **significant** |
| clinical 5-feat vs fusion | **+0.097** | [+0.029, +0.169] | **0.004** | **significant** |
| imaging vs fusion | **+0.059** | [+0.003, +0.118] | **0.039** | **significant** |

The test was verified to detect a genuine difference when one exists (a
synthetic strong/weak pair returned p < 0.001), so "not distinguishable" is a
substantive result rather than an artefact of low power.

**Clinical features significantly outperform the full multimodal model**, and
the result does not depend on which clinical baseline is used: the 2-feature
version beats fusion by +0.094 (p=0.003) and the stronger 5-feature version
beats it by slightly *more*, +0.097 (p=0.004). The "you compared against a weak
baseline" objection is therefore closed from both directions.

**Fusion is also significantly worse than imaging alone** (+0.059, p=0.039)
despite having strictly more information available to it. A fusion model that is
significantly worse than both of its own components is not a marginal effect —
it is a failure of naive concatenation-based fusion at this scale.

### Checkpoint selection

Selecting on validation AUC repeatedly chose miscalibrated models:

- **imaging fold 2** — checkpoint taken at epoch 10, where val AUC was 0.947
  (the highest in the run) while val balanced accuracy was **already 0.500** and
  val ECE **0.500**. Test result: AUC 0.846 (highest of the imaging folds),
  balanced accuracy 0.500, ECE 0.480, sensitivity 0.000. The criterion selected
  a model that was visibly broken on the validation set it was selecting from.
- **clinical fold 3** — best val AUC 0.844 at **epoch 1**; test AUC 0.703, the
  worst fold. **clinical fold 5** — best val AUC 0.731 at epoch 32; test AUC
  0.855, the best fold. Across folds, validation and test AUC were *inversely*
  related.

---

## Findings (Phase 2)

1. **Per-fold means conceal cross-model probability incoherence.** Reporting
   mean per-fold AUC — near-universal in this literature — ranks fusion first.
   Pooled out-of-fold AUC over the same predictions ranks it last, significantly
   so. The two measures disagree because only the second requires fold-models to
   agree on what a given probability means.

2. **Naive fusion is significantly worse than either single modality.** Not
   "no better" — significantly worse, against both branches, with more
   information available. Adding a well-behaved 2-feature clinical branch to the
   imaging model degraded it.

3. **Discrimination and calibration decouple.** Across all three modes, adding
   modalities monotonically improved per-fold AUC (0.786 → 0.849 → 0.869) while
   monotonically worsening calibration (ECE 0.0955 → 0.225 → 0.309) and pooled
   coherence (0.764 → 0.729 → 0.670).

4. **AUC-based checkpoint selection actively selects broken models.** It is not
   merely insensitive to calibration failure; in imaging fold 2 it preferred a
   model whose validation balanced accuracy had already collapsed to chance.

5. **Single test splits at this size cannot rank models.** Bootstrap noise on
   n=73 is ±0.060 AUC; the observed between-mode spread was 0.023. Even pooling
   all 365 subjects, the resolvable difference is ~0.053 — which is why the
   clinical-vs-imaging comparison remains undecided while both clear fusion.

6. **Imaging does not clear the non-imaging baseline.** Pooled imaging AUC
   (0.729) sits below both the 2-feature clinical model (0.764) and the
   5-feature one (0.768). A 3.57M-parameter 3D CNN over 128³ volumes, at ~5
   hours per fold, does not beat APOE e4 count and age.

7. **The non-imaging signal is APOE and age, and nothing else.** Adding sex,
   education and socioeconomic status to the clinical branch changed pooled AUC
   by +0.004 (p=0.856). This matters for the comparison above: the imaging model
   fails to beat the *stronger* clinical baseline as well as the weaker one, so
   the result cannot be attributed to a deliberately handicapped comparator.
## Limitations (Phase 2)

- **n = 365** caps absolute performance and widens every interval. The clinical
  branch has now been run at both 2 and 5 features, so the non-imaging baseline
  is no longer understated, but the cohort remains small.
- **Single architecture.** These findings are demonstrated for one custom 3D CNN
  trained from scratch. Whether they persist with pretrained backbones
  (MedicalNet) or a different input representation (2D-slice, ImageNet-pretrained)
  is not yet established.
- **Single cohort.** All results are internal to OASIS-3. Whether the instability
  is a small-cohort artefact or a property of the approach cannot be settled
  without an independent cohort — ADNI (access approved) is the next step.
- **OASIS-3 labels derive from CDR**, whereas ADNI/AIBL use physician-assigned
  NINCDS-ADRDA criteria. There is no strict equivalence, and this must be stated
  explicitly in any cross-cohort comparison.

## Artifacts (Phase 2)

- Code: `dataset_3d.py`, `model_3d.py`, `train.py`, `cross_validate.py`
- CV results: `outputs/cv/cv_{clinical_only,imaging_only,fusion}_{folds,summary,oof_predictions}.*`
- Single-split logs: `outputs/logs/`
- Reproduce the comparison:
  `python cross_validate.py --compare cv_clinical_only cv_imaging_only cv_fusion`

---

## Phase 3 — ADNI external validation (OASIS-3 → ADNI)

### Purpose

Everything in Phase 2 is internal to OASIS-3 (n=365). The central claim — that
per-fold AUC reporting conceals cross-model probability incoherence, and that
calibration-aware checkpoint selection largely repairs it — could not be
distinguished from a small-cohort artefact without an independent cohort. This
phase applies the frozen OASIS-3 fold-models to ADNI (n=1,287) and asks whether
both the failure and the remedy reproduce.

### Cohort

- **1,287 ADNI subjects** (433 AD / 854 CN) with usable T1 and complete
  APOE + age, out of 1,294 strict-target subjects (scan within 365 days of
  diagnosis). Coverage 99.5%.
- **7 subjects excluded**: sites 057, 098 and 126 returned **PET series rather
  than T1 structural scans** (64×64×64×2 volumes at ~4.7×4.7×5.7 mm, versus
  ~240×256×176 at ~1 mm for a valid T1). These were caught by a
  dimensionality/voxel-size audit run over the entire converted cohort *before*
  preprocessing. They had initially failed an ID-matching check for an unrelated
  reason (protocol-name suffixes in the dcm2niix output filenames), and the
  obvious remedy — correcting the filenames — would have admitted seven PET
  volumes into the external validation set, where they would have been resampled
  to 160³ and become indistinguishable from valid scans in the `.npy` cache.
- **1 subject recovered**: `037_S_4001` converted to `037_V_4001` because of a
  data-entry variant in the DICOM `PatientID` field. The IDA directory name was
  treated as authoritative over the free-text DICOM field.
- APOE ε4 carrier rate 65% AD / 32% CN, against OASIS-3's 62% / 30% — consistent
  enough to confirm correct label linkage.
- **2-feature clinical branch only.** ADNI has no SES equivalent (PTWORK covers
  ~21% of records and is a different construct), so the 5-feature branch cannot
  be run cross-cohort. Given that the 5-feature branch changed pooled AUC by
  +0.004 (p=0.856) on OASIS-3, little is lost.

### Design decisions

Three decisions were fixed before any inference was run.

1. **Each of the five fold-models is applied to ADNI separately** — not
   ensembled, not replaced by a single model retrained on all 365 subjects.
   Ensembling averages away precisely the between-fold variance this paper
   measures, and could produce a well-behaved aggregate from individually
   degenerate members. Retraining on all 365 would require a fresh ad-hoc
   validation split for checkpoint selection, inconsistent with the rest of the
   protocol, and would confound training-set size with cohort.
2. **Normalization is frozen.** Age is standardised using each fold's stored
   OASIS-3 *training-split* `age_mean`/`age_std`; statistics are never
   recomputed on ADNI. The normalization parameters are learned preprocessing
   parameters of the trained model: re-estimating them on the external cohort
   would constitute test-time distribution adaptation and would silently change
   the model's input representation while leaving its weights fixed. It would
   also centre ADNI's age distribution at zero by construction, erasing the very
   cohort shift external validation exists to expose. APOE ε4 count is passed
   raw (0/1/2), matching training exactly — it was never standardised.
3. **Both `auc`- and `neg_brier`-selected checkpoints are run**, so the
   cross-cohort question is asked of standard practice and of the proposed
   remedy under identical conditions.

### Age-shift diagnostic (run before inference, not used to correct anything)

| Cohort | Group | N | Mean age | SD age |
|---|---|---|---|---|
| OASIS-3 | All | 365 | 73.41 | 8.96 |
| OASIS-3 | CN | 172 | 69.64 | 8.63 |
| OASIS-3 | AD | 193 | 76.77 | 7.87 |
| ADNI | All | 1287 | 72.10 | 7.74 |
| ADNI | CN | 854 | 70.46 | 7.07 |
| ADNI | AD | 433 | 75.33 | 8.00 |

ADNI expressed in each fold's frozen OASIS-3 normalization:

| Fold | train age_mean | train age_std | z(ADNI all) | z(ADNI CN) | z(ADNI AD) |
|---|---|---|---|---|---|
| 1 | 73.64 | 8.62 | -0.179 | -0.369 | 0.197 |
| 2 | 73.52 | 9.36 | -0.152 | -0.327 | 0.194 |
| 3 | 73.27 | 9.03 | -0.130 | -0.312 | 0.228 |
| 4 | 73.57 | 9.13 | -0.161 | -0.341 | 0.193 |
| 5 | 73.24 | 9.10 | -0.125 | -0.306 | 0.230 |

ADNI is ~1.1–1.5 years younger overall (mean z ≈ -0.15, roughly one sixth of a
fold SD), consistently across all five folds. The subgroup structure is the more
informative part: ADNI CN sits well below the OASIS-3 training mean
(z ≈ -0.31 to -0.37) while ADNI AD sits slightly above it (z ≈ +0.19 to +0.23).
Age could therefore **contribute to apparent AD/CN separability in ADNI through
the age component of the clinical model**, rather than explaining a performance
drop. The shift is reported, not corrected.

### Primary result — per-fold mean vs cross-fold pooled-style AUC on ADNI

| mode | selection | per-fold mean AUC | pooled-style AUC | gap |
|---|---|---|---|---|
| clinical | auc | 0.7795 | 0.7407 | 0.0388 |
| clinical | neg_brier | 0.7877 | **0.7759** | **0.0118** |
| imaging | auc | 0.7699 | 0.6661 | 0.1039 |
| imaging | neg_brier | 0.7711 | **0.7535** | **0.0176** |
| fusion | auc | 0.8062 | 0.6330 | **0.1732** |
| fusion | neg_brier | 0.8221 | **0.8014** | **0.0208** |

**Terminology.** "Cross-fold pooled-style AUC" is *not* a pooled AUC over
independent observations. On OASIS-3 each subject receives exactly one
out-of-fold prediction, so pooling is clean. On ADNI every subject is external
to all five fold-models and therefore contributes five correlated rows. The
quantity measures the same property pooled OOF AUC measured internally — whether
the five fold-models place probabilities on a comparable scale — but it must be
reported under this name, not as a conventional pooled AUC.

### Side-by-side with OASIS-3

Recomputed from the committed per-fold JSON (`outputs/cv/sel_*_folds.json`):

| mode | selection | OASIS-3 gap | ADNI gap |
|---|---|---|---|
| clinical | auc | 0.0222 | 0.0388 |
| clinical | neg_brier | -0.0053 | 0.0118 |
| imaging | auc | 0.1200 | 0.1039 |
| imaging | neg_brier | 0.0227 | 0.0176 |
| fusion | auc | 0.1989 | 0.1732 |
| fusion | neg_brier | 0.0408 | 0.0208 |

### Findings (Phase 3)

1. **The failure reproduces on an independent cohort.** Under standard
   AUC-based checkpoint selection, the per-fold-to-pooled gap on ADNI is 0.0388
   (clinical), 0.1039 (imaging), 0.1732 (fusion) — the same ordering and
   substantially the same magnitude as OASIS-3 (0.0222 / 0.1200 / 0.1989),
   on a cohort 3.5× larger, from different sites and scanners. The instability
   is not a small-cohort artefact.

2. **The gap scales with reliance on the imaging branch, again.** clinical <
   imaging < fusion holds on both cohorts, consistent with the imaging branch
   being the source of the unstable probability scale.

3. **The remedy transfers, and transfers well.** Calibration-aware
   (`neg_brier`) checkpoint selection reduces the ADNI gap by 70% (clinical),
   83% (imaging) and **88% (fusion)**. On OASIS-3 the equivalent fusion
   reduction was 79%. The remedy was designed on OASIS-3 and applied unchanged;
   it works at least as well on a cohort it was never tuned against.

4. **Fusion becomes the best mode once selection is fixed.** Under `neg_brier`,
   ADNI pooled-style AUC is fusion 0.8014 > clinical 0.7759 > imaging 0.7535.
   Under `auc` selection the same models rank clinical 0.7407 > imaging 0.6661 >
   fusion 0.6330. The apparent inferiority of multimodal fusion is an artefact
   of checkpoint selection, not a property of fusion — and this now holds on two
   independent cohorts.

5. **Absolute external performance is credible rather than inflated.** Best
   external pooled-style AUC is 0.8014 (fusion, `neg_brier`), against 0.8195
   internally. A drop of that size moving to an unseen cohort, with frozen
   preprocessing and no adaptation, is what honest external validation looks
   like.

### Limitations (Phase 3)

- **Cross-fold pooled-style AUC is not an independent-observation statistic**
  (see terminology note). It is a coherence diagnostic, not a deployment metric.
- **Label definitions differ between cohorts.** OASIS-3 labels derive from CDR;
  ADNI uses physician-assigned criteria via DXSUM (`DXAD` for ADNI1, `DXDDUE`
  for ADNI2/GO/3/4). There is no strict equivalence.
- **Class balance differs** — ADNI is CN-heavy (1:1.97) against OASIS-3's near
  balance (1:0.89), which affects any threshold-dependent metric.
- **Scan-availability selection bias.** The ADNI cohort was 879 AD / 949 CN
  before the usable-scan requirement and 433 AD / 854 CN after: AD subjects are
  substantially more likely to lack a usable T1 (motion, dropout).
- **Brain fraction runs counter to atrophy.** ADNI AD mean brain fraction
  (0.1177) slightly exceeds CN (0.1130). Brain fraction here is brain voxels as
  a proportion of the resampled volume and is therefore FOV-dependent; site 168
  uses a notably wider FOV and is CN-heavy, and the four highest-brain-fraction
  subjects are all AD with visibly enlarged ventricles. This is site composition,
  not biology, and the quantity is a QC metric never seen by the model.
- **Single architecture still.** Phase 3 establishes cross-cohort reproducibility
  for one custom 3D CNN. Whether the effect persists across architectures remains
  open.

### Artifacts (Phase 3)

- Code: `inference_adni.py`, `preprocess_adni.py`, `build_adni_cohort.py`,
  `select_adni_scans.py`, `check_adni_coverage.py`
- Results: `outputs/adni_external/adni_external_results.json`,
  `outputs/adni_external/adni_preds_{mode}_fold{k}_{criterion}.csv`
- Reproduce: `python inference_adni.py`
- The inference script discovers checkpoints by embedded metadata rather than
  filename, validates every checkpoint against an explicit schema, sanity-checks
  every loaded volume, and asserts identical subject sets across folds before
  pooling — it fails loudly rather than producing a number from malformed input.

---

## Phase 4 — Grad-CAM interpretability (null result)

### Purpose

Phases 2-3 establish that checkpoint selection changes model behaviour, but all
of that evidence is scalar. This phase asked whether the difference is visible
spatially: does the degenerate `auc`-selected checkpoint attend to different
regions than the calibration-aware pick, and do either attend to hippocampal /
medial temporal structures at all?

The centrepiece comparison was fusion fold 4, where the two criteria diverge
most sharply:

| criterion | epoch | test AUC | balanced acc | ECE | sens / spec |
|---|---|---|---|---|---|
| auc | 10 | 0.8974 | 0.5000 | 0.4860 | 0.000 / 1.000 |
| neg_brier | 17 | 0.8597 | 0.7285 | 0.1380 | 0.692 / 0.765 |

The `auc` checkpoint predicts CN for **every** test subject while scoring
AUC 0.897.

### Method

Grad-CAM on the imaging branch, hooking `block5` (4³ feature map at 128³ input)
and `block4` (8³). Subjects for the individual-example figures were chosen by a
rule fixed in code before any output was inspected — the two most confident
correct predictions per class plus the two most confident errors, ties broken by
subject ID — so no figure could be reached by browsing for a striking one. Group
means over all 73 fold-4 subjects are the primary evidence; individual examples
are illustration only.

Attention was quantified rather than only eyeballed:

- **brain selectivity** = (attention inside the brain mask) ÷ (brain's share of
  the volume). **1.0 is chance**; >1 means preferential attention to tissue.
  Reporting the raw in-brain fraction alone would be meaningless because the
  brain occupies only ~21% of a skull-stripped volume.
- **entropy**, normalised: 1.0 = attention spread uniformly.
- Dead CAMs (all-zero after ReLU) are **excluded** from all averages rather than
  counted as zero, and reported separately as a rate.

Group-mean maps are per-subject min-max normalised before averaging, so they
show **where** attention lands, not **how much** — magnitude claims come only
from the per-subject statistics above.

### Result — no interpretable localisation

`block5` (the only layer with usable coverage, 0% dead CAMs, all 73 subjects):

| mode | criterion | brain selectivity AD | CN | entropy AD |
|---|---|---|---|---|
| fusion | auc | 0.493 | 0.586 | 0.964 |
| fusion | neg_brier | 0.781 | 0.630 | 0.959 |
| imaging_only | auc | 0.624 | 0.508 | 0.966 |
| imaging_only | neg_brier | 0.624 | 0.508 | 0.966 |

**Every value is below 1.0** — attention falls inside brain tissue *less* often
than chance, and entropy near 0.96 confirms it is close to uniform. Visual
inspection agrees: peak attention sits in the empty background corners, and the
AD and CN group means are nearly indistinguishable.

`block4` is unusable: 51-65 of 73 subjects (70-89%) produced completely dead
CAMs. The surviving 8-22 subjects show selectivity above 1.0, but that is a
self-selected sample — the subjects that survive are exactly those with unusual
gradients — and those numbers are **not** reported as a finding.

### Interpretation

The likely mechanism is architectural. `ImagingBranch3D` ends with
`AdaptiveAvgPool3d(1)`, which averages the entire 4³ feature map to one value
per channel before the classifier. Spatial location is discarded by
construction, so no individual cell is under gradient pressure to matter and
Grad-CAM has little to recover. This is the spatial counterpart of the effect
already documented in Phase 2a, where the same pooling stage left imaging
features ~45× weaker than clinical features.

Two controls support the architectural reading rather than a
model-specific one:

1. `imaging_only` shows the same below-chance selectivity as `fusion`, so the
   clinical branch is not diluting the gradient signal.
2. For imaging fold 4 both criteria selected the same epoch (12), so the
   identical rows above are expected, not a bug.

**This is reported as a null result.** Grad-CAM did not yield interpretable
localisation for this architecture, and no anatomical claim is made from it.
Notably it is *consistent* with the performance findings — a model whose pooled
AUC falls to 0.670 under standard selection would not be expected to have
learned crisp anatomical features. A striking hippocampal figure alongside that
pooled AUC would have been the more troubling outcome.

Fixing this would require replacing global average pooling with a
spatially-preserving alternative and retraining, which would invalidate every
Phase 2 and Phase 3 result. It is not pursued. Occlusion sensitivity, which does
not depend on spatial feature maps, remains available as a gradient-free
alternative.

### Artifacts (Phase 4)

- Code: `gradcam_analysis.py`
- Results: `figures_gradcam/gradcam_summary.json` (imaging_only),
  `figures_gradcam_fusion/gradcam_summary.json` (fusion), 24 figures
- Reproduce: `python gradcam_analysis.py --cohort oasis --mode fusion --fold 4`

---

## Phase 5 — Temperature scaling vs calibration-aware selection

### The question

Temperature scaling is the standard post-hoc calibration fix: fit one scalar T
on validation data, divide all logits by it. It is cheaper and better known than
changing checkpoint selection, so the obvious objection to this paper is *"why
not just temperature-scale?"*

### Hypothesis, stated before running

Dividing every logit by the same positive scalar is a strictly monotonic
transform, so temperature scaling **cannot change AUC at all** and cannot change
which class is predicted. It therefore cannot rescue a checkpoint that has
collapsed to one class — it can only make that checkpoint predict the same class
less confidently, improving ECE and Brier while balanced accuracy stays at
0.500.

The falsifier was specified in advance: if temperature scaling closed the
per-fold/pooled gap as well as `neg_brier` selection *and* repaired the
degenerate folds, the contribution would narrow to "selection matters only where
it goes degenerate".

### Method

T fitted by LBFGS on NLL, on each fold's **validation split only** — never on
test. Fold splits are reconstructed by importing `build_cv_folds` from
`cross_validate.py` rather than reimplementing it; all five reconstructed test
splits match the subject IDs in the committed CV JSON exactly, so the 37-subject
validation splits are equally correct. `log T` is optimised rather than T so
temperature cannot go non-positive.

### Result

| mode | condition | gap | balanced acc | ECE | degenerate folds |
|---|---|---|---|---|---|
| clinical | auc, no scaling | 0.0222 | 0.6765 | 0.1905 | 1/5 |
| clinical | auc + temperature | 0.0110 | 0.6765 | 0.2424 | 1/5 |
| clinical | **neg_brier** | **-0.0053** | **0.7354** | 0.2602 | **0/5** |
| clinical | neg_brier + temp | 0.0165 | 0.7354 | 0.3353 | 0/5 |
| imaging | auc, no scaling | 0.1201 | 0.6719 | 0.3606 | 1/5 |
| imaging | auc + temperature | 0.0735 | 0.6719 | 0.2720 | 1/5 |
| imaging | **neg_brier** | **0.0227** | **0.7581** | 0.3667 | **0/5** |
| imaging | neg_brier + temp | 0.0159 | 0.7581 | 0.3564 | 0/5 |
| fusion | auc, no scaling | 0.1989 | 0.6231 | 0.3905 | 1/5 |
| fusion | auc + temperature | 0.1426 | 0.6231 | 0.2219 | 1/5 |
| fusion | **neg_brier** | **0.0408** | **0.7506** | 0.3514 | **0/5** |
| fusion | neg_brier + temp | 0.0330 | 0.7506 | 0.3362 | 0/5 |

AUC was identical before and after scaling in 29 of 30 fold-runs, as
monotonicity requires — a useful internal check that the implementation is
correct. The single exception is clinical fold 2 under `neg_brier`, where AUC
moved by 0.0004 (0.7624 → 0.7628). This is a floating-point artefact rather than
a monotonicity violation: the fitted T was 0.020, and dividing logits by 0.02
saturates the softmax to exactly 0.0 / 1.0 in float64, creating ties that AUC
scores at half credit. It occurs only on the one fold where temperature fitting
had already visibly failed (validation NLL rose 0.544 → 4.070), so it is a
symptom of the small-validation-set problem noted under limitations, not a
separate defect.

### Findings (Phase 5)

1. **Temperature scaling improves calibration metrics without fixing anything.**
   On fusion it cut mean ECE from 0.3905 to 0.2219 — a 43% improvement — while
   balanced accuracy stayed at **0.6231, unchanged to four decimal places**, and
   the degenerate fold count stayed at 1/5.

2. **The mechanism is visible in the fitted temperatures.** On fusion fold 4
   (sensitivity 0.000, predicts CN for all 73 subjects) the optimiser drove
   **T = 145,474**, which flattens every probability toward 0.5. ECE fell from
   0.4353 to 0.0342 — a 92% "improvement" — while balanced accuracy stayed
   exactly 0.5000. Imaging fold 2 behaved identically (T = 576,062,
   ECE 0.4387 → 0.0205, balanced accuracy 0.5000 throughout). **Temperature
   scaling can make a model that predicts one class for every subject appear
   well calibrated.**

3. **Calibration-aware selection is 3-5× more effective on the gap and is the
   only one of the two that repairs the failure.** On fusion, selection closes
   the gap to 0.0408 against temperature's 0.1426, raises balanced accuracy from
   0.6231 to 0.7506, and eliminates the degenerate fold. Temperature does none
   of the latter two.

4. **Temperature scaling is not useless — it is insufficient.** It does close
   part of the gap (fusion 0.1989 → 0.1426, imaging 0.1201 → 0.0735) by
   rescaling probabilities across folds. It simply cannot address a selection
   failure, because it operates after selection has already happened.

5. **The two do not meaningfully stack.** Adding temperature on top of
   `neg_brier` moves the fusion gap only 0.0408 → 0.0330 and leaves balanced
   accuracy untouched. Once selection is corrected there is little left for
   post-hoc scaling to do, which indicates the two are not doing the same job.

6. **The criteria genuinely disagree.** `auc` and `neg_brier` selected different
   epochs in **13 of 15** mode-fold combinations (clinical 5/5 differ, fusion
   5/5, imaging 3/5). The comparison is not a distinction without a difference.

### Limitations (Phase 5)

- **T is fitted on 37 validation subjects**, which is small enough to overfit.
  On clinical fold 2 the fit produced T = 0.020 and validation NLL *rose* from
  0.544 to 4.070; mean clinical ECE worsened (0.1905 → 0.2424). That same fold
  is where the AUC float-saturation artefact above appears. Temperature scaling
  assumes a validation set large enough to estimate one parameter stably, and at
  this cohort size that assumption is marginal — which is itself a point against
  temperature scaling as the remedy for a small-cohort study.
- Only single-parameter temperature scaling is tested. Vector scaling, Platt
  scaling and isotonic regression are not, though all share the property that
  they act after checkpoint selection.
- OASIS-3 only; the temperature comparison has not been repeated on ADNI.

### Artifacts (Phase 5)

- Code: `temperature_scaling.py`
- Results: `outputs/temperature/temperature_results.json`,
  `outputs/temperature/temperature_preds_{mode}_fold{k}_{criterion}.csv`
- Reproduce: `python temperature_scaling.py`
---

## Phase 6 — Sample-size sweep (is the gap a test-set-size artefact?)

### The objection this answers

"You trained on 365 subjects. The instability is a small-data artefact — measure
it on more subjects and it will disappear."

### Method

The ADNI predictions are subsampled at increasing n and both statistics are
recomputed at each size. No new inference is run: all 30 ADNI prediction CSVs
already exist, so this is pure resampling of numbers already on disk.

- Sampling is **without replacement** — genuine subsets of the real cohort, not
  bootstrap resamples.
- **Stratified by label.** ADNI is CN-heavy (1:1.97); an unstratified draw at
  n=100 could land with too few AD subjects for a stable AUC.
- **The identical subject subset is used across all five folds within a draw.**
  This is required, not cosmetic: the pooled-style statistic compares the five
  fold-models on the *same* subjects, so drawing different subsets per fold
  would confound subject composition with model disagreement.
- 20 draws per size, reported as mean ± SD with a 5th–95th percentile band. A
  single draw at small n could land anywhere by chance. The full cohort has
  exactly one possible draw, so its spread is zero by construction.

### Result — the gap does not move

| mode | selection | n=100 | n=200 | n=400 | n=800 | n=1287 |
|---|---|---|---|---|---|---|
| clinical | auc | 0.0388 | 0.0406 | 0.0378 | 0.0394 | 0.0388 |
| clinical | neg_brier | 0.0123 | 0.0119 | 0.0113 | 0.0120 | 0.0118 |
| imaging | auc | 0.1033 | 0.1044 | 0.1032 | 0.1043 | 0.1039 |
| imaging | neg_brier | 0.0184 | 0.0168 | 0.0173 | 0.0176 | 0.0176 |
| fusion | auc | **0.1763** | 0.1715 | 0.1730 | 0.1737 | **0.1732** |
| fusion | neg_brier | 0.0242 | 0.0194 | 0.0213 | 0.0210 | 0.0208 |

Across a 13× range of test-set size, the largest change in any of the six curves
is 0.0034 (fusion, `neg_brier`). Fusion under standard selection moves from
0.1763 to 0.1732 — a change of 0.0031 against a gap of 0.17.

What *does* change is precision. The fusion `auc` 5th–95th band narrows from
[0.1397, 0.2118] at n=100 to a point at n=1287, and the SD falls from 0.0261 to
0.0040 at n=800. **More evaluation data measures the gap more precisely without
moving it.** Measurement noise would shrink toward zero; this converges on a
stable non-zero value, which is the signature of a real property of the
fold-models.

The separation between selection criteria is visible at every sample size: the
`auc` curve sits above the `neg_brier` curve in all three modes at all five
sizes, so the remedy's effect is not a large-sample phenomenon either.

### What this does NOT establish

This concerns **test-set size only**. It says nothing about whether training on
more data would fix the instability — that would require retraining on
progressively larger training sets, which on ADNI would permanently burn the
clean external test set. The supported claim is the narrow one: *the gap is not
an artefact of the size of the set it is measured on.* Combined with the gap
persisting across two independent cohorts of very different sizes (Phase 3),
this is a reasonable answer to the small-data objection, but it should not be
stated more strongly than that.

### Artifacts (Phase 6)

- Code: `sample_size_sweep.py`
- Results: `outputs/sample_size/sample_size_results.json`,
  `outputs/sample_size/sample_size_curve.png`
- Reproduce: `python sample_size_sweep.py` (no GPU, no inference — seconds)