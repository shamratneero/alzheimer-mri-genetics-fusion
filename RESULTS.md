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

### Paired comparison (identical subjects, identical bootstrap resamples)

| comparison | ΔAUC | 95% CI | p | verdict |
|---|---|---|---|---|
| clinical vs imaging | +0.035 | [−0.024, +0.093] | 0.253 | not distinguishable |
| clinical vs fusion  | **+0.094** | [+0.033, +0.158] | **0.003** | **significant** |
| imaging vs fusion   | **+0.059** | [+0.003, +0.118] | **0.039** | **significant** |

The test was verified to detect a genuine difference when one exists (a
synthetic strong/weak pair returned p < 0.001), so "not distinguishable" is a
substantive result rather than an artefact of low power.

**Two APOE/age scalars significantly outperform the full multimodal model**
(+0.094, p=0.003). **Fusion is also significantly worse than imaging alone**
(+0.059, p=0.039) despite having strictly more information available to it.
A fusion model that is significantly worse than both of its own components is
not a marginal effect — it is a failure of naive concatenation-based fusion at
this scale.

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
   (0.729) sits below the 2-feature clinical model (0.764) and well below the
   5-feature demographics logistic regression (0.826). A 3.57M-parameter 3D CNN
   over 128³ volumes, at ~5 hours per fold, does not beat APOE e4 count and age.

## Limitations (Phase 2)

- **n = 365** caps absolute performance and widens every interval. The clinical
  branch currently uses 2 features against the EDA baseline's 5, so the
  non-imaging comparison is understated; a 5-feature rerun is pending.
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