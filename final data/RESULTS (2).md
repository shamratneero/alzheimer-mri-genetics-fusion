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
significantly worse than both of its own components is not a marginal effect.

**Important qualification, established in Phase 2d below.** The natural reading
of this result is that concatenation-based fusion fails at this scale. That
reading turns out to be wrong, or at least incomplete: re-selecting the
checkpoint from the *same training runs* using a calibration-aware criterion
raises pooled fusion AUC from 0.670 to 0.819 and removes the inversion. The
deficit reported here is therefore largely a property of **how the checkpoint
was chosen**, not of the fusion architecture. Everything above describes what
standard practice produces; Section 2d describes why.

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

## Phase 2d — Checkpoint selection is the cause, and it is fixable

Everything above uses the conventional rule: keep the checkpoint with the
highest validation AUC. That rule is standard — a recent cardiac-MRI
calibration study notes that most current work identifies the best model by
highest validation performance — and it is implemented correctly here. The
question this section asks is whether the rule itself is the problem.

### Retrospective evidence (free, from the existing logs)

The per-epoch validation history was already recorded for every fold. Replaying
it shows that AUC-based selection chose an epoch whose validation balanced
accuracy had **already collapsed to ~0.500** in 4 of 10 imaging/fusion
fold-runs, and that in each case a better-behaved epoch existed in the same run
and was passed over. The evidence of degeneracy was present in the very data
used for selection; the criterion ignored it because it only measures ranking.

### Design

Four criteria were tracked **simultaneously within a single training run**, each
keeping its own best checkpoint. Same folds, same weight trajectory, same random
state, so differences are attributable to the selection rule alone and the
comparison is paired rather than across separate runs.

| criterion | rule | rationale |
|---|---|---|
| `auc` | max validation AUC | baseline: current practice |
| `auc_minus_ece` | max (AUC − 0.5·ECE) | explicit calibration penalty; weight is arbitrary |
| `gated_bacc` | max AUC among epochs with bacc > 0.55 | rejects degenerate models, no arbitrary weighting |
| `neg_brier` | min Brier | Brier decomposes into calibration + refinement, so it balances both by construction |

Early stopping still keys off AUC in all cases, so the training trajectory is
identical to the runs reported above — otherwise differences could reflect run
length rather than selection. All criteria are evaluated on validation data
only; the test folds remain untouched.

### Results

Pooled out-of-fold AUC (n=365), by mode and criterion:

| criterion | clinical | imaging | fusion |
|---|---|---|---|
| `auc` (standard) | 0.7638 | 0.7292 | 0.6702 |
| `auc_minus_ece` | 0.7455 | 0.8032 | 0.8190 |
| `gated_bacc` | 0.7691 | 0.7971 | 0.7915 |
| **`neg_brier`** | **0.7983** | **0.8392** | **0.8195** |

Paired bootstrap against the `auc` baseline (2000 resamples, identical indices):

| mode | criterion | ΔAUC | 95% CI | p |
|---|---|---|---|---|
| clinical | `neg_brier` | **+0.0345** | [+0.0160, +0.0537] | **<0.001** |
| clinical | `gated_bacc` | **+0.0053** | [+0.0024, +0.0087] | **<0.001** |
| clinical | `auc_minus_ece` | −0.0183 | [−0.0464, +0.0105] | 0.204 |
| imaging | `neg_brier` | **+0.1100** | [+0.0666, +0.1577] | **<0.001** |
| imaging | `auc_minus_ece` | **+0.0740** | [+0.0130, +0.1371] | **0.018** |
| imaging | `gated_bacc` | **+0.0679** | [+0.0346, +0.1037] | **<0.001** |
| fusion | `neg_brier` | **+0.1493** | [+0.0924, +0.2123] | **<0.001** |
| fusion | `auc_minus_ece` | **+0.1488** | [+0.0882, +0.2140] | **<0.001** |
| fusion | `gated_bacc` | **+0.1213** | [+0.0455, +0.1988] | **<0.001** |

### The incoherence gap largely closes

| mode | gap under `auc` | gap under `neg_brier` |
|---|---|---|
| clinical | 0.0222 | **−0.0053** |
| imaging | 0.1200 | **0.0227** |
| fusion | 0.1989 | **0.0408** |

The per-fold-to-pooled gap is the quantity Section 2c identified as diagnostic of
cross-model probability incoherence. Under Brier selection it nearly vanishes.
Degenerate folds (test balanced accuracy < 0.55) go from one per mode to **zero**
under every calibration-aware criterion.

### The single clearest case

Fusion fold 4, one training run, four checkpoints:

| criterion | epoch | test AUC | bacc | ECE | Brier |
|---|---|---|---|---|---|
| `auc` | 10 | **0.897** | **0.500** | **0.486** | **0.474** |
| `auc_minus_ece` | 5 | 0.855 | 0.739 | 0.119 | 0.172 |
| `gated_bacc` | 5 | 0.855 | 0.739 | 0.119 | 0.172 |
| `neg_brier` | 17 | 0.860 | 0.729 | 0.138 | 0.174 |

The AUC-selected checkpoint has the **highest AUC of the four and is unusable**:
balanced accuracy exactly 0.500, i.e. one class predicted for all 73 test
subjects, with near-maximal ECE and Brier. Conceding 0.042 AUC yields a working
model with a 4x lower calibration error. This is not a case of the criterion
failing by its own standard — it is a case of the standard being wrong.
Two independent criteria selected the same epoch 5, which argues the choice is
robust rather than an artefact of one weighting.

### Two alternative explanations, ruled out

- **"Brier just trains longer."** On clinical, `neg_brier` did pick later epochs
  (30, 20, 16, 38, 47 vs 27, 5, 1, 23, 32). But on imaging it picked epoch 7
  where AUC picked 10, and on fusion fold 4 it picked 17 where AUC picked 10 in
  a run that stopped at 25. The direction is not consistent, so the benefit is
  not a training-duration effect.
- **"The penalty weight was tuned."** `auc_minus_ece` uses an arbitrary λ=0.5
  and is the *worst* of the three alternatives — significantly negative on
  clinical. `gated_bacc` introduces no weighting at all and still improves
  significantly on all three modes. The gain does not come from tuning.

### What this changes

The effect scales with how much the model depends on imaging: clinical +0.034,
imaging +0.110, fusion +0.149. Under calibration-aware selection the ordering
reported in Section 2c **reverses**: imaging (0.839) and fusion (0.819) both
exceed clinical (0.798).

The ranking inversion documented in Section 2c is therefore **substantially an
artefact of checkpoint selection**, not an inherent property of multimodal
fusion or of imaging at this sample size. Section 2c reports what standard
practice produces and remains a valid description of it. This section shows that
a one-line change to the selection rule removes most of the pathology.

---

## Findings (Phase 2)

Findings 1-3 describe what the **conventional pipeline** produces. Findings 4-6
identify the cause and are the transferable contribution. Findings 7-8 hold
regardless of selection rule.

1. **Per-fold means conceal cross-model probability incoherence.** Reporting
   mean per-fold AUC - near-universal in this literature - ranks fusion first.
   Pooled out-of-fold AUC over the same predictions ranks it last, significantly
   so. The two measures disagree because only the second requires fold-models to
   agree on what a given probability means. Report both; the gap between them is
   informative in its own right.

2. **Under standard selection, fusion is significantly worse than either single
   modality** (-0.094 vs clinical, p=0.003; -0.059 vs imaging, p=0.039). The
   obvious inference - that concatenation-based fusion fails at this scale - is
   **not supported**. Re-selecting checkpoints from the same training runs on a
   calibration-aware criterion lifts pooled fusion AUC from 0.670 to 0.819. The
   deficit was a selection artefact, not an architectural one.

3. **Under standard selection, discrimination and calibration decouple.** Adding
   modalities monotonically improved per-fold AUC (0.786 -> 0.849 -> 0.869) while
   monotonically worsening calibration (ECE 0.0955 -> 0.225 -> 0.309) and pooled
   coherence (0.764 -> 0.729 -> 0.670). Under Brier selection the coherence
   collapse largely disappears.

4. **AUC-based checkpoint selection actively selects broken models.** It is not
   merely insensitive to calibration failure. In 4 of 10 imaging/fusion
   fold-runs it chose an epoch whose validation balanced accuracy had already
   collapsed to ~0.500, while a better-behaved epoch existed in the same run and
   was passed over. In fusion fold 4 the selected checkpoint had the highest AUC
   of the four candidates and predicted a single class for all 73 test subjects.

5. **Calibration-aware selection repairs most of it, and costs one line.**
   Selecting on Brier score rather than AUC improved pooled out-of-fold AUC
   significantly in all three modes (+0.034 / +0.110 / +0.149, all p<0.001),
   eliminated every degenerate fold, and cut the per-fold-to-pooled gap from
   0.199 to 0.041 in fusion. Simply requiring validation balanced accuracy above
   0.55 before maximising AUC - which introduces no free parameter - achieves
   most of the same benefit.

6. **The harm scales with reliance on imaging.** Switching criteria gained
   +0.034 for clinical-only, +0.110 for imaging-only and +0.149 for fusion.
   Models depending more on the CNN are hurt more by AUC-based selection, which
   is consistent with the imaging branch being the source of the unstable
   probability scale.

7. **Single test splits at this size cannot rank models.** Bootstrap noise on
   n=73 is +/-0.060 AUC; the observed between-mode spread was 0.023. Even
   pooling all 365 subjects, the resolvable difference is ~0.053.

8. **The non-imaging signal is APOE and age, and nothing else.** Adding sex,
   education and socioeconomic status changed pooled AUC by +0.004 (p=0.856),
   so the imaging comparison is not made against a handicapped baseline.

## Limitations (Phase 2)

- **n = 365** caps absolute performance and widens every interval. The clinical
  branch has now been run at both 2 and 5 features, so the non-imaging baseline
  is no longer understated, but the cohort remains small.
- **The selection criteria were not themselves cross-validated.** Four criteria
  were evaluated on the same test folds. With four candidates the risk of
  selecting a criterion that happens to suit these folds is real, even though
  the effects are large and consistent across three independent modes. A nested
  design, choosing the criterion on an inner loop, would settle it.
- **λ = 0.5 in `auc_minus_ece` is arbitrary** and was not tuned. It is reported
  because it is the obvious first thing to try and because it performs *worst*
  of the three alternatives, which is itself informative — but it should not be
  read as an optimised weighting.
- **Brier selection is not obviously the right default.** It wins here, but
  Brier is threshold-dependent in a way AUC is not, and a cohort with different
  prevalence might behave differently. `gated_bacc` is more conservative,
  introduces no free parameter, and captures most of the gain.
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
- CV results (standard AUC selection):
  `outputs/cv/cv_{clinical_only,clinical_5feat,imaging_only,fusion}_*`
- CV results (multi-criterion selection):
  `outputs/cv/sel_{clinical,imaging,fusion}_*` — each fold's `by_criterion`
  block holds the epoch, validation metrics at selection, test metrics and
  out-of-fold predictions for all four criteria
- Single-split logs: `outputs/logs/`

Reproduce:

```bash
# mode comparison under standard selection
python cross_validate.py --compare cv_clinical_only cv_clinical_5feat \
                                   cv_imaging_only cv_fusion

# multi-criterion runs (all four criteria tracked in one pass per mode)
python cross_validate.py --mode clinical_only --tag sel_clinical
python cross_validate.py --mode imaging_only  --tag sel_imaging --resume
python cross_validate.py --mode fusion        --tag sel_fusion  --resume

# selection-criterion comparison within a mode
python cross_validate.py --compare_criteria sel_fusion
```