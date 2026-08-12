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

## Phase 2 — OASIS-3 (3D, imaging + genetics) — IN PROGRESS

Access granted. Planned as the primary training cohort:
- ~1,378 subjects (vs 192 usable in OASIS-1), full 3D T1 NIfTI volumes.
- Includes APOE status → enables the genetics branch and fusion architecture.
- 3D preprocessing pipeline (NIfTI loading, downsampling to fit ~8GB VRAM,
  intensity normalization) under construction (`dataset_3d.py`).

(Results to be added as experiments are run.)
