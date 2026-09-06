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

### Analysis (Phases 4-8)

| script | purpose |
|---|---|
| `gradcam_analysis.py` | Grad-CAM on the imaging branch, comparing `auc`- and `neg_brier`-selected checkpoints; reports brain selectivity against a chance baseline (Phase 4, null result) |
| `temperature_scaling.py` | fits temperature on each fold's validation split and tests whether post-hoc calibration substitutes for calibration-aware selection (Phase 5) |
| `sample_size_sweep.py` | subsamples the ADNI predictions at increasing n to test whether the per-fold/pooled gap is an artefact of test-set size (Phase 6) |
| `resnet2d_cv.py` | ResNet-18 on the identical 5-fold protocol as a negative control; `--train_seed` varies training only, fold construction is locked (Phase 7) |
| `scale_incoherence_test.py` | rank transform and permutation null testing whether the gap is genuine cross-model scale incoherence or the documented negative bias of pooled AUC (Airola et al.) |
| `calibration_metrics.py` | ECE, AECE, MCE, OE and Brier on both cohorts plus per-fold reliability diagrams; imports ECE from `train.py` rather than reimplementing it (Phase 8) |
| `oe_degeneracy_check.py` | tests whether overconfidence error scores one-class degenerate models as well calibrated; reports direction of degeneracy separately and can falsify the hypothesis |
| `build_experiment_c_inputs.py`, `experiment_c_analysis.py` | tests whether any internal measure predicts a fold-model's external AUC; **null result**, no predictor is sign-consistent and all pooled correlations fall inside a permutation null |
| `tools/verify_results_numbers.py` | cross-checks Phase 3-7 numbers in `RESULTS.md` against their committed JSON (116 values) |

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
- `outputs/sample_size/` — sample-size sweep results and curve
- `outputs/resnet2d/`, `outputs/resnet2d_scratch*/` — ResNet-18 negative control, pretrained and four scratch seeds
- `outputs/scale_test/` — rank transform and permutation null results
- `outputs/calibration/` — five-metric calibration panel and the OE degeneracy check
- `figures_calibration/` — per-fold reliability diagrams, one panel per mode and criterion
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
python sample_size_sweep.py                 # no GPU, seconds

# architecture negative control (trains; ~2h per seed)
python resnet2d_cv.py --pretrained
python resnet2d_cv.py --train_seed 1 --out_dir outputs/resnet2d_scratch_s1

# estimator-bias control and calibration panel (no GPU, seconds)
python scale_incoherence_test.py            # rank transform + permutation null
python calibration_metrics.py               # 5 metrics + reliability diagrams
python oe_degeneracy_check.py               # OE blind-spot test

# does any internal measure forecast external AUC? (null result)
python build_experiment_c_inputs.py
python experiment_c_analysis.py

python tools/verify_results_numbers.py      # checks Phases 3-7, 116 values
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

**The gap is not an artefact of test-set size** (Phase 6). Subsampling the ADNI
predictions from n=100 to n=1,287 leaves it essentially unchanged — fusion under
standard selection moves only 0.1763 → 0.1732 across a 13× range — while the
5th–95th band narrows from [0.140, 0.212] to a point. More evaluation data
measures the gap more precisely without moving it, which is what a stable
property looks like rather than measurement noise. This concerns test-set size
only; it does not establish what more *training* data would do.

**The instability is architecture-dependent** (Phase 7). ResNet-18 was run on
the identical 5-fold protocol as a negative control. Under standard selection
the gap is 0.0054 pretrained and 0.0344 ± 0.0313 across four scratch seeds,
against 0.1200 for the 3D CNN on the same data — large and reproducible in the
custom CNN, not reliably present in ResNet-18 under any of five configurations.
A diagnostic that found instability everywhere would be uninformative, so this
negative control matters. Note that an earlier conclusion drawn from the
seed-0-versus-pretrained pair — that ImageNet pretraining removes the
instability — **was withdrawn** after three further seeds showed seed 0 was a
5× outlier. The retraction is recorded in `RESULTS.md` deliberately rather than
deleted: it is the same failure this work documents, a single confident-looking
number with a plausible mechanism available, caught only by repeating and
reporting spread.

**Five calibration metrics are reported, and they do not agree** (Phase 8).
ECE, AECE, MCE, OE and Brier were computed on both cohorts from the saved
predictions — four selection criteria on OASIS-3, two on ADNI (only those two
checkpoints were pushed through external inference). Calibration-aware
selection clearly helps where imaging reliance is high: OASIS-3 fusion ECE
falls 0.3093 → 0.1566 and imaging 0.2254 → 0.1363 under `neg_brier`, with the
same direction on ADNI. But no criterion wins on all five metrics anywhere.
`auc_minus_ece` beats `neg_brier` on OASIS-3 fusion ECE (0.1503 vs 0.1566) and
MCE (0.4932 vs 0.5764), and on clinical — the mode with least to repair —
`neg_brier` makes ECE *worse* (0.0955 → 0.1246). Calibration-aware selection is
therefore presented as a **family**, never as a prescription for one criterion.

**Overconfidence error is defeated by the failure mode under study.** OE is
asymmetric by construction: it sums `weight × mean_prob × max(mean_prob −
positive_rate, 0)`, so under-confidence contributes exactly zero. A model
predicting the negative class for every subject therefore scores OE = 0 — a
perfect result — while scoring worst on every other metric. All three
degenerate folds in this study (OASIS-3 imaging fold 2, OASIS-3 fusion fold 4,
ADNI imaging fold 2) score OE exactly 0.0000 against a working-model median of
0.0299, alongside ECE ≈ 0.48 and Brier ≈ 0.47 on OASIS-3. This does not rest on
n=3: OE = 0 for an all-CN model is an algebraic consequence, not an empirical
coincidence. The narrow claim is that OE should not be read as evidence of
reliability without a companion metric sensitive to under-confidence — not that
OE is broken, since it never claimed to measure anything else.

Remaining:

1. Seed repeats on the 3D CNN (imaging_only). **Seed 1 complete** (tag
   `seed1_imaging`, 403 min). Per-fold auc-selected AUC: 0.825 / 0.835 /
   0.849 / 0.885 / 0.891 (mean 0.857), pooled OOF AUC 0.804. The remedy
   replicates: neg_brier beats auc on 3 of 5 folds, ties 1, loses 1 (by
   0.013). No degenerate fold under seed 1 — confirming that degeneracy
   occurrence is stochastic (fold identity varies across seeds) while the
   gap magnitude reproduces reliably. Seed 2 and one fusion seed remain.
2. Reverse direction (train ADNI → test OASIS-3) and pooled-cohort CV. Note that
   once ADNI enters training it is permanently unusable as a clean external test
   set, so this comes last.

MedicalNet-pretrained 3D ResNet was considered and **deprioritised**: roughly
125 GPU-hours for one run, and Phase 7 established that single runs here are
unreliable (SD comparable to the mean), so doing it properly would need ~375
hours. It is stated as future work rather than run, and would be revisited only
if a reviewer requires broader 3D architecture validation or a manuscript claim
cannot be defended without pretrained-architecture evidence.

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