# 📋 Model Card: Parkinson’s Voice Feature Screening (Subject-Level)

## 1. Model Overview

- **Model Name:** Parkinson’s Voice Feature Screening Pipeline
- **Model Type:** `StandardScaler` $\to$ L2 regularized `LogisticRegression` (`C=0.01`, `class_weight='balanced'`)
- **Aggregation:** Subject-level `median` score aggregation across repeated recordings
- **Input Scope:** 20 precomputed acoustic voice features from sustained phonation `/a/` (UCI Parkinsons dataset)
- **Audio Limitation:** **The model does NOT process raw audio files (WAV, MP3, FLAC).** It operates strictly on tabular acoustic measurements.

---

## 2. Intended Use & Boundaries

### ✅ Intended Use:
- **Academic & Portfolio Research:** Demonstrating leakage-aware evaluation, subject-level stratification, nested cross-validation, and bootstrap uncertainty estimation on grouped biomedical tabular data.
- **Methodological Benchmark:** Illustrating the difference between naive row-level splits and subject-level evaluation.

### ❌ Out-of-Scope / Non-Intended Uses:
- **NOT a Clinical Diagnostic Device:** Cannot diagnose Parkinson’s Disease or replace neurological examination, DaTscan, or clinical motor scoring (MDS-UPDRS).
- **NOT an Audio Processing Tool:** Does not extract features from microphone signals.
- **NOT Validated for Clinical Practice:** No clinical trial validation or regulatory clearance (e.g., FDA 510(k), CE-MDR).

---

## 3. Dataset & Preprocessing

- **Dataset Source:** [UCI Machine Learning Repository: Parkinsons Dataset](https://archive.ics.uci.edu/dataset/174/parkinsons)
- **Cohort Size:** 195 sustained vowel recordings across **32 unique subjects** (24 positive, 8 healthy controls).
- **Grouping:** Each subject has multiple recordings (typically 6 per person). All recordings of the same subject share the same label.
- **Feature Selection:** 22 original acoustic features $\to$ 20 features used. `Jitter:DDP` and `Shimmer:DDA` are excluded because they are deterministic algebraic multiples of `MDVP:RAP` ($3\times$) and `Shimmer:APQ3` ($3\times$), not due to statistical feature selection.

---

## 4. Evaluation Protocol

- **Protocol:** Nested Stratified Subject-Level Cross-Validation (4 outer folds $\times$ 3 inner folds).
- **Zero-Leakage Invariant:** All recordings from any single subject belong strictly to train or test within each fold; subject sets between fit and test are strictly disjoint.
- **Inner Folds:** Tune regularization parameter $C \in [0.01, 0.1, 1, 10, 100]$, `class_weight \in [None, "balanced"]`, and internal research decision threshold on inner out-of-fold predictions.
- **Outer Folds:** Evaluate generalization performance on completely held-out subjects.
- **Subject Aggregation:** Individual recording probabilities are aggregated by subject using **median**, minimizing sensitivity to noisy outlier recordings.
- **Uncertainty Estimation:** 95% Confidence Intervals computed via subject-level cluster bootstrap.

---

## 5. Performance Summary

> **Statistical Uncertainty Caveat:** With only 32 subjects (24 PD vs 8 controls), performance estimates exhibit wide variance. Reported metrics reflect internal nested CV, not multi-center clinical validation.

| Metric | Cross-Fitted Subject Estimate | 95% Bootstrap CI | Notes |
| :--- | :---: | :---: | :--- |
| **Balanced Accuracy** | **0.625** | [0.44 – 0.81] | Primary metric accounting for 3:1 class imbalance |
| **ROC-AUC** | **0.740** | [0.55 – 0.90] | Discriminative capacity across thresholds |
| **Sensitivity (Recall)** | **0.750** | [0.54 – 0.92] | True positive rate on PD subjects |
| **Specificity** | **0.500** | [0.12 – 0.88] | True negative rate on healthy controls |
| **Macro-F1** | **0.614** | [0.42 – 0.80] | Unweighted harmonic mean across classes |
| **Brier Score** | **0.215** | [0.12 – 0.31] | Mean squared probability error |

*Decision Threshold: An internal research threshold ($\approx 0.42$) selected to maximize out-of-fold balanced accuracy. It is a statistical research parameter, not a clinical operating cutoff.*

---

## 6. Inference Guardrails

- **Training-Range Plausibility Warnings:** Compares input features against observed training 1st–99th percentiles ($P_1, P_{99}$). If an input value falls outside this range, a `FEATURE_OUTSIDE_TRAINING_RANGE` warning is issued.
- **Recording Count Warning:** Emits a warning when fewer recordings than the training protocol are supplied for a subject.
- **Target Status Rejection:** Inference APIs reject inputs containing the training label `status` (`HTTP 422`).

---

## 7. Limitations

1. **Small Cohort Size:** Only 32 individuals (8 controls). Sub-cohorts in cross-validation folds have few control patients, leading to substantial metric uncertainty.
2. **No Demographic Covariates:** The UCI dataset lacks age, biological sex, recording hardware, and acquisition site details. Fairness across demographic groups cannot be audited.
3. **No External Cohort Validation:** Not validated on independent datasets (e.g., PC-GITA, mPower) due to measurement protocol variations.
4. **Artifact Security:** Models are serialized using `joblib`. Artifacts should only be loaded from trusted local sources.
