# 📋 Model Card: Parkinson’s Voice Feature Screening Prototype

## 1. Model Overview & Positioning

- **Model Name:** Parkinson’s Voice Feature Screening Pipeline (Leakage-Aware Prototype)
- **Version:** `1.0.0`
- **Model Architecture:** `StandardScaler` $\to$ L2 `LogisticRegression` with fixed Subject-Level `median` Aggregation.
- **Input Scope:** 20 model features derived from 22 pre-extracted tabular acoustic voice features from sustained phonation `/a/` (UCI Parkinsons).
- **Audio Limitation:** **The model does NOT process raw audio files (WAV, MP3, FLAC).** It operates strictly as a tabular acoustic feature screening model.

---

## 2. Intended Use & Clinical Boundaries

### ✅ Intended Uses:
- **Academic & Portfolio Research:** Demonstrating leakage-aware validation, patient-level stratification, nested CV and subject bootstrap uncertainty on grouped biomedical tabular data.
- **Experimental Screening Signal:** Generating recording `screening_score` values and a subject-level `model-positive` or `model-negative` result under explicit research caveats.

### ❌ Non-Intended Uses:
- **NOT a Medical Diagnostic System:** The model cannot diagnose Parkinson’s Disease or replace neurological examination, dopamine transporter imaging (DaTscan), or clinical motor scoring (MDS-UPDRS).
- **NOT an End-to-End Voice Diagnosis Tool:** It does not extract features from raw microphone recordings or perform acoustic signal processing.
- **NOT Validated for Clinical Deployment:** The prototype has not undergone clinical trial validation, multi-site external validation, or regulatory clearance (e.g., FDA 510(k), CE-MDR).

---

## 3. Training & Evaluation Data

- **Dataset Source:** [UCI Machine Learning Repository: Parkinsons Telemonitoring / Voice Dataset](https://archive.ics.uci.edu/dataset/174/parkinsons).
- **Cohort Size:** 195 acoustic recordings from **32 distinct subjects** (24 subjects with `status=1`, 8 with `status=0`). Each subject provided 6–7 sustained vowel phonations.
- **Data Partitioning (Zero-Leakage Invariant):**
- **Evaluation Cohort:** All 32 subjects are used by nested 4-fold outer CV; each subject is outer-test exactly once.
- **Inner Selection:** Three subject-level inner folds tune only `C` and `class_weight`.
- **Constraint:** A subject never crosses train/validation boundaries inside an outer or inner fold.

---

## 4. Canonical 8-Stage Architecture

```
1. DATA INGESTION
   UCI Parkinsons Tabular Acoustic Features (195 recordings / 32 subjects)
          ↓
2. SUBJECT IDENTITY & SCHEMA AUDIT
   Schema validation, subject_id extraction, remove algebraic redundancies (Jitter:DDP, Shimmer:DDA)
          ↓
3. NESTED PATIENT-LEVEL CROSS-VALIDATION
   4 Outer Folds × 3 Inner Folds (32 Subjects, Zero Leakage)
          ↓
4. MODEL DEVELOPMENT INSIDE OUTER TRAIN
   Subject-Stratified Inner Folds → StandardScaler → Logistic Regression
          ↓
5. FIXED MODEL CONFIGURATION
   Only C and class_weight are selected; production architecture stays logistic
          ↓
6. SCORE & DECISION LAYER
   Recording scores → median per subject → threshold from outer-train OOF
          ↓
7. CROSSFITTED EVALUATION
   32 Cross-Fitted Subject Predictions → Metrics + Subject Bootstrap 95% CI (5,000x)
          ↓
8. SERVING & RELIABILITY LAYER
   FastAPI & Streamlit → training-range warning → INSUFFICIENT_RECORDINGS
```

---

## 5. Performance Metrics & Statistical Uncertainty

### ⚠️ Critical Sample Size Caveat
The dataset contains only **32 subjects (24 positive, 8 control)**. Cross-fitted
metrics and bootstrap intervals are internal research estimates with high
variance, not clinical validation. There is no external patient cohort.

### Performance Summary Table

| Evaluation Layer | Cohort | F1-Macro | Balanced Accuracy | Sensitivity | Specificity | ROC-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nested Subject CV** | 32 subjects (4 Outer × 3 Inner) | generated in `artifacts/metrics.json` | generated | generated | generated | generated | generated |
| **Subject Bootstrap 95% CI** | 32 cross-fitted subjects (5,000 replicates) | see `artifacts/evaluation/bootstrap_ci.csv` | see artifact | see artifact | see artifact | see artifact | see artifact |

---

## 6. Serving Guardrails & Training-Range Policy

To prevent silent failures and deceptive predictions during inference, the serving runtime incorporates three automated guardrails:

1. **P1–P99 Feature Range Checks:**
   During training, the 1st and 99th percentiles ($P_1, P_{99}$) of all 20 modeling features are recorded. If any input feature falls outside $[P_1, P_{99}]$, the system emits a `FEATURE_OUTSIDE_TRAINING_RANGE` warning and marks subject reliability as `"limited"`. This is a plausibility warning, not an OOD detector.
2. **Minimum Recordings Policy:**
   The minimum is read from the training distribution and stored in the artifact. When fewer recordings are supplied, the system issues `INSUFFICIENT_RECORDINGS` and classifies reliability as `"limited"`.
3. **Training Label Rejection:**
   The canonical inference API (`POST /v1/screen/subject`) rejects payloads containing the training target `status` with a `422 Unprocessable Entity` status.

---

## 7. Limitations & Technical Debt

1. **Severe Sample Size Limitations:** 32 subjects total (only 8 healthy controls in the entire dataset). Small validation folds are prone to high metric variance.
2. **Lack of Demographic Covariates:** The UCI dataset omits age, biological sex, recording hardware, and clinical site metadata. Subgroup fairness and demographic parity cannot be audited.
3. **Absence of External Cohort Validation:** The pipeline has not been tested against external speech datasets (e.g., PC-GITA, mPower) due to acoustic schema differences.
4. **Artifact Serialization Security:** Current deployment bundles use `joblib`. While standard for local portfolios, enterprise deployments should migrate to hardened formats like `skops` or `ONNX` to eliminate arbitrary code execution vulnerabilities.
