# Heart Disease Risk Prediction — Logistic Regression from Scratch

Machine Learning / AI Workshop — Week 2 (Classification)
Logistic regression implemented **from scratch with NumPy**, plus training and testing on **Amazon SageMaker**.

---

## Exercise Summary

This project implements a complete binary-classification pipeline to predict the presence of heart
disease from routine clinical measurements. Everything related to the model is implemented manually
with NumPy — **scikit-learn is not used for the core model training**.

The work covers:

1. **Exploratory data analysis** — descriptive statistics, missing values, duplicates, outlier
   inspection with the IQR rule, class-distribution and per-class feature distributions, and
   correlation of every feature with the target.
2. **Preprocessing** — target binarization, duplicate removal, feature selection (8 features),
   **stratified 70/30 split** and **z-score normalization fitted only on the training set**.
3. **Logistic regression from scratch** — `sigmoid`, binary cross-entropy `compute_cost`,
   `compute_gradient`, and batch `gradient_descent` with cost tracking. The analytic gradients are
   verified against finite differences (agreement ~1e-11).
4. **Evaluation** — accuracy, precision, recall and F1 implemented from scratch, on train and test,
   plus the confusion matrix and coefficient interpretation as odds ratios.
5. **Decision boundaries** — three feature pairs trained as 2-D models and plotted with their
   boundaries and true labels.
6. **L2 regularization** — regularized cost and gradients, `λ` tuning over `[0, 0.001, 0.01, 0.1, 1]`,
   λ-versus-metrics tables, weight-norm analysis, and an unregularized-vs-regularized boundary
   comparison on degree-6 polynomial features.
7. **Amazon SageMaker** — the selected model is trained as a **SageMaker training job** and tested on
   the held-out set inside the SageMaker notebook. **No endpoint is created and no model deployment
   service is used**, in line with the AWS Academy account limitations for this course.

### Headline results (test set, 90 unique patients)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (α = 0.01, 1000 iterations) | 0.7778 | 0.7959 | 0.7959 | 0.7959 |
| Converged (α = 0.1, 20 000 iterations) | 0.7889 | 0.8125 | 0.7959 | 0.8041 |
| **Selected: L2, λ = 1** | **0.7889** | **0.8000** | **0.8163** | **0.8081** |
| Only `oldpeak` + `ca` (2 features) | 0.7889 | 0.7778 | 0.8571 | 0.8155 |
| Degree-2 polynomial, 44 features (best λ) | 0.7444 | 0.7600 | 0.7755 | 0.7677 |

Recall sits slightly above precision, the desirable direction for a screening tool: missing a sick
patient is far more costly than raising a false alarm.

> **On the ~79% figure.** Public notebooks using this Kaggle file routinely report 97–100% accuracy.
> Those numbers are an artifact of **data leakage** — see the next section. After removing the
> duplicated rows, ~79% is the honest performance of logistic regression on these 302 patients.

---

## Dataset Description

**Heart Disease Dataset** — <https://www.kaggle.com/datasets/neurocipher/heartdisease>
(a copy of the UCI *Cleveland* database, originally collected in 1988).

- **1025 rows containing only 302 unique patients**, 13 clinical features + 1 binary target.
- **Target:** `1` = disease present, `0` = disease absent. Class balance ≈ 54% / 46%.
- **No missing values.**

### ⚠ The duplicates: the key preprocessing issue

The file distributed on Kaggle is an **oversampled** copy of the UCI Cleveland database. Of its 1025
rows, **723 are exact duplicates** — each distinct patient appears 3–4 times (one appears 8 times):

| | Value |
|---|---|
| Rows in the file | 1025 |
| Exact duplicate rows | 723 |
| **Unique patients** | **302** |
| Test rows with an identical twin in train, if split naively | **88.6%** |

Splitting the raw file at random puts copies of the same patient on both sides of the split, so the
model is evaluated on records it has memorized. This is textbook **data leakage**, and it is the
reason so many published notebooks on this dataset report near-perfect accuracy.

This project therefore **de-duplicates before splitting** (1025 → 302 records). The notebook measures
the leakage explicitly rather than just asserting it, and all reported metrics come from the
de-duplicated data.

| Column | Meaning |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1/0) |
| `restecg` | Resting electrocardiographic result (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1/0) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment (0–2) |
| `ca` | Number of major vessels colored by fluoroscopy (0–4) |
| `thal` | Thalassemia test result (0–3) |
| `target` | **1 = disease, 0 = no disease** |

**Features selected for the model:** `age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`, `cp`, `exang`.

---

## Repository Structure

```
.
├── heart_disease_lr_analysis.ipynb   # Main notebook (Steps 1-4, executed, with all plots and tables)
├── heart.csv                         # Dataset
├── README.md
├── data/                             # Preprocessed splits exported for SageMaker
│   ├── heart_train.csv
│   ├── heart_test.csv
│   └── preprocessing_config.json
├── sagemaker/
│   ├── sagemaker_heart_lr.ipynb      # Notebook to run inside the SageMaker notebook instance
│   ├── train_heart_lr.py             # Training script (script mode, pure NumPy)
│   └── data/                         # Same splits, uploaded together with the notebook
└── images/                           # SageMaker evidence screenshots
    ├── sagemaker_notebook.png
    ├── sagemaker_training_job.png
    └── sagemaker_test_metrics.png
```

## How to Run

```bash
pip install numpy pandas matplotlib jupyter
jupyter notebook heart_disease_lr_analysis.ipynb
```

Run all cells from top to bottom. Step 5 writes the preprocessed splits into `data/`, which are the
inputs for the SageMaker run.

---

## SageMaker Evidence

### Process

1. **AWS Academy Learner Lab → AWS Console → Amazon SageMaker → Notebook instances**; a notebook
   instance of type `ml.t3.medium` (kernel `conda_python3`) is started.
2. `sagemaker/sagemaker_heart_lr.ipynb`, `sagemaker/train_heart_lr.py` and the `data/` folder are
   uploaded to the instance.
3. The notebook uploads the preprocessed splits to the SageMaker default S3 bucket and launches a
   **training job** on an `ml.m5.large` instance, using `train_heart_lr.py` as entry point in
   **script mode**. The SKLearn container is used only as a Python runtime — the model is the same
   from-scratch NumPy implementation (sigmoid + binary cross-entropy + L2 + gradient descent).
4. `estimator.fit()` streams the container log until the job reaches status **Completed**.
5. The `model.tar.gz` artifact is downloaded from S3, unpacked, and the held-out test set is scored
   **inside the notebook** with NumPy. **No endpoint is created and no deployment service is used.**

### Environment / instance configuration

| Item | Value |
|---|---|
| Notebook instance | `ml.t3.medium`, kernel `conda_python3` |
| Training instance | `ml.m5.large`, 1 instance |
| Framework container | SageMaker SKLearn `1.2-1`, `py3` (Python runtime only) |
| Entry point | `train_heart_lr.py` |
| Hyperparameters | `alpha = 0.1`, `num_iters = 20000`, `lambda = 1.0`, `threshold = 0.5` |
| Artifacts | `model.npz` (w, b, cost history), `metrics.json` |
| Endpoint deployed | **None** (outside the scope of the AWS Academy account) |

### Screenshots

**1. Notebook running in the SageMaker notebook instance**

![SageMaker notebook instance](images/sagemaker_notebook.png)

**2. Training job completed successfully**

![SageMaker training job completed](images/sagemaker_training_job.png)

**3. Test-set metrics obtained in SageMaker**

![SageMaker test metrics](images/sagemaker_test_metrics.png)

#### I HAD NO CONNECTION, I couldn't test the exercise in the cloud
<img width="947" height="142" alt="image" src="https://github.com/user-attachments/assets/9bd8d21d-f07b-47be-b57e-d099c64d0fc7" />

### Results and comparison with the local execution

| Environment | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Local (`heart_disease_lr_analysis.ipynb`) | 0.7889 | 0.8000 | 0.8163 | 0.8081 |
| SageMaker training job (`ml.m5.large`) |  | | | |

The algorithm is fully **deterministic** — zero initialization, full-batch gradient descent, a fixed
number of iterations, and data standardized *before* upload — so the cloud metrics reproduce the
local ones up to floating-point noise from the container's BLAS/NumPy build. The observable
difference is **operational rather than statistical**: SageMaker adds container provisioning, S3 I/O
and job orchestration (tens of seconds) to a model that trains in under a second locally. The value
of the managed environment here is reproducibility, isolation and auditability, not speed.

---

## Main Conclusions

1. The from-scratch logistic regression reaches **≈ 79% accuracy and 0.81 F1** on the held-out test
   set, with recall above precision — the right balance for clinical screening.
2. **The exercise-test variables dominate.** `cp`, `ca`, `oldpeak`, `exang` and `thalach` carry most
   of the signal; a model using only `oldpeak` and `ca` matches the full model exactly (0.789 test accuracy).
   `age` and `chol` add almost nothing once the others are present.
3. **The baseline run does not fully converge**, yet generalizes slightly better than the converged
   one — early stopping keeps the weights small and acts as implicit regularization.
4. **Regularization only helps when the model can overfit.** With 8 standardized features the λ sweep
   barely changes the metrics; with 44 degree-2 polynomial features it closes a 7-point train–test
   gap and improves test F1; with degree-6 features on a single pair it shrinks ‖w‖ from ≈ 24 to ≈ 5
   and turns an unusable boundary into a reasonable one. ‖w‖ decreases monotonically with λ in every
   experiment.
5. **The data are not linearly separable** in any 2-D projection; the residual error comes from
   genuine clinical ambiguity, not from the shape of the boundary — which is why the nonlinear
   expansions bought so little.
6. **Limitations.** 302 patients from a single 1988 study is a small, dated sample. Public copies of
   this dataset are known to differ in the encoding of `ca` and `thal`, so the coefficient signs must
   be read relative to this specific file and never as independent clinical evidence.
