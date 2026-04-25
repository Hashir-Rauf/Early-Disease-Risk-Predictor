# Early Disease Risk Predictor

An AI-powered, explainable health risk assessment system for early detection of chronic diseases, built with XGBoost, SHAP, and a multi-dataset clinical pipeline.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Track](https://img.shields.io/badge/Track-Type%20A%20Application%20Development-purple?style=flat-square)
![HF Spaces](https://img.shields.io/badge/Deployed-Hugging%20Face%20Spaces-orange?style=flat-square&logo=huggingface)

---

## Live Demo

| Link | Description |
|---|---|
| **[https://hrm05-early-disease-risk-predictor.hf.space](https://hrm05-early-disease-risk-predictor.hf.space)** | Live Gradio app (real-time inference) |
| **[https://huggingface.co/spaces/hrm05/early-disease-risk-predictor](https://huggingface.co/spaces/hrm05/early-disease-risk-predictor)** | Hugging Face Space (source + logs) |

Enter patient vitals, click **Predict Risk**, and receive instant risk scores for Diabetes, Heart Disease, and Hypertension with full SHAP waterfall explanations.

---

## Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Features](#features)
- [Diseases Targeted](#diseases-targeted)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Progress](#progress)
- [Model Performance](#model-performance)
- [Deployment](#deployment)
- [Getting Started](#getting-started)
- [Team](#team)

---

## Overview

Chronic non-communicable diseases such as Type 2 Diabetes, Cardiovascular Disease, and Hypertension remain widely underdiagnosed until they reach advanced stages. This project builds an AI-powered risk assessment tool that takes a user's vitals and lab results and returns an interpretable risk score for all three conditions simultaneously.

Explainability (XAI) is treated as a primary output, not an afterthought. The system surfaces SHAP-based feature contributions interactively so users understand not just their risk level but the specific factors driving it.

---

## Features

- Multi-disease risk scoring for Diabetes, Heart Disease, and Hypertension from a single input
- Integration of three distinct clinical datasets (PIMA, UCI Heart Disease, Framingham Heart Study)
- Interactive SHAP waterfall explanations per disease per patient
- XGBoost primary model tuned via RandomizedSearchCV with class-imbalance correction
- Bias analysis across age, BMI, and dataset-source subgroups
- Fully deployed Gradio frontend on Hugging Face Spaces

---

## Diseases Targeted

| Disease | Key Input Features |
|---|---|
| Type 2 Diabetes | Glucose, BMI, Insulin, Age, Blood Pressure |
| Cardiovascular Disease | Cholesterol, Blood Pressure, Age |
| Hypertension | Systolic BP, BMI, Age, Glucose |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Environment | Jupyter Notebook (.ipynb) |
| ML Models | XGBoost 3.0, scikit-learn 1.6 |
| Explainability | SHAP 0.47 |
| Data Processing | pandas, NumPy, SciPy |
| Visualisation | matplotlib, seaborn |
| Serialisation | joblib |
| Frontend | Gradio 5 |
| Deployment | Hugging Face Spaces |

---

## Repository Structure

```
Early-Disease-Risk-Predictor/
|
+-- deployment/                        # Hugging Face Spaces deployment package
|   +-- app.py                         # Gradio frontend + inference logic
|   +-- requirements.txt               # Space dependencies
|   +-- README.md                      # HF Spaces metadata card
|   +-- models/                        # Serialised XGBoost models + scaler
|       +-- xgb_diabetes.pkl
|       +-- xgb_heart_disease.pkl
|       +-- xgb_hypertension.pkl
|       +-- standard_scaler.pkl
|
+-- Documents/                         # Technical report (LaTeX PDF)
|
+-- Scripts/
|   +-- code/
|   |   +-- 01_data_collection.ipynb
|   |   +-- 02_cleaning.ipynb
|   |   +-- 03_eda.ipynb
|   |   +-- 04_feature_engineering.ipynb
|   |   +-- 05_model_architecture.ipynb
|   |   +-- 06_model_training.ipynb
|   |   +-- 07_xai_dashboard.ipynb
|   |
|   +-- data/
|   |   +-- raw/                       # Downloaded source datasets
|   |   +-- processed/                 # Cleaned, merged, scaled outputs
|   |   +-- models/                    # Trained model weights (.pkl)
|   |
|   +-- reports/
|       +-- figures/                   # EDA, feature engineering, evaluation, SHAP plots
|
+-- .gitignore
+-- README.md
+-- requirements.txt
```

---

## Progress

| Notebook | Description | Status |
|---|---|---|
| `01_data_collection.ipynb` | Download PIMA, UCI Heart, Framingham datasets via URL | Done |
| `02_cleaning.ipynb` | Missing values, duplicates, outliers, integration | Done |
| `03_eda.ipynb` | Distributions, correlations, class balance, pairplot | Done |
| `04_feature_engineering.ipynb` | Log transform, encoding, scaling, per-disease X/y export | Done |
| `05_model_architecture.ipynb` | Framework selection, model configs, hyperparameter tables | Done |
| `06_model_training.ipynb` | Training, RandomizedSearchCV, metrics, error analysis, bias check | Done |
| `07_xai_dashboard.ipynb` | SHAP waterfall, beeswarm, force, decision, dependence plots + patient interface | Done |
| **Deployment** | Gradio app live on Hugging Face Spaces | **Done** |

---

## Model Performance

Test-set results for the primary model (XGBoost, tuned via RandomizedSearchCV):

| Disease | Accuracy | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Diabetes | 0.934 | 0.981 | 0.422 | 0.875 | 0.569 |
| Heart Disease | 0.686 | 0.758 | 0.268 | 0.658 | 0.381 |
| Hypertension | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Recall is prioritised as the primary clinical metric (minimising missed diagnoses). Class imbalance is handled via `scale_pos_weight` per disease.

---

## Deployment

The app is deployed on **Hugging Face Spaces** using **Gradio 5**.

| Item | Detail |
|---|---|
| Live app | https://hrm05-early-disease-risk-predictor.hf.space |
| HF Space | https://huggingface.co/spaces/hrm05/early-disease-risk-predictor |
| Framework | Gradio 5 (`gr.Blocks`) |
| Models served | XGBoost (diabetes, heart disease, hypertension) |
| Model size | ~730 KB total |
| Explainability | SHAP waterfall charts via XGBoost native `pred_contribs` |
| Inputs | Age, BMI, Fasting Glucose, Systolic BP, Insulin, Cholesterol |
| Outputs | Risk % per disease + SHAP waterfall per disease |

### Running the app locally

```bash
cd deployment
pip install -r requirements.txt
python app.py
```

---

## Getting Started

```bash
git clone https://github.com/Hashir-Rauf/Early-Disease-Risk-Predictor.git
cd Early-Disease-Risk-Predictor

python -m venv venv
venv\Scripts\activate        # On Mac/Linux: source venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

Open notebooks from the `Scripts/code/` directory and run them sequentially from `01` through `07`. Each notebook saves its outputs to `Scripts/data/processed/` or `Scripts/reports/figures/` for the next notebook to consume.

---

## Team

| Name | Student ID | University |
|---|---|---|
| Hashir Rauf | 23L-2572 | FAST-NUCES, Lahore |
| Minahil Mir | 23L-2517 | FAST-NUCES, Lahore |

**Project track:** Type A -- Application Development  
**Supervisor approval status:** Approved
