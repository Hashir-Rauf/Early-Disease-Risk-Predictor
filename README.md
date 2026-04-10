# Early Disease Risk Predictor

An AI-powered, explainable health risk assessment system for early detection of chronic diseases, built with XGBoost, SHAP, and a multi-dataset clinical pipeline.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange?style=flat-square)
![Track](https://img.shields.io/badge/Track-Type%20A%20Application%20Development-purple?style=flat-square)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Diseases Targeted](#diseases-targeted)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Progress](#progress)
- [Getting Started](#getting-started)
- [Team](#team)

---

## Overview

Chronic non-communicable diseases such as Type 2 Diabetes, Cardiovascular Disease, and Hypertension remain widely underdiagnosed until they reach advanced stages. This project builds an AI-powered risk assessment tool that takes a user's vitals, lab results, and lifestyle inputs and returns an interpretable risk score for one or more of these conditions.

Explainability (XAI) is treated as a primary output, not an afterthought. The system surfaces SHAP-based feature contributions interactively so users understand not just their risk level but the specific factors driving it.

---

## Features

- Multi-disease risk scoring for diabetes, cardiovascular disease, and hypertension from a single input
- Integration of three distinct clinical datasets (no single pre-cleaned CSV)
- Interactive SHAP explainability as a primary UI component
- Comorbidity-aware modelling to capture interactions between conditions

---

## Diseases Targeted

| Disease | Key Input Features |
|---|---|
| Type 2 Diabetes | Glucose, BMI, Insulin, Age, Blood Pressure |
| Cardiovascular Disease | Cholesterol, BP, ECG results, Chest pain type |
| Hypertension | Systolic BP, BMI, Glucose, Age |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Environment | Jupyter Notebook (.ipynb) |
| ML Models | XGBoost, scikit-learn |
| Explainability | SHAP |
| Data Processing | pandas, NumPy, SciPy |
| Visualisation | matplotlib, seaborn |
| Serialisation | joblib |

---

## Repository Structure

```
Early-Disease-Risk-Predictor/
|
+-- Documents/                         # Technical report (LaTeX PDF)
|
+-- Scripts/
|   +-- code/
|   |   +-- 01_data_collection.ipynb
|   |   +-- 02_cleaning.ipynb
|   |   +-- 03_eda.ipynb
|   |   +-- 04_feature_engineering.ipynb
|   |   +-- 05_model_training.ipynb    # (upcoming)
|   |   +-- 06_evaluation.ipynb        # (upcoming)
|   |   +-- 07_xai_dashboard.ipynb     # (upcoming)
|   |
|   +-- data/
|   |   +-- raw/                       # Downloaded source datasets
|   |   +-- processed/                 # Cleaned, merged, scaled outputs
|   |
|   +-- reports/
|       +-- figures/                   # EDA and feature engineering plots
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
| `06_model_training.ipynb` | XGBoost + MLP training, RandomizedSearchCV | Upcoming |
| `07_evaluation.ipynb` | Metrics, ROC curves, cross-validation | Upcoming |
| `08_xai_dashboard.ipynb` | Interactive SHAP explainability interface | Upcoming |

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
