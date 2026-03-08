# Early Disease Risk Predictor

An AI-powered, explainable health risk assessment system for early detection of chronic diseases.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Status](https://img.shields.io/badge/Status-Planning-lightgrey?style=flat-square)
![Track](https://img.shields.io/badge/Track-Type%20A%20Application%20Development-purple?style=flat-square)

---

## Table of Contents

- [Overview](#overview)
- [Planned Features](#planned-features)
- [Diseases Targeted](#diseases-targeted)
- [Proposed Tech Stack](#proposed-tech-stack)
- [Notebook Structure](#notebook-structure)
- [Getting Started](#getting-started)
- [Team](#team)

---

## Overview

Chronic non-communicable diseases such as Type 2 Diabetes, Cardiovascular Disease, and Hypertension remain widely underdiagnosed until they reach advanced stages. This project aims to build an AI-powered risk assessment tool that takes a user's vitals, lab results, and lifestyle inputs and returns an interpretable risk score for one or more of these conditions.

A core requirement of the project is that explainability (XAI) is treated as a primary output, not an afterthought. The system will surface SHAP-based feature contributions interactively so users understand not just their risk level but the specific factors driving it.

This repository is in its initial planning phase. Notebooks and model files will be added as development progresses.

---

## Planned Features

- Multi-disease risk scoring for diabetes, cardiovascular disease, and hypertension from a single input
- Integration of multiple distinct clinical datasets (no single pre-cleaned CSV)
- Interactive SHAP explainability as a primary UI component
- Comorbidity-aware modelling to capture interactions between conditions

---

## Diseases Targeted

| Disease | Key Input Features |
|---|---|
| Type 2 Diabetes | Glucose, HbA1c, BMI, Insulin, Age |
| Cardiovascular Disease | Cholesterol, BP, ECG results, Chest pain type |
| Hypertension | Systolic/Diastolic BP, BMI, Sodium intake, Stress level |

---

## Proposed Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Environment | Jupyter Notebook (.ipynb) |
| ML Models | XGBoost, scikit-learn |
| Explainability | SHAP |
| Data Processing | pandas, NumPy |
| Visualisation | matplotlib, seaborn, Plotly |

---

## Notebook Structure

The project will be implemented as a series of Jupyter notebooks run sequentially. This structure is tentative and may evolve.

```
early-disease-risk-predictor/
|
+-- data/
|   +-- raw/                        # Raw datasets (to be added)
|   +-- processed/                  # Cleaned and merged outputs (to be added)
|
+-- notebooks/
|   +-- 01_data_loading.ipynb       # (planned)
|   +-- 02_eda.ipynb                # (planned)
|   +-- 03_preprocessing.ipynb      # (planned)
|   +-- 04_feature_engineering.ipynb# (planned)
|   +-- 05_model_training.ipynb     # (planned)
|   +-- 06_evaluation.ipynb         # (planned)
|   +-- 07_xai_dashboard.ipynb      # (planned)
|
+-- reports/
|   +-- phase1_theoretical_basis.pdf
|
+-- requirements.txt
+-- README.md
+-- .gitignore
```

---

## Getting Started

Setup instructions will be added once the initial notebooks are in place. For now, clone the repository and install the dependencies below.

```bash
git clone https://github.com/<your-username>/early-disease-risk-predictor.git
cd early-disease-risk-predictor

python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
```

---

## Team

| Name | Student ID | University |
|---|---|---|
| Hashir Rauf | 23L-2572 | FAST-NUCES, Lahore |
| Minahil Mir | 23L-2517 | FAST-NUCES, Lahore |
 
**Project track:** Type A -- Application Development
