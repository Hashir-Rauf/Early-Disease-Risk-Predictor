---
title: Early Disease Risk Predictor
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.23.3
app_file: app.py
pinned: false
license: mit
short_description: Disease risk predictor with SHAP explanations
---

# Early Disease Risk Predictor

Predicts risk for **Diabetes**, **Heart Disease**, and **Hypertension** from six
clinical measurements using XGBoost models trained on 5,311 patient records
(PIMA, UCI Heart Disease, Framingham Heart Study datasets).

## Features

- Real-time risk prediction for three chronic diseases
- SHAP waterfall explanations for every prediction
- Interactive sliders for clinical input
- Example patients (high risk, low risk, borderline)

## How It Works

1. Enter patient vitals using the sliders
2. Click **Predict Risk**
3. View risk percentages with color-coded indicators
4. Explore SHAP explanations to understand which features drove each prediction

## Models

| Model | AUC-ROC | Recall |
|---|---|---|
| XGBoost - Diabetes | 0.981 | 0.875 |
| XGBoost - Heart Disease | 0.758 | 0.658 |
| XGBoost - Hypertension | 1.000 | 1.000 |

All models use RandomizedSearchCV-tuned hyperparameters and class-imbalance
correction via `scale_pos_weight`.

## Disclaimer

This tool is for **educational purposes only** and does not constitute medical
advice. Consult a qualified healthcare professional for diagnosis.
