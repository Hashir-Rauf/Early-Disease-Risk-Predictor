import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import xgboost as xgb
import gradio as gr

# Constants
FEATURE_NAMES = [
    "age", "bmi", "glucose", "blood_pressure",
    "insulin", "cholesterol", "source_pima", "source_uci_heart",
]
FEAT_DISPLAY = [
    "Age", "BMI (log)", "Glucose (log)", "Blood Pressure",
    "Insulin", "Cholesterol", "Src: PIMA", "Src: UCI Heart",
]
LOG_FEATS   = ["bmi", "glucose"]
SCALE_FEATS = ["age", "bmi", "glucose", "blood_pressure", "insulin", "cholesterol"]
DISEASES    = ["diabetes", "heart_disease", "hypertension"]
DIS_LABEL   = {
    "diabetes":     "Diabetes",
    "heart_disease": "Heart Disease",
    "hypertension": "Hypertension",
}
DIS_EMOJI   = {"diabetes": "Diabetes", "heart_disease": "Heart Disease", "hypertension": "Hypertension"}
DIS_ICON    = {"diabetes": "drop-of-blood", "heart_disease": "heart", "hypertension": "syringe"}

THRESHOLDS = {"diabetes": 0.50, "heart_disease": 0.50, "hypertension": 0.50}

# Load models (once at startup)
MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
std_scaler = joblib.load(os.path.join(MODEL_DIR, "standard_scaler.pkl"))
xgb_models = {d: joblib.load(os.path.join(MODEL_DIR, f"xgb_{d}.pkl")) for d in DISEASES}
print("Models loaded successfully.")

# Inference helpers
def preprocess(age, bmi, glucose, blood_pressure, insulin, cholesterol):
    df = pd.DataFrame([{
        "age": float(age), "bmi": float(bmi), "glucose": float(glucose),
        "blood_pressure": float(blood_pressure), "insulin": float(insulin),
        "cholesterol": float(cholesterol),
    }])
    for col in LOG_FEATS:
        df[col] = np.log1p(df[col])
    df[SCALE_FEATS] = std_scaler.transform(df[SCALE_FEATS])
    df["source_pima"]      = 0
    df["source_uci_heart"] = 0
    return df[FEATURE_NAMES]


def xgb_shap_exp(model, X_df):
    """XGBoost native pred_contribs -- works with XGBoost 3.x + SHAP 0.47."""
    booster  = model.get_booster()
    dmat     = xgb.DMatrix(X_df, feature_names=list(X_df.columns))
    contribs = booster.predict(dmat, pred_contribs=True)
    return shap.Explanation(
        values        = contribs[:, :-1],
        base_values   = contribs[:, -1],
        data          = X_df.values,
        feature_names = FEAT_DISPLAY,
    )


def waterfall_figure(sv_single, disease, prob):
    plt.close("all")
    shap.plots.waterfall(sv_single, show=False, max_display=8)
    fig = plt.gcf()

    if prob >= 0.50:
        risk, color = "HIGH RISK", "#c0392b"
    elif prob >= 0.30:
        risk, color = "MODERATE",  "#e67e22"
    else:
        risk, color = "LOW RISK",  "#27ae60"

    fig.suptitle(
        f"{DIS_LABEL[disease]}: {prob * 100:.1f}%  --  {risk}",
        fontsize=12, fontweight="bold", color=color, y=1.01,
    )
    fig.tight_layout()
    return fig


def risk_html(probs):
    cards = ""
    for disease, prob in probs.items():
        pct = prob * 100
        if pct >= 50:
            bg, border, badge_bg, label = "#fff0f0", "#e74c3c", "#c0392b", "HIGH RISK"
        elif pct >= 30:
            bg, border, badge_bg, label = "#fff8ed", "#e67e22", "#d35400", "MODERATE"
        else:
            bg, border, badge_bg, label = "#f0fff4", "#2ecc71", "#27ae60", "LOW RISK"

        bar_pct = min(100, pct)
        cards += f"""
        <div style="margin:10px 0;padding:14px 16px;background:{bg};
                    border-left:5px solid {border};border-radius:8px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-size:16px;font-weight:700;color:#2c3e50;">{DIS_LABEL[disease]}</span>
            <span style="background:{badge_bg};color:#fff;padding:3px 12px;
                         border-radius:20px;font-size:12px;font-weight:600;">{label}</span>
          </div>
          <div style="background:#ddd;border-radius:6px;height:12px;overflow:hidden;">
            <div style="background:{badge_bg};height:12px;width:{bar_pct:.0f}%;
                        border-radius:6px;transition:width 0.4s;"></div>
          </div>
          <div style="margin-top:5px;font-size:26px;font-weight:800;color:{badge_bg};">
            {pct:.1f}%
          </div>
        </div>"""
    return f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                max-width:480px;padding:4px;">
      {cards}
      <p style="font-size:11px;color:#888;margin-top:12px;line-height:1.5;">
        This tool is for educational purposes only and does not constitute
        medical advice. Please consult a qualified healthcare professional.
      </p>
    </div>"""


# Main prediction function
def predict(age, bmi, glucose, blood_pressure, insulin, cholesterol):
    X_pat  = preprocess(age, bmi, glucose, blood_pressure, insulin, cholesterol)
    probs  = {}
    figs   = {}

    for disease in DISEASES:
        clf  = xgb_models[disease]
        prob = float(clf.predict_proba(X_pat)[0, 1])
        probs[disease] = prob

        sv_exp    = xgb_shap_exp(clf, X_pat)
        sv_single = shap.Explanation(
            values        = sv_exp.values[0],
            base_values   = float(sv_exp.base_values[0]),
            data          = sv_exp.data[0],
            feature_names = FEAT_DISPLAY,
        )
        figs[disease] = waterfall_figure(sv_single, disease, prob)

    return (
        risk_html(probs),
        figs["diabetes"],
        figs["heart_disease"],
        figs["hypertension"],
    )


def reset_inputs():
    return 45, 27.0, 100, 80, 125, 200, "", None, None, None


# ---------------------------------------------------------------------------
# Clinical reference panel
# ---------------------------------------------------------------------------
REFERENCE_MD = """
**Normal Ranges (adults)**
| Measurement | Normal |
|---|---|
| BMI | 18.5 - 24.9 kg/m2 |
| Fasting Glucose | 70 - 99 mg/dL |
| Systolic BP | < 120 mmHg |
| Total Cholesterol | < 200 mg/dL |
| Insulin | 2 - 25 uU/mL |
"""

DISCLAIMER_MD = """
> **Disclaimer:** This predictor is built for educational purposes as part of
> an academic AI project. It does not replace clinical diagnosis.
> Results are based on population-level statistical patterns in PIMA, UCI Heart
> Disease, and Framingham Heart Study datasets.
"""

SHAP_EXPLAINER_MD = """
**How to read the SHAP chart**
- Each bar shows one feature's contribution to this prediction.
- **Red bars** push the risk score higher.
- **Blue bars** push the risk score lower.
- The final prediction equals the base value + sum of all SHAP values.
"""

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
.predict-btn { background: #2980b9 !important; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="Early Disease Risk Predictor",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=CUSTOM_CSS,
) as demo:

    # Header
    gr.Markdown("""
    # Early Disease Risk Predictor
    ### Diabetes - Heart Disease - Hypertension
    *Powered by XGBoost + SHAP Explainable AI*

    Enter a patient's clinical measurements on the left, then click **Predict Risk** to
    see real-time risk scores and SHAP explanations for all three conditions.
    """)

    with gr.Row():
        # ----------------------------------------------------------------
        # Left column: inputs
        # ----------------------------------------------------------------
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("## Patient Vitals")

            age            = gr.Slider(18,  90,  value=45,  step=1,   label="Age (years)")
            bmi            = gr.Slider(15.0, 60.0, value=27.0, step=0.1, label="BMI (kg/m2)")
            glucose        = gr.Slider(50,  400, value=100, step=1,   label="Fasting Glucose (mg/dL)")
            blood_pressure = gr.Slider(40,  220, value=80,  step=1,   label="Systolic Blood Pressure (mmHg)")
            insulin        = gr.Slider(0,   500, value=125, step=1,   label="Insulin (uU/mL)")
            cholesterol    = gr.Slider(100, 450, value=200, step=1,   label="Total Cholesterol (mg/dL)")

            with gr.Row():
                btn_reset   = gr.Button("Reset",       variant="secondary")
                btn_predict = gr.Button("Predict Risk", variant="primary",
                                        elem_classes=["predict-btn"])

            gr.Markdown(REFERENCE_MD)
            gr.Markdown(DISCLAIMER_MD)

        # ----------------------------------------------------------------
        # Right column: outputs
        # ----------------------------------------------------------------
        with gr.Column(scale=2, min_width=420):
            gr.Markdown("## Risk Assessment")
            risk_display = gr.HTML(label="Risk Summary", value="")

            gr.Markdown("## SHAP Explanations")
            gr.Markdown(SHAP_EXPLAINER_MD)

            with gr.Tabs():
                with gr.Tab("Diabetes"):
                    plot_diabetes = gr.Plot(label="Diabetes - SHAP Waterfall")
                with gr.Tab("Heart Disease"):
                    plot_heart    = gr.Plot(label="Heart Disease - SHAP Waterfall")
                with gr.Tab("Hypertension"):
                    plot_hyper    = gr.Plot(label="Hypertension - SHAP Waterfall")

    # Example patients
    gr.Examples(
        examples=[
            [58, 34.5, 165, 95,  180, 265],
            [28, 22.0,  82, 70,   90, 180],
            [45, 27.5, 108, 82,  125, 220],
            [62, 31.0, 140, 100, 150, 240],
        ],
        inputs=[age, bmi, glucose, blood_pressure, insulin, cholesterol],
        label="Example Patients  |  High Risk  |  Low Risk  |  Borderline  |  Older High-Risk",
        examples_per_page=4,
    )

    # Wire events
    btn_predict.click(
        fn=predict,
        inputs=[age, bmi, glucose, blood_pressure, insulin, cholesterol],
        outputs=[risk_display, plot_diabetes, plot_heart, plot_hyper],
    )

    btn_reset.click(
        fn=reset_inputs,
        inputs=[],
        outputs=[age, bmi, glucose, blood_pressure, insulin, cholesterol,
                 risk_display, plot_diabetes, plot_heart, plot_hyper],
    )


if __name__ == "__main__":
    demo.launch()
