import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Visa Processing Predictor",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("rf_model.pkl")
    return model

model = load_model()

# -----------------------------
# Dashboard Header
# -----------------------------
st.title("AI Visa Processing Time Intelligence Platform")
st.markdown(
"""
Predict visa processing timelines using machine learning insights.

This AI tool analyzes application patterns and estimates processing delays.
"""
)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Application Details")

visa_type = st.sidebar.selectbox(
    "Visa Type",
    ["F-1", "H-1B", "B-2", "E-2", "L-1"]
)

application_month = st.sidebar.slider(
    "Application Month",
    min_value=1,
    max_value=12,
    value=6
)

filing_year = st.sidebar.slider(
    "Application Year",
    min_value=2008,
    max_value=2016,
    value=2013
)

monthly_volume = st.sidebar.slider(
    "Application Volume",
    min_value=1,
    max_value=100,
    value=30
)

# -----------------------------
# Prepare Input Data
# -----------------------------
input_data = pd.DataFrame({
    "visa_type": [visa_type],
    "month": [application_month],
    "filing_year": [filing_year],
    "monthly_volume": [monthly_volume]
})

# -----------------------------
# Prediction Button
# -----------------------------
if st.sidebar.button("Run Prediction"):

    prediction = model.predict(input_data)[0]

    # -----------------------------
    # Risk Score
    # -----------------------------
    risk_score = min(prediction / 2500, 1)

    # -----------------------------
    # Dashboard Results
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Estimated Processing Time",
            f"{prediction:.0f} days"
        )

    with col2:
        st.metric(
            "Delay Risk Score",
            f"{risk_score*100:.1f}%"
        )

    # -----------------------------
    # Risk Indicator
    # -----------------------------
    st.subheader("Delay Risk Assessment")

    if risk_score > 0.7:
        st.error("High Delay Risk")
    elif risk_score > 0.4:
        st.warning("Moderate Delay Risk")
    else:
        st.success("Low Delay Risk")

    st.progress(risk_score)

    # -----------------------------
    # Visualization Chart
    # -----------------------------
    st.subheader("Predicted Processing Time")

    fig, ax = plt.subplots()

    ax.bar(
        ["Estimated Processing Days"],
        [prediction],
        color="steelblue"
    )

    ax.set_ylabel("Days")
    ax.set_title("Processing Time Forecast")

    st.pyplot(fig)

    # -----------------------------
    # SHAP Explainability
    # -----------------------------
    st.subheader("AI Model Explainability")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(input_data)

    shap.summary_plot(
        shap_values,
        input_data,
        show=False
    )

    st.pyplot(bbox_inches="tight")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
"""
AI-Powered Visa Delay Forecasting Platform  
Machine Learning + Predictive Analytics
"""
)
