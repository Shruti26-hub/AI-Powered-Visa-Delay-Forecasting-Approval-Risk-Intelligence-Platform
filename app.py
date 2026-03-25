import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Visa Processing Intelligence Platform", layout="wide")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("rf_model.pkl")

model = load_model()

# -----------------------------
# LOAD DATASET
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_visa_dataset.csv")
    return df

df = load_data()

# -----------------------------
# HEADER
# -----------------------------
st.title("AI Visa Processing Time Intelligence Platform")

st.write(
"""
Predict visa processing timelines using **machine learning analytics**.
This dashboard estimates visa decision delays based on historical case patterns.
"""
)

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("Application Details")

visa_type = st.sidebar.selectbox(
    "Visa Type",
    ["H-1B", "F-1", "B-2", "E-2", "L-1"]
)

application_month = st.sidebar.slider(
    "Application Month",
    1, 12, 6
)

filing_year = st.sidebar.slider(
    "Application Year",
    2007, 2016, 2012
)

workload_level = st.sidebar.slider(
    "Application Workload Level",
    1, 100, 30
)

# -----------------------------
# FEATURE PREPARATION
# -----------------------------
filing_quarter = (application_month - 1) // 3 + 1

input_data = pd.DataFrame({
    "filing_year":[filing_year],
    "filing_month":[application_month],
    "filing_quarter":[filing_quarter],
    "monthly_volume":[workload_level]
})

# Ensure features match model
if hasattr(model, "feature_names_in_"):
    expected_features = list(model.feature_names_in_)
    for col in expected_features:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[expected_features]

# -----------------------------
# RUN PREDICTION
# -----------------------------
if st.sidebar.button("Run Prediction"):

    prediction = model.predict(input_data)[0]

    # Confidence interval
    tree_preds = np.array([tree.predict(input_data)[0] for tree in model.estimators_])
    std = np.std(tree_preds)

    lower = prediction - 1.96 * std
    upper = prediction + 1.96 * std

    risk_score = min(prediction / 2500, 1)

    # -----------------------------
    # METRICS
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Estimated Processing Time",
        f"{int(prediction)} Days"
    )

    col2.metric(
        "Delay Risk Score",
        f"{risk_score*100:.1f}%"
    )

    col3.metric(
        "Confidence Interval",
        f"{int(lower)} - {int(upper)} days"
    )

    # -----------------------------
    # RISK INDICATOR
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
    # PREDICTION CHART
    # -----------------------------
    st.subheader("Processing Time Forecast")

    fig, ax = plt.subplots()

    ax.bar(
        ["Predicted Processing Days"],
        [prediction],
        color="#4CAF50"
    )

    ax.set_ylabel("Days")
    ax.set_title("AI Forecast")

    st.pyplot(fig)

    # -----------------------------
    # SHAP EXPLANATION
    # -----------------------------
    st.subheader("Model Explanation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_values[0],
        feature_names=input_data.columns
    )

    st.pyplot(bbox_inches="tight")

# -----------------------------
# HISTORICAL TREND CHART
# -----------------------------
st.subheader("Historical Visa Processing Trends")

# Convert date columns
df["case_received_date"] = pd.to_datetime(df["case_received_date"], errors="coerce")
df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce")

# Create processing_days
df["processing_days"] = (df["decision_date"] - df["case_received_date"]).dt.days

# Extract year
df["year"] = df["case_received_date"].dt.year

trend = df.groupby("year")["processing_days"].mean()

fig2, ax2 = plt.subplots()

ax2.plot(
    trend.index,
    trend.values,
    marker="o",
    color="#00C2A8",
    linewidth=3
)

ax2.set_xlabel("Year")
ax2.set_ylabel("Average Processing Days")
ax2.set_title("Visa Processing Trends Over Time")

st.pyplot(fig2)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("AI-Powered Visa Delay Forecasting Platform")
