import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Visa Processing Intelligence", layout="wide")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("rf_model.pkl")
    return model

model = load_model()

# Load dataset for trend charts
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_visa_dataset.csv")

df = load_data()

# -----------------------------
# PAGE HEADER
# -----------------------------
st.title("AI Visa Processing Time Intelligence Platform")
st.markdown(
"""
Predict visa processing timelines using **machine learning analytics**.
This dashboard estimates visa decision delays based on historical case patterns.
"""
)

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("Application Details")

visa_type = st.sidebar.selectbox(
    "Visa Type",
    df["class_of_admission"].dropna().unique()
)

application_month = st.sidebar.slider(
    "Application Month",
    1, 12, 6
)

filing_year = st.sidebar.slider(
    "Application Year",
    int(df["filing_year"].min()),
    int(df["filing_year"].max()),
    int(df["filing_year"].median())
)

workload = st.sidebar.slider(
    "Application Workload Level",
    1, 100, 30
)

# -----------------------------
# PREPARE INPUT
# -----------------------------
filing_quarter = (application_month - 1)//3 + 1

input_data = pd.DataFrame({
    "filing_year":[filing_year],
    "filing_month":[application_month],
    "filing_quarter":[filing_quarter],
    "monthly_volume":[workload]
})

# Align features with model expectation
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

    # Confidence interval using tree variance
    tree_preds = np.array([t.predict(input_data)[0] for t in model.estimators_])
    std = np.std(tree_preds)

    lower = prediction - 1.96*std
    upper = prediction + 1.96*std

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
        "Confidence Range",
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
        ["Predicted Days"],
        [prediction],
        color="#00C2A8"
    )

    ax.set_ylabel("Days")
    ax.set_title("AI Forecast")

    st.pyplot(fig)

    # -----------------------------
    # SHAP WATERFALL EXPLANATION
    # -----------------------------
    st.subheader("AI Decision Explanation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    fig2 = plt.figure()

    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_values[0],
        feature_names=input_data.columns
    )

    st.pyplot(fig2)

# -----------------------------
# HISTORICAL TREND CHART
# -----------------------------
st.subheader("Historical Visa Processing Trends")

trend = df.groupby("filing_year")["processing_days"].mean()

fig3, ax = plt.subplots()

ax.plot(
    trend.index,
    trend.values,
    marker="o",
    color="#4CAF50"
)

ax.set_xlabel("Year")
ax.set_ylabel("Average Processing Days")
ax.set_title("Visa Processing Trends Over Time")

st.pyplot(fig3)

st.caption("AI-Powered Visa Delay Forecasting Platform")
