import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Visa Processing Intelligence",
    page_icon="🌍",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.metric-card {
    background-color:#1E1E1E;
    padding:20px;
    border-radius:12px;
    text-align:center;
}
.metric-title {
    font-size:18px;
    color:#9aa0a6;
}
.metric-value {
    font-size:32px;
    color:#00C2A8;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

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
    return pd.read_csv("cleaned_visa_dataset.csv")

df = load_data()

# -----------------------------
# HEADER
# -----------------------------
st.title("🌍 AI Visa Processing Time Intelligence Platform")

st.write(
"""
Predict visa processing timelines using **machine learning analytics**.

This AI system analyzes historical visa processing patterns to estimate
future delays and processing timelines.
"""
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("📋 Application Details")

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

st.sidebar.info(
"""
Application workload represents how many visa cases immigration
offices are processing during that time period.
"""
)

# -----------------------------
# FEATURE PREPARATION
# -----------------------------
filing_quarter = (application_month - 1)//3 + 1

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
if st.sidebar.button("🔍 Run Prediction"):

    prediction = model.predict(input_data)[0]

    tree_preds = np.array([tree.predict(input_data)[0] for tree in model.estimators_])
    std = np.std(tree_preds)

    lower = prediction - 1.96*std
    upper = prediction + 1.96*std

    risk_score = min(prediction/2500,1)

    # -----------------------------
    # METRICS DASHBOARD
    # -----------------------------
    st.header("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">Estimated Processing Time</div>
        <div class="metric-value">{int(prediction)} Days</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">Delay Risk Score</div>
        <div class="metric-value">{risk_score*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">Confidence Interval</div>
        <div class="metric-value">{int(lower)} - {int(upper)} days</div>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------
    # APPROVAL RISK
    # -----------------------------
    st.header("⚠️ Approval Risk Assessment")

    if risk_score > 0.7:
        st.error("High delay risk detected.")
    elif risk_score > 0.4:
        st.warning("Moderate delay risk.")
    else:
        st.success("Low delay risk.")

    st.progress(risk_score)

    # -----------------------------
    # PREDICTION CHART
    # -----------------------------
    colA, colB = st.columns(2)

    with colA:
        st.subheader("📈 Processing Time Forecast")

      avg_days = df["processing_days"].mean()

fig, ax = plt.subplots()

labels = ["Predicted", "Dataset Average"]
values = [prediction, avg_days]

ax.bar(labels, values, color=["#00C2A8","#FFA500"])

ax.set_ylabel("Days")
ax.set_title("Processing Time Comparison")

st.pyplot(fig)
    # -----------------------------
    # SHAP EXPLANATION
    # -----------------------------
    with colB:
      st.subheader("🧠 AI Explanation")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(input_data)

# Handle expected value safely
expected_value = explainer.expected_value
if isinstance(expected_value, list) or isinstance(expected_value, np.ndarray):
    expected_value = expected_value[0]

# Create explanation object
explanation = shap.Explanation(
    values=shap_values[0],
    base_values=expected_value,
    data=input_data.iloc[0],
    feature_names=input_data.columns
)

fig, ax = plt.subplots()

shap.waterfall_plot(explanation, show=False)

st.pyplot(fig)
# -----------------------------
# HISTORICAL TREND
# -----------------------------
st.header("📅 Historical Visa Processing Trends")

df["case_received_date"] = pd.to_datetime(df["case_received_date"], errors="coerce")
df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce")

df["processing_days"] = (df["decision_date"] - df["case_received_date"]).dt.days
df["year"] = df["case_received_date"].dt.year

trend = df.groupby("year")["processing_days"].mean()

fig2, ax2 = plt.subplots()

ax2.plot(
    trend.index,
    trend.values,
    marker="o",
    color="#4CAF50",
    linewidth=3
)

ax2.set_xlabel("Year")
ax2.set_ylabel("Average Processing Days")

st.pyplot(fig2)

# -----------------------------
# BUSINESS INSIGHTS
# -----------------------------
st.header("💡 Business Insights")

st.markdown("""
• Visa processing delays tend to increase when **application workload rises**.

• Certain visa types historically experience **longer processing times**.

• Seasonal filing patterns may impact decision timelines.

• Predictive analytics can help applicants estimate **expected processing delays**.
""")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("AI-Powered Visa Delay Forecasting Platform | Machine Learning Dashboard")
