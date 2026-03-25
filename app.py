import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="AI Visa Processing Intelligence",
    layout="wide"
)

# -----------------------
# CUSTOM STYLE
# -----------------------
st.markdown("""
<style>
.main {
    background-color:#0E1117;
}
.stButton>button {
    background-color:#4CAF50;
    color:white;
    border-radius:8px;
}
.metric-box {
    background-color:#1E1E1E;
    padding:20px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_model():
    return joblib.load("rf_model.pkl")

model = load_model()

# -----------------------
# TITLE
# -----------------------
st.title("AI Visa Processing Time Intelligence Platform")
st.write(
"""
Predict visa processing timelines using **machine learning analytics**.
This dashboard estimates visa decision delays based on historical case patterns.
"""
)

# -----------------------
# SIDEBAR INPUTS
# -----------------------
st.sidebar.header("Application Details")

visa_type = st.sidebar.selectbox(
    "Visa Category",
    ["F-1", "H-1B", "B-2", "E-2", "L-1"]
)

application_month = st.sidebar.slider(
    "Application Month",
    1, 12, 6
)

filing_year = st.sidebar.slider(
    "Application Year",
    2007, 2015, 2012
)

workload_level = st.sidebar.slider(
    "Immigration Workload Level",
    1, 100, 30
)

st.sidebar.write(
"Workload level represents the number of applications processed during that period."
)

# -----------------------
# FEATURE PREPARATION
# -----------------------

filing_quarter = (application_month - 1) // 3 + 1

input_data = pd.DataFrame({
    "filing_year":[filing_year],
    "filing_month":[application_month],
    "filing_quarter":[filing_quarter],
    "monthly_volume":[workload_level]
})

# -----------------------
# PREDICTION
# -----------------------
if st.sidebar.button("Run Prediction"):

    prediction = model.predict(input_data)[0]

    risk_score = min(prediction / 2500, 1)

    # -----------------------
    # METRICS
    # -----------------------
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Estimated Processing Time",
            f"{int(prediction)} Days"
        )

    with col2:
        st.metric(
            "Delay Risk Score",
            f"{risk_score*100:.1f}%"
        )

    # -----------------------
    # RISK INDICATOR
    # -----------------------
    st.subheader("Delay Risk Assessment")

    if risk_score > 0.7:
        st.error("High Delay Risk")
    elif risk_score > 0.4:
        st.warning("Moderate Delay Risk")
    else:
        st.success("Low Delay Risk")

    st.progress(risk_score)

    # -----------------------
    # VISUALIZATION
    # -----------------------
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

    # -----------------------
    # SHAP EXPLAINABILITY
    # -----------------------
    st.subheader("Model Explainability")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    shap.summary_plot(
        shap_values,
        input_data,
        show=False
    )

    st.pyplot(bbox_inches="tight")

# -----------------------
# FOOTER
# -----------------------
st.markdown("---")
st.caption("AI-Powered Visa Delay Forecasting Platform")
