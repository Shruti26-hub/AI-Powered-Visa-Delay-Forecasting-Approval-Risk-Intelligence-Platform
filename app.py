import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="AI Visa Processing Intelligence", layout="wide")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("rf_model.pkl")

model = load_model()

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_visa_dataset.csv")
    return df

df = load_data()

# -----------------------------
# CREATE PROCESSING DAYS
# -----------------------------
if "processing_days" not in df.columns:
    df["case_received_date"] = pd.to_datetime(df["case_received_date"], errors="coerce")
    df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce")
    df["processing_days"] = (df["decision_date"] - df["case_received_date"]).dt.days

# -----------------------------
# TITLE
# -----------------------------
st.title("🌍 AI Visa Processing Time Intelligence Platform")

st.write(
"""
Predict visa processing timelines using machine learning insights.
This system analyzes historical visa application patterns.
"""
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Application Details")

visa_type = st.sidebar.selectbox(
    "Visa Type",
    ["H-1B","F-1","B-2","E-2","L-1"]
)

application_month = st.sidebar.slider(
    "Application Month",
    1,12,6
)

filing_year = st.sidebar.slider(
    "Application Year",
    2007,2016,2012
)

workload_level = st.sidebar.slider(
    "Application Workload Level",
    1,100,30
)

# -----------------------------
# FEATURE CREATION
# -----------------------------
filing_quarter = (application_month-1)//3+1

input_data = pd.DataFrame({
    "filing_year":[filing_year],
    "filing_month":[application_month],
    "filing_quarter":[filing_quarter],
    "monthly_volume":[workload_level]
})

if hasattr(model,"feature_names_in_"):
    expected_features=list(model.feature_names_in_)
    for col in expected_features:
        if col not in input_data.columns:
            input_data[col]=0
    input_data=input_data[expected_features]

# -----------------------------
# PREDICTION
# -----------------------------
if st.sidebar.button("Run Prediction"):

    prediction=model.predict(input_data)[0]

    tree_preds=np.array([tree.predict(input_data)[0] for tree in model.estimators_])
    std=np.std(tree_preds)

    lower=prediction-1.96*std
    upper=prediction+1.96*std

    risk_score=min(prediction/2500,1)

    st.header("Prediction Results")

    col1,col2,col3=st.columns(3)

    col1.metric("Predicted Processing Time",f"{int(prediction)} days")
    col2.metric("Delay Risk Score",f"{risk_score*100:.1f}%")
    col3.metric("Confidence Range",f"{int(lower)} - {int(upper)} days")

    # -----------------------------
    # PROCESSING FORECAST CHART
    # -----------------------------
    avg_days=df["processing_days"].mean()

    fig,ax=plt.subplots()

    labels=["Predicted","Dataset Average"]
    values=[prediction,avg_days]

    ax.bar(labels,values,color=["#00C2A8","#FFA500"])

    ax.set_ylabel("Days")
    ax.set_title("Processing Time Comparison")

    st.pyplot(fig)

    # -----------------------------
    # RISK GAUGE
    # -----------------------------
    st.subheader("Visa Delay Risk Meter")

    fig_gauge=go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score*100,
        title={"text":"Delay Risk (%)"},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":"red"},
            "steps":[
                {"range":[0,30],"color":"green"},
                {"range":[30,70],"color":"yellow"},
                {"range":[70,100],"color":"red"}
            ],
        }
    ))

    st.plotly_chart(fig_gauge)

    # -----------------------------
    # SHAP EXPLANATION
    # -----------------------------
    st.subheader("AI Explanation")

    explainer=shap.TreeExplainer(model)
    shap_values=explainer.shap_values(input_data)

    expected_value=explainer.expected_value
    if isinstance(expected_value,(list,np.ndarray)):
        expected_value=expected_value[0]

    explanation=shap.Explanation(
        values=shap_values[0],
        base_values=expected_value,
        data=input_data.iloc[0],
        feature_names=input_data.columns
    )

    fig2,ax2=plt.subplots()

    shap.waterfall_plot(explanation,show=False)

    st.pyplot(fig2)

# -----------------------------
# HISTORICAL TREND
# -----------------------------
st.header("Historical Visa Processing Trends")

df["year"]=pd.to_datetime(df["case_received_date"],errors="coerce").dt.year

trend=df.groupby("year")["processing_days"].mean()

fig3,ax3=plt.subplots()

ax3.plot(trend.index,trend.values,marker="o",color="green",linewidth=3)

ax3.set_xlabel("Year")
ax3.set_ylabel("Average Processing Days")

st.pyplot(fig3)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.header("Feature Importance")

importances=model.feature_importances_
features=model.feature_names_in_

importance_df=pd.DataFrame({
    "feature":features,
    "importance":importances
}).sort_values("importance",ascending=False)

fig4,ax4=plt.subplots()

ax4.barh(importance_df["feature"],importance_df["importance"],color="purple")

ax4.invert_yaxis()

st.pyplot(fig4)

# -----------------------------
# SCENARIO SIMULATOR
# -----------------------------
st.header("Interactive Scenario Simulator")

sim_month=st.slider("Simulate Filing Month",1,12,6)

sim_volume=st.slider("Simulate Workload",1,100,40)

sim_quarter=(sim_month-1)//3+1

sim_data=pd.DataFrame({
    "filing_year":[filing_year],
    "filing_month":[sim_month],
    "filing_quarter":[sim_quarter],
    "monthly_volume":[sim_volume]
})

for col in model.feature_names_in_:
    if col not in sim_data.columns:
        sim_data[col]=0

sim_data=sim_data[model.feature_names_in_]

sim_prediction=model.predict(sim_data)[0]

st.metric("Simulated Processing Time",f"{int(sim_prediction)} days")

# -----------------------------
# BUSINESS INSIGHTS
# -----------------------------
st.header("Business Insights")

st.markdown("""
• Visa processing delays decrease in recent years.

• Application workload significantly impacts processing time.

• Seasonal filing patterns influence immigration backlog.

• AI forecasting helps applicants estimate expected delays.
""")
