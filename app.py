import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

model = joblib.load("rf_model.pkl")

st.title("AI Visa Processing Time Predictor")

country = st.selectbox("Country", ["India","USA","UK"])
visa_type = st.selectbox("Visa Type", ["Student","Work","Tourist"])
month = st.slider("Application Month",1,12)

if st.button("Predict"):

    data = pd.DataFrame({
        "country":[country],
        "visa_type":[visa_type],
        "month":[month]
    })

    prediction = model.predict(data)[0]

    st.success(f"Estimated Processing Time: {prediction:.0f} days")

    # Risk score
    risk = prediction/2500
    st.subheader("Delay Risk Score")
    st.progress(risk)

    # Prediction chart
    fig, ax = plt.subplots()
    ax.bar(["Prediction"], [prediction])
    ax.set_ylabel("Processing Days")
    st.pyplot(fig)
