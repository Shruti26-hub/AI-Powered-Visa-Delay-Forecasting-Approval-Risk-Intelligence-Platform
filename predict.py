import joblib
import pandas as pd

model = joblib.load("rf_model.pkl")

def predict_processing_time(data):

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    return round(prediction,2)
