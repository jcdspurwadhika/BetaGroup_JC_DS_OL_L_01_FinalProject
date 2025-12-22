import os
import pickle
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")

# Load pickle (format: model_name, pipeline, threshold)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "churn_model.pkl")

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

pipe = bundle["pipeline"]
default_thr = float(bundle["threshold"])
model_name = bundle.get("model_name", "Unknown")

st.title("Telco Churn Predictor (Single Customer)")
st.caption(f"Model: **{model_name}** | Default threshold: **{default_thr:.2f}**")

# Threshold Slider
thr = st.slider("Threshold", 0.05, 0.95, default_thr, 0.01)


# Form input 1 customer

with st.form("customer_form"):
    gender = st.selectbox("gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("SeniorCitizen", ["No", "Yes"])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("tenure", min_value=0, max_value=100, value=12, step=1)

    PhoneService = st.selectbox("PhoneService", ["Yes", "No"])
    MultipleLines = st.selectbox("MultipleLines", ["No phone service", "No", "Yes"])

    InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("OnlineSecurity", ["No internet service", "No", "Yes"])
    OnlineBackup = st.selectbox("OnlineBackup", ["No internet service", "No", "Yes"])
    DeviceProtection = st.selectbox("DeviceProtection", ["No internet service", "No", "Yes"])
    TechSupport = st.selectbox("TechSupport", ["No internet service", "No", "Yes"])
    StreamingTV = st.selectbox("StreamingTV", ["No internet service", "No", "Yes"])
    StreamingMovies = st.selectbox("StreamingMovies", ["No internet service", "No", "Yes"])

    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "PaymentMethod",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

    MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0, max_value=500.0, value=70.0, step=1.0)
    TotalCharges = st.number_input("TotalCharges", min_value=0.0, max_value=20000.0, value=1000.0, step=10.0)

    submit = st.form_submit_button("Predict")

if submit:
    # Buat 1 baris data
    X_one = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
    }])

    # Prediksi
    proba = pipe.predict_proba(X_one)[:, 1][0]
    pred = int(proba >= thr)

    st.subheader("Result")
    st.metric("Churn Probability", f"{proba:.4f}")
    st.metric("Prediction", "CHURN (1)" if pred == 1 else "NO CHURN (0)")
    st.write(f"Rule: proba ≥ **{thr:.2f}** → churn")

    with st.expander("Show input dataframe"):
        st.dataframe(X_one)
