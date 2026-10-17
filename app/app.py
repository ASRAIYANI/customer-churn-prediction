
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "churn_pipeline.joblib"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")
st.title("📊 Customer Churn Prediction")
st.write("Enter customer information to estimate churn probability.")

if not MODEL_PATH.exists():
    st.error("Model not found. Run: python src/train.py")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("prediction_form"):
    age = st.number_input("Age", 18, 100, 35)
    tenure = st.number_input("Tenure (months)", 1, 120, 12)
    monthly = st.number_input("Monthly charges (£)", 0.0, 500.0, 70.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "None"])
    tech = st.selectbox("Tech support", ["Yes", "No"])
    security = st.selectbox("Online security", ["Yes", "No"])
    payment = st.selectbox("Payment method", ["Electronic check", "Card", "Bank transfer", "Mailed check"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    calls = st.number_input("Support calls", 0, 30, 2)
    submitted = st.form_submit_button("Predict churn")

if submitted:
    row = pd.DataFrame([{
        "age": age,
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "contract": contract,
        "internet_service": internet,
        "tech_support": tech,
        "online_security": security,
        "payment_method": payment,
        "senior_citizen": 1 if senior == "Yes" else 0,
        "support_calls": calls
    }])
    p = float(model.predict_proba(row)[0,1])
    st.metric("Churn probability", f"{p:.1%}")
    if p >= 0.65:
        st.error("HIGH RISK — consider a retention action.")
    elif p >= 0.40:
        st.warning("MEDIUM RISK — monitor and consider an offer.")
    else:
        st.success("LOW RISK — customer is less likely to churn.")
