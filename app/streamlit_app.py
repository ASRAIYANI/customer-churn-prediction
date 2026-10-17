
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.joblib"

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer information to estimate whether the customer "
    "is likely to churn, based on the trained machine learning model."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Please run src/train.py first.")
    st.stop()

model = load_model()

with st.form("customer_form"):
    st.subheader("Customer information")

    age = st.slider("Age", min_value=18, max_value=100, value=35)

    tenure_months = st.slider(
        "Tenure (months)", min_value=1, max_value=72, value=24
    )

    monthly_charges = st.number_input(
        "Monthly charges",
        min_value=0.0,
        max_value=1000.0,
        value=70.0,
        step=5.0,
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"],
    )

    internet_service = st.selectbox(
        "Internet service",
        ["DSL", "Fiber optic", "Unknown"],
    )

    tech_support = st.selectbox(
        "Tech support",
        ["Yes", "No", "Unknown"],
    )

    online_security = st.selectbox(
        "Online security",
        ["Yes", "No"],
    )

    payment_method = st.selectbox(
        "Payment method",
        ["Card", "Electronic check", "Bank transfer", "Mailed check"],
    )

    senior_citizen = st.selectbox(
        "Senior citizen",
        options=[0, 1],
        format_func=lambda value: "Yes" if value == 1 else "No",
    )

    support_calls = st.slider(
        "Support calls",
        min_value=0,
        max_value=20,
        value=2,
    )

    submitted = st.form_submit_button(
        "Predict customer churn",
        type="primary",
        use_container_width=True,
    )

if submitted:
    customer = pd.DataFrame([{
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "contract": contract,
        "internet_service": (
            None if internet_service == "Unknown" else internet_service
        ),
        "tech_support": (
            None if tech_support == "Unknown" else tech_support
        ),
        "online_security": online_security,
        "payment_method": payment_method,
        "senior_citizen": senior_citizen,
        "support_calls": support_calls,
    }])

    prediction = int(model.predict(customer)[0])
    probabilities = model.predict_proba(customer)[0]
    churn_probability = float(probabilities[list(model.classes_).index(1)])

    st.subheader("Prediction result")
    st.metric("Estimated churn probability", f"{churn_probability:.1%}")

    if prediction == 1:
        st.warning(
            "The model predicts that this customer belongs to the churn class."
        )
    else:
        st.success(
            "The model predicts that this customer belongs to the "
            "non-churn class."
        )

    st.caption(
        "This is a model estimate, not a guarantee. Results depend on "
        "the training dataset and should not be used as the sole basis "
        "for real customer decisions."
    )

st.divider()
st.caption(
    "Portfolio project | Random Forest | Educational use"
)
