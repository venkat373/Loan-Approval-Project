import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")

ARTIFACTS_DIR = "."


@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{ARTIFACTS_DIR}/model.pkl")
    scaler = joblib.load(f"{ARTIFACTS_DIR}/scaler.pkl")
    encoders = joblib.load(f"{ARTIFACTS_DIR}/encoders.pkl")
    feature_columns = joblib.load(f"{ARTIFACTS_DIR}/feature_columns.pkl")
    scale_columns = joblib.load(f"{ARTIFACTS_DIR}/scale_columns.pkl")
    return model, scaler, encoders, feature_columns, scale_columns


try:
    model, scaler, encoders, feature_columns, scale_columns = load_artifacts()
    artifacts_ready = True
except FileNotFoundError:
    artifacts_ready = False

st.title("🏦 Loan Approval Predictor")
st.write(
    "Fill in the applicant's details below to get an instant loan approval "
    "prediction from the trained Random Forest model."
)

if not artifacts_ready:
    st.error(
        "Model artifacts not found in `artifacts/`. Run `python train_model.py` "
        "first (with `loan_approval_dataset.csv` present) to generate "
        "`model.pkl`, `scaler.pkl`, `encoders.pkl`, `feature_columns.pkl`, "
        "and `scale_columns.pkl`."
    )
    st.stop()

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        no_of_dependents = st.number_input("Number of dependents", min_value=0, max_value=10, value=0, step=1)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self employed", ["No", "Yes"])
        income_annum = st.number_input("Annual income (₹)", min_value=0, value=5000000, step=100000)
        loan_amount = st.number_input("Loan amount requested (₹)", min_value=0, value=10000000, step=100000)
        loan_term_years = st.number_input("Loan term (years)", min_value=1, max_value=30, value=10, step=1)

    with col2:
        cibil_score = st.slider("CIBIL score", min_value=300, max_value=900, value=700)
        residential_assets_value = st.number_input("Residential assets value (₹)", min_value=0, value=3000000, step=100000)
        commercial_assets_value = st.number_input("Commercial assets value (₹)", min_value=0, value=2000000, step=100000)
        luxury_assets_value = st.number_input("Luxury assets value (₹)", min_value=0, value=8000000, step=100000)
        bank_asset_value = st.number_input("Bank asset value (₹)", min_value=0, value=4000000, step=100000)

    submitted = st.form_submit_button("Predict loan status")

if submitted:
    raw_input = {
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term_years": loan_term_years,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
    }
    input_df = pd.DataFrame([raw_input])

    # Apply the same label encoding used at train time
    for col in ["education", "self_employed"]:
        le = encoders[col]
        input_df[col] = le.transform(input_df[col])

    # Apply the same scaling used at train time
    input_df[scale_columns] = scaler.transform(input_df[scale_columns])

    # Ensure column order matches training
    input_df = input_df[feature_columns]

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    approved_prob = proba[1]

    if pred == 1:
        st.success(f"✅ Loan likely **Approved** (confidence: {approved_prob:.1%})")
    else:
        st.error(f"❌ Loan likely **Rejected** (confidence: {1 - approved_prob:.1%})")

    with st.expander("See prediction probabilities"):
        st.write(f"Rejected: {proba[0]:.1%}")
        st.write(f"Approved: {proba[1]:.1%}")

st.caption(
    "This tool provides an automated, data-driven estimate and should be "
    "used to support — not replace — human loan review."
)
