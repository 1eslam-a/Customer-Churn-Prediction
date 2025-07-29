import streamlit as st
import pandas as pd
import joblib
import time

# Load model
pipeline = joblib.load("../Model/fraud_model.pkl")
model_columns = joblib.load("../Model/model_columns.pkl")
target_encoder = joblib.load("../Model/target_encoder.pkl")

# Page config
st.set_page_config(page_title="Churn Insight Predictor", layout="wide")
st.markdown("<h1 style='text-align: center; color: #003262;'>📉 Churn Insight Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Fill in customer details below to predict churn risk.</p>", unsafe_allow_html=True)

# --- Layout with 3 columns for input ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Customer Info")
    gender = st.radio("Gender", ['Female', 'Male'])
    senior = st.radio("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Partner", ['Yes', 'No'])
    dependents = st.selectbox("Dependents", ['Yes', 'No'])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    st.subheader("☎️ Services")
    phone = st.selectbox("Phone Service", ['Yes', 'No'])
    multiple = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
    internet = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    security = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
    backup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
    device = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
    support = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
    tv = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
    movies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])

with col3:
    st.subheader("💳 Billing")
    monthly = st.slider("Monthly Charges ($)", 0.0, 120.0, 65.0)
    total = st.slider("Total Charges ($)", 0.0, 10000.0, 2000.0)
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    paperless = st.selectbox("Paperless Billing", ['Yes', 'No'])
    payment = st.selectbox("Payment Method", [
        'Electronic check', 'Mailed check',
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ])

# --- Prediction ---
st.markdown("----")
if st.button("🔮 Predict Now"):
    with st.spinner("Analyzing..."):
        time.sleep(1)

        input_data = pd.DataFrame([{
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multiple,
            'InternetService': internet,
            'OnlineSecurity': security,
            'OnlineBackup': backup,
            'DeviceProtection': device,
            'TechSupport': support,
            'StreamingTV': tv,
            'StreamingMovies': movies,
            'MonthlyCharges': monthly,
            'TotalCharges': total,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment
        }])

        try:
            input_data = input_data[model_columns]
            prediction = pipeline.predict(input_data)[0]
            label = target_encoder.inverse_transform([prediction])[0]

            prob = pipeline.predict_proba(input_data)[0][1] * 100 if hasattr(pipeline, "predict_proba") else None

            if label == "Yes":
                st.error("⚠️ The customer is **likely to churn**.")
            else:
                st.success("✅ The customer is **likely to stay**.")

            if prob is not None:
                st.info(f"📊 Estimated churn probability: **{prob:.2f}%**")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# --- Optional Insights Section ---
st.markdown("---")
with st.expander("📊 Show Churn Overview Table"):
    st.markdown("##### Example Churn Analysis")
    df_report = pd.DataFrame({
        "Segment": ["Stayed", "Churned"],
        "Count": [510, 235],
        "Avg Monthly Charges": [64.2, 78.9],
        "Avg Tenure": [32.5, 9.2]
    })
    st.dataframe(df_report, use_container_width=True)
    st.bar_chart(df_report.set_index("Segment")["Count"])
