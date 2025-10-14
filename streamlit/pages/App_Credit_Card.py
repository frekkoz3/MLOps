import streamlit as st
import numpy as np
import pickle
import pandas as pd

# -------------------------------
# 1. Load model and scaler
# -------------------------------
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Example accuracy value
MODEL_ACCURACY = 0.9422

# -------------------------------
# 2. Page configuration
# -------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("💳 Credit Card Fraud Detection System")
st.markdown("""
This application predicts whether a **credit card transaction** is likely to be *fraudulent* or *legitimate*, 
based on behavioral and transaction-related features.
""")


# -------------------------------
# 3. User input section
# -------------------------------
st.header("🧾 Transaction Feature Inputs")

st.write("Adjust the sliders or inputs to simulate a transaction and evaluate its fraud risk.")

# Divide inputs into logical groups (for readability)
st.subheader("Transaction Information")
col1, col2 = st.columns(2)
with col1:
    transaction_time = st.number_input("Transaction Time (seconds)", min_value=0.0, value=10000.0, step=100.0)
with col2:
    transaction_amount = st.number_input("Transaction Amount (€)", min_value=0.0, max_value=5000.0, value=120.0, step=10.0)

st.subheader("Behavioral and Risk Indicators")
cols = st.columns(3)
user_inputs = []

# Loop over V1–V28 for completeness
for i in range(1, 29):
    col = cols[(i - 1) % 3]
    user_inputs.append(
        col.slider(f"Feature V{i}", -5.0, 5.0, 0.0, step=0.1)
    )

# Combine all features into an array (Time, V1–V28, Amount)
input_data = np.array([[transaction_time] + user_inputs + [transaction_amount]])

# Scale input
if hasattr(scaler, "feature_names_in_"):
    input_df = pd.DataFrame(input_data, columns=scaler.feature_names_in_)
    scaled = scaler.transform(input_df)
else:
    scaled = scaler.transform(input_data)
# -------------------------------
# 4. Prediction section
# -------------------------------
st.markdown("---")
st.header("🔍 Prediction")

if st.button("Predict Fraud Risk"):
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    st.subheader("📊 Model Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Fraud Detected!\n\nFraud Probability: **{probability:.2%}**")
    else:
        st.success(f"✅ Transaction Appears Legitimate.\n\nFraud Probability: **{probability:.2%}**")

# -------------------------------
# 5. Display model performance
# -------------------------------
st.markdown("---")
st.subheader("📈 Model Performance")
st.write(f"**Model Accuracy:** {MODEL_ACCURACY * 100:.2f}%")
st.caption("Accuracy calculated using the validation dataset from training.")

st.markdown(
    "<small>Developed for 440MI – University of Trieste | Demonstration of scalable ML model deployment using Streamlit.</small>",
    unsafe_allow_html=True,
)
