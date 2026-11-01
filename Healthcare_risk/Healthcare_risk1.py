"""
Healthcare Risk Stratification - Streamlit App (FIXED)

Loads risk_model1.pkl, which is now a full Pipeline (StandardScaler +
LogisticRegression) saved by train_risk_model.py. Because scaling is baked
into the pipeline, this app doesn't need to scale inputs manually - just
pass raw values in, same as before.
"""
import streamlit as st
import pandas as pd
import joblib

model = joblib.load('risk_model1.pkl')

st.title("Healthcare Risk Stratification App")

age = st.number_input("Age", min_value=0)
length_of_stay = st.number_input("Length of stay (days)", min_value=0)
treatment_cost = st.number_input("Treatment Cost", min_value=0)
abnormal_lab_count = st.number_input("Abnormal Lab Count", min_value=0)

if st.button("Predict"):
    input_data = pd.DataFrame(
        [[age, length_of_stay, treatment_cost, abnormal_lab_count]],
        columns=['Age', 'LengthOfstay', 'TreatmentCost', 'AbnormalLabCount']
    )
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    st.write(f"Risk Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}")
    st.write(f"Risk Probability: {round(probability, 2)}")