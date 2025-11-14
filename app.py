import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Page setup
st.set_page_config(page_title="AI Health Risk Predictor", page_icon="hp.png", layout="centered")
st.title("AI Health Risk Predictor")
st.write("Enter your health and lifestyle details to predict your disease risk level.")

# Load the trained model
model = joblib.load("small_model.pkl")

# Input fields (all in main area)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=100, value=25)
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=22.5)
daily_steps = st.number_input("Average Daily Steps", min_value=0, max_value=30000, value=8000)
sleep_hours = st.number_input("Average Sleep Hours", min_value=0.0, max_value=12.0, value=7.0)
water_intake_l = st.number_input("Daily Water Intake (liters)", min_value=0.0, max_value=10.0, value=2.0)
calories_consumed = st.number_input("Daily Calories Consumed", min_value=500, max_value=6000, value=2500)
resting_hr = st.number_input("Resting Heart Rate (bpm)", min_value=40, max_value=120, value=72)
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=50, max_value=130, value=80)
cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=180)
smoker = st.selectbox("Do you smoke?", ["No", "Yes"])
alcohol = st.selectbox("Do you consume alcohol?", ["No", "Yes"])
family_history = st.selectbox("Any family history of disease?", ["No", "Yes"])
screen_time = st.number_input("Average Screen Time (hours/day)", min_value=0.0, max_value=15.0, value=5.0)
mood_score = st.slider("Mood Score (0 = poor mood, 10 = great mood)", 0.0, 10.0, 7.0)

# Encode categorical data
le = LabelEncoder()
gender_encoded = 1 if gender == "Male" else 0
smoker_encoded = 1 if smoker == "Yes" else 0
alcohol_encoded = 1 if alcohol == "Yes" else 0
family_history_encoded = 1 if family_history == "Yes" else 0

# Prepare input dataframe
input_data = pd.DataFrame([[
    age, gender_encoded, bmi, daily_steps, sleep_hours, water_intake_l,
    calories_consumed, resting_hr, systolic_bp, diastolic_bp, cholesterol,
    smoker_encoded, alcohol_encoded, family_history_encoded,
    screen_time, mood_score
]], columns=[
    'age','gender','bmi','daily_steps','sleep_hours','water_intake_l',
    'calories_consumed','resting_hr','systolic_bp','diastolic_bp',
    'cholesterol','smoker','alcohol','family_history',
    'screen_time','mood_score'
])

# Scale numerical features (same scaling logic used in training)
scaler = StandardScaler()
numeric_cols = ['age','bmi','daily_steps','sleep_hours','water_intake_l','calories_consumed',
                'resting_hr','systolic_bp','diastolic_bp','cholesterol']
input_data[numeric_cols] = scaler.fit_transform(input_data[numeric_cols])

# Predict
if st.button("Predict Disease Risk"):
    prediction = model.predict(input_data)[0]
    risk_label = "High Risk 🚨" if prediction == 1 else "Low Risk ✅"
    st.subheader(f"Predicted Health Status: {risk_label}")
