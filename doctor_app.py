"""Streamlit app for predicting heart disease using a pretrained LightGBM model.
This tool is for research and educational purposes only and must not be used as a
substitute for professional medical diagnosis or treatment."""

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = 'lgbm_model.pkl'

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title('Heart Disease Prediction (Prototype)')
st.write('This tool uses a pretrained LightGBM model to predict the heart disease '
         'status of a patient. It is intended for demonstration only and does not '
         'provide medical advice. Always consult qualified healthcare professionals '
         'for medical decisions.')

st.header('Patient Information')

age = st.number_input('Age', min_value=0, max_value=120, value=50)
gender = st.selectbox('Gender', ['Male', 'Female'])
blood_pressure = st.number_input('Blood Pressure', value=120.0)
cholesterol_level = st.number_input('Cholesterol Level', value=200.0)
exercise_habits = st.selectbox('Exercise Habits', ['Low', 'Medium', 'High'])
smoking = st.selectbox('Smoking', ['Yes', 'No'])
family_history = st.selectbox('Family Heart Disease', ['Yes', 'No'])
diabetes = st.selectbox('Diabetes', ['Yes', 'No'])
bmi = st.number_input('BMI', value=25.0)
high_bp = st.selectbox('High Blood Pressure', ['Yes', 'No'])
low_hdl = st.selectbox('Low HDL Cholesterol', ['Yes', 'No'])
high_ldl = st.selectbox('High LDL Cholesterol', ['Yes', 'No'])
alcohol = st.selectbox('Alcohol Consumption', ['Low', 'Medium', 'High'])
stress = st.selectbox('Stress Level', ['Low', 'Medium', 'High'])
sleep_hours = st.number_input('Sleep Hours', value=7.0)
sugar_consumption = st.selectbox('Sugar Consumption', ['Low', 'Medium', 'High'])
triglyceride_level = st.number_input('Triglyceride Level', value=150.0)
fasting_blood_sugar = st.number_input('Fasting Blood Sugar', value=100.0)
crp_level = st.number_input('CRP Level', value=10.0)
homocysteine_level = st.number_input('Homocysteine Level', value=10.0)

if st.button('Predict'):
    input_df = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Blood Pressure': [blood_pressure],
        'Cholesterol Level': [cholesterol_level],
        'Exercise Habits': [exercise_habits],
        'Smoking': [smoking],
        'Family Heart Disease': [family_history],
        'Diabetes': [diabetes],
        'BMI': [bmi],
        'High Blood Pressure': [high_bp],
        'Low HDL Cholesterol': [low_hdl],
        'High LDL Cholesterol': [high_ldl],
        'Alcohol Consumption': [alcohol],
        'Stress Level': [stress],
        'Sleep Hours': [sleep_hours],
        'Sugar Consumption': [sugar_consumption],
        'Triglyceride Level': [triglyceride_level],
        'Fasting Blood Sugar': [fasting_blood_sugar],
        'CRP Level': [crp_level],
        'Homocysteine Level': [homocysteine_level]
    })

    prediction = model.predict(input_df)[0]
    st.subheader('Prediction')
    st.write(f'Heart Disease Status: **{prediction}**')
