import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the KNN model using joblib
model_path = "model/knn.joblib"  # Update with your actual path
knn_model = joblib.load(model_path)

# Streamlit UI
st.title("Heart Disease Prediction with KNN")

# User Input for Prediction
st.sidebar.header("User Input")
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)
sex = st.sidebar.radio("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, value=120)
chol = st.sidebar.number_input("Serum Cholesterol (mg/dL)", min_value=50, max_value=600, value=200)
fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", ["Normal", "Abnormal", "Hypertrophy"])
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=70, max_value=200, value=150)
exang = st.sidebar.radio("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0)

# Map categorical variables to numerical values
sex = 1 if sex == "Male" else 0
cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
fbs = 1 if fbs == "Yes" else 0
restecg_mapping = {"Normal": 0, "Abnormal": 1, "Hypertrophy": 2}
exang = 1 if exang == "Yes" else 0

# Create a DataFrame with the input data
input_data = pd.DataFrame({
    "Age": [age],
    "Sex": [sex],
    "Chest Pain Type": [cp_mapping[cp]],
    "Resting Blood Pressure (mm Hg)": [trestbps],
    "Serum Cholesterol (mg/dL)": [chol],
    "Fasting Blood Sugar > 120 mg/dL": [fbs],
    "Resting Electrocardiographic Results": [restecg_mapping[restecg]],
    "Maximum Heart Rate Achieved": [thalach],
    "Exercise Induced Angina": [exang],
    "ST Depression Induced by Exercise": [oldpeak]
})

# Prediction
if st.button("Predict"):
    prediction = knn_model.predict(input_data)
    st.write("Prediction Result:", prediction[0])

# You can further customize the layout and styling based on your preferences.
