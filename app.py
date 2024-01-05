import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

# Load the model
model_path = "model/hungarian_heart.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Helper function for prediction
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]])
    prediction = model.predict(input_data)
    return prediction

# Streamlit UI
st.title("Heart Disease Prediction")

# Sidebar for user input
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

# Predict button
if st.sidebar.button("Predict"):
    sex = 1 if sex == "Male" else 0
    cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    fbs = 1 if fbs == "Yes" else 0
    restecg_mapping = {"Normal": 0, "Abnormal": 1, "Hypertrophy": 2}
    exang = 1 if exang == "Yes" else 0

    prediction = predict_heart_disease(age, sex, cp_mapping[cp], trestbps, chol, fbs, restecg_mapping[restecg], thalach, exang, oldpeak)

    st.write("Prediction Result:")
    st.write(prediction)
