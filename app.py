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
def predict_heart_disease(inputs):
    predictions = model.predict(inputs)
    return predictions

# Streamlit UI
st.title("Heart Disease Prediction")

# Tabs for single prediction and multi-prediction
tab_selector = st.sidebar.radio("Select Prediction Mode", ["Single Prediction", "Multi Prediction"])

if tab_selector == "Single Prediction":
    st.sidebar.header("User Input (Single Prediction)")

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

    # Predict button for single prediction
    # Predict button for single prediction
if st.sidebar.button("Predict (Single)"):
    sex = 1 if sex == "Male" else 0
    cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    fbs = 1 if fbs == "Yes" else 0
    restecg_mapping = {"Normal": 0, "Abnormal": 1, "Hypertrophy": 2}
    exang = 1 if exang == "Yes" else 0

    input_data = np.array([[age, sex, cp_mapping[cp], trestbps, chol, fbs, restecg_mapping[restecg], thalach, exang, oldpeak]])
    
    # Debug prints
    st.write("Debug - Input Data:")
    st.write(input_data)

    prediction = predict_heart_disease(input_data)

    # Debug prints
    st.write("Debug - Raw Prediction:")
    st.write(prediction)

    st.write("Single Prediction Result:")
    st.write(prediction)


elif tab_selector == "Multi Prediction":
    st.sidebar.header("User Input (Multi Prediction)")

    st.sidebar.write("Upload a CSV file containing multiple rows of input data.")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type="csv")

    # Predict button for multi-prediction
    if st.sidebar.button("Predict (Multi)"):
        if uploaded_file is not None:
            # Read the CSV file
            uploaded_df = pd.read_csv(uploaded_file)

            # Ensure the DataFrame has the correct column names and data types
            # Adjust this part based on your input requirements
            uploaded_df["Sex"] = uploaded_df["Sex"].map({"Male": 1, "Female": 0})
            uploaded_df["Chest Pain Type"] = uploaded_df["Chest Pain Type"].map({"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3})
            uploaded_df["Fasting Blood Sugar > 120 mg/dL"] = uploaded_df["Fasting Blood Sugar > 120 mg/dL"].map({"No": 0, "Yes": 1})
            uploaded_df["Resting Electrocardiographic Results"] = uploaded_df["Resting Electrocardiographic Results"].map({"Normal": 0, "Abnormal": 1, "Hypertrophy": 2})
            uploaded_df["Exercise Induced Angina"] = uploaded_df["Exercise Induced Angina"].map({"No": 0, "Yes": 1})

            # Ensure the DataFrame has the correct columns and order based on the model
            # Adjust this part based on your model input requirements
            input_columns = ["Age", "Sex", "Chest Pain Type", "Resting Blood Pressure (mm Hg)", "Serum Cholesterol (mg/dL)", "Fasting Blood Sugar > 120 mg/dL",
                             "Resting Electrocardiographic Results", "Maximum Heart Rate Achieved", "Exercise Induced Angina", "ST Depression Induced by Exercise"]
            uploaded_df = uploaded_df[input_columns]

            # Predict
            predictions = predict_heart_disease(uploaded_df.values)

            st.write("Multi Prediction Results:")
            st.write(predictions)
        else:
            st.write("Please upload a CSV file for multi-prediction.")

# You can customize the layout and styling further based on your preferences.
