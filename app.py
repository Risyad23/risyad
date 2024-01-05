import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the XGBoost model using pickle
model_path = "model/xgb.pkl"  # Update with your actual path
xgb_model = pickle.load(open(model_path, "rb"))

# Helper function for single prediction
def predict_single_heart_disease(inputs):
    prediction = xgb_model.predict(inputs)
    return prediction

# Helper function for multi-prediction
def predict_multi_heart_disease(inputs):
    predictions = xgb_model.predict(inputs)
    return predictions

# Streamlit UI
st.title("Heart Disease Prediction with XGBoost")

# Tabs for single prediction and multi-prediction
tab_selector = st.sidebar.radio("Select Prediction Mode", ["Single Prediction", "Multi Prediction"])

if tab_selector == "Single Prediction":
    st.sidebar.header("User Input (Single Prediction)")
    # User Input for Single Prediction
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

    # Prediction for Single Prediction
    if st.sidebar.button("Predict (Single)"):
        prediction = predict_single_heart_disease(input_data.values)
        st.write("Single Prediction Result:", prediction[0])

elif tab_selector == "Multi Prediction":
    st.sidebar.header("User Input (Multi Prediction)")
    # User Input for Multi Prediction
    st.sidebar.write("Upload a CSV file containing multiple rows of input data.")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type="csv")

    # Prediction for Multi Prediction
    if st.sidebar.button("Predict (Multi)"):
        if uploaded_file is not None:
            # Read the CSV file
            uploaded_df = pd.read_csv(uploaded_file)

            # Ensure the DataFrame has the correct column names and data types
            # Adjust this part based on your input requirements
            # ...

            # Ensure the DataFrame has the correct columns and order based on the model
            # Adjust this part based on your model input requirements
            # ...

            # Predict
            predictions = predict_multi_heart_disease(uploaded_df.values)

            st.write("Multi Prediction Results:")
            st.write(predictions)
        else:
            st.write("Please upload a CSV file for multi-prediction.")

# You can further customize the layout and styling based on your preferences.
