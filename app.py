import pandas as pd
import numpy as np
import streamlit as st
import time
import pickle

# Load the XGBoost model using pickle
model_path = "model/xgb.pkl"  # Update with your actual path
xgb_model = pickle.load(open(model_path, "rb"))

# Map values for dropdowns
sex_mapping = {"Male": 1, "Female": 0}
cp_mapping = {"Typical angina": 1, "Atypical angina": 2, "Non-anginal pain": 3, "Asymptomatic": 4}
fbs_mapping = {"False": 0, "True": 1}
restecg_mapping = {"Normal": 0, "Having ST-T wave abnormality": 1, "Showing left ventricular hypertrophy": 2}
exang_mapping = {"No": 0, "Yes": 1}

# Streamlit UI
st.title("Heart Disease Prediction")

# Tabs for single prediction and multi-prediction
tab_selector = st.sidebar.radio("Select Prediction Mode", ["Single Prediction", "Multi Prediction"])

if tab_selector == "Single Prediction":
    st.sidebar.header("User Input (Single Prediction)")

    age = st.sidebar.number_input("Age", min_value=29, max_value=77, value=44)
    sex_sb = st.sidebar.selectbox("Sex", ["Male", "Female"])
    cp_sb = st.sidebar.selectbox("Chest pain type", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
    trestbps = st.sidebar.number_input("Resting blood pressure (in mm Hg on admission to the hospital)", min_value=94, max_value=200, value=124)
    chol = st.sidebar.number_input("Serum cholestoral (in mg/dl)", min_value=126, max_value=564, value=240)
    fbs_sb = st.sidebar.selectbox("Fasting blood sugar > 120 mg/dl?", ["False", "True"])
    restecg_sb = st.sidebar.selectbox("Resting electrocardiographic results", ["Normal", "Having ST-T wave abnormality", "Showing left ventricular hypertrophy"])
    thalach = st.sidebar.number_input("Maximum heart rate achieved", min_value=71, max_value=202, value=144)
    exang_sb = st.sidebar.selectbox("Exercise induced angina?", ["No", "Yes"])
    oldpeak = st.sidebar.number_input("ST depression induced by exercise relative to rest", min_value=0.0, max_value=6.2, value=0.0)

    # Map values to numerical
    sex = sex_mapping[sex_sb]
    cp = cp_mapping[cp_sb]
    fbs = fbs_mapping[fbs_sb]
    restecg = restecg_mapping[restecg_sb]
    exang = exang_mapping[exang_sb]

    # Create input dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Chest pain type': [cp],
        'Resting blood pressure': [trestbps],
        'Serum cholestoral': [chol],
        'Fasting blood sugar': [fbs],
        'Resting electrocardiographic results': [restecg],
        'Maximum heart rate achieved': [thalach],
        'Exercise induced angina': [exang],
        'ST depression': [oldpeak]
    })

    # Prediction for Single Prediction
    if st.sidebar.button("Predict (Single)"):
        prediction = xgb_model.predict(input_data)[0]

        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()

        if prediction == 0:
            result = ":green[**Healthy**]"
        elif prediction == 1:
            result = ":orange[**Heart disease level 1**]"
        elif prediction == 2:
            result = ":orange[**Heart disease level 2**]"
        elif prediction == 3:
            result = ":red[**Heart disease level 3**]"
        elif prediction == 4:
            result = ":red[**Heart disease level 4**]"

        st.write("")
        st.write("")
        st.subheader("Prediction:")
        st.subheader(result)

elif tab_selector == "Multi Prediction":
    st.sidebar.header("User Input (Multi Prediction)")

    st.sidebar.write("Upload a CSV file containing multiple rows of input data.")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type="csv")

    # Prediction for Multi Prediction
    if st.sidebar.button("Predict (Multi)"):
        if uploaded_file is not None:
            # Read the CSV file
            uploaded_df = pd.read_csv(uploaded_file)

            # Map categorical values to numerical
            uploaded_df["Sex"] = uploaded_df["Sex"].map(sex_mapping)
            uploaded_df["Chest pain type"] = uploaded_df["Chest pain type"].map(cp_mapping)
            uploaded_df["Fasting blood sugar"] = uploaded_df["Fasting blood sugar"].map(fbs_mapping)
            uploaded_df["Resting electrocardiographic results"] = uploaded_df["Resting electrocardiographic results"].map(restecg_mapping)
            uploaded_df["Exercise induced angina"] = uploaded_df["Exercise induced angina"].map(exang_mapping)

            # Ensure the DataFrame has the correct columns and order based on the model
            # Adjust this part based on your model input requirements
            input_columns = ["Age", "Sex", "Chest pain type", "Resting blood pressure", "Serum cholestoral", "Fasting blood sugar",
                             "Resting electrocardiographic results", "Maximum heart rate achieved", "Exercise induced angina", "ST depression"]
            uploaded_df = uploaded_df[input_columns]

            # Predict
            predictions = xgb_model.predict(uploaded_df)

            bar = st.progress(0)
            status_text = st.empty()

            for i in range(1, 101):
                status_text.text(f"{i}% complete")
                bar.progress(i)
                time.sleep(0.01)
                if i == 100:
                    time.sleep(1)
                    status_text.empty()
                    bar.empty()

            result_arr = []
            for prediction in predictions:
                if prediction == 0:
                    result = "Healthy"
                elif prediction == 1:
                    result = "Heart disease level 1"
                elif prediction == 2:
                    result = "Heart disease level 2"
                elif prediction == 3:
                    result = "Heart disease level 3"
                elif prediction == 4:
                    result = "Heart disease level 4"
                result_arr.append(result)

            uploaded_result = pd.DataFrame({'Prediction Result': result_arr})

            st.write("Multi Prediction Results:")
            st.write(uploaded_result)
        else:
            st.write("Please upload a CSV file for multi-prediction.")
