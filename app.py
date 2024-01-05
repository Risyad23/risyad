import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the XGBoost model using pickle
model_path = "model/xgb.pkl"
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
    # ... (same as the original code)

    # Prediction for Single Prediction
    if st.sidebar.button("Predict (Single)"):
        # ... (same as the original code)
        prediction = predict_single_heart_disease(input_data)
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
