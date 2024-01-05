import streamlit as st
import pandas as pd
import numpy as np
import itertools
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
@st.cache
def load_data():
    with open("dataset/hungarian.data", encoding='Latin1') as file:
        lines = [line.strip() for line in file]

    data = itertools.takewhile(
        lambda x: len(x) == 76,
        (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
    )

    df = pd.DataFrame.from_records(data)
    df = df.iloc[:, :-1]
    df = df.drop(df.columns[0], axis=1)
    df = df.astype(float)

    df.replace(-9.0, np.NaN, inplace=True)

    df_selected = df.iloc[:, [1, 2, 7, 8, 10, 14, 17, 30, 36, 38, 39, 42, 49, 56]]

    column_mapping = {
        2: 'age',
        3: 'sex',
        8: 'cp',
        9: 'trestbps',
        11: 'chol',
        15: 'fbs',
        18: 'restecg',
        31: 'thalach',
        37: 'exang',
        39: 'oldpeak',
        40: 'slope',
        43: 'ca',
        50: 'thal',
        57: 'num'
    }

    df_selected.rename(columns=column_mapping, inplace=True)

    columns_to_drop = ['ca', 'slope', 'thal']
    df_selected = df_selected.drop(columns_to_drop, axis=1)

    meanTBPS = df_selected['trestbps'].dropna()
    # ...

    fill_values = {
        'trestbps': meanTBPS,
        'chol': meanChol,
        'fbs': meanfbs,
        'thalach': meanthalach,
        'exang': meanexang,
        'restecg': meanRestCG
    }

    df_clean = df_selected.fillna(value=fill_values)
    df_clean.drop_duplicates(inplace=True)

    X = df_clean.drop("num", axis=1)
    y = df_clean['num']

    return X, y


# Load the model
@st.cache
def load_model():
    return pickle.load(open("model/hungarian_heart.pkl", 'rb'))


# Main function
def main():
    st.set_page_config(
        page_title="Heart Disease Prediction",
        page_icon=":heart:"
    )

    st.title("Heart Disease Prediction App")
    st.sidebar.header("User Input")

    X, y = load_data()
    model = load_model()

    age = st.sidebar.number_input("Age", X['age'].min(), X['age'].max())
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    # ... (similar for other features)

    if sex == "Male":
        sex_val = 1
    else:
        sex_val = 0

    # ... (similar mapping for other features)

    # User input as a DataFrame
    user_input = pd.DataFrame({
        'age': [age],
        'sex': [sex_val],
        # ... (similar for other features)
    })

    # Display User Input
    st.subheader("User Input:")
    st.write(user_input)

    # Make Prediction
    if st.button("Predict"):
        prediction = model.predict(user_input)[0]

        st.subheader("Prediction Result:")
        if prediction == 0:
            st.success("Healthy")
        elif prediction == 1:
            st.warning("Heart disease level 1")
        elif prediction == 2:
            st.warning("Heart disease level 2")
        elif prediction == 3:
            st.error("Heart disease level 3")
        elif prediction == 4:
            st.error("Heart disease level 4")

    # Model Performance Metrics
    st.sidebar.subheader("Model Performance Metrics")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.text(f"Accuracy: {accuracy:.2%}")

    # ... (you can display other metrics like confusion matrix, classification report, etc.)

    # Display Dataset
    if st.checkbox("Show Dataset"):
        st.subheader("Heart Disease Dataset")
        st.write(pd.concat([X, y], axis=1))

if __name__ == "__main__":
    main()
