# Saya akan membuat aplikasi prediksi penyakit jantung menggunakan machine learning
# dan di deploy menggunakan streamlit

# Import library yang dibutuhkan
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

st.title("Heart Disease Prediction")

st.write("""
Aplikasi ini dapat memprediksi apakah seseorang terkena penyakit jantung atau tidak
""")

st.sidebar.header("User Input Features")

# Fungsi untuk mengambil input dari user


def user_input_features():
    age = int(st.sidebar.number_input("Age: "))
    sex = int(st.sidebar.selectbox("Sex: ", (0, 1)))
    cp = int(st.sidebar.selectbox("chest pain type: ", (0, 1, 2, 3)))
    trestbps = int(st.sidebar.number_input("resting blood pressure: "))
    chol = int(st.sidebar.number_input("serum cholestoral in mg/dl: "))
    fbs = int(st.sidebar.selectbox("fasting blood sugar: ", (0, 1)))
    restecg = int(st.sidebar.selectbox(
        "Masukan resting electrocardiographic results: ", (0, 1, 2)))
    thalach = int(st.sidebar.number_input(
        "Masukan maximum heart rate achieved: "))
    exang = int(st.sidebar.selectbox(
        "Masukan exercise induced angina: ", (0, 1)))
    oldpeak = int(st.sidebar.number_input("oldpeak: "))
    slope = int(st.sidebar.selectbox(
        "Masukan the slope of the peak exercise ST segment: ", (0, 1, 2)))
    ca = int(st.sidebar.selectbox(
        "Masukan number of major vessels: ", (0, 1, 2, 3)))
    thal = int(st.sidebar.selectbox("thal: ", (0, 1, 2, 3)))

    data = {"age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal}

    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

heart_dataset = pd.read_csv("datsets/hungarian.data")

heart = df.copy()

st.write(df)

# Menyiapkan data latih dan target
X = heart_dataset.drop(columns=["target"])
y = heart_dataset["target"]

# Memisaahkan data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Membuat model KNN
model = KNeighborsClassifier(n_neighbors=3)
knn = model.fit(X_train, y_train)

# prediksi dengan algoritma KNN
btn = st.button("Prediksi")

if btn:
    # Memeriksa apakah semua nilai input adalah 0
    if df.values.sum() == 0:
        st.write("Silahkan masukan nilai input terlebih dahulu")
    else:
        # Melakukan prediksi
        prediksi = knn.predict(df)
        # Menampilkan hasil prediksi
        if prediksi == 0:
            st.write("Anda tidak terkena penyakit jantung")
        else:
            st.write("Anda terkena penyakit jantung")
