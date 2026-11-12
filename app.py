import streamlit as st
import joblib
import numpy as np

# Load trained machine learning model
model = joblib.load("iris_model.pkl")

# Flower names
flower_names = ["Setosa", "Versicolor", "Virginica"]

# Application title
st.title("🌸 Iris Flower Predictor")

# Application description
st.write(
    "Enter the flower measurements below "
    "to predict its species using Machine Learning."
)

st.subheader("Flower Measurements")

# Input fields

sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2
)

# Prediction button

if st.button("Predict Flower"):

    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(input_data)

    predicted_flower = flower_names[prediction[0]]

    st.success(f"Predicted Flower: {predicted_flower}")

    