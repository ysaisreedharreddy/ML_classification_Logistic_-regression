import streamlit as st
import numpy as np
import pickle

# Define the path where your model and scaler are stored
model_path = r'C:\Users\prasu\DS2\git\classification\1. Logistic regression\2.LOGISTIC REGRESSION CODE\classifier.pkl'
scaler_path = r'C:\Users\prasu\DS2\git\classification\1. Logistic regression\2.LOGISTIC REGRESSION CODE\scaler.pkl'

# Load the trained model and scaler
with open(model_path, 'rb') as f:
    classifier = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Setting up the Streamlit interface
st.title("Car Purchase Prediction Using Logistic Regression")

# User inputs for the features
age = st.number_input('Enter your age', min_value=18, max_value=100, value=30)
salary = st.number_input('Enter your estimated salary', min_value=10000, max_value=1000000, value=50000)

# Button to perform prediction
if st.button('Predict Purchase'):
    # Prepare the feature vector for prediction
    features = np.array([[age, salary]])
    features_scaled = scaler.transform(features)

    # Perform prediction
    prediction = classifier.predict(features_scaled)
    proba = classifier.predict_proba(features_scaled)

    # Display the prediction
    if prediction[0] == 1:
        st.success(f'You are likely to purchase the car! Probability: {proba[0][1]:.2f}')
    else:
        st.error(f'You are unlikely to purchase the car. Probability: {proba[0][1]:.2f}')

# Additional details
st.write("This model predicts whether a customer is likely to purchase a car based on their age and estimated salary.")
