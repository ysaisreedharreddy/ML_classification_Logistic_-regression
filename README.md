Logistic Regression Car Purchase Predictor
This repository contains a Python application using Streamlit that demonstrates how logistic regression can be used to predict whether a person will purchase a car based on their age and estimated salary. The application allows users to input their age and estimated salary, and it uses a pre-trained logistic regression model to predict the likelihood of a car purchase.

Features
Interactive Prediction: Users can input their age and estimated salary to predict the likelihood of purchasing a car.
Pre-trained Model: Utilizes a logistic regression model trained on a dataset of social network advertisement responses.
Real-time Results: Instantly displays the probability of the user purchasing a car.

Technologies Used
Python: The project is implemented in Python, highlighting its versatility in data processing and machine learning.
Streamlit: This application leverages Streamlit for creating the web interface, making it user-friendly and interactive.
Pickle: Used for loading the pre-trained logistic regression model and scaler.

Project Structure
logistic_regression_app.py: Main application script for running the Streamlit interface.
classifier.pkl: Pickle file containing the trained logistic regression model.
scaler.pkl: Pickle file containing the scaler for normalizing input data.



Run the application:
streamlit run logistic_regression_app.py

Usage
Open the Streamlit application, and enter your age and estimated salary in the provided input fields. Click the "Predict Purchase" button to see the prediction results displayed on the screen.
