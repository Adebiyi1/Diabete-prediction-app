# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:53:38 2024

@author: ADEBIYI I
"""

import numpy as np
import pickle
import streamlit as st

# Loading the saved model
load_model = pickle.load(open('trained model.sav', 'rb'))

# Creating function for prediction
def diabetes_prediction(input_data):
    # Changing the input to a numpy array
    data_array = np.asarray(input_data)

    # Reshaping the np array as we predict
    reshape_input_data = data_array.reshape(1, -1)

    # Predicting diabetes
    predict = load_model.predict(reshape_input_data)

    if predict[0] == 0:
        return 'is not diabetic'
    else:
        return 'is diabetic'


def main():
    # Web app title
    st.title("Biyi's Diabetes Prediction Web App")

    # User's inputted data
    Pregnancies = st.text_input('What is the number of pregnancies') 
    Glucose = st.text_input('What is the glucose level')
    BloodPressure = st.text_input('What is the Blood Pressure (BP)')
    SkinThickness = st.text_input('What is the skin thickness')
    Insulin = st.text_input('What is the amount of insulin used')
    BMI = st.text_input('What is the BMI level')
    DiabetesPedigreeFunction = st.text_input('What is the Diabetes Pedigree Function')
    Age = st.text_input('What is the age of the patient')

    # Code for prediction
    diagnosis = ''

    # Ensure that all inputs are provided before making a prediction
    if st.button('Diabetes test result'):
        try:
            # Convert inputs to float if provided
            input_data = [
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age)
            ]
            # Make prediction
            diagnosis = diabetes_prediction(input_data)
        except ValueError:
            st.error("Please ensure all input fields are filled with valid numbers.")
    
    st.success(diagnosis)


if __name__ == '__main__':
    main()
