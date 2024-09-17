# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:30:55 2024

@author: ADEBIYI I
"""

#Importing libraries
import numpy as np
import pandas as pd
import pickle

 #loading the saved model
load_model = pickle.load(open('C:/Users/ADEBIYI I/Documents/road-to-machine-learning/projects/2/trained model.sav', 'rb'))


## The Predictive system
input_data = (1,85,66,29,0,26.6,0.351,31)

#changing of input to a numpy array
data_array = np.asarray(input_data)

#reshaping the np array as we predict
reshape_input_data = data_array.reshape(1,-1)

#predicting diabetes
predict = load_model.predict(reshape_input_data)

if (predict==0):
    print('is not diabetic')
else :
    print('is diabetic')

print(predict)