# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# loading the saved model

loaded_model = pickle.load(open("D:/HK!_2023-2024/newtech/diabetes/benh_tieu_duong/trained_model.sav", "rb"))

input_data = (5,166,72,19,175,25.8,0.587,51)

#changing the input data to a numpy array

input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance

input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

# Standarize nahi kar rahe hai

prediction = loaded_model.predict(input_data_reshaped)

print(prediction)

if (prediction[0])==0:
    print("Người này không bị tiểu đường")
else:
    print("Người này bị tiểu đường")