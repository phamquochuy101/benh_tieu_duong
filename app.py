# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 09:52:03 2022

@author: ANURAG
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model

loaded_model = pickle.load(open("D:/HK!_2023-2024/newtech/diabetes/benh_tieu_duong/trained_model.sav", "rb"))

# Creating a function for Prediction

def diabetes_prediction(input_data):
    
    

    #changing the input data to a numpy array

    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance

    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    # Standarize nahi kar rahe hai

    prediction = loaded_model.predict(input_data_reshaped)

    print(prediction)

    if (prediction[0])==0:
        return "Người này không bị tiểu đường"
    else:
        return "Người này bị tiểu đường"
        
        
        
def main():
    
    
    #Giving a title 
    
    st.title("Ứng dụng dự đoán bệnh tiểu đường")
    
    # Getting the input data from the user
    
        
    Pregnancies = st.text_input("Số lần mang thai")
    Glucose = st.text_input("Mức đường huyết")
    BloodPressure = st.text_input("Huyết áp")
    SkinThickness = st.text_input("Độ dày của da")
    Insulin = st.text_input("Mức độ Insulin")
    BMI = st.text_input("Chỉ số BMI")
    DiabetesPedigreeFunction = st.text_input("Chỉ số mức dộ di truyền")
    Age = st.text_input("Tuổi")
    
    
    #Code for prediction
    
    diagnosis = ''
    
    
    #Creating a button for prediction
    
    if st.button("Kết qủa kiểm tra bệnh tiểu đường"):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    