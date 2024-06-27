import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open("calories_burnt_prediction.sav", 'rb'))

# Creating a function for Prediction 
def calories_prediction(input_data):
    # Changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    
    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction[0]

def main():
    # Giving a title 
    st.title('Calories Burnt Prediction')
    
    # Getting the input from the user
    Gender = st.selectbox('Select your gender', ['Male', 'Female'])
    Age = st.number_input('Enter your age', min_value=0, max_value=120, value=25, step=1)
    Height = st.number_input('Enter your height (in cm)', min_value=0, max_value=300, value=170, step=1)
    Weight = st.number_input('Enter your weight (in kg)', min_value=0, max_value=300, value=70, step=1)
    Duration = st.number_input('Enter the duration (in minutes)', min_value=0, max_value=500, value=30, step=1)
    Heart_Rate = st.number_input('Enter your heart rate', min_value=0, max_value=200, value=70, step=1)
    Body_Temp = st.number_input('Enter your body temperature (in °C)', min_value=30.0, max_value=45.0, value=36.5, step=0.1)
    
    # Encode Gender: assuming 0 for Male and 1 for Female
    Gender_encoded = 0 if Gender == 'Male' else 1
    
    # Creating a button for prediction
    if st.button('Check Calories'):
        Calories = calories_prediction([Gender_encoded, Age, Height, Weight, Duration, Heart_Rate, Body_Temp])
        st.success(f'Calories Burnt: {Calories}')

if __name__ == '__main__':
    main()
