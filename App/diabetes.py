import streamlit as st
import joblib
import numpy as np
import pandas as pd

maxVals = pd.Series({
    "Age": 13.0,
    "Sex": 1.0,
    "HighChol": 1.0,
    "CholCheck": 1.0,
    "BMI": 98.0,
    "Fruits": 1.0,
    "Veggies": 1.0,
    "HvyAlcoholConsump": 1.0
})

minVals = pd.Series({
    "Age": 1.0,
    "Sex": 0.0,
    "HighChol": 0.0,
    "CholCheck": 0.0,
    "BMI": 12.0,
    "Fruits": 0.0,
    "Veggies": 0.0,
    "HvyAlcoholConsump": 0.0
})

def normalize(res):
    return (res - minVals) / (maxVals-minVals)



# Function to map age to age group
def map_age_to_age_group(age):
    if age >= 0 and age <= 17:
        return 0
    elif age >= 18 and age <= 24:
        return 1
    elif age >= 25 and age <= 29:
        return 2
    elif age >= 30 and age <= 34:
        return 3
    elif age >= 35 and age <= 39:
        return 4
    elif age >= 40 and age <= 44:
        return 5
    elif age >= 45 and age <= 49:
        return 6
    elif age >= 50 and age <= 54:
        return 7
    elif age >= 55 and age <= 59:
        return 8
    elif age >= 60 and age <= 64:
        return 9
    elif age >= 65 and age <= 69:
        return 10
    elif age >= 70 and age <= 74:
        return 11
    elif age >= 75 and age <= 79:
        return 12
    else:
        return 13  # 80 or older

def map_heavy_alcohol_consumption(sex, drinks_per_week):
    if sex == 'Male' and drinks_per_week >= 14:
        return 1  # Heavy alcohol consumption
    elif sex == 'Female' and drinks_per_week >= 7:
        return 1  # Heavy alcohol consumption
    else:
        return 0  # No heavy alcohol consumption

# Function to get user input features
def user_input_features():
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    age = st.sidebar.slider('Age', 0, 100, 50)
    high_chol = st.sidebar.selectbox('High Cholesterol', ('No', 'Yes'))
    chol_check = st.sidebar.selectbox('Cholesterol Check in 5yrs', ('No', 'Yes'))
    bmi = st.sidebar.slider('BMI', 12.0, 98.0, 25.0)
    fruits = st.sidebar.slider('Fruits', 0, 10, 5)
    veggies = st.sidebar.slider('Veggies', 0, 10, 5)
    drinks_per_week = st.sidebar.slider('Drinks Per Week', 0, 30, 0)

    # Convert categorical variables to numerical
    sex_encoded = 1 if sex == 'Male' else 0
    high_chol_encoded = 1 if high_chol == 'Yes' else 0
    chol_check_encoded = 1 if chol_check == 'Yes' else 0
    heavy_alcohol_encoded = map_heavy_alcohol_consumption(sex, drinks_per_week)
    fruits_encoded = 1 if fruits >= 1 else 0
    veggies_encoded = 1 if veggies >= 1 else 0
    
    # Map age to age group
    age_encoded = map_age_to_age_group(age)
    
    # Create feature vector
    X = np.array([[age_encoded, sex_encoded, high_chol_encoded, chol_check_encoded, 
                   bmi, fruits_encoded, veggies_encoded, heavy_alcohol_encoded]])
    
    X = pd.DataFrame(X, columns=minVals.index.tolist())
    
    X = normalize(X)

    return X

def diabetes_page():

    # Display app title
    st.title("Diabetes Prediction")

    # Sidebar header
    st.sidebar.header('User Input Features')

    # Get user input
    input_features = user_input_features()

    # Load the trained model
    model = joblib.load('models/diabetes/DiabetesCLF.pkl')

    # Predict probability of diabetes
    diabetes_probability = model.predict_proba(input_features)[:, 1][0]

    # Display results
    st.write("Diabetes")
    st.progress(diabetes_probability)
    st.write(round(diabetes_probability * 100, 2))
