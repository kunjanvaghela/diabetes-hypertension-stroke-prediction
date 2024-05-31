import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Function to map work type to numerical values
def work_type_mapper(work_type):
    if work_type == 'Never_worked':
        return 0
    elif work_type == 'Children':
        return 1
    elif work_type == 'Govt_job':
        return 2
    elif work_type == 'Self-employed':
        return 3
    else:
        return 4

maxVals = pd.Series({
    "sex": 1.00,
    "age": 103.00,
    "hypertension": 1.00,
    "heart_disease": 1.00,
    "ever_married": 1.00,
    "work_type": 4.00,
    "Residence_type": 1.00,
    "avg_glucose_level": 271.74,
    "bmi": 92.00,
    "smoking_status": 1.00
})

minVals = pd.Series({
    "sex": 0.00,
    "age": -9.00,
    "hypertension": 0.00,
    "heart_disease": 0.00,
    "ever_married": 0.00,
    "work_type": 0.00,
    "Residence_type": 0.00,
    "avg_glucose_level": 55.12,
    "bmi": 11.50,
    "smoking_status": 0.00
})

def normalize(res):
    return (res - minVals) / (maxVals-minVals)

# Function to get user input features
def user_input_features():
    sex = st.sidebar.selectbox('Sex',('Male','Female'))
    age = st.sidebar.slider('Age', 0, 100, 50)
    hypertension = st.sidebar.selectbox('Hypertension',('No','Yes'))
    heart_disease = st.sidebar.selectbox('Heart Disease',('No','Yes'))
    ever_married = st.sidebar.selectbox('Ever Married',('No','Yes'))
    work_type = st.sidebar.selectbox('Work Type',('Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked'))
    residence_type = st.sidebar.selectbox('Residence Type',('Urban','Rural'))
    avg_glucose_level = st.sidebar.slider('Average Glucose Level', 55.0, 272.0, 150.0)
    bmi = st.sidebar.slider('BMI', 11.0, 92.0, 25.0)
    smoking_status = st.sidebar.selectbox('Smoking Status',('Smokes', 'Never Smoked'))
    
    # Convert categorical variables to numerical
    sex_encoded = 1 if sex == 'Male' else 0
    hypertension_encoded = 1 if hypertension == 'Yes' else 0
    heart_disease_encoded = 1 if heart_disease == 'Yes' else 0
    ever_married_encoded = 1 if ever_married == 'Yes' else 0
    residence_type_encoded = 1 if residence_type == 'Urban' else 0
    smoking_status_encoded = 1 if smoking_status == 'Smokes' else 0
    
    # Map work type to numerical value
    work_type_encoded = work_type_mapper(work_type)
    
    # Create feature vector
    X = np.array([[sex_encoded, age, hypertension_encoded, heart_disease_encoded, 
                   ever_married_encoded, work_type_encoded, residence_type_encoded, 
                   avg_glucose_level, bmi, smoking_status_encoded]])
    

    X = pd.DataFrame(X, columns=minVals.index.tolist())

    X = normalize(X)

    return X

def stroke_page():

    # Display app title
    st.title("Stroke Prediction")

    # Sidebar header
    st.sidebar.header('User Input Features')

    # Get user input
    input_features = user_input_features()

    # Load the trained model
    model = joblib.load('models/stroke/StrokeCLF.pkl')

    # Predict probability of stroke
    strokeProbability = model.predict_proba(input_features)[:, 1][0]

    # Display results
    st.write("Stroke")
    st.progress(strokeProbability)
    st.write(round(strokeProbability * 100, 2))
