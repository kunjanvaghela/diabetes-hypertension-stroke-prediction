import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import cv2

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
    
maxVals = pd.Series({
    "Age" : 13.0,
    "Sex" : 1.0,
    "Smoker" : 1.0,
    "HeartDiseaseorAttack" : 1.0,
    "PhysActivity" : 1.0,
    "GenHlth" : 5.0,
    "MentHlth" : 30.0,
    "PhysHlth" : 30.0,
    "DiffWalk" : 1.0,
})

minVals = pd.Series({
    "Age" : 1.0,
    "Sex" : 0.0,
    "Smoker" : 0.0,
    "HeartDiseaseorAttack" : 0.0,
    "PhysActivity" : 0.0,
    "GenHlth" : 1.0,
    "MentHlth" : 0.0,
    "PhysHlth" : 0.0,
    "DiffWalk" : 0.0,
})

def normalize(X):
    return (X - minVals) / (maxVals - minVals)

def health_prediction_page():
    st.title('Common Health Prediction')

    st.sidebar.write("Please enter the following information:")

    # User inputs
    age = st.sidebar.slider("Age", min_value=0, max_value=120, value=30, step=1)
    sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    smoker = st.sidebar.selectbox("Smoker", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    heart_disease = st.sidebar.selectbox("Heart Disease or Attack", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    physical_activity = st.sidebar.selectbox("Physical Activity", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    general_health = st.sidebar.slider("General Health", min_value=1, max_value=5, value=3, step=1)
    mental_health = st.sidebar.slider("Mental Health", min_value=0, max_value=30, value=15, step=1)
    physical_health = st.sidebar.slider("Physical Health", min_value=0, max_value=30, value=15, step=1)
    diff_walk = st.sidebar.selectbox("Difficulty Walking", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

    # Create feature vector
    age = map_age_to_age_group(age)
    X = np.array([[age, sex, smoker, heart_disease, physical_activity, general_health, mental_health, physical_health, diff_walk]])
    
    X = pd.DataFrame(X, columns=minVals.index.tolist())
    
    X = normalize(X)

    # Fit logistic regression model
    model = joblib.load("models/common/DiabetesCLF.pkl")

    # Predict probability of diabetes
    diabetesProbability = model.predict_proba(X)[:, 1][0]

    st.write("Diabetes")
    st.progress(diabetesProbability)
    st.write(round(diabetesProbability * 100, 2))
    # st.text("0                                        50                                      100")
    
    # Fit logistic regression model
    model = joblib.load("models/common/StrokeCLF.pkl")

    # Predict probability of stroke
    strokeProbability = model.predict_proba(X)[:, 1][0]

    st.write("Stroke")
    st.progress(strokeProbability)
    st.write(round(strokeProbability * 100, 2))
    # st.text("0                                        50                                      100")
    
    # Fit logistic regression model
    model = joblib.load("models/common/HighBPCLF.pkl")

    # Predict probability of HighBP
    HighBPProbability = model.predict_proba(X)[:, 1][0]

    st.write("HighBP")
    st.progress(HighBPProbability)
    st.write(round(HighBPProbability * 100, 2))
    # st.text("0                                        50                                      100")
    
    cols = st.columns(3)
    folder = "Images"
    
    i = 0
    for subFolder in os.listdir(folder):
        with cols[i]:
            subFolderPath = os.path.join(folder, subFolder)
            for image in os.listdir(subFolderPath):
                imagePath = os.path.join(subFolderPath, image)
                
                H, W = cv2.imread(imagePath).shape[:2]
                expectedWidth = int(W / H * 200)
                st.image(imagePath, width=expectedWidth, use_column_width=False)
        i += 1
