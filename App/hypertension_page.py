import streamlit as st
import pandas as pd
import numpy as np
import joblib

maxVals = pd.Series({
	"age": 98.0,
	"sex": 1.0,
	"cp": 3.0,
	"trestbps": 200.0,
	"chol": 564.0,
	"fbs": 1.0,
	"restecg": 2.0,
	"thalach": 202.0,
	"exang": 1.0,
	"oldpeak": 6.2,
	"slope": 2.0,
	"ca": 4.0,
	"thal": 3.0,
})

minVals = pd.Series({
	"age" : 11.0,
	"sex" : 0.0,
	"cp" : 0.0,
	"trestbps" : 94.0,
	"chol" : 126.0,
	"fbs" : 0.0,
	"restecg" : 0.0,
	"thalach" : 71.0,
	"exang" : 0.0,
	"oldpeak" : 0.0,
	"slope" : 0.0,
	"ca" : 0.0,
	"thal" : 0.0,
})

def normalize(X):
    return (X - minVals) / (maxVals - minVals)

def hypertension_page():
    st.title('Hypertension')

    st.sidebar.write("Please enter the following information:")

    # Display progress value
    st.sidebar.write("Please adjust the input values below:")

    # User inputs in sidebar
    age = st.sidebar.slider("Age", min_value=0, max_value=120, value=30, step=1)
    sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    cp_options = {0: 'Asymptomatic', 1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-anginal Pain'}
    cp = st.sidebar.selectbox("Chest Pain Type", options=cp_options, format_func=lambda x: cp_options[x])
    trestbps = st.sidebar.slider("Resting Blood Pressure", min_value=90, max_value=200, value=120, step=1)
    chol = st.sidebar.slider("Serum Cholesterol", min_value=100, max_value=600, value=200, step=1)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    restecg = st.sidebar.slider("Resting Electrocardiographic Results", min_value=0, max_value=2, value=0, step=1)
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150, step=1)
    exang = st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=6.2, value=0.0, step=0.1)
    slope_options = {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}
    slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", options=slope_options, format_func=lambda x: slope_options[x])
    ca = st.sidebar.slider("Number of Major Vessels (0-3) Colored by Flourosopy", min_value=0, max_value=3, value=0, step=1)
    thal_options = {3: 'Normal', 6: 'Fixed defect', 7: 'Reversable defect'}
    thal = st.sidebar.selectbox("Thalassemia", options=thal_options, format_func=lambda x: thal_options[x])

    # Create feature vector
    X = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    X = pd.DataFrame(X, columns=minVals.index.tolist())
    
    X = normalize(X)

    # Fit logistic regression model
    model = joblib.load("models/hypertension/HyperTensionCLF.pkl")

    # Predict probability of Hypertension
    hypertensionProbability = model.predict_proba(X)[:, 1][0]

    st.write("Hypertension")
    st.progress(hypertensionProbability)
    st.write(round(hypertensionProbability * 100, 2))
    # st.text("0                                        50                                      100")
