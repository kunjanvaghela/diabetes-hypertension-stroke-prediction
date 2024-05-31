import streamlit as st
from health_prediction_page import health_prediction_page
from hypertension_page import hypertension_page
from stroke import stroke_page
from diabetes import diabetes_page

def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['User All Rounds', 'Hypertension', 'Stroke', 'Diabetes'])

    if page == 'User All Rounds':
        health_prediction_page()
    elif page == 'Hypertension':
        hypertension_page()
    elif page == 'Stroke':
        stroke_page()
    else:
        diabetes_page()

if __name__ == "__main__":
    main()
