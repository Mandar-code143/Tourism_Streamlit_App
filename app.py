import streamlit as st
import pandas as pd
import numpy as np
import joblib

# LOAD DATASET
df = pd.read_csv("final_tourism_dataset.csv")

# LOAD MODELS
classifier = joblib.load("visit_mode_model.pkl")
regressor = joblib.load("rating_prediction_model.pkl")

# TITLE
st.title("Tourism Experience Analytics System")

st.write("Machine Learning Based Tourism Prediction App")

# SHOW DATASET
st.subheader("Dataset Preview")
st.dataframe(df.head())

# USER INPUTS
visit_year = st.number_input("Visit Year", 2020, 2030, 2025)

visit_month = st.slider("Visit Month", 1, 12, 6)

user_id = st.number_input("User ID", 1, 100000, 14)

attraction_id = st.number_input("Attraction ID", 1, 100000, 100)

rating = st.slider("Rating", 1.0, 5.0, 3.0)

# PREDICT BUTTON
if st.button("Predict"):

    # CREATE SAMPLE INPUT
    sample_input = np.zeros((1, classifier.n_features_in_))

    # PUT VALUES
    sample_input[0][0] = user_id
    sample_input[0][1] = attraction_id
    sample_input[0][2] = visit_year
    sample_input[0][3] = visit_month
    sample_input[0][4] = rating

    # CLASSIFICATION
    visit_mode_prediction = classifier.predict(sample_input)

    # REGRESSION
    rating_prediction = regressor.predict(sample_input)

    # OUTPUT
    st.subheader("Prediction Results")

    st.write("Predicted Visit Mode:", visit_mode_prediction[0])

    st.write("Predicted Rating:", rating_prediction[0])