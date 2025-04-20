import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# === Load model and preprocessor === #
model = tf.keras.models.load_model("structured_price_model.h5", compile=False)
preprocessor = joblib.load("structured_preprocessor.pkl")

# === UI === #
st.title("🏠 Melbourne Property Price Predictor (Structured Data Only)")
st.caption("Powered by Neural Networks — Estimate house prices instantly")

# === Property Details Input === #
st.header("Enter Property Information")
rooms = st.slider("Number of rooms", 1, 10, 3)
bathroom = st.slider("Number of bathrooms", 1, 5, 1)
car = st.slider("Car parking spaces", 0, 5, 1)
land = st.number_input("Land size (sq.m)", value=150.0)
build = st.number_input("Building area (sq.m)", value=100.0)
year = st.number_input("Year built", min_value=1800, max_value=2025, value=2000)
dist = st.number_input("Distance to CBD (km)", value=5.0)
lat = st.number_input("Latitude", value=-37.8)
lng = st.number_input("Longitude", value=144.96)
suburb = st.text_input("Suburb", value="Abbotsford")
ptype = st.selectbox("Property type", ['h', 'u', 't'])
region = st.selectbox("Region", [
    'Northern Metropolitan', 'Southern Metropolitan',
    'Eastern Victoria', 'Western Victoria', 'South-Eastern Metropolitan'
])

# === Create input dataframe === #
input_df = pd.DataFrame([{
    'Rooms': rooms,
    'Bathroom': bathroom,
    'Car': car,
    'Landsize': land,
    'BuildingArea': build,
    'YearBuilt': year,
    'Distance': dist,
    'Lattitude': lat,
    'Longtitude': lng,
    'Suburb': suburb,
    'Type': ptype,
    'Regionname': region
}])

# === Transform structured input === #
try:
    structured_input = preprocessor.transform(input_df)
except Exception as e:
    st.error(f"❌ Input processing error: {e}")
    st.stop()

# === Predict === #
if st.button("Predict Price"):
    prediction = model.predict(structured_input)[0][0]
    # Show in dollars with formatting
    st.success(f"💵 Estimated Property Price: **${prediction:,.2f} USD**")
