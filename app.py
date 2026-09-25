import streamlit as st
import pandas as pd
import joblib



@st.cache_resource
def load_model():
    return joblib.load("models/best_model.joblib")


pipeline = load_model()

st.title("Flight Price Predictor")

st.write("Enter the flight information to estimate its price.")



col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox(
        "Airline",
        ["Air India", "Indigo", "SpiceJet", "Vistara", "GO FIRST"]
    )

with col2:
    flight = st.text_input(
        "Flight",
        placeholder="Example: UK-706"
    )


col1, col2 = st.columns(2)

with col1:
    source_city = st.selectbox(
        "Source City",
        ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"]
    )

with col2:
    destination_city = st.selectbox(
        "Destination City",
        ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"]
    )


col1, col2 = st.columns(2)

with col1:
    departure_time = st.selectbox(
        "Departure Time",
        [
            "Early_Morning",
            "Morning",
            "Afternoon",
            "Evening",
            "Night",
            "Late_Night"
        ]
    )

with col2:
    arrival_time = st.selectbox(
        "Arrival Time",
        [
            "Early_Morning",
            "Morning",
            "Afternoon",
            "Evening",
            "Night",
            "Late_Night"
        ]
    )


col1, col2 = st.columns(2)

with col1:
    stops = st.selectbox(
        "Stops",
        ["zero", "one", "two_or_more"]
    )

with col2:
    flight_class = st.selectbox(
        "Class",
        ["Economy", "Business"]
    )


col1, col2 = st.columns(2)

with col1:
    duration = st.number_input(
        "Duration (hours)",
        min_value=0.0,
        max_value=50.0,
        value=2.5
    )

with col2:
    days_left = st.number_input(
        "Days Left",
        min_value=1,
        max_value=50,
        value=15
    )


st.divider()



if st.button("Predict Price", type="primary"):

    input_data = pd.DataFrame([{
        "airline": airline,
        "flight": flight,
        "source_city": source_city,
        "departure_time": departure_time,
        "stops": stops,
        "arrival_time": arrival_time,
        "destination_city": destination_city,
        "class": flight_class,
        "duration": duration,
        "days_left": days_left
    }])

    prediction = pipeline.predict(input_data)

    st.success(f"Estimated price: {prediction[0]:,.2f}")