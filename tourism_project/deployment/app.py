import streamlit as st
import pandas as pd
import joblib
import os

# Define the path to the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved.")
        return None

model = load_model()

st.title("Tourism Package Purchase Prediction")
st.write("Predict whether a customer will purchase the Wellness Tourism Package.")

if model:
    st.header("Customer Information")

    # Input features (based on numerical and categorical features from train.py)
    age = st.slider("Age", 18, 90, 30)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.slider("Duration of Pitch (minutes)", 5, 60, 15)
    number_of_person_visiting = st.slider("Number of Persons Visiting", 1, 10, 2)
    number_of_followups = st.slider("Number of Follow-ups", 0, 10, 3)
    preferred_property_star = st.slider("Preferred Property Star", 1, 5, 3)
    number_of_trips = st.slider("NumberOfTrips (annually)", 0, 50, 5)
    passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    number_of_children_visiting = st.slider("Number of Children Visiting", 0, 5, 0)
    monthly_income = st.number_input("Monthly Income", 0.0, 100000.0, 30000.0, step=1000.0)

    typeof_contact = st.selectbox("Type of Contact", ['Company Invited', 'Self Inquiry'])
    occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer', 'Government Sector'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    product_pitched = st.selectbox("Product Pitched", ['Domestic', 'International', 'Adventure', 'Resorts', 'Cruise', 'Luxury'])
    marital_status = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced', 'Unmarried'])
    designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP', 'Director', 'CEO'])

    # Create a DataFrame from inputs
    input_data = pd.DataFrame({
        'Unnamed: 0': [0], # Dummy column to match expected features if present in trained model
        'Age': [age],
        'CityTier': [city_tier],
        'DurationOfPitch': [duration_of_pitch],
        'NumberOfPersonVisiting': [number_of_person_visiting],
        'NumberOfFollowups': [number_of_followups],
        'PreferredPropertyStar': [preferred_property_star],
        'NumberOfTrips': [number_of_trips],
        'Passport': [passport],
        'PitchSatisfactionScore': [pitch_satisfaction_score],
        'OwnCar': [own_car],
        'NumberOfChildrenVisiting': [number_of_children_visiting],
        'MonthlyIncome': [monthly_income],
        'TypeofContact': [typeof_contact],
        'Occupation': [occupation],
        'Gender': [gender],
        'ProductPitched': [product_pitched],
        'MaritalStatus': [marital_status],
        'Designation': [designation]
    })

    if st.button("Predict Purchase"):
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[:, 1][0]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.success(f"The customer is likely to purchase the package (Probability: {prediction_proba:.2f})")
        else:
            st.info(f"The customer is not likely to purchase the package (Probability: {prediction_proba:.2f})")
else:
    st.warning("Model could not be loaded. Please ensure the model training pipeline has run successfully.")
