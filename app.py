import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('regression_model_kfold.pkl')  # Replace this with your valid model file
scaler = joblib.load('scaler.pkl')  # Make sure to include your scaler or preprocess accordingly

# Streamlit app title
st.title("Phone Price Prediction")

# Define input fields for each feature
battery_power = st.number_input("Battery Power (mAh)", min_value=500, max_value=5000, step=10)
blue = st.selectbox("Bluetooth", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
clock_speed = st.number_input("Clock Speed (GHz)", min_value=0.5, max_value=3.0, step=0.1)
dual_sim = st.selectbox("Dual SIM", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
fc = st.number_input("Front Camera (MP)", min_value=0, max_value=30, step=1)
four_g = st.selectbox("4G", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
int_memory = st.number_input("Internal Memory (GB)", min_value=2, max_value=256, step=1)
m_dep = st.number_input("Mobile Depth (cm)", min_value=0.1, max_value=2.0, step=0.1)
mobile_wt = st.number_input("Mobile Weight (g)", min_value=80, max_value=300, step=10)
n_cores = st.number_input("Number of Cores", min_value=1, max_value=8, step=1)
pc = st.number_input("Primary Camera (MP)", min_value=0, max_value=100, step=1)
px_height = st.number_input("Pixel Resolution Height", min_value=100, max_value=2000, step=50)
px_width = st.number_input("Pixel Resolution Width", min_value=100, max_value=2000, step=50)
ram = st.number_input("RAM (MB)", min_value=512, max_value=8192, step=128)
sc_h = st.number_input("Screen Height (cm)", min_value=5, max_value=20, step=1)
sc_w = st.number_input("Screen Width (cm)", min_value=5, max_value=20, step=1)
talk_time = st.number_input("Talk Time (hours)", min_value=2, max_value=30, step=1)
three_g = st.selectbox("3G", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
touch_screen = st.selectbox("Touch Screen", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
wifi = st.selectbox("Wi-Fi", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Create a button for prediction
if st.button("Predict"):
    # Create a list of input values
    input_data = np.array([[battery_power, blue, clock_speed, dual_sim, fc, four_g,
                            int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
                            px_width, ram, sc_h, sc_w, talk_time, three_g,
                            touch_screen, wifi]])

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make a prediction
    prediction = model.predict(input_data_scaled)[0]

    # Define thresholds for classifying into 'low', 'medium', 'high', 'very high'
    if prediction == 0:
        classification = 'Low'
    elif prediction == 1:
        classification = 'Medium'
    elif prediction == 2:
        classification = 'High'
    else:
        classification = 'Very High'

    st.success(f'The predicted phone price range is: {classification}')
