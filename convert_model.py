import streamlit as st
import numpy as np
import tensorflow as tf
import time

# Load TFLite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model/ECG_CNN_MODEL_reduced.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Predict function
def predict_ecg(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.array(input_data, dtype=np.float32).reshape(1, -1, 1)  # adjust based on your model's input

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Streamlit UI
st.set_page_config(page_title="ECG Analyzer", layout="centered")
st.title("🫀 ECG Signal Analyzer")

st.markdown("Upload a preprocessed ECG signal (as CSV or input manually) to classify it.")

option = st.radio("Choose input method:", ["Upload CSV", "Enter Manually"])

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your ECG file", type=["csv"])
    if uploaded_file:
        try:
            ecg_data = np.loadtxt(uploaded_file, delimiter=",")
            if ecg_data.ndim == 1:
                ecg_data = ecg_data[:187]  # use first 187 points (adjust to your model)
            st.line_chart(ecg_data)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            ecg_data = None
    else:
        ecg_data = None
else:
    ecg_input = st.text_area("Enter comma-separated ECG values (e.g., 0.1, 0.2, ...)")
    try:
        ecg_data = np.array([float(x) for x in ecg_input.strip().split(",")])
        if ecg_data.ndim == 1:
            ecg_data = ecg_data[:187]  # ensure consistent input size
        st.line_chart(ecg_data)
    except:
        ecg_data = None
        st.warning("Please enter valid numeric ECG values.")

# Predict button
if ecg_data is not None and len(ecg_data) >= 100:
    if st.button("🔍 Analyze ECG"):
        st.info("Analyzing ECG signal...")
        with st.spinner("Running inference..."):
            model = load_model()
            prediction = predict_ecg(model, ecg_data)
            pred_class = np.argmax(prediction)

            st.success(f"Prediction: **Class {pred_class}**")
            st.json({"Raw Output": prediction.tolist()})
else:
    st.info("Waiting for valid ECG input...")

st.caption("Made by Umar Farooq 🚀 | Model: ECG_CNN_MODEL_reduced.tflite")
