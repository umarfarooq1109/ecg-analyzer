import cv2
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
from scipy.signal import find_peaks
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=r"C:\Users\Admin\OneDrive\Desktop\ecg\model\ECG_CNN_MODEL_reduced.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels and descriptions
class_labels = ['MI', 'HB', 'PMI', 'NORMAL']
descriptions = {
    'MI': 'Myocardial Infarction detected — ST elevation and abnormal Q waves observed.',
    'HB': 'Heartbeat irregularity detected — signs of arrhythmia or conduction abnormality.',
    'PMI': 'Past Myocardial Infarction — historical evidence of infarction remains.',
    'NORMAL': 'Normal sinus rhythm — no major abnormalities detected.'
}

# Function to extract 1D signal from ECG lead image
def extract_1d_from_lead(lead_img):
    gray = np.array(lead_img.convert('L'))
    h, w = gray.shape
    cropped = gray[int(h*0.3):int(h*0.7), int(w*0.05):int(w*0.95)]
    resized = cv2.resize(cropped, (500, 100))
    signal = np.mean(resized, axis=0)
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    return signal

# Estimate BPM using peaks in the signal
def estimate_bpm(signal, min_bpm=40, max_bpm=180):
    smoothed = np.convolve(signal, np.ones(5)/5, mode='same')
    height_thresh = 0.6 * (np.max(smoothed) - np.min(smoothed)) + np.min(smoothed)
    min_rr_interval = 60 / max_bpm
    min_distance = int(min_rr_interval * 300)
    peaks, _ = find_peaks(smoothed, height=height_thresh, distance=min_distance)
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / 300.0
        mean_rr = np.mean(rr_intervals)
        bpm = int(60 / mean_rr)
        if bpm < min_bpm or bpm > max_bpm:
            bpm = 0
    else:
        bpm = 0
    return bpm, peaks

# Main function to analyze ECG image
def analyze_ecg_adaptive(img_pil, patient_name="Name", record_no="0"):
    try:
        # Preprocess image for model
        img_resized = img_pil.resize((224, 224))
        img_array = keras_image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        # Run TFLite model prediction
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])

        class_idx = np.argmax(pred)
        condition = class_labels[class_idx]
        description = descriptions[condition]

        # Signal extraction and visualization
        img_np = np.array(img_pil.convert('RGB'))
        h, w, _ = img_np.shape
        lead_height = h // 12
        is_12lead = h > 800

        bpm_list = []

        if is_12lead:
            fig, axs = plt.subplots(6, 2, figsize=(12, 8))
            axs = axs.flatten()
            for i in range(12):
                lead_img = Image.fromarray(img_np[i*lead_height:(i+1)*lead_height])
                signal = extract_1d_from_lead(lead_img)
                bpm, peaks = estimate_bpm(signal)
                bpm_list.append(bpm)
                axs[i].plot(signal)
                axs[i].scatter(peaks, signal[peaks], color='red')
                axs[i].set_title(f"Lead {i+1} - BPM: {bpm}")
                axs[i].set_ylim([0, 1])
                axs[i].axis('off')
        else:
            signal = extract_1d_from_lead(img_pil)
            bpm, peaks = estimate_bpm(signal)
            bpm_list.append(bpm)
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(signal)
            ax.scatter(peaks, signal[peaks], color='red')
            ax.set_title(f"Single Lead - BPM: {bpm}")
            ax.set_ylim([0, 1])
            ax.axis('off')

        plt.tight_layout()
        plt.savefig("12lead_plot.png")
        plt.close()

        avg_bpm = int(np.mean([b for b in bpm_list if b > 0])) if bpm_list else 0

        report = (
            f"👤 Name: {patient_name}\n"
            f"🏷️ Record No: {record_no}\n"
            f"❤️ Condition: {condition}\n"
            f"📈 Estimated BPM: {avg_bpm}\n"
            f"📋 Description: {description}"
        )

        return report, "12lead_plot.png"

    except Exception as e:
        return f"❌ Error: {str(e)}", None

# Gradio Interface
gr.Interface(
    fn=analyze_ecg_adaptive,
    inputs=[
        gr.Image(type="pil", label="Upload ECG Image"),
        gr.Textbox(label="Patient Name", value="Name"),
        gr.Textbox(label="Record Number", value="0")
    ],
    outputs=[
        gr.Textbox(label="ECG Report"),
        gr.Image(label="ECG Visualization")
    ],
    title="ECG Analyzer (Single or 12-Lead)"
).launch()
