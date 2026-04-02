import streamlit as st
import requests
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os

BACKEND_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Speech Emotion Detection", layout="centered")

st.title("🎤 Speech Emotion Detection")
st.markdown("Record your voice or upload a WAV file to detect emotion.")

# ----------------------
# Upload Option
# ----------------------

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    with open(temp_path, "rb") as f:
        response = requests.post(
            BACKEND_URL,
            files={"file": f}
        )

    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Emotion: {result['predicted_emotion']}")
        st.write("Confidence:", round(result["confidence"], 4))
        st.bar_chart(result["all_probabilities"])
    else:
        st.error("Prediction failed.")

    os.remove(temp_path)

# ----------------------
# Microphone Option
# ----------------------

st.subheader("🎙 Record from Microphone")

duration = st.slider("Recording Duration (seconds)", 2, 10, 5)

if st.button("Start Recording"):
    st.info("Recording...")
    recording = sd.rec(int(duration * 22050), samplerate=22050, channels=1)
    sd.wait()
    st.success("Recording complete!")

    temp_file = "temp_record.wav"
    sf.write(temp_file, recording, 22050)

    with open(temp_file, "rb") as f:
        response = requests.post(
            BACKEND_URL,
            files={"file": f}
        )

    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Emotion: {result['predicted_emotion']}")
        st.write("Confidence:", round(result["confidence"], 4))
        st.bar_chart(result["all_probabilities"])
    else:
        st.error("Prediction failed.")

    os.remove(temp_file)
