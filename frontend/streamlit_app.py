import streamlit as st
import requests
import tempfile
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/predict")

st.set_page_config(
    page_title="Emotion recognition from speech using multi headed attention and deep learning/federated learning",
    layout="centered",
)

st.title(
    "🎤 Emotion recognition from speech using multi headed attention and "
    "deep learning/federated learning"
)
st.markdown("Provide a speech recording in WAV format to analyze its emotion.")

# ----------------------
# Upload Option
# ----------------------

uploaded_file = st.file_uploader("", type=["wav"])

if uploaded_file is not None:
    if st.button("Predict Emotion"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        with st.spinner("Analyzing audio..."):
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
            st.error("Prediction failed. Please try again later.")

        os.remove(temp_path)
