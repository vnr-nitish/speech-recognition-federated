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

backend_url = st.text_input("Backend /predict URL", value=BACKEND_URL)

# ----------------------
# Input options
# ----------------------

mode = st.radio("Input source", ["Upload WAV", "Record from microphone"], horizontal=True)
audio_bytes = None
audio_name = "audio.wav"

if mode == "Upload WAV":
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        audio_name = uploaded_file.name
else:
    if hasattr(st, "audio_input"):
        recorded_file = st.audio_input("Record audio")
        if recorded_file is not None:
            audio_bytes = recorded_file.read()
            audio_name = "recorded_audio.wav"
    else:
        st.info("Your Streamlit version does not support `st.audio_input`. Please upgrade Streamlit or use Upload WAV.")

if audio_bytes is not None:
    if st.button("Predict Emotion", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        with st.spinner("Analyzing audio..."):
            try:
                with open(temp_path, "rb") as f:
                    response = requests.post(
                        backend_url,
                        files={"file": (audio_name, f, "audio/wav")},
                        timeout=60,
                    )

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Predicted Emotion: {result['predicted_emotion']}")
                    st.write("Confidence:", round(result["confidence"], 4))
                    st.bar_chart(result["all_probabilities"])
                else:
                    detail = response.text
                    try:
                        detail_json = response.json()
                        detail = detail_json.get("detail", detail)
                    except Exception:
                        pass
                    st.error(f"Prediction failed (HTTP {response.status_code}): {detail}")
            except requests.exceptions.RequestException as exc:
                st.error(
                    "Could not connect to backend. "
                    "Start FastAPI first and check the URL above. "
                    f"Error: {exc}"
                )
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
