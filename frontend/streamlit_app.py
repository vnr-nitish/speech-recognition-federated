import streamlit as st
import requests
import tempfile
import os


# Decide behavior based on environment.
# LOCAL (default): show backend URL input + recording option.
# CLOUD (Streamlit Cloud): hide backend URL input and recording option.
IS_CLOUD = os.getenv("STREAMLIT_CLOUD", "false").lower() == "true"

DEFAULT_LOCAL_BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/predict")
DEFAULT_CLOUD_BACKEND = os.getenv(
    "BACKEND_URL_CLOUD",
    "https://speech-recognition-federated.onrender.com/predict",
)

st.set_page_config(
    page_title="Emotion recognition from speech using multi headed attention and deep learning/federated learning",
    layout="centered",
)

st.title(
    "🎤 Emotion recognition from speech using multi headed attention and "
    "deep learning/federated learning"
)
st.markdown("Provide a speech recording in WAV format to analyze its emotion.")


def warmup_backend(url: str) -> None:
    """Best-effort warmup to reduce cold-start timeouts on Render.

    Called only on Streamlit Cloud. We hit the backend root (GET /)
    with a small timeout, ignoring failures; this is just to wake
    the container before the first /predict call.
    """

    try:
        base_url = url.rsplit("/predict", 1)[0]
        requests.get(base_url or url, timeout=5)
    except Exception:
        # Ignore; regular prediction path will handle real errors.
        pass

if IS_CLOUD:
    # Fixed backend URL in Streamlit Cloud: no editable field.
    backend_url = DEFAULT_CLOUD_BACKEND
    # Try to warm up the backend once when the page loads.
    with st.spinner("Warming up backend (first request may take longer)..."):
        warmup_backend(backend_url)
else:
    # Local development: allow overriding the backend URL.
    backend_url = st.text_input("Backend /predict URL", value=DEFAULT_LOCAL_BACKEND)

# ----------------------
# Input options
# ----------------------

if IS_CLOUD:
    # On Streamlit Cloud we only allow upload.
    mode = "Upload WAV"
else:
    # Locally allow either upload or microphone recording.
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
                        timeout=180,
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
