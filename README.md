# Emotion Detection App

A lightweight local application that detects emotions from audio using a pre-trained Keras model. It includes a backend service and a Streamlit frontend for recording/uploading audio, audio preprocessing utilities, and a bundled model for inference.

## Features
- Real-time emotion detection from audio clips
- Streamlit frontend for recording/uploading audio and viewing predictions
- Flask backend serving inference endpoints
- Audio preprocessing utilities and emotion label mapping
- Local model file included for offline use

## Repository Structure
- [backend/app.py](backend/app.py) - Flask backend API for model inference.
- [backend/test_units.py](backend/test_units.py) - unit test examples for backend logic.
- [frontend/streamlit_app.py](frontend/streamlit_app.py) - Streamlit UI for recording/uploading audio and viewing predictions.
- [model/global_federated_model_grouped.keras](model/global_federated_model_grouped.keras) - pretrained Keras model used for inference.
- [utils/audio_processing.py](utils/audio_processing.py) - audio processing helpers.
- [utils/emotion_labels.py](utils/emotion_labels.py) - emotion label definitions.
- `requirements.txt` - top-level Python dependencies.
- `backend/requirements.txt` - backend-specific dependencies.

## Requirements
- Python 3.8+
- Recommended: create and use a virtual environment.
- Install dependencies with `pip install -r requirements.txt` (or the backend-specific file for the backend only).

## Installation
1. Clone repo:
   - `git clone <repo-url>`
   - `cd emotion_detection_app`
2. Create and activate a virtual environment (Windows PowerShell example):
   - `python -m venv .venv`
   - `.venv\Scripts\Activate.ps1`
3. Install dependencies:
   - `pip install -r requirements.txt`
   - If running backend separately: `pip install -r backend/requirements.txt`

## Running Locally
- Start the backend (optional):
  - `cd backend`
  - `python app.py`
  - By default the Flask server runs on `http://127.0.0.1:5000` (see `backend/app.py` for exact settings).
- Start the Streamlit frontend:
  - `streamlit run frontend/streamlit_app.py`
  - Open the provided Streamlit URL in your browser.

Notes:
- The Streamlit app may call the local backend for inference. If you run the frontend and backend separately, ensure the backend URL in `frontend/streamlit_app.py` (or config) matches the Flask server address.
- The app uses the local model file at `model/global_federated_model_grouped.keras` for predictions. Keep it in place unless you update paths in the code.

## Usage
- Use the Streamlit UI to record from microphone or upload an audio file.
- The app preprocesses audio via `utils/audio_processing.py` and maps model outputs to human-readable labels in `utils/emotion_labels.py`.
- Predictions are displayed live on the frontend with confidence scores.

## Model
- Pretrained Keras model: [model/global_federated_model_grouped.keras](model/global_federated_model_grouped.keras).
- If you retrain or replace the model, keep the same input feature shape expected by the preprocessing pipeline in `utils/audio_processing.py`.

## Testing
- Run backend unit tests:
  - `python backend/test_units.py`
  - Or, if using pytest: `pytest backend`
- Add tests as needed to validate preprocessing and inference.

## Configuration
- No external API keys required by default — the model and processing are local.
- If you add environment-based configuration (ports, model path), document it and use environment variables (e.g., `MODEL_PATH`, `FLASK_PORT`).

## Development Notes
- Helpers:
  - `utils/audio_processing.py` — adapt sample rate, windowing, or feature extraction here.
  - `utils/emotion_labels.py` — change or extend label mapping.
- Frontend:
  - `frontend/streamlit_app.py` — customize UI, add user settings, or change inference behavior.
- Backend:
  - `backend/app.py` — inference endpoint; adjust batching or concurrency settings for production.

## Contributing
- Fork the repo and open a pull request with a clear description.
- Add unit tests for new functionality and ensure existing tests pass.
- Update this README if you change the public behavior or interfaces.

## Privacy
- Audio processing and inference are performed locally by default using the bundled model. No audio is uploaded to third-party services unless you explicitly change the app to do so.

## License
- Add a license file (e.g., MIT) if you want open-source distribution. Replace this block with the chosen license details.

## Author
- Name: Vinnakota Nitish Raj
- LinkedIn: https://www.linkedin.com/in/vnr-nitish
