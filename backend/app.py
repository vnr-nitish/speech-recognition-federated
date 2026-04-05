# app.py

from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import shutil
import os
from pathlib import Path

from .utils.audio_processing import extract_mfcc
from .utils.emotion_labels import EMOTION_MAP

app = FastAPI(title="Speech Emotion Detection API")

# Resolve paths relative to this file so imports work from any cwd.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "global_federated_model_grouped.keras"
model = None
model_load_error = None

try:
    import tensorflow as tf

    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(str(MODEL_PATH))
    else:
        model_load_error = f"Model file not found at {MODEL_PATH}"
except Exception as exc:
    model_load_error = str(exc)

@app.get("/")
def home():
    return {"message": "Emotion Detection Backend is running"}

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    if model is None:
        detail = "Model unavailable."
        if model_load_error:
            detail = f"Model unavailable: {model_load_error}"
        raise HTTPException(status_code=503, detail=detail)

    temp_file = f"temp_{file.filename}"

    # Save uploaded file temporarily
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Preprocess
        mfcc = extract_mfcc(temp_file)

        # Predict
        predictions = model.predict(mfcc)[0]
        emotion_index = int(np.argmax(predictions))
        emotion = EMOTION_MAP[emotion_index]

        return {
            "predicted_emotion": emotion,
            "confidence": float(predictions[emotion_index]),
            "all_probabilities": {
                EMOTION_MAP[i]: float(predictions[i])
                for i in range(len(predictions))
            }
        }

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
