# app.py

from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
import shutil
import os

from utils.audio_processing import extract_mfcc
from utils.emotion_labels import EMOTION_MAP

app = FastAPI(title="Speech Emotion Detection API")

# Load model once at startup
MODEL_PATH = "model/global_federated_model_grouped.keras"
model = tf.keras.models.load_model(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "Emotion Detection Backend is running"}

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
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
        os.remove(temp_file)
