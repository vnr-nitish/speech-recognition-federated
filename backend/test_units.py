import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings

from utils.audio_processing import extract_mfcc
import tensorflow as tf

# Step 1: Extract MFCC
mfcc = extract_mfcc("03-01-05-02-01-01-23.wav")
print("MFCC Shape:", mfcc.shape)

# Step 2: Load Model
MODEL_PATH = "model/global_federated_model_grouped.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Step 3: Predict
pred = model.predict(mfcc, verbose=0)

print("Prediction Shape:", pred.shape)
print("Prediction Values:", pred)