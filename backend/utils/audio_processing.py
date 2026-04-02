# utils/audio_processing.py

import librosa
import numpy as np

SAMPLE_RATE = 22050

# MUST match training
N_MFCC = 120
MAX_LEN = 94

def extract_mfcc(file_path):
    # Load audio
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc = mfcc.T  # (time, features)

    # Pad or truncate time dimension
    if mfcc.shape[0] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN, :]

    # Ensure feature dimension matches 120
    if mfcc.shape[1] != N_MFCC:
        mfcc = mfcc[:, :N_MFCC]

    # Add batch dimension
    return np.expand_dims(mfcc, axis=0)