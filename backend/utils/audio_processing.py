# utils/audio_processing.py

import librosa
import numpy as np

# These settings mirror your Colab training pipeline
SAMPLE_RATE = 16000
DURATION_SECONDS = 3.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)  # 48000

# In training you used 40 MFCCs + delta + delta-delta -> 120 features
N_MFCC = 40


def _load_and_normalize_audio(file_path: str) -> np.ndarray:
    """Load audio at 16 kHz, fix length to 3 seconds, normalize amplitude.

    This matches the `preprocess_audio` function from the Colab:
    - librosa.load(..., sr=16000, mono=True)
    - trim or pad to exactly 48000 samples
    - divide by max absolute value so samples are in [-1, 1]
    """

    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    if len(audio) > NUM_SAMPLES:
        audio = audio[:NUM_SAMPLES]
    else:
        audio = np.pad(audio, (0, NUM_SAMPLES - len(audio)))

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    return audio


def extract_mfcc(file_path: str) -> np.ndarray:
    """Extract MFCC + delta + delta-delta to match training features.

    Output shape: (1, 94, 120) — batch of one sample.
    """

    audio = _load_and_normalize_audio(file_path)

    # Base MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)

    # First and second order deltas (same as Colab `extract_mfcc_features`)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Stack along the feature axis -> (3 * N_MFCC, time)
    features = np.vstack([mfcc, delta, delta2])

    # Transpose to (time, features) so final shape is (94, 120)
    features = features.T

    # Add batch dimension -> (1, 94, 120)
    return np.expand_dims(features, axis=0)