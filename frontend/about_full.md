# About — Emotion Detection App

Emotion Detection App provides an easy-to-use local interface to analyze emotion in speech. The app combines a Streamlit frontend for capturing or uploading audio with a lightweight backend that runs inference using a bundled Keras model.

Key points:
- Local-first: audio processing and inference run on your machine using the included model. No external audio uploads are required by default.
- Modular: audio preprocessing lives in `utils/audio_processing.py`, and emotion labels are mapped in `utils/emotion_labels.py` so you can swap or extend components easily.
- Extensible: replace the bundled model at `model/global_federated_model_grouped.keras` with your own trained model if you need different labels or higher accuracy.

Use cases:
- Prototyping emotion recognition UIs and demos
- Research experiments that require fast local feedback
- Educational demos and class projects

Author: hp — update the contact details in the repository README if you want a different name or email.
