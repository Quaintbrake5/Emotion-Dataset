import gradio as gr
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.svm import SVC
import joblib
import os
from pathlib import Path

# Disable GPU to prevent issues in Spaces
tf.config.set_visible_devices([], 'GPU')

# Constants
SAMPLE_RATE = 22050
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

# Load models
BASE_DIR = Path(__file__).parent
extractor_path = BASE_DIR / "best_cnn.keras"
svm_path = BASE_DIR / "best_svm.pkl"

extractor = None
svm_model = None

if extractor_path.exists():
    try:
        cnn = load_model(str(extractor_path))
        extractor = tf.keras.Model(cnn.input, cnn.get_layer("embedding").output)
        print("CNN model loaded successfully")
    except Exception as e:
        print(f"Failed to load CNN model: {e}")
else:
    print(f"CNN model not found at {extractor_path}")

if svm_path.exists():
    try:
        svm_model = joblib.load(str(svm_path))
        print("SVM model loaded successfully")
    except Exception as e:
        print(f"Failed to load SVM model: {e}")
else:
    print(f"SVM model not found at {svm_path}")

def generate_spectrogram(signal):
    """Generate spectrogram from audio signal."""
    # Parameters to match model input shape (180, 120)
    n_fft = 1024
    hop_length = 368  # Adjusted to get ~180 time frames for 3s audio
    n_mels = 120

    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=signal, sr=SAMPLE_RATE, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize
    mel_spec_db = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db))

    # Transpose to (time_frames, n_mels)
    mel_spec_db = mel_spec_db.T

    # Ensure exact shape (180, 120)
    if mel_spec_db.shape[0] < 180:
        # Pad time dimension (first dimension)
        pad_width = ((0, 180 - mel_spec_db.shape[0]), (0, 0))
        mel_spec_db = np.pad(mel_spec_db, pad_width, mode='constant')
    else:
        mel_spec_db = mel_spec_db[:180, :]

    return mel_spec_db.astype(np.float32)

def get_embedding(spectrogram):
    """Get prediction from CNN model."""
    if extractor is None:
        raise ValueError("CNN model not loaded.")
    # spectrogram should be in shape (height, width, channels)
    if len(spectrogram.shape) == 2:
        spectrogram = spectrogram[..., np.newaxis]
    spectrogram = spectrogram[np.newaxis, ...]  # Add batch dimension
    prediction = extractor.predict(spectrogram)
    return prediction

def predict_emotion(embedding):
    """Extract emotion probabilities from SVM prediction on CNN embedding."""
    if svm_model is None:
        raise ValueError("SVM model not loaded.")

    # Get probabilities from SVM
    probabilities = svm_model.predict_proba(embedding.reshape(1, -1))[0]

    # Return all emotions with their probabilities
    emotion_probabilities = {}
    for i, emotion in enumerate(EMOTION_LABELS):
        if i < len(probabilities):
            emotion_probabilities[emotion] = float(probabilities[i])
        else:
            emotion_probabilities[emotion] = 0.0

    return emotion_probabilities

def process_audio(audio):
    """Process audio file and return emotion predictions as JSON string."""
    if audio is None:
        return '{"error": "No audio provided"}'

    try:
        # Load audio
        signal, sr = librosa.load(audio, sr=SAMPLE_RATE)

        # Generate spectrogram
        spectrogram = generate_spectrogram(signal)

        # Get embedding
        embedding = get_embedding(spectrogram)

        # Predict emotion
        emotion_probabilities = predict_emotion(embedding)

        # Return as JSON string
        import json
        return json.dumps(emotion_probabilities)

    except Exception as e:
        import json
        return json.dumps({"error": str(e)})

# Create Gradio interface
iface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath", label="Upload Audio File"),
    outputs=gr.Textbox(label="Emotion Prediction"),
    title="Emotion Recognition from Audio",
    description="Upload an audio file to predict the emotion using CNN-SVM hybrid model."
)

if __name__ == "__main__":
    iface.launch(show_error=True)
