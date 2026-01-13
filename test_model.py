#!/usr/bin/env python3
"""
Test script for the emotion recognition model.
Loads the trained CNN and SVM models and tests them on sample audio files.
"""

import os
import sys
from pathlib import Path
import random

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Set environment variables before any TF imports
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Testing emotion recognition model...")

try:
    import numpy as np
    import librosa
    import tensorflow as tf
    import joblib
    from sklearn.metrics import classification_report
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Disable GPU usage
tf.config.set_visible_devices([], 'GPU')

# Configuration
SCRIPT_DIR = Path(__file__).parent
SAMPLE_RATE = 16000
MAX_DURATION = 3.0
MAX_LEN = int(SAMPLE_RATE * MAX_DURATION)
MAX_MFCC_LEN = 120

EMOTION_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad"
}

# MFCC parameters (matching the trained model)
N_MFCC = 50  # 50 * 3 (mfcc + delta + delta2) = 150 features
N_FFT = 512
HOP_LENGTH = 256

def load_audio(path):
    """Load and preprocess audio file."""
    signal, _ = librosa.load(path, sr=SAMPLE_RATE)
    signal, _ = librosa.effects.trim(signal)

    if len(signal) > MAX_LEN:
        signal = signal[:MAX_LEN]
    else:
        signal = np.pad(signal, (0, MAX_LEN - len(signal)))

    return signal.astype(np.float32)

def extract_mfcc(signal):
    """Extract MFCC features from audio signal."""
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    stacked = np.vstack([mfcc, delta, delta2])

    # Normalize
    stacked = librosa.util.normalize(stacked)

    # Fix length
    if stacked.shape[1] < MAX_MFCC_LEN:
        pad_width = MAX_MFCC_LEN - stacked.shape[1]
        stacked = np.pad(stacked, ((0, 0), (0, pad_width)), mode="constant")
    else:
        stacked = stacked[:, :MAX_MFCC_LEN]

    return stacked.astype(np.float32)

def build_cnn():
    """Rebuild the CNN model architecture (same as training)."""
    inputs = tf.keras.layers.Input(shape=(150, 120, 1))

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-5))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="embedding")(x)  # 128-dim embedding to match trained SVM
    outputs = tf.keras.layers.Dense(6, activation="softmax")(x)
    return tf.keras.models.Model(inputs, outputs)

def load_models():
    """Load the trained CNN and SVM models."""
    script_dir = Path(__file__).parent

    # Try to load CNN model, fallback to rebuilding if incompatible
    cnn = None
    try:
        cnn_path = script_dir / "best_cnn.keras"
        cnn = tf.keras.models.load_model(str(cnn_path), safe_mode=False)
        print("✓ CNN model loaded from file")
    except Exception as e:
        print(f"⚠ Failed to load CNN model from file: {str(e)[:100]}...")
        print("   Rebuilding model architecture...")
        try:
            cnn = build_cnn()
            # Try to load weights if they exist
            try:
                cnn.load_weights(str(cnn_path.with_suffix('.weights.h5')))
                print("✓ CNN weights loaded")
            except:
                print("⚠ Could not load weights, using untrained model")
        except Exception as e2:
            print(f"✗ Failed to rebuild CNN model: {e2}")
            return None, None

    try:
        svm_path = script_dir / "best_svm.pkl"
        svm = joblib.load(str(svm_path))
        print("✓ SVM model loaded")
    except Exception as e:
        print(f"✗ Failed to load SVM model: {e}")
        return None, None

    # Create feature extractor from CNN
    extractor = tf.keras.Model(cnn.input, cnn.get_layer("embedding").output)

    return extractor, svm

def get_test_files(num_samples=5):
    """Get random test audio files from the dataset."""
    ravdess_root = SCRIPT_DIR / "data" / "RAVDESS"
    cremad_root = SCRIPT_DIR / "data" / "CREMA-D" / "AudioWAV"

    test_files = []

    # Get RAVDESS files
    if ravdess_root.exists():
        ravdess_files = []
        for actor in ravdess_root.glob("Actor_*"):
            ravdess_files.extend(list(actor.glob("*.wav")))
        test_files.extend(ravdess_files)

    # Get CREMA-D files
    if cremad_root.exists():
        cremad_files = list(cremad_root.glob("*.wav"))
        test_files.extend(cremad_files)

    # Randomly select files
    if len(test_files) > num_samples:
        test_files = random.sample(test_files, num_samples)

    return test_files

def predict_emotion(audio_path, extractor, svm):
    """Predict emotion for a single audio file."""
    try:
        # Load and process audio
        signal = load_audio(audio_path)
        mfcc = extract_mfcc(signal)

        # Add channel dimension for CNN
        mfcc = mfcc[np.newaxis, ..., np.newaxis]

        # Extract features
        features = extractor.predict(mfcc, verbose=0)

        # Predict with SVM
        prediction = svm.predict(features)[0]
        probabilities = svm.predict_proba(features)[0]

        emotion = EMOTION_MAP[prediction]
        confidence = probabilities[prediction]

        return emotion, confidence, probabilities

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, None, None

def main():
    print("\n" + "="*50)
    print("EMOTION RECOGNITION MODEL TEST")
    print("="*50)

    # Load models
    extractor, svm = load_models()
    if extractor is None or svm is None:
        print("Cannot proceed without models.")
        return

    # Get test files
    test_files = get_test_files(10)  # Test on 10 random files
    if not test_files:
        print("No test files found!")
        return

    print(f"\nTesting on {len(test_files)} audio files:")
    print("-" * 50)

    results = []
    for i, audio_path in enumerate(test_files, 1):
        print(f"\n{i}. Testing: {audio_path.name}")

        emotion, confidence, probabilities = predict_emotion(audio_path, extractor, svm)

        if emotion:
            print(f"   Predicted: {emotion.upper()}")
            print(".2f")

            # Show top 3 probabilities
            prob_pairs = list(zip(EMOTION_MAP.values(), probabilities))
            prob_pairs.sort(key=lambda x: x[1], reverse=True)

            print("   Top predictions:")
            for j, (emo, prob) in enumerate(prob_pairs[:3]):
                marker = "→" if j == 0 else "  "
                print(".2f")
        else:
            print("   Failed to predict")

    print("\n" + "="*50)
    print("TEST COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main()
