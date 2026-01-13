#!/usr/bin/env python3
"""
Retrain SVM with the new CNN model architecture.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Set environment variables before any TF imports
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Retraining SVM with new CNN model...")

try:
    import numpy as np
    import joblib
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import tensorflow as tf
    from tensorflow.keras import layers, models
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Rebuild the CNN model (same as quick_train.py)
def build_cnn():
    inputs = layers.Input(shape=(120, 120, 1))
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu", name="embedding")(x)  # 64-dim embedding
    outputs = layers.Dense(6, activation="softmax")(x)
    return models.Model(inputs, outputs)

# Load training data (reuse from quick_train.py logic)
SCRIPT_DIR = Path(__file__).parent
RAVDESS_ROOT = SCRIPT_DIR / "data" / "RAVDESS"
CREMAD_ROOT = SCRIPT_DIR / "data" / "CREMA-D"

EMOTION_MAP = {
    'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5
}

RAVDESS_CODE_MAP = {
    '01': 'neutral', '02': None, '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fear', '07': 'disgust', '08': None
}

CREMAD_CODE_MAP = {
    'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear', 'HAP': 'happy',
    'NEU': 'neutral', 'SAD': 'sad'
}

def index_datasets():
    data = []
    # RAVDESS
    for actor in RAVDESS_ROOT.glob("Actor_*"):
        for wav in actor.glob("*.wav"):
            parts = wav.stem.split("-")
            emotion = RAVDESS_CODE_MAP.get(parts[2])
            if emotion:
                data.append((str(wav), EMOTION_MAP[emotion]))
    # CREMA-D
    for wav in (CREMAD_ROOT / "AudioWAV").glob("*.wav"):
        parts = wav.stem.split("_")
        emotion = CREMAD_CODE_MAP.get(parts[2])
        if emotion:
            data.append((str(wav), EMOTION_MAP[emotion]))
    return data

print("Loading datasets...")
data = index_datasets()
print(f"Found {len(data)} audio samples")

# Use subset for faster training
subset_size = min(2000, len(data))
subset_data = data[:subset_size]
print(f"Using {subset_size} samples for training")

# Build and compile CNN
print("Building CNN model...")
cnn = build_cnn()
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Generate training data
print("Generating training data...")
X, y = [], []
for path, label in subset_data:
    try:
        # Simple MFCC extraction (same as quick_train.py)
        import librosa
        signal, _ = librosa.load(path, sr=16000)
        signal = signal[:16000*3]  # 3 seconds
        signal = np.pad(signal, (0, 16000*3 - len(signal)))

        mfcc = librosa.feature.mfcc(y=signal, sr=16000, n_mfcc=40, n_fft=512, hop_length=256)
        mfcc = librosa.util.normalize(mfcc)

        # Pad/truncate to fixed size
        if mfcc.shape[1] < 120:
            mfcc = np.pad(mfcc, ((0, 0), (0, 120 - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :120]

        X.append(mfcc)
        y.append(label)
    except Exception as e:
        print(f"Warning: Failed to process {path}: {e}")
        continue

X = np.array(X)[..., np.newaxis]
y = np.array(y)
print(f"Generated {len(X)} training samples")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train CNN
print("Training CNN...")
cnn.fit(X_train, y_train, epochs=5, batch_size=16, validation_split=0.1, verbose=1)

# Extract features for SVM
print("Extracting features for SVM...")
extractor = models.Model(cnn.input, cnn.get_layer("embedding").output)
X_train_features = extractor.predict(X_train)
X_test_features = extractor.predict(X_test)

print(f"CNN embedding shape: {X_train_features.shape}")

# Train SVM
print("Training SVM...")
svm = SVC(kernel='rbf', probability=True, class_weight='balanced', C=1.0)
svm.fit(X_train_features, y_train)

# Evaluate
print("Evaluating...")
y_pred = svm.predict(X_test_features)
print(classification_report(y_test, y_pred))

# Save models
print("Saving models...")
cnn.save("best_cnn.keras")

# Save SVM using pickle instead of joblib
import pickle
with open("best_svm.pkl", "wb") as f:
    pickle.dump(svm, f)

print("✅ SVM retraining complete!")
print("Models saved as best_cnn.keras and best_svm.pkl")
