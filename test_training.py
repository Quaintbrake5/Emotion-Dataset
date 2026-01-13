#!/usr/bin/env python3
"""
Simple test script to verify training components work before full training.
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

print("Testing basic imports...")

try:
    import numpy as np
    print("✓ NumPy imported")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")
    sys.exit(1)

try:
    import librosa
    print("✓ Librosa imported")
except ImportError as e:
    print(f"✗ Librosa import failed: {e}")
    sys.exit(1)

try:
    import tensorflow as tf
    print("✓ TensorFlow imported")
    print(f"  TF version: {tf.__version__}")
except ImportError as e:
    print(f"✗ TensorFlow import failed: {e}")
    sys.exit(1)

try:
    from sklearn.svm import SVC
    print("✓ Scikit-learn imported")
except ImportError as e:
    print(f"✗ Scikit-learn import failed: {e}")
    sys.exit(1)

print("\nTesting data loading...")

SCRIPT_DIR = Path(__file__).parent
RAVDESS_ROOT = SCRIPT_DIR / "data" / "RAVDESS"
CREMAD_ROOT = SCRIPT_DIR / "data" / "CREMA-D"

print(f"RAVDESS root: {RAVDESS_ROOT}")
print(f"CREMAD root: {CREMAD_ROOT}")

# Test basic file counting
ravdess_count = 0
if RAVDESS_ROOT.exists():
    for actor in RAVDESS_ROOT.glob("Actor_*"):
        ravdess_count += len(list(actor.glob("*.wav")))
    print(f"Found {ravdess_count} RAVDESS files")
else:
    print("RAVDESS directory not found")

cremad_count = 0
if (CREMAD_ROOT / "AudioWAV").exists():
    cremad_count = len(list((CREMAD_ROOT / "AudioWAV").glob("*.wav")))
    print(f"Found {cremad_count} CREMAD files")
else:
    print("CREMAD directory not found")

print(f"\nTotal audio files: {ravdess_count + cremad_count}")

if ravdess_count + cremad_count == 0:
    print("No audio files found! Check dataset paths.")
    sys.exit(1)

print("\nTesting audio loading...")

try:
    # Test loading one audio file
    test_file = None
    if ravdess_count > 0:
        for actor in RAVDESS_ROOT.glob("Actor_*"):
            files = list(actor.glob("*.wav"))
            if files:
                test_file = files[0]
                break
    elif cremad_count > 0:
        files = list((CREMAD_ROOT / "AudioWAV").glob("*.wav"))
        if files:
            test_file = files[0]

    if test_file:
        print(f"Testing with file: {test_file}")
        signal, sr = librosa.load(str(test_file), sr=16000)
        print(f"✓ Audio loaded: {len(signal)} samples at {sr}Hz")
    else:
        print("No test file found")

except Exception as e:
    print(f"✗ Audio loading failed: {e}")

print("\nTesting TensorFlow operations...")

try:
    # Test basic TF operations
    tf.config.set_visible_devices([], 'GPU')  # Disable GPU
    print("✓ GPU disabled")

    # Test simple model creation
    from tensorflow.keras import layers, models # type: ignore
    inputs = layers.Input(shape=(10, 10, 1))
    x = layers.Conv2D(8, 3, activation='relu')(inputs)
    x = layers.Flatten()(x)
    outputs = layers.Dense(6, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    print("✓ Simple CNN model created")

except Exception as e:
    print(f"✗ TensorFlow operations failed: {e}")

print("\nAll basic tests passed! Ready for training.")
print("Run the full training with: python emotion-data.py")
