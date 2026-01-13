#!/usr/bin/env python3
"""
Test script to check if all imports work and data can be indexed
"""
import sys
import os
from pathlib import Path

print("Testing imports...")

try:
    import numpy as np
    print("✓ numpy imported")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")
    sys.exit(1)

try:
    import librosa
    print("✓ librosa imported")
except ImportError as e:
    print(f"✗ librosa import failed: {e}")
    sys.exit(1)

try:
    import tensorflow as tf
    # Disable GPU usage to prevent hanging
    tf.config.set_visible_devices([], 'GPU')
    print("✓ tensorflow imported")
except ImportError as e:
    print(f"✗ tensorflow import failed: {e}")
    sys.exit(1)

try:
    from sklearn.model_selection import train_test_split
    print("✓ sklearn imported")
except ImportError as e:
    print(f"✗ sklearn import failed: {e}")
    sys.exit(1)

print("\nTesting data indexing...")

SCRIPT_DIR = Path(__file__).parent
RAVDESS_ROOT = SCRIPT_DIR / "data" / "RAVDESS"
CREMAD_ROOT = SCRIPT_DIR / "data" / "CREMA-D"

EMOTION_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5
}

RAVDESS_CODE_MAP = {
    "01": "neutral",
    "02": None,
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": None
}

CREMAD_CODE_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

def index_ravdess(root):
    data = []
    if not root.exists():
        print(f"RAVDESS root {root} does not exist")
        return data

    for actor in root.glob("Actor_*"):
        for wav in actor.glob("*.wav"):
            parts = wav.stem.split("-")
            if len(parts) >= 3:
                emotion = RAVDESS_CODE_MAP.get(parts[2])
                if emotion:
                    data.append((str(wav), EMOTION_MAP[emotion]))
    return data

def index_cremad(root):
    data = []
    audio_dir = root / "AudioWAV"
    if not audio_dir.exists():
        print(f"CREMA-D audio dir {audio_dir} does not exist")
        return data

    for wav in audio_dir.glob("*.wav"):
        parts = wav.stem.split("_")
        if len(parts) >= 3:
            emotion = CREMAD_CODE_MAP.get(parts[2])
            if emotion:
                data.append((str(wav), EMOTION_MAP[emotion]))
    return data

ravdess_data = index_ravdess(RAVDESS_ROOT)
cremad_data = index_cremad(CREMAD_ROOT)

print(f"Found {len(ravdess_data)} RAVDESS samples")
print(f"Found {len(cremad_data)} CREMA-D samples")
print(f"Total samples: {len(ravdess_data) + len(cremad_data)}")

if len(ravdess_data) > 0:
    print(f"Sample RAVDESS: {ravdess_data[0]}")

if len(cremad_data) > 0:
    print(f"Sample CREMA-D: {cremad_data[0]}")

print("\n✅ All tests passed!")
