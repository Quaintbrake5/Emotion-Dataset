# ===============================================================
# IMPORTS
# ===============================================================
import os
import random
import logging
from pathlib import Path
import numpy as np # type: ignore
import subprocess

import librosa # type: ignore
from tqdm import tqdm # type: ignore

import tensorflow as tf # type: ignore
# Disable GPU usage to prevent hanging (CPU priority)
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras import layers, models, regularizers, callbacks # type: ignore

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from joblib import Parallel, delayed # type: ignore
import joblib # type: ignore

# Disable oneDNN for numerical stability and set memory growth
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

# ===============================================================
# PROCESSOR DETECTION
# ===============================================================
def detect_processor():
    try:
        result = subprocess.run(
            ["wmic", "cpu", "get", "name"],
            capture_output=True, text=True, shell=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                return lines[1].strip()
    except Exception as e:
        print(f"Error detecting processor: {e}")
    return None

# ===============================================================
# LOGGING
# ===============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===============================================================
# CONFIG
# ===============================================================
global_parallel_jobs_ultra = 4  # default parallel jobs
SCRIPT_DIR = Path(__file__).parent
RAVDESS_ROOT = SCRIPT_DIR / "data" / "RAVDESS"
CREMAD_ROOT = SCRIPT_DIR / "data" / "CREMA-D"

SAMPLE_RATE = 16000
MAX_DURATION = 3.0
MAX_LEN = int(SAMPLE_RATE * MAX_DURATION)

CACHE_DIR = Path("cache_mfcc")
CACHE_DIR.mkdir(exist_ok=True)

# MFCC SHAPE
MAX_MFCC_LEN = 120 

# ===============================================================
# FIXED EMOTION MAPS (SOLVING IMBALANCE)
# ===============================================================
EMOTION_MAP = {
    "angry": 0, "disgust": 1, "fear": 2, 
    "happy": 3, "neutral": 4, "sad": 5
}

RAVDESS_CODE_MAP = {
    "01": "neutral",
    "02": "neutral",  # FIX: Merged Calm into Neutral
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "happy"     # FIX: Mapped Surprise to Happy (High Arousal)
}

CREMAD_CODE_MAP = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fear", 
    "HAP": "happy", "NEU": "neutral", "SAD": "sad"
}

# ===============================================================
# DATASET INDEXING
# ===============================================================
def index_ravdess(root):
    data = []
    print(f"Indexing RAVDESS from {root}")
    actors = list(root.glob("Actor_*"))
    for actor in actors:
        wavs = list(actor.glob("*.wav"))
        for wav in wavs:
            parts = wav.stem.split("-")
            emotion = RAVDESS_CODE_MAP.get(parts[2])
            if emotion:
                data.append((str(wav.resolve()), EMOTION_MAP[emotion]))
    return data

def index_cremad(root):
    data = []
    for wav in (root / "AudioWAV").glob("*.wav"):
        parts = wav.stem.split("_")
        emotion = CREMAD_CODE_MAP.get(parts[2])
        if emotion:
            data.append((str(wav.resolve()), EMOTION_MAP[emotion]))
    return data

# ===============================================================
# AUDIO PROCESSING
# ===============================================================
def load_audio(path):
    signal, _ = librosa.load(path, sr=SAMPLE_RATE)
    signal, _ = librosa.effects.trim(signal)
    if len(signal) > MAX_LEN:
        signal = signal[:MAX_LEN]
    else:
        signal = np.pad(signal, (0, MAX_LEN - len(signal)))
    return signal.astype(np.float32)

def fix_mfcc_length(mfcc, max_len=MAX_MFCC_LEN):
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def extract_mfcc(signal, n_mfcc, n_fft, hop_length):
    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    stacked = np.vstack([mfcc, delta, delta2])
    stacked = librosa.util.normalize(stacked)
    stacked = fix_mfcc_length(stacked)
    return stacked.astype(np.float32)

# ===============================================================
# AUGMENTATION
# ===============================================================
def add_noise(signal, noise_factor=0.01):
    rng = np.random.default_rng()
    return signal + noise_factor * rng.standard_normal(len(signal))

def change_volume(signal, factor=0.5):
    return signal * factor

def speed_perturbation(signal, rate=1.1):
    return librosa.effects.time_stretch(signal, rate=rate)

def augment_audio(signal):
    # Reduced list for efficiency, but kept high variety
    return [
        signal,
        add_noise(signal),
        librosa.effects.pitch_shift(signal, sr=SAMPLE_RATE, n_steps=2),
        librosa.effects.time_stretch(signal, rate=0.9),
        change_volume(signal, 0.7),
        speed_perturbation(signal, 1.2)
    ]

# ===============================================================
# MFCC CACHING & GENERATION
# ===============================================================
def cached_mfcc(path, n_mfcc, n_fft, hop_length):
    dataset = "ravdess" if "Actor" in path else "cremad"
    cache_path = CACHE_DIR / dataset
    cache_path.mkdir(exist_ok=True)
    
    fname = f"{Path(path).stem}_{n_mfcc}_{n_fft}_{hop_length}.npy"
    fpath = cache_path / fname
    
    if fpath.exists():
        return np.load(fpath)
    
    signal = load_audio(path)
    mfcc = extract_mfcc(signal, n_mfcc, n_fft, hop_length)
    np.save(fpath, mfcc)
    return mfcc

# ---------------------------------------------------------------
# NEW: SPLIT-SAFE GENERATORS
# ---------------------------------------------------------------
def process_train_data(file_list, n_mfcc, n_fft, hop_length, max_samples=25000):
    """Augments ONLY training data. Checks limits to prevent OOM."""
    X, y = [], []
    print(f"🚀 Processing Training Data: {len(file_list)} files...")
    
    # Calculate augmentation budget
    aug_budget = max(1, max_samples // len(file_list))
    
    for path, label in tqdm(file_list):
        if len(X) >= max_samples: break
        try:
            signal = load_audio(path)
            # Generate augmentations
            augs = augment_audio(signal)
            # Limit per file to fit budget
            for aug_sig in augs[:aug_budget]:
                if len(X) >= max_samples: break
                mfcc = extract_mfcc(aug_sig, n_mfcc, n_fft, hop_length)
                X.append(mfcc)
                y.append(label)
        except Exception as e:
            continue
            
    return np.array(X)[..., np.newaxis], np.array(y)

def process_test_data(file_list, n_mfcc, n_fft, hop_length):
    """No augmentation for test data."""
    print(f"🔍 Processing Test Data: {len(file_list)} files...")
    X, y = [], []
    for path, label in tqdm(file_list):
        try:
            # We can use cache for test data since it's unmodified
            mfcc = cached_mfcc(path, n_mfcc, n_fft, hop_length)
            X.append(mfcc)
            y.append(label)
        except Exception as e:
            continue
    return np.array(X)[..., np.newaxis], np.array(y)

# ===============================================================
# MFCC TUNING (IMPROVED)
# ===============================================================
def tune_mfcc(data, n_jobs=4):
    print("🎛️ Tuning MFCC parameters...")
    
    # 1. Use a larger, stratified subset (2500 samples)
    paths, labels = zip(*data)
    try:
        # Stratify to ensure we tune on ALL emotions, not just the majority class
        subset_paths, _, subset_labels, _ = train_test_split(
            paths, labels, train_size=2500, stratify=labels, random_state=42
        )
        subset_data = list(zip(subset_paths, subset_labels))
    except ValueError:
        # Fallback if dataset is smaller than 2500
        subset_data = data 

    candidates = [
        (40, 1024, 512),  # Standard speech setting
        (60, 2048, 512),  # Higher resolution (good for subtle emotions)
        (30, 512, 256),   # Low latency / compact
    ]

    best_acc = 0
    best_params = (40, 1024, 512) # Safe default

    for n_mfcc, n_fft, hop in candidates:
        print(f"   Testing: MFCC={n_mfcc}, FFT={n_fft}, HOP={hop}")
        
        try:
            # Generate features for the subset (Reuse your feature generation logic)
            X_sub, y_sub = [], []
            # FIX: Iterate over entire subset, not just first 500
            for path, label in tqdm(subset_data, desc="Tuning subset", leave=False): 
                 sig = load_audio(path)
                 mfcc = extract_mfcc(sig, n_mfcc, n_fft, hop)
                 X_sub.append(mfcc.mean(axis=1)) # Flatten to 1D for quick SVM check
                 y_sub.append(label)
            
            # Quick SVM check
            clf = SVC(kernel='linear', class_weight='balanced')
            # Simple split for validation
            X_tr, X_val, y_tr, y_val = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)
            clf.fit(X_tr, y_tr)
            acc = clf.score(X_val, y_val)
            
            print(f"   -> Accuracy: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_params = (n_mfcc, n_fft, hop)
                
        except Exception as e:
            print(f"   Skipping config due to error: {e}")
            continue

    print(f"✅ Best Configuration: {best_params}")
    return best_params

# ===============================================================
# CNN MODEL
# ===============================================================
def build_cnn(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=regularizers.l2(1e-5))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(256, 3, padding="same", activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu", name="embedding")(x)
    outputs = layers.Dense(6, activation="softmax")(x)
    
    return models.Model(inputs, outputs)

# ===============================================================
# MAIN
# ===============================================================
def main():
    processor = detect_processor()
    print(f"Detected processor: {processor}")

    if processor and "Intel(R) Core(TM) Ultra 5 235" in processor:
        print("🚀 Intel Core Ultra 5 235 detected! Optimizing...")
        os.environ["TF_NUM_INTEROP_THREADS"] = "14"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "14"
        global_parallel_jobs_ultra = 14
    else:
        global_parallel_jobs_ultra = 4

    # 1. Index Datasets
    print("📂 Indexing datasets...")
    ravdess_data = index_ravdess(RAVDESS_ROOT)
    cremad_data = index_cremad(CREMAD_ROOT)
    all_data = ravdess_data + cremad_data
    
    if not all_data:
        print("❌ No data found! Check paths.")
        return

    print(f"Total files: {len(all_data)}")

    # 2. MFCC Parameters (TUNING ENABLED)
    best_mfcc = tune_mfcc(all_data)
    print(f"Using Best MFCC params: {best_mfcc}")

    # 3. SPLIT FIRST (CRITICAL FIX FOR DATA LEAKAGE)
    print("✂️ splitting data...")
    paths, labels = zip(*all_data)
    train_paths, test_paths, y_train_orig, y_test_orig = train_test_split(
        paths, labels, stratify=labels, test_size=0.2, random_state=42
    )

    # Reassemble tuples
    train_files = list(zip(train_paths, y_train_orig))
    test_files = list(zip(test_paths, y_test_orig))

    # 4. Generate Features
    # Augment ONLY train
    x_train, y_train = process_train_data(train_files, *best_mfcc)
    # Clean Test
    x_test, y_test = process_test_data(test_files, *best_mfcc)

    print(f"Training Shape: {x_train.shape}")
    print(f"Testing Shape: {x_test.shape}")

    # 5. CNN Training
    # Recalculate weights on the augmented y_train
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    
    cnn = build_cnn(x_train.shape[1:])
    cnn.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("🏋️ Training CNN...")
    cnn.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=60, # Reduced slightly as we have better data now
        batch_size=32,
        class_weight=class_weights,
        callbacks=[
            callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=4)
        ]
    )

    # 6. Hybrid SVM Training
    print("🤖 Training SVM Head...")
    extractor = models.Model(cnn.input, cnn.get_layer("embedding").output)
    
    # Extract features from CNN
    x_train_feat = extractor.predict(x_train, batch_size=32)
    x_test_feat = extractor.predict(x_test, batch_size=32)
    
    svm = SVC(kernel="rbf", probability=True, class_weight="balanced")
    svm.fit(x_train_feat, y_train)

    # 7. Evaluation
    print("\n📊 Final Evaluation (SVM on Unseen Test Data):")
    y_pred = svm.predict(x_test_feat)
    print(classification_report(y_test, y_pred, target_names=list(EMOTION_MAP.keys())))

    # 8. Save
    cnn.save("best_cnn.keras")
    joblib.dump(svm, "best_svmpkl")
    print("\n✅ ALL DONE")

if __name__ == "__main__":
    main()