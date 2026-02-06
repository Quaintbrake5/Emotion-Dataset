# ===============================================================
# IMPORTS
# ===============================================================
import os
import random
import logging
from pathlib import Path
import numpy as np
import subprocess

import librosa # type: ignore
from tqdm import tqdm # type: ignore

import tensorflow as tf
# Disable GPU usage to prevent hanging
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras import layers, models, regularizers, callbacks # type: ignore

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize

from joblib import Parallel, delayed
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Disable oneDNN for numerical stability and set memory growth
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

# Configure memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ===============================================================
# PROCESSOR DETECTION
# ===============================================================
def detect_processor():
    """
    Detects the CPU model using WMIC command on Windows.
    Returns the processor name or None if detection fails.
    """
    try:
        result = subprocess.run(
            ["wmic", "cpu", "get", "name"],
            capture_output=True,
            text=True,
            shell=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                processor_name = lines[1].strip()
                return processor_name
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
SCRIPT_DIR = Path(__file__).parent
RAVDESS_ROOT = SCRIPT_DIR / "data" / "RAVDESS"
CREMAD_ROOT = SCRIPT_DIR / "data" / "CREMA-D"

SAMPLE_RATE = 16000
MAX_DURATION = 3.0
MAX_LEN = int(SAMPLE_RATE * MAX_DURATION)

CACHE_DIR = Path("cache_mfcc")
CACHE_DIR.mkdir(exist_ok=True)

# ===============================================================
# MFCC SHAPE FIX
# ===============================================================
MAX_MFCC_LEN = 120  # fixed time dimension for MFCCs (adjustable)

# ===============================================================
# EMOTION MAPS
# ===============================================================
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

# ===============================================================
# DATASET INDEXING
# ===============================================================
def index_ravdess(root):
    data = []
    print(f"Indexing RAVDESS from {root}")
    print(f"Root exists: {root.exists()}")
    actors = list(root.glob("Actor_*"))
    print(f"Found {len(actors)} actors")
    for actor in actors:
        print(f"Found actor: {actor}")
        wavs = list(actor.glob("*.wav"))
        print(f"Actor {actor.name} has {len(wavs)} wav files")
        for wav in wavs:
            parts = wav.stem.split("-")
            emotion = RAVDESS_CODE_MAP.get(parts[2])
            if emotion:
                data.append((str(wav.resolve()), EMOTION_MAP[emotion]))
            else:
                print(f"Skipped {wav}: emotion code {parts[2]} not in map")
    print(f"RAVDESS: found {len(data)} valid samples")
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
    """
    Ensures MFCC has fixed time dimension for CNN compatibility
    """
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def extract_mfcc(signal, n_mfcc, n_fft, hop_length):
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=SAMPLE_RATE,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
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
    rng = np.random.default_rng(seed=42)
    return signal + noise_factor * rng.standard_normal(len(signal))

def change_volume(signal, factor=0.5):
    return signal * factor

def speed_perturbation(signal, rate=1.1):
    return librosa.effects.time_stretch(signal, rate=rate)

def augment_audio(signal):
    return [
        signal,
        add_noise(signal),
        librosa.effects.pitch_shift(signal, sr=SAMPLE_RATE, n_steps=2),
        librosa.effects.time_stretch(signal, rate=0.9),
        change_volume(signal, 0.7),
        change_volume(signal, 1.3),
        speed_perturbation(signal, 0.8),
        speed_perturbation(signal, 1.2)
    ]

# ===============================================================
# MFCC CACHING (NO AUGMENTATION)
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

# ===============================================================
# MFCC FEATURE GENERATION (PARALLEL, NO AUG)
# ===============================================================
def generate_mfcc_features(data, n_mfcc, n_fft, hop_length, n_jobs=global_parallel_jobs_ultra):
    def process(item):
        path, label = item
        return cached_mfcc(path, n_mfcc, n_fft, hop_length), label

    results = Parallel(n_jobs=n_jobs)(
        delayed(process)(item) for item in tqdm(data)
    )

    X, y = zip(*results)
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    return X, y

# ===============================================================
# MFCC TUNING (SVM-BASED, BATCHED, MEMORY EFFICIENT)
# ===============================================================

def tune_mfcc(data, n_jobs=2):
    candidates = [
        (30, 512, 256),
        (40, 512, 256),
        (50, 1024, 512),
        (60, 1024, 512)
    ]  # More candidates for better tuning

    best_acc = 0
    best_params = None

    # Use larger subset for better tuning
    subset_size = min(2000, len(data))  # Increased to 2000 for better accuracy
    subset_data = data[:subset_size]

    for n_mfcc, n_fft, hop in candidates:
        print(f"\nTesting MFCC: n_mfcc={n_mfcc}, n_fft={n_fft}, hop={hop}")

        try:
            # Process in smaller batches to avoid memory issues
            batch_size = 200  # Smaller batch size
            x_list, y_list = [], []

            for i in range(0, len(subset_data), batch_size):
                batch = subset_data[i:i+batch_size]
                x_batch, y_batch = generate_mfcc_features(batch, n_mfcc, n_fft, hop, n_jobs=n_jobs)  # Use provided n_jobs
                x_list.append(x_batch)
                y_list.append(y_batch)

            x = np.concatenate(x_list, axis=0)
            y = np.concatenate(y_list, axis=0)

            x_stat = x.mean(axis=(2, 3))
            x_tr, x_va, y_tr, y_va = train_test_split(
                x_stat, y, stratify=y, test_size=0.2, random_state=42
            )

            svm = SVC(C=10, gamma="scale", class_weight="balanced")
            svm.fit(x_tr, y_tr)
            acc = svm.score(x_va, y_va)

            print(f"Validation accuracy: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_params = (n_mfcc, n_fft, hop)

        except Exception as e:
            print(f"Error with MFCC config {n_mfcc},{n_fft},{hop}: {e}")
            continue

    if best_params is None:
        best_params = (40, 512, 256)  # Default fallback

    print("✅ Best MFCC:", best_params)
    return best_params

# ===============================================================
# CNN MODEL
# ===============================================================
def build_cnn(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-5))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(512, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu", name="embedding")(x)
    outputs = layers.Dense(6, activation="softmax")(x)

    return models.Model(inputs, outputs)

# ===============================================================
# AUGMENTED CNN DATA (MEMORY EFFICIENT - STREAMING APPROACH)
# ===============================================================

def generate_augmented_data_streaming(data, n_mfcc, n_fft, hop_length, max_samples=20000):
    """
    Generate augmented data using streaming approach to avoid memory issues.
    Limits total samples to prevent memory overflow.
    """
    X, y = [], []
    target_samples = min(max_samples, len(data) * 8)  # 8 augmentations per sample
    samples_per_original = max(1, target_samples // len(data))

    print(f"Generating up to {target_samples} augmented samples from {len(data)} original samples")

    for path, label in tqdm(data):
        if len(X) >= target_samples:
            break

        try:
            signal = load_audio(path)
            # Limit augmentations per sample
            augmentations = augment_audio(signal)[:samples_per_original]

            for aug_signal in augmentations:
                if len(X) >= target_samples:
                    break
                mfcc = extract_mfcc(aug_signal, n_mfcc, n_fft, hop_length)
                X.append(mfcc)
                y.append(label)
        except Exception as e:
            print(f"Warning: Failed to process {path}: {e}")
            continue

    print(f"After processing, X has {len(X)} samples, y has {len(y)} samples")

    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    print(f"Generated {len(X)} augmented samples")
    return X, y

# ===============================================================
# MAIN
# ===============================================================
def main():
    # Detect processor for optimized training on Intel Core Ultra 5 235
    processor = detect_processor()
    print(f"Detected processor: {processor}")

    if processor and "Intel(R) Core(TM) Ultra 5 235" in processor:
        print("🚀 Intel Core Ultra 5 235 detected! Optimizing TensorFlow and parallel processing for 14 cores/threads...")
        # Adjust TensorFlow settings for Intel Core Ultra 5 235 (14 cores/threads)
        os.environ["TF_NUM_INTEROP_THREADS"] = "14"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "14"
        # Use all 14 threads for parallel processing
        global_parallel_jobs_ultra = 14
        print(f"Set TF_NUM_INTEROP_THREADS to 14, TF_NUM_INTRAOP_THREADS to 14, parallel jobs to {global_parallel_jobs_ultra}")
    else:
        print("Using default settings for current processor (optimized for Intel Core i5-10310U with 4 cores/8 threads)")
        global_parallel_jobs_ultra = 4  # Default for i5-10310U

    print("📂 Indexing datasets...")
    print(f"RAVDESS root: {RAVDESS_ROOT}")
    print(f"CREMAD root: {CREMAD_ROOT}")

    ravdess_data = index_ravdess(RAVDESS_ROOT)
    print(f"Indexed {len(ravdess_data)} RAVDESS samples")
    if ravdess_data:
        print(f"Sample RAVDESS: {ravdess_data[0]}")
    else:
        print("No RAVDESS data found!")

    cremad_data = index_cremad(CREMAD_ROOT)
    print(f"Indexed {len(cremad_data)} CREMAD samples")
    if cremad_data:
        print(f"Sample CREMAD: {cremad_data[0]}")
    else:
        print("No CREMAD data found!")

    data = ravdess_data + cremad_data
    print(f"Total data: {len(data)} samples")
    if not data:
        print("No data found! Exiting.")
        return

    # 1️⃣ MFCC tuning
    best_mfcc = tune_mfcc(data)

    # 2️⃣ CNN training
    x, y = generate_augmented_data_streaming(data, *best_mfcc)
    x_train, x_test, y_tr, y_te = train_test_split(
        x, y, stratify=y, test_size=0.2, random_state=42
    )

    # Compute class weights for balanced training
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    class_weights = dict(enumerate(class_weights))

    cnn = build_cnn(x_train.shape[1:])
    cnn.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    cnn.fit(
        x_train, y_tr,
        validation_split=0.1,
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=5)
        ]
    )

    # 3️⃣ CNN → SVM
    extractor = models.Model(cnn.input, cnn.get_layer("embedding").output)
    x_feat = extractor.predict(x_train)
    svm = SVC(kernel="rbf", probability=True, class_weight="balanced")
    svm.fit(x_feat, y_tr)

    # 4️⃣ Evaluation
    x_te_feat = extractor.predict(x_test)
    y_pred = svm.predict(x_te_feat)
    print(classification_report(y_te, y_pred))

    cnn.save("best_cnn.keras")
    joblib.dump(svm, "best_svm.pkl")

    print("\n✅ ALL DONE")

if __name__ == "__main__":
    main()
