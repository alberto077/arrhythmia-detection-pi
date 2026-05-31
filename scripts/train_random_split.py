import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from collections import Counter

from ecg_arrhythmia.model import (
    build_cnn_model,
    compile_binary_classifier,
    export_tflite_models,
)

os.makedirs("models/random_split", exist_ok=True)

data = np.load("mitbih_windows.npz")
X_train = data["X_train"]
y_train = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]
X_test = data["X_test"]
y_test = data["y_test"]

X_all = np.concatenate([X_train, X_val, X_test], axis=0)
y_all = np.concatenate([y_train, y_val, y_test], axis=0)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

X_dev, X_test_rand, y_dev, y_test_rand = train_test_split(
    X_all, y_all,
    test_size=0.20,
    random_state=SEED,
    stratify=y_all
)

X_train_rand, X_val_rand, y_train_rand, y_val_rand = train_test_split(
    X_dev, y_dev,
    test_size=0.125,  
    random_state=SEED,
    stratify=y_dev
)


cnt = Counter(y_train_rand.tolist())
w0 = len(y_train_rand) / (len(cnt) * max(1, cnt[0]))
w1 = len(y_train_rand) / (len(cnt) * max(1, cnt[1]))
class_weight = {0: w0, 1: w1}

np.savez_compressed(
    "models/random_split/random_split_data.npz",
    X_val_rand=X_val_rand,
    y_val_rand=y_val_rand,
    X_test_rand=X_test_rand,
    y_test_rand=y_test_rand
)

model = build_cnn_model((X_train_rand.shape[1], 1), name="random_split_ecg_classifier")
compile_binary_classifier(model)

model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "models/random_split/best.keras",
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
]

history = model.fit(
    X_train_rand, y_train_rand,
    validation_data=(X_val_rand, y_val_rand),
    epochs=50,
    batch_size=256,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

model.save("models/random_split/best_trained.keras")

export_tflite_models(
    model,
    X_train_rand,
    y_train_rand,
    "models/random_split/ecg_float32.tflite",
    "models/random_split/ecg_int8.tflite",
)

print("Saved models/random_split")
