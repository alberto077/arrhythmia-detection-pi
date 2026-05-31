import numpy as np
import random
import tensorflow as tf
from tensorflow import keras

from ecg_arrhythmia.model import (
    build_cnn_model,
    compile_binary_classifier,
    export_tflite_models,
)

data = np.load('mitbih_windows.npz')
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
class_weight = {0: data['w0'], 1: data['w1']}

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


model = build_cnn_model((X_train.shape[1], 1), name='improved_ecg_classifier')
compile_binary_classifier(model)

model.summary()

# Training
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'models/best.keras',
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=256,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)
export_tflite_models(
    model,
    X_train,
    y_train,
    "models/ecg_float32.tflite",
    "models/ecg_int8.tflite",
)
print("Saved ecg_float32.tflite")
print("Saved ecg_int8.tflite")
