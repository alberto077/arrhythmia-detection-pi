import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data = np.load('mitbih_windows.npz')
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']
class_weight = {0: data['w0'], 1: data['w1']}

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


inp = keras.Input(shape=(X_train.shape[1], 1), name='ecg_input')

x = layers.Conv1D(64, 7, padding='same', name='conv1')(inp)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv1D(128, 5, padding='same', name='conv2')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv1D(256, 3, padding='same', name='conv3')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(0.3)(x)

x = layers.Conv1D(128, 3, padding='same', name='conv4')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(0.3)(x)

x = layers.Flatten()(x)

x = layers.Dense(128, activation='relu', name='dense1')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(64, activation='relu', name='dense2')(x)
x = layers.Dropout(0.3)(x)


out = layers.Dense(1, activation= 'sigmoid', name='output')(x)

model = keras.Model(inp, out, name='improved_ecg_classifier')

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=[keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')]
)

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




# Float32
conv = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float32 = conv.convert()
with open("models/ecg_float32.tflite", "wb") as f:
    f.write(tflite_float32)
print("Saved ecg_float32.tflite")


# Full INT8 with stratified representative dataset
def representative_dataset():
    rng = np.random.default_rng(0)

    samples_per_class = 100
    chosen = []
    
    for cls in range(2):
        cls_idx = np.where(y_train == cls)[0]
        n = min(samples_per_class, len(cls_idx))
        chosen.append(rng.choice(cls_idx, size=n, replace=False))
    
    chosen = np.concatenate(chosen)
    rng.shuffle(chosen)
    
    for i in chosen:
        yield [X_train[i:i+1].astype(np.float32)]


conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = representative_dataset
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tflite_int8 = conv.convert()
with open("models/ecg_int8.tflite", "wb") as f:
    f.write(tflite_int8)

