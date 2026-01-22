import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

# Load preprocessed data
data = np.load('mitbih_windows.npz')
X_train, y_train = data['X_train'], data['y_train']
X_val,   y_val   = data['X_val'],   data['y_val']
X_test,  y_test  = data['X_test'],  data['y_test']
fs, win = int(data['fs']), int(data['win'])

# class weights computed during prepare (patient-wise)
cw0 = float(data['class_weight_0']) if 'class_weight_0' in data else 1.0
cw1 = float(data['class_weight_1']) if 'class_weight_1' in data else 1.0
class_weight = {0: cw0, 1: cw1}

print("Shapes:",
      "X_train", X_train.shape, "y_train", y_train.shape,
      "| X_val", X_val.shape, "y_val", y_val.shape,
      "| X_test", X_test.shape, "y_test", y_test.shape)
print("Class weight:", class_weight)

# Build the compact 1D CNN
# Input shape: (time=win, channels=1) e.g., (720, 1)

inp = keras.Input(shape=(X_train.shape[1], 1))

# Local feature detectors (edges, QRS morphology)
x = layers.Conv1D(16, 7, padding='same', activation='relu')(inp)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(2)(x)

# Higher-level patterns
x = layers.Conv1D(32, 5, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(2)(x)

# Fine distinctions
x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)

# Global summary over time
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.25)(x)
x = layers.Dense(32, activation='relu')(x)

# Binary output: ventricular (1) vs normal (0)
out = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inp, out)

# Use metrics that matter for imbalanced detection tasks
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=[keras.metrics.AUC(name='auc'),
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

model.summary()

# -----------------------------
# Training: early stop on val AUC, reduce LR, save best
# -----------------------------
callbacks = [
    keras.callbacks.ModelCheckpoint('best.keras', monitor='val_auc', mode='max', save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=256,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

# -----------------------------
# Choose a decision threshold on the VALIDATION subject
# (0.5 is often suboptimal; we pick the F1-optimal threshold here)
# -----------------------------
val_probs = model.predict(X_val, batch_size=1024, verbose=0).ravel()
prec, rec, th = precision_recall_curve(y_val, val_probs)
f1 = 2 * prec * rec / (prec + rec + 1e-9)
best_idx = np.nanargmax(f1)
# precision_recall_curve returns len(th) = len(prec)-1; align safely:
best_th = float(th[max(0, best_idx-1)]) if len(th) > 0 else 0.5
print(f"Chosen threshold (val F1-opt): {best_th:.3f} | F1={f1[best_idx]:.3f} | P={prec[best_idx]:.3f} | R={rec[best_idx]:.3f}")

# -----------------------------
# Final evaluation on TEST subject (one shot, no tuning here)
# -----------------------------
test_probs = model.predict(X_test, batch_size=1024, verbose=0).ravel()
test_pred  = (test_probs >= best_th).astype(int)
test_auc   = roc_auc_score(y_test, test_probs)
cm         = confusion_matrix(y_test, test_pred)
print("TEST AUC:", f"{test_auc:.4f}")
print("TEST Confusion matrix:\n", cm)
print(classification_report(y_test, test_pred, digits=3))

# -----------------------------
# Export TFLite models
# 1) Float32 baseline
# 2) Dynamic-range quantized (weights int8, activations float)
# 3) Full INT8 with representative dataset (fastest/lowest power on Pi)
# -----------------------------

# 1) Float32
conv = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float32 = conv.convert()
open("ecg_float32.tflite", "wb").write(tflite_float32)
print("Wrote ecg_float32.tflite")

# 2) Dynamic-range quantization
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_dr = conv.convert()
open("ecg_dr.tflite", "wb").write(tflite_dr)
print("Wrote ecg_dr.tflite")

# 3) Full INT8: needs a representative dataset (few hundred train windows)
def representative_dataset():
    rng = np.random.default_rng(0)
    n = len(X_train)
    # Up to 300 samples for calibration is usually enough
    for i in rng.choice(n, size=min(300, n), replace=False):
        # Input must be float32 with shape (1, time, channels)
        yield [X_train[i:i+1].astype(np.float32)]

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = representative_dataset
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tflite_int8 = conv.convert()
open("ecg_int8.tflite", "wb").write(tflite_int8)
print("Wrote ecg_int8.tflite")

# Save threshold for the runtime script on the Pi
with open("threshold.txt", "w") as f:
    f.write(str(best_th))
print("Saved threshold.txt")
