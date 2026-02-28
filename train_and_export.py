#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

# Load preprocessed data
data = np.load('mitbih_windows.npz')
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']
class_weight = {0: float(data['class_weight_0']), 1: float(data['class_weight_1'])}

print("Class weight:", class_weight)
print("Train shape:", X_train.shape, y_train.shape)
print("Val shape:  ", X_val.shape, y_val.shape)
print("Test shape: ", X_test.shape, y_test.shape)

# BUILD ARCHITECTURE
inp = keras.Input(shape=(X_train.shape[1], 1), name='ecg_input')

# Wider filters for better feature learning
# Ventricular beats have distinctive wide QRS complexes - need capacity to learn this
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

# Bottleneck fix: reduce temporal size and channel count before flattening
x = layers.Conv1D(128, 3, padding='same', name='conv4')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(0.3)(x)


x = layers.Flatten()(x)

# Deeper decision layers
x = layers.Dense(128, activation='relu', name='dense1')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(64, activation='relu', name='dense2')(x)
x = layers.Dropout(0.3)(x)

# Binary output
out = layers.Dense(1, activation='sigmoid', name='output')(x)

model = keras.Model(inp, out, name='improved_ecg_classifier')

# Compile with same metrics
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=[
        keras.metrics.AUC(name='auc'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
    ]
)

model.summary()

total_params = model.count_params()
print(f"Total parameters: {total_params:,}")

# Training
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'models/best.keras',
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        mode='max',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=7,
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

# THRESHOLD SELECTION
print(f"\n4. Selecting optimal threshold...")

val_probs = model.predict(X_val, batch_size=1024, verbose=0).ravel()

if (y_val == 1).sum() > 0:
    prec, rec, th = precision_recall_curve(y_val, val_probs)

    if len(th) > 0:
        f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
        best_idx = np.nanargmax(f1)
        best_th = float(th[best_idx])

        print(f"Optimal threshold: {best_th:.4f}")
        print(f"F1 at threshold: {f1[best_idx]:.4f}")
        print(f"Precision: {prec[:-1][best_idx]:.4f}")
        print(f"Recall: {rec[:-1][best_idx]:.4f}")
    else:
        best_th = 0.5
        print(f"Using default threshold: {best_th}")
else:
    best_th = 0.5
    print(f"Using default threshold: {best_th}")

# TEST SET EVALUATION
print(f"\n5. Final test set evaluation...")

test_probs = model.predict(X_test, batch_size=1024, verbose=0).ravel()
neg_probs = test_probs[y_test == 0]
pos_probs = test_probs[y_test == 1]

print("Negative probs:")
print(f"  mean={neg_probs.mean():.4f} p90={np.quantile(neg_probs, 0.90):.4f} max={neg_probs.max():.4f}")

if len(pos_probs) > 0:
    print("Positive probs:")
    print(f"  mean={pos_probs.mean():.4f} p10={np.quantile(pos_probs, 0.10):.4f} min={pos_probs.min():.4f}")
test_pred = (test_probs >= best_th).astype(int)

if (y_test == 1).sum() > 0:
    test_auc = roc_auc_score(y_test, test_probs)
    print(f"   Test AUC: {test_auc:.4f}")
else:
    print(f"   ⚠️  Test set has no positives - cannot compute AUC")

cm = confusion_matrix(y_test, test_pred)
print(f"\n   Confusion Matrix:")
print(cm)
print(f"\n   Classification Report:")
print(classification_report(y_test, test_pred, digits=4))

# EXPORT MODELS
print(f"\n6. Exporting TFLite models...")

# Float32
conv = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float32 = conv.convert()
with open("models/ecg_float32.tflite", "wb") as f:
    f.write(tflite_float32)
print(f"   ✅ ecg_float32.tflite")

# Dynamic range
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_dr = conv.convert()
with open("models/ecg_dr.tflite", "wb") as f:
    f.write(tflite_dr)
print(f"   ✅ ecg_dr.tflite")


# Full INT8 with stratified representative dataset
def representative_dataset():
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    rng = np.random.default_rng(0)

    # Keep calibration balanced: same number of positives and negatives.
    # Cap the size so we stay in the "small subset" range for TFLite calibration.
    n = min(len(pos_idx), len(neg_idx), 100)

    if n == 0:
        # Fallback: if one class is missing, use up to 100 random training samples.
        all_idx = np.arange(len(y_train))
        chosen = rng.choice(all_idx, size=min(100, len(all_idx)), replace=False)
    else:
        chosen_pos = rng.choice(pos_idx, size=n, replace=False)
        chosen_neg = rng.choice(neg_idx, size=n, replace=False)
        chosen = np.concatenate([chosen_pos, chosen_neg])
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
print(f"   ✅ ecg_int8.tflite")

# Save threshold
with open("threshold.txt", "w") as f:
    f.write(str(best_th))
print(f"   ✅ threshold.txt")

# VALIDATION SET SUMMARY
print(f"\n7. Summary of splits used:")
print(f"   Train:      Records 100,102,103,108,112,113 (weights learned here)")
print(f"   Validation: Record 105 (threshold tuned here, val_auc monitored)")
print(f"   Test:       Record 109 (held-out, evaluated once in step 5)")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"\n✅ Models saved to models/")

