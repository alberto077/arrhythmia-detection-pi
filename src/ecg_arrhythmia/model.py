import numpy as np


def build_cnn_model(input_shape, name="improved_ecg_classifier"):
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input(shape=input_shape, name="ecg_input")

    x = layers.Conv1D(64, 7, padding="same", name="conv1")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, 5, padding="same", name="conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(256, 3, padding="same", name="conv3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(128, 3, padding="same", name="conv4")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu", name="dense2")(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    return keras.Model(inp, out, name=name)


def representative_dataset(X_train, y_train, samples_per_class=100, seed=0):
    rng = np.random.default_rng(seed)
    chosen = []

    for cls in range(2):
        cls_idx = np.where(y_train == cls)[0]
        n = min(samples_per_class, len(cls_idx))
        chosen.append(rng.choice(cls_idx, size=n, replace=False))

    chosen = np.concatenate(chosen)
    rng.shuffle(chosen)

    for i in chosen:
        yield [X_train[i:i + 1].astype(np.float32)]


def compile_binary_classifier(model, learning_rate=1e-3):
    from tensorflow import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def export_tflite_models(model, X_train, y_train, float32_path, int8_path):
    import tensorflow as tf

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float32 = conv.convert()
    with open(float32_path, "wb") as f:
        f.write(tflite_float32)

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: representative_dataset(X_train, y_train)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tflite_int8 = conv.convert()
    with open(int8_path, "wb") as f:
        f.write(tflite_int8)
