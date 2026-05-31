import numpy as np
import time
from sklearn.metrics import classification_report, confusion_matrix

from ecg_arrhythmia.inference import ECGDetector
from ecg_arrhythmia.metrics import best_f1_threshold


def run_experiment(model_path, X, y, static_th):
    detector = ECGDetector(model_path)

    all_times = []
    all_actuals = []
    all_preds = []

    for i in range(len(X)):
        start = time.time()
        prob = detector.predict(X[i, :, 0])
        elapsed = (time.time() - start) * 1000

        pred = 1 if prob >= static_th else 0

        all_times.append(elapsed)
        all_actuals.append(int(y[i]))
        all_preds.append(pred)

    print(f"Model: {model_path}")
    print(f"Threshold: static ({static_th:.4f})")
    print(classification_report(
        all_actuals, all_preds,
        target_names=["Normal", "Arrhythmia"],
        digits=4
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(all_actuals, all_preds))
    print(f"Avg latency: {np.mean(all_times):.2f} ms/window")


if __name__ == "__main__":
    data = np.load("models/random_split/random_split_data.npz")
    X_val = data["X_val_rand"]
    y_val = data["y_val_rand"]
    X_test = data["X_test_rand"]
    y_test = data["y_test_rand"]


    f32_detector = ECGDetector("models/random_split/ecg_float32.tflite")
    f32_probs = np.array([f32_detector.predict(X_val[i, :, 0]) for i in range(len(X_val))])
    f32_th = best_f1_threshold(y_val, f32_probs)
    print(f"Float32 threshold: {f32_th:.4f}")
    np.save("models/random_split/static_threshold_float32.npy", np.array(f32_th, dtype=np.float32))


    int8_detector = ECGDetector("models/random_split/ecg_int8.tflite")
    int8_probs = np.array([int8_detector.predict(X_val[i, :, 0]) for i in range(len(X_val))])
    int8_th = best_f1_threshold(y_val, int8_probs)
    print(f"INT8 threshold: {int8_th:.4f}")
    np.save("models/random_split/static_threshold_int8.npy", np.array(int8_th, dtype=np.float32))

    print("\nEXP 1: Float32 + Static")
    run_experiment("models/random_split/ecg_float32.tflite", X_test, y_test, f32_th)

    print("\nEXP 2: INT8 + Static")
    run_experiment("models/random_split/ecg_int8.tflite", X_test, y_test, int8_th)    
