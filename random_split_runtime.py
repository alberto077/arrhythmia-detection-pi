import numpy as np
import time
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        raise ImportError("No lightweight TFLite runtime found")


class ECGDetector:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_scale, self.input_zero_point = self.input_details[0]["quantization"]
        self.output_scale, self.output_zero_point = self.output_details[0]["quantization"]

    def predict(self, window_1d):
        w = np.asarray(window_1d, dtype=np.float32)
        w = (w - w.mean()) / (w.std() + 1e-8)
        x = w[None, :, None].astype(np.float32)

        if self.input_details[0]["dtype"] == np.int8:
            x = x / self.input_scale + self.input_zero_point
            x = np.clip(np.round(x), -128, 127).astype(np.int8)

        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()

        y = self.interpreter.get_tensor(self.output_details[0]["index"])
        if self.output_details[0]["dtype"] == np.int8:
            y = (y.astype(np.float32) - self.output_zero_point) * self.output_scale

        return float(y.reshape(-1)[0])


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
    prec, rec, th = precision_recall_curve(y_val, f32_probs)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    
    f32_th = float(th[np.nanargmax(f1)])
    print(f"Float32 threshold: {f32_th:.4f}")
    np.save("models/random_split/static_threshold_float32.npy", np.array(f32_th, dtype=np.float32))


    int8_detector = ECGDetector("models/random_split/ecg_int8.tflite")
    int8_probs = np.array([int8_detector.predict(X_val[i, :, 0]) for i in range(len(X_val))])
    prec, rec, th = precision_recall_curve(y_val, int8_probs)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    int8_th = float(th[np.nanargmax(f1)])
    print(f"INT8 threshold: {int8_th:.4f}")
    np.save("models/random_split/static_threshold_int8.npy", np.array(int8_th, dtype=np.float32))

    print("\nEXP 1: Float32 + Static")
    run_experiment("models/random_split/ecg_float32.tflite", X_test, y_test, f32_th)

    print("\nEXP 2: INT8 + Static")
    run_experiment("models/random_split/ecg_int8.tflite", X_test, y_test, int8_th)    