import numpy as np
import time
from dynamic_threshold import DynamicThreshold
import ai_edge_litert.interpreter as tflite
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve




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


def run_experiment(model_path, X, y, rids, records, threshold_mode, static_th):
    detector = ECGDetector(model_path)

    all_times = []
    all_actuals = []
    all_preds = []

    for record in records:
        mask = rids == record
        X_rec = X[mask]
        y_rec = y[mask]

        if len(y_rec) == 0:
            continue

        if threshold_mode == "adaptive":
            dyn = DynamicThreshold(
                global_threshold=static_th,
                buffer_size=150,
                margin=0.15,
                floor=0.40,
                ceil=0.95,
                warmup_length=50,
                update_freq=10,
                admit_ceil=0.85
            )

        for i in range(len(X_rec)):
            start = time.time()
            prob = detector.predict(X_rec[i, :, 0])
            elapsed = (time.time() - start) * 1000

            if threshold_mode == "static":
                pred = 1 if prob >= static_th else 0
            else:
                result = dyn.update(prob)
                pred = 1 if result["is_alert"] else 0

            all_times.append(elapsed)
            actual = int(y_rec[i])
            all_actuals.append(actual)
            all_preds.append(pred)  
   
   
    report = classification_report(
    all_actuals, all_preds,
    target_names=["Normal", "Arrhythmia"],
    digits=4,
    output_dict=True
)

    cm = confusion_matrix(all_actuals, all_preds)

    prec = report["Arrhythmia"]["precision"]
    rec = report["Arrhythmia"]["recall"]
    f1 = report["Arrhythmia"]["f1-score"]

    print(f"Model: {model_path}")
    print(f"Threshold mode: {threshold_mode}")
    print(classification_report(
        all_actuals, all_preds,
        target_names=["Normal", "Arrhythmia"],
        digits=4
        ))
    
    print("Confusion Matrix:")
    print(cm)
    print(f"Avg latency: {np.mean(all_times):.2f} ms/window")

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "avg_latency_ms": np.mean(all_times),
        "confusion_matrix": cm,
    }


if __name__ == "__main__":


    data = np.load("mitbih_windows.npz")
    X_test = data["X_test"]
    y_test = data["y_test"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    rids_test = data["rids_test"]

    data2 = np.load("incart_windows.npz")
    X_all_incart = data2["X_all"]
    y_all_incart = data2["y_all"]
    rids_all_incart = data2["rids_all"]
     

    DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
            210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
    INCART_RECORDS = [f'I{i:02d}' for i in range(1, 76)]


   
    f32_detector = ECGDetector("models/ecg_float32.tflite")
    f32_probs = np.array([f32_detector.predict(X_val[i, :, 0]) for i in range(len(X_val))])
    prec, rec, th = precision_recall_curve(y_val, f32_probs)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    f32_th = float(th[np.nanargmax(f1)])
    print(f"Float32 threshold: {f32_th:.4f}")

    int8_detector = ECGDetector("models/ecg_int8.tflite")
    int8_probs = np.array([int8_detector.predict(X_val[i, :, 0]) for i in range(len(X_val))])
    prec, rec, th = precision_recall_curve(y_val, int8_probs)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    int8_th = float(th[np.nanargmax(f1)])
    print(f"INT8 threshold: {int8_th:.4f}")

    print("\nEXP 3: Float32 + Static")
    run_experiment("models/ecg_float32.tflite", X_test, y_test, rids_test, DS2, "static", f32_th)

    print("\nEXP 4: INT8 + Static")
    run_experiment("models/ecg_int8.tflite", X_test, y_test, rids_test, DS2, "static", int8_th)

    print("\nEXP 5: Float32 + Adaptive")
    run_experiment("models/ecg_float32.tflite", X_test, y_test, rids_test, DS2, "adaptive", f32_th)

    print("\nEXP 6: INT8 + Adaptive")
    run_experiment("models/ecg_int8.tflite", X_test, y_test, rids_test, DS2, "adaptive", int8_th)

    print("\nEXP 7: Float32 + Static (INCART)")
    run_experiment("models/ecg_float32.tflite", X_all_incart, y_all_incart, rids_all_incart, INCART_RECORDS, "static", f32_th)

    print("\nEXP 8: INT8 + Static (INCART)")
    run_experiment("models/ecg_int8.tflite", X_all_incart, y_all_incart, rids_all_incart, INCART_RECORDS, "static", int8_th)

    print("\nEXP 9: Float32 + Adaptive (INCART)")
    run_experiment("models/ecg_float32.tflite", X_all_incart, y_all_incart, rids_all_incart, INCART_RECORDS, "adaptive", f32_th)

    print("\nEXP 10: INT8 + Adaptive (INCART)")
    run_experiment("models/ecg_int8.tflite", X_all_incart, y_all_incart, rids_all_incart, INCART_RECORDS, "adaptive", int8_th)