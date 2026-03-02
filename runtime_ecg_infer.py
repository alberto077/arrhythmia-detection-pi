import numpy as np
import time

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

# Configuration
MODEL_PATH = "models/ecg_int8.tflite"

# Load threshold
with open("threshold.txt", "r") as f:
    THRESHOLD_OPTIMAL = float(f.read().strip())

THRESHOLD_SAFE = 0.5  # Higher recall, lower precision (safer for medical)
THRESHOLD_BALANCED = 0.7  # Balance between precision and recall

# Choose which threshold to use
THRESHOLD = THRESHOLD_OPTIMAL  # Change this as needed

print(f"Loaded thresholds:")
print(f"  Optimal (F1-max): {THRESHOLD_OPTIMAL:.4f}")
print(f"  Safe (high recall): {THRESHOLD_SAFE:.4f}")
print(f"  Balanced: {THRESHOLD_BALANCED:.4f}")
print(f"  → Using: {THRESHOLD:.4f}\n")


class ECGDetector:
    def __init__(self, model_path=MODEL_PATH):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Quantization params
        self.input_scale, self.input_zero_point = self.input_details[0]["quantization"]
        self.output_scale, self.output_zero_point = self.output_details[0]["quantization"]

    def _prepare_window(self, window_1d: np.ndarray) -> np.ndarray:
        """Per-window z-score normalization"""
        w = np.asarray(window_1d, dtype=np.float32)
        w = (w - w.mean()) / (w.std() + 1e-8)
        return w[None, :, None].astype(np.float32)

    def predict(self, window_1d: np.ndarray):
        """Returns probability and label"""
        x = self._prepare_window(window_1d)

        # Handle INT8 quantization
        input_dtype = self.input_details[0]["dtype"]
        if input_dtype == np.int8:
            x_q = x / self.input_scale + self.input_zero_point
            x_q = np.clip(np.round(x_q), -128, 127).astype(np.int8)
            self.interpreter.set_tensor(self.input_details[0]["index"], x_q)
        else:
            self.interpreter.set_tensor(self.input_details[0]["index"], x.astype(np.float32))

        # Run inference
        self.interpreter.invoke()

        # Get output
        y = self.interpreter.get_tensor(self.output_details[0]["index"])
        if self.output_details[0]["dtype"] == np.int8:
            y = (y.astype(np.float32) - self.output_zero_point) * self.output_scale

        prob = float(y.reshape(-1)[0])
        label = int(prob >= THRESHOLD)

        return prob, label


def evaluate_record(detector, X, y, record_name):
    print(f"EVALUATING RECORD {record_name}")
    print(f"Total windows: {len(y)}")
    print(f"  Negative (normal): {(y==0).sum()}")
    print(f"  Positive (ventricular): {(y==1).sum()}")

    preds = []
    probs = []
    times = []

    for i in range(len(X)):
        w = X[i, :, 0]

        start = time.time()
        p, yhat = detector.predict(w)
        elapsed = (time.time() - start) * 1000  # ms

        probs.append(p)
        preds.append(yhat)
        times.append(elapsed)

    preds = np.array(preds)
    probs = np.array(probs)

    # Compute metrics
    tn = int(((y==0) & (preds==0)).sum())
    fp = int(((y==0) & (preds==1)).sum())
    fn = int(((y==1) & (preds==0)).sum())
    tp = int(((y==1) & (preds==1)).sum())

    acc = (tn + tp) / len(y) if len(y) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"\nResults (threshold={THRESHOLD:.4f}):")
    print(f"  Confusion Matrix: [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")

    print(f"\nProbability distribution:")
    print(f"  All:      mean={probs.mean():.4f}, std={probs.std():.4f}")
    if (y==0).sum() > 0:
        print(f"  Normal:   mean={probs[y==0].mean():.4f}, std={probs[y==0].std():.4f}")
    if (y==1).sum() > 0:
        print(f"  Ventri:   mean={probs[y==1].mean():.4f}, std={probs[y==1].std():.4f}")

    avg_time = np.mean(times)
    print(f"\nInference time: {avg_time:.2f} ms/window (avg of {len(times)})")

    return f1, prec, rec, avg_time


def threshold_sweep(detector, X, y, record_name):
    print(f"THRESHOLD SWEEP ON RECORD {record_name}")

    # Get all probabilities
    probs = np.array([detector.predict(X[i, :, 0])[0] for i in range(len(X))])

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, THRESHOLD_OPTIMAL]

    print(f"\n{'Threshold':>10} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'TP':>3} | {'FP':>3} | {'FN':>3}")
    print("-" * 60)

    best_f1 = 0
    best_th = 0.5

    for th in sorted(set(thresholds)):
        preds = (probs >= th).astype(int)

        tp = ((y==1) & (preds==1)).sum()
        fp = ((y==0) & (preds==1)).sum()
        fn = ((y==1) & (preds==0)).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        marker = " ← CURRENT" if abs(th - THRESHOLD) < 0.01 else ""
        marker = marker or (" ← OPTIMAL" if abs(th - THRESHOLD_OPTIMAL) < 0.01 else "")
        marker = marker or (" ← BEST" if f1 > best_f1 else "")

        print(f"{th:>10.4f} | {prec:>6.4f} | {rec:>6.4f} | {f1:>6.4f} | {tp:>3d} | {fp:>3d} | {fn:>3d}{marker}")

        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    print(f"\nBest threshold: {best_th:.4f} (F1={best_f1:.4f})")
    return best_th, best_f1


if __name__ == "__main__":
    # Load data
    data = np.load("mitbih_windows.npz", allow_pickle=True)
    X_all = data["X_all"]
    y_all = data["y_all"].astype(int)
    rids_all = data["rids_all"].astype(str)

    # Initialize detector
    print(f"Loading model: {MODEL_PATH}")
    detector = ECGDetector(MODEL_PATH)

    # TEST ON SPECIFIC RECORD
    TARGET_RECORD = "109"
    mask = (rids_all == TARGET_RECORD)
    X_target = X_all[mask]
    y_target = y_all[mask]

    # Main evaluation
    f1, prec, rec, avg_time = evaluate_record(detector, X_target, y_target, TARGET_RECORD)

    # Threshold sweep (to find optimal threshold for this record)
    best_th, best_f1 = threshold_sweep(detector, X_target, y_target, TARGET_RECORD)

    # Summary
    print("SUMMARY:")
    print(f"Model: {MODEL_PATH}")
    print(f"Record: {TARGET_RECORD} ({len(y_target)} windows, {(y_target==1).sum()} ventricular)")
    print(f"\nPerformance at current threshold ({THRESHOLD:.4f}):")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"Performance at optimal threshold ({best_th:.4f}):")
    print(f"  F1:        {best_f1:.4f}")
    print(f"Inference: {avg_time:.2f} ms/window")
