import numpy as np

# On Raspberry Pi you usually use tflite_runtime. Fall back to tf.lite if needed.
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

# -----------------------------
# Load TFLite model + threshold
# -----------------------------
MODEL_PATH = "ecg_int8.tflite"   # or "ecg_dr.tflite" / "ecg_float32.tflite"

with open("threshold.txt", "r") as f:
    THRESHOLD = float(f.read().strip())


class ECGDetector:
    def __init__(self, model_path=MODEL_PATH):
        # Create interpreter
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Cache quantization info (for INT8 models)
        self.input_scale, self.input_zero_point = self.input_details[0]["quantization"]
        self.output_scale, self.output_zero_point = self.output_details[0]["quantization"]

        # Input shape should be (1, win, 1)
        self.expected_shape = self.input_details[0]["shape"]

    def _prepare_window(self, window_1d: np.ndarray) -> np.ndarray:
        """
        Takes a 1D window (length win),
        z-scores it, and reshapes to (1, time, channels).
        """
        window = np.asarray(window_1d, dtype=np.float32)

        # Z-score normalization per window (like prepare script, but per-window)
        mean = window.mean()
        std = window.std() + 1e-8
        window = (window - mean) / std

        # Add channel and batch dims: (time,) -> (1, time, 1)
        window = window[None, :, None].astype(np.float32)
        return window

    def predict(self, window_1d: np.ndarray):
        """
        Returns:
          prob: float in [0,1] – model's probability of ventricular (class 1)
          label: int 0 or 1 using THRESHOLD
        """
        # Prepare window
        x = self._prepare_window(window_1d)

        # Handle different model types
        input_dtype = self.input_details[0]["dtype"]

        if input_dtype == np.int8:
            # Full INT8 model: we must quantize the float input
            x_q = x / self.input_scale + self.input_zero_point
            x_q = np.clip(np.round(x_q), -128, 127).astype(np.int8)
            self.interpreter.set_tensor(self.input_details[0]["index"], x_q)
        else:
            # Float or dynamic-range model: accepts float32 directly
            self.interpreter.set_tensor(self.input_details[0]["index"], x.astype(np.float32))

        # Run inference
        self.interpreter.invoke()

        # Get output
        y = self.interpreter.get_tensor(self.output_details[0]["index"])

        if self.output_details[0]["dtype"] == np.int8:
            # Dequantize to float probability
            y = (y.astype(np.float32) - self.output_zero_point) * self.output_scale

        prob = float(y.reshape(-1)[0])  # probability of class 1 (ventricular)
        label = int(prob >= THRESHOLD)
        return prob, label


if __name__ == "__main__":
    data = np.load("mitbih_windows.npz", allow_pickle=True)

    # Pick a record that you KNOW has ventricular beats (from the counting script)
    TARGET_RECORD = "105"  # change after you see the counts

    X_all = data["X_all"]
    y_all = data["y_all"].astype(int)
    rids  = data["rids_all"].astype(str)

    mask = (rids == TARGET_RECORD)
    X_pat = X_all[mask]
    y_pat = y_all[mask]

    print(f"Target record: {TARGET_RECORD}")
    print(f"Windows: {len(X_pat)} | Positives: {int((y_pat==1).sum())}")

    detector = ECGDetector(MODEL_PATH)

    # Evaluate this patient record
    preds = []
    probs = []
    for i in range(len(X_pat)):
        w = X_pat[i, :, 0]
        p, yhat = detector.predict(w)
        probs.append(p)
        preds.append(yhat)

    preds = np.array(preds)
    y_true = y_pat

    tn = int(((y_true==0) & (preds==0)).sum())
    fp = int(((y_true==0) & (preds==1)).sum())
    fn = int(((y_true==1) & (preds==0)).sum())
    tp = int(((y_true==1) & (preds==1)).sum())

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    print("\n=== Patient Record Results ===")
    print(f"Confusion Matrix: [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")