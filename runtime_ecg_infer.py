import numpy as np
import time
import matplotlib.pyplot as plt
from dynamic_threshold import DynamicThreshold

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    print("ai_edge_litert not installed, trying legacy interpreter")
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        raise ImportError("No lightweight TFLite runtime found")


class ECGDetector:
    def __init__(self, model_path: str):
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

        # output
        y = self.interpreter.get_tensor(self.output_details[0]["index"])
        if self.output_details[0]["dtype"] == np.int8:
            y = (y.astype(np.float32) - self.output_zero_point) * self.output_scale

        return float(y.reshape(-1)[0])


def evaluate_record(detector, dyn_thresh, X, y, record_name, verbose=False):
    print(f"\nEVALUATING RECORD {record_name}")
    print("=" * 40)
    print(f"Total windows: {len(y)}")
    print(f"  Negative (normal): {(y==0).sum()}")
    print(f"  Positive (ventricular): {(y==1).sum()}")

    dyn_thresh.reset()

    preds = []
    probs = []
    times = []
    dynamic_thresholds = []

    for i in range(len(X)):
        w = X[i, :, 0]

        start = time.time()
        prob = detector.predict(w)
        result = dyn_threshold.update(prob)
        elapsed = (time.time() - start) * 1000

        probs.append(prob)
        preds.append(1 if result['is_alert'] else 0)
        times.append(elapsed)
        dynamic_thresholds.append(result['current_threshold'])

        if verbose:
            if dyn_thresh.is_warmup and (i + 1) % 10 == 0:
                print(f"  Warmup: {i + 1}/{dyn_thresh.warmup_length} windows, "
                       f"buffer={result['buffer_size']}")

            if not dyn_thresh.is_warmup and (i + 1) % 100 == 0:
                diag = dyn_thresh.get_diagnostics()
                print(f"\n[Window {i + 1}] Diagnostics:")
                print(f"Threshold: {diag['current_threshold']:.4f}"
                      f"(P95={diag['buffer_p95']:.4f}")
                print(f"Buffer: {diag['buffer_size']}/{diag['buffer_capacity']}")
                print(f"Alerts: {diag['alert_count']}")

    preds = np.array(preds)
    probs = np.array(probs)
    dynamic_thresholds = np.array(dynamic_thresholds)

    # cm
    tn = int(((y==0) & (preds==0)).sum())
    fp = int(((y==0) & (preds==1)).sum())
    fn = int(((y==1) & (preds==0)).sum())
    tp = int(((y==1) & (preds==1)).sum())

    acc = (tn + tp) / len(y) if len(y) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    avg_time = np.mean(times)
    final_diag = dyn_thresh.get_diagnostics()

    print("\nDYNAMIC THRESHOLD RESULTS")

    print("\nThreshold Dynamics")

    print(f"Final Threshold: {dynamic_thresholds[-1]:.4f}")
    print(f"Operating Range: [{dynamic_thresholds.min():.4f} - {dynamic_thresholds.max():.4f}]")
    print(f"Buffer P95: {final_diag['buffer_p95']:.4f}")

    print("\nClinical Performance")
    print(f"  TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")

    print("\nHardware Metrics")
    print(f"Avg Inference: {avg_time:.2f} ms/window")

    print(f"\nProbability distribution:")
    print(f"All: mean={probs.mean():.4f}, std={probs.std():.4f}")
    if (y==0).sum() > 0:
        print(f"Normal: mean={probs[y==0].mean():.4f}, std={probs[y==0].std():.4f}")
    if (y==1).sum() > 0:
        print(f"Ventricular: mean={probs[y==1].mean():.4f}, std={probs[y==1].std():.4f}")

    return tp, tn, fp, fn, avg_time, dynamic_thresholds, probs


def threshold_sweep(detector, X, y, record_name):
    print(f"THRESHOLD SWEEP ON RECORD {record_name}")

    # Get all probabilities
    probs = np.array([detector.predict(X[i, :, 0]) for i in range(len(X))])

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

        marker = " ← CURRENT" if abs(th - GLOBAL_THRESHOLD) < 0.01 else ""
        marker = marker or (" ← OPTIMAL" if abs(th - THRESHOLD_OPTIMAL) < 0.01 else "")
        marker = marker or (" ← BEST" if f1 > best_f1 else "")

        print(f"{th:>10.4f} | {prec:>6.4f} | {rec:>6.4f} | {f1:>6.4f} | {tp:>3d} | {fp:>3d} | {fn:>3d}{marker}")

        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    print(f"\nBest threshold: {best_th:.4f} (F1={best_f1:.4f})")
    return best_th, best_f1

def plot_threshold_trace(probs, y_true, thresholds, record_name):

    alerts = (probs >= thresholds).astype(int)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    # Top panel: Probabilities and threshold
    ax1.plot(probs, label='Probability', alpha=0.7, linewidth=1)
    ax1.plot(thresholds, label='Dynamic Threshold', linewidth=2, color='orange')

    # Mark true ventricular beats
    ventri_idx = np.where(y_true == 1)[0]
    ax1.scatter(ventri_idx, probs[ventri_idx], color='red',
                s=100, marker='x', label='True Ventricular', zorder=5)

    # Mark alerts
    alert_idx = np.where(alerts == 1)[0]
    ax1.scatter(alert_idx, probs[alert_idx], color='red',
                s=50, alpha=0.3, label='Alerts', zorder=4)

    ax1.set_ylabel('Probability')
    ax1.set_ylim([0, 1])
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Dynamic Threshold Performance - Record {record_name}')

    # Bottom panel: Threshold evolution
    ax2.plot(thresholds, linewidth=2, color='orange')
    ax2.set_xlabel('Window Index')
    ax2.set_ylabel('Threshold Value')
    ax2.set_ylim([0.3, 1.0])
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.40, color='gray', linestyle='--', label='Floor')
    ax2.axhline(y=0.95, color='gray', linestyle='--', label='Ceil')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'dynamic_threshold_record_{record_name}.png', dpi=150)
    print(f"Saved plot: dynamic_threshold_record_{record_name}.png")
    plt.close()

if __name__ == "__main__":
    # Config
    MODEL_PATH = "models/ecg_int8.tflite"

    with open("threshold.txt", "r") as f:
        THRESHOLD_OPTIMAL = float(f.read().strip())

    THRESHOLD_SAFE = 0.5
    THRESHOLD_BALANCED = 0.7
    GLOBAL_THRESHOLD = THRESHOLD_OPTIMAL

    # Initialize dynamic threshold
    dyn_threshold = DynamicThreshold(
        global_threshold=GLOBAL_THRESHOLD,
        buffer_size=150,
       margin=0.05,
        floor=0.40,
        ceil=0.95,
        warmup_length=50,
        update_freq=10
    )

    print(f"Loaded thresholds:")
    print(f"  Optimal (F1-max): {THRESHOLD_OPTIMAL:.4f}")
    print(f"  Safe (high recall): {THRESHOLD_SAFE:.4f}")
    print(f"  Balanced: {THRESHOLD_BALANCED:.4f}")
    print(f"  Using: {GLOBAL_THRESHOLD:.4f}\n")

    # Load data
    data = np.load("mitbih_windows.npz", allow_pickle=True)
    X_all = data["X_all"]
    y_all = data["y_all"].astype(int)
    rids_all = data["rids_all"].astype(str)

    # Initialize detector
    print(f"Loading model: {MODEL_PATH}")
    detector = ECGDetector(MODEL_PATH)

    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    all_times = []

    # TEST ON SPECIFIC RECORD
    TARGET_RECORDS = ["109", "205"]
    RUN_THRESHOLD_SWEEP = False
    RUN_PLOT = False
    for TARGET_RECORD in TARGET_RECORDS:
        mask = (rids_all == TARGET_RECORD)
        X_target = X_all[mask]
        y_target = y_all[mask]

        # Main eval
        tp, tn, fp, fn, avg_time, dynamic_thresholds, probs = evaluate_record(detector,dyn_threshold, X_target, y_target, TARGET_RECORD)

        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn
        all_times.append(avg_time)

        if RUN_PLOT:
            plot_threshold_trace(probs, y_target, dynamic_thresholds, TARGET_RECORD)

        if RUN_THRESHOLD_SWEEP:
            print("Threshold sweep (to find optimal threshold for this record")
            best_th, best_f1 = threshold_sweep(detector, X_target, y_target, TARGET_RECORD)
            print(f"Performance at optimal threshold ({best_th:.4f}):")
            print(f"  F1:        {best_f1:.4f}")
            print(f"Inference: {avg_time:.2f} ms/window")


    print(f"\n\nFULL SUMMARY")
    if (total_tp + total_fp) > 0 and (total_tp + total_tn) > 0:
        mean_prec = total_tp / (total_tp + total_fp)
        mean_rec = total_tp / (total_tp + total_fn)
        mean_f1 = (2 * mean_prec * mean_rec / (mean_prec + mean_rec)
                   if (mean_prec + mean_rec) > 0 else 0)
        print(f"Micro Precision: {mean_prec:.4f}")
        print(f"Micro Recall: {mean_rec:.4f}")
        print(f"Micro F1: {mean_f1:.4f}")
    else:
        print("No positive predictions")

    print(f"Avg Inference: {np.mean(all_times):.4f} ms/window")







