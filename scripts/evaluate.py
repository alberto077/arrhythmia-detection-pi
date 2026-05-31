import argparse
import csv
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from ecg_arrhythmia.inference import ECGDetector
from ecg_arrhythmia.metrics import best_f1_threshold, binary_metrics
from ecg_arrhythmia.threshold import DynamicThreshold


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ECG TFLite threshold experiments."
    )
    parser.add_argument(
        "--mode",
        choices=["trace", "sweep", "experiments", "all"],
        default="trace",
        help="Which runtime workflow to execute.",
    )
    return parser.parse_args()


def trace_record(detector, X_rec, y_rec, static_th,
                 margin, buffer_size,
                 admit_ceil, warmup_length,
                 floor, ceil, update_freq):

    dyn = DynamicThreshold(
        global_threshold=static_th,
        buffer_size=buffer_size,
        margin=margin,
        floor=floor,
        ceil=ceil,
        warmup_length=warmup_length,
        update_freq=update_freq,
        admit_ceil=admit_ceil,
    )

    probs = []
    thresholds = []
    actuals = []
    preds = []

    for i in range(len(X_rec)):
        prob = detector.predict(X_rec[i, :, 0])
        result = dyn.update(prob)

        probs.append(prob)
        thresholds.append(result["current_threshold"])
        actuals.append(int(y_rec[i]))
        preds.append(1 if result["is_alert"] else 0)

    return {
        "probs": np.array(probs),
        "thresholds": np.array(thresholds),
        "actuals": np.array(actuals),
        "preds": np.array(preds),
        "history": list(dyn.threshold_history),
    }


def run_experiment(detector, X, y, rids, records,
                   threshold_mode, static_th,
                   margin=0.15, buffer_size=150,
                   admit_ceil=0.85, warmup_length=50,
                   floor=0.40, ceil=0.95, update_freq=10,
                   verbose=False):

    per_record = {}

    for record in records:
        mask = rids == record
        X_rec = X[mask]
        y_rec = y[mask]

        if threshold_mode == "adaptive":
            dyn = DynamicThreshold(
                global_threshold=static_th,
                buffer_size=buffer_size,
                margin=margin,
                floor=floor,
                ceil=ceil,
                warmup_length=warmup_length,
                update_freq=update_freq,
                admit_ceil=admit_ceil,
                verbose=verbose,
            )

        actuals = []
        preds = []
        updates = 0

        for i in range(len(X_rec)):
            prob = detector.predict(X_rec[i, :, 0])

            if threshold_mode == "static":
                pred = 1 if prob >= static_th else 0
            else:
                result = dyn.update(prob)
                if result["threshold_updated"]:
                    updates += 1
                pred = 1 if result["is_alert"] else 0

            actuals.append(int(y_rec[i]))
            preds.append(pred)

        metrics = binary_metrics(actuals, preds)
        per_record[record] = {
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "threshold_updates": updates,
        }

    return per_record


def plot_trace(record_name, f32_trace, int8_trace, floor=0.40, ceil=0.95):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Float32
    ax1.plot(f32_trace["probs"], alpha=0.5, linewidth=0.5, color="grey", label="Probability")
    ax1.plot(f32_trace["thresholds"], linewidth=2, color="orange", label="Adaptive Threshold")

    arrhy_idx = np.where(f32_trace["actuals"] == 1)[0]
    ax1.scatter(arrhy_idx, f32_trace["probs"][arrhy_idx],
                color="red", s=20, marker="x", label="True Arrhythmia", zorder=5)

    tp_idx = np.where((f32_trace["actuals"] == 1) & (f32_trace["preds"] == 1))[0]
    fn_idx = np.where((f32_trace["actuals"] == 1) & (f32_trace["preds"] == 0))[0]
    fp_idx = np.where((f32_trace["actuals"] == 0) & (f32_trace["preds"] == 1))[0]

    ax1.set_ylabel("Probability")
    ax1.set_ylim([0, 1.05])
    ax1.set_title(f"Record {record_name} — Float32")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=floor, color="grey", linestyle="--", alpha=0.5, label="Floor")
    ax1.axhline(y=ceil, color="grey", linestyle="--", alpha=0.5, label="Ceil")

    tp = len(tp_idx)
    fp = len(fp_idx)
    fn = len(fn_idx)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    ax1.text(0.02, 0.95, f"TP={tp} FP={fp} FN={fn} F1={f1:.3f}",
             transform=ax1.transAxes, fontsize=8, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # INT8
    ax2.plot(int8_trace["probs"], alpha=0.5, linewidth=0.5, color="grey", label="Probability")
    ax2.plot(int8_trace["thresholds"], linewidth=2, color="blue", label="Adaptive Threshold")

    ax2.scatter(arrhy_idx, int8_trace["probs"][arrhy_idx],
                color="red", s=20, marker="x", label="True Arrhythmia", zorder=5)

    tp_idx = np.where((int8_trace["actuals"] == 1) & (int8_trace["preds"] == 1))[0]
    fn_idx = np.where((int8_trace["actuals"] == 1) & (int8_trace["preds"] == 0))[0]
    fp_idx = np.where((int8_trace["actuals"] == 0) & (int8_trace["preds"] == 1))[0]

    ax2.set_ylabel("Probability")
    ax2.set_xlabel("Beat Index")
    ax2.set_ylim([0, 1.05])
    ax2.set_title(f"Record {record_name} — INT8")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=floor, color="grey", linestyle="--", alpha=0.5)
    ax2.axhline(y=ceil, color="grey", linestyle="--", alpha=0.5)

    tp = len(tp_idx)
    fp = len(fp_idx)
    fn = len(fn_idx)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    ax2.text(0.02, 0.95, f"TP={tp} FP={fp} FN={fn} F1={f1:.3f}",
             transform=ax2.transAxes, fontsize=8, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    out_path = f"threshold_trace_{record_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    args = parse_args()

    data = np.load("mitbih_windows.npz")
    X_test = data["X_test"]
    y_test = data["y_test"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    rids_test = data["rids_test"]
    rids_eval = data["rids_eval"]

    data2 = np.load("incart_windows.npz")
    X_all_incart = data2["X_all"]
    y_all_incart = data2["y_all"]
    rids_all_incart = data2["rids_all"]

    DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
    INCART_RECORDS = [f'I{i:02d}' for i in range(1, 76)]
    eval_records = np.unique(rids_eval)


    f32_detector = ECGDetector("models/ecg_float32.tflite")
    f32_probs = np.array([f32_detector.predict(X_val[i, :, 0]) for i in range(len(X_val))])
    f32_th = best_f1_threshold(y_val, f32_probs)
    print(f"Float32 threshold: {f32_th:.4f}")

    int8_detector = ECGDetector("models/ecg_int8.tflite")
    int8_probs = np.array([int8_detector.predict(X_val[i, :, 0]) for i in range(len(X_val))])
    int8_th = best_f1_threshold(y_val, int8_probs)
    print(f"INT8 threshold: {int8_th:.4f}")


    defaults = {
        "margin": 0.15,
        "buffer_size": 150,
        "admit_ceil": 0.85,
        "warmup_length": 50,
        "floor": 0.40,
        "ceil": 0.95,
        "update_freq": 10
    }

    sweeps = {
        "margin": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "buffer_size": [50, 100, 150, 200, 300],
        "admit_ceil": [0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        "warmup_length": [20, 30, 50, 75, 100],
        "floor": [0.20, 0.30, 0.40, 0.50, 0.60],
        "ceil": [0.85, 0.90, 0.95, 0.99],
        "update_freq": [5, 10, 20, 35, 50],
    }

    models = [
        ("Float32", f32_detector, f32_th),
        ("INT8", int8_detector, int8_th),
    ]


    run_csv = args.mode in {"sweep", "all"}
    run_exps = args.mode in {"experiments", "all"}
    run_trace = args.mode in {"trace", "all"}

    if run_csv:
        rows = []
        total_cells = sum(len(v) for v in sweeps.values()) * len(models)
        pbar = tqdm(total=total_cells, desc="sweep")
        for param_name, values in sweeps.items():
            for model_label, detector, static_th in models:
                value_to_record_results = {}
                for val in values:
                    pbar.set_postfix(param=param_name, val=val, model=model_label)
                    params = {**defaults, param_name: val}
                    per_record = run_experiment(
                        detector, X_val, y_val, rids_eval, eval_records,
                        "adaptive", static_th, **params,
                    )
                    value_to_record_results[val] = per_record
                    for rec, m in per_record.items():
                        rows.append({
                            "param": param_name,
                            "value": val,
                            "model": model_label,
                            "record": rec,
                            "f1": m["f1"],
                            "precision": m["precision"],
                            "recall": m["recall"],
                            "tp": m["tp"],
                            "fp": m["fp"],
                            "fn": m["fn"],
                            "threshold_updates": m["threshold_updates"],
                        })
                    pbar.update(1)
        pbar.close()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = f"sweep_results_val_{timestamp}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out_path}")

    if run_trace:
        for r in DS2:
            mask = rids_test == r
            y_r = y_test[mask]
            arrhy = (y_r == 1).sum()
            total = len(y_r)
            pct = 100 * arrhy / total if total > 0 else 0
            print(f"Record {r}: {total} beats, {arrhy} arrhythmia ({pct:.1f}%)")

        rep_recs = [212, 202, 210, 200, 232]
        for record in rep_recs:
            mask = rids_test == record
            X_rec = X_test[mask]
            y_rec = y_test[mask]

            arrhy = int((y_rec == 1).sum())
            total = int(len(y_rec))
            print(f"\nRecord {record}: {total} beats, {arrhy} arrhythmia ({100*arrhy/total:.1f}%)")

            f32_trace = trace_record(f32_detector, X_rec, y_rec, f32_th, **defaults)
            int8_trace = trace_record(int8_detector, X_rec, y_rec, int8_th, **defaults)

            plot_trace(record, f32_trace, int8_trace)

    if run_exps:
        print("\nEXP 3: Float32 + Static")
        run_experiment(f32_detector, X_test, y_test, rids_test, DS2, "static", f32_th)

        print("\nEXP 4: INT8 + Static")
        run_experiment(int8_detector, X_test, y_test, rids_test, DS2, "static", int8_th)

        print("\nEXP 5: Float32 + Adaptive")
        run_experiment(f32_detector, X_test, y_test, rids_test, DS2, "adaptive", f32_th)

        print("\nEXP 6: INT8 + Adaptive")
        run_experiment(int8_detector, X_test, y_test, rids_test, DS2, "adaptive", int8_th)

        print("\nEXP 7: Float32 + Static (INCART)")
        run_experiment(f32_detector, X_all_incart, y_all_incart, rids_all_incart, INCART_RECORDS, "static", f32_th)

        print("\nEXP 8: INT8 + Static (INCART)")
        run_experiment(int8_detector, X_all_incart, y_all_incart, rids_all_incart, INCART_RECORDS, "static", int8_th)

        print("\nEXP 9: Float32 + Adaptive (INCART)")
        run_experiment(f32_detector, X_all_incart, y_all_incart, rids_all_incart, INCART_RECORDS, "adaptive", f32_th)

        print("\nEXP 10: INT8 + Adaptive (INCART)")
        run_experiment(int8_detector, X_all_incart, y_all_incart, rids_all_incart, INCART_RECORDS, "adaptive", int8_th)
