import numpy as np
from sklearn.metrics import confusion_matrix


def binary_metrics(actuals, preds):
    cm = confusion_matrix(actuals, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tn": int(tn),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "confusion_matrix": cm,
    }


def best_f1_threshold(y_true, probs):
    from sklearn.metrics import precision_recall_curve

    prec, rec, thresholds = precision_recall_curve(y_true, probs)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    return float(thresholds[np.nanargmax(f1)])
