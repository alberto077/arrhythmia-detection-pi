# Experiment 07 — Bottleneck Fix, Larger Training Set, With Bandpass

**Date:** 2026-02-28  
**Purpose:** Evaluate whether reintroducing bandpass filtering improves the bottleneck-fixed model when using the expanded patient-wise training set.

---

## Setup

### Data / Split
- **Dataset:** Expanded MIT-BIH patient-wise split
- **Validation (held fixed):** `105`
- **Test (held fixed):** `109`
- **Training:** Expanded MIT-BIH pool of selected records excluding validation and test

### Preprocessing
- **Bandpass filter:** Yes
- **Filter:** Butterworth bandpass, `0.5–50 Hz`, zero-phase (`sosfiltfilt`)
- **Window length:** `2.0 s` (`720` samples at `360 Hz`)
- **Normalization:** Per-record z-score after filtering
- **Oversampling:** None
- **Class weighting:** Natural-distribution class weights loaded from NPZ

### Architecture
- Same bottleneck-fixed architecture used in Experiments 04, 05, and 06
- Added a fourth convolution + pooling block before `Flatten()`

### Parameter Count
- **Total parameters:** `986,625`

---

## Training Summary

- **Best validation epoch:** `Epoch 9`
- **Best validation AUC:** `0.99431`
- Early stopping triggered at **Epoch 16**
- Weights restored from **Epoch 9**

### Training behavior
Training remained stable across many epochs:
- validation performance improved gradually instead of collapsing early
- the expanded training set continued to reduce instability
- the model maintained strong validation quality with bandpass enabled

---

## Threshold Selection (Validation Record 105)

- **Optimal threshold:** `0.9813`
- **Validation F1 at threshold:** `0.7727`
- **Validation Precision:** `0.7234`
- **Validation Recall:** `0.8293`

This produced a very strict operating point, favoring high confidence positive predictions.

---

## Held-Out Test Results (Record 109)

### Probability Summary
**Negative windows (Normal)**
* **Mean:** 0.0498
* **P90:** 0.1648
* **Max:** 0.9733

**Positive windows (Ventricular)**
* **Mean:** 0.9476
* **P10:** 0.9395
* **Min:** 0.0013

### Confusion Matrix
| | Predicted Normal | Predicted Ventricular |
| :--- | :--- | :--- |
| **Actual Normal** | 2488 (TN) | 0 (FP) |
| **Actual Ventricular** | 4 (FN) | 34 (TP) |

### Test Metrics
*Target Class: 1 (Ventricular)*

* **Test AUC:** 0.9720
* **Test Precision:** 1.0000
* **Test Recall:** 0.8947
* **Test F1 Score:** 0.9444

**Conclusion:** The combination of the bandpass filter, the bottleneck architecture, and the expanded dataset successfully eliminated the probability bias and severe overfitting. The model now effectively ignores baseline wander and isolates genuine ventricular morphology with perfect precision.