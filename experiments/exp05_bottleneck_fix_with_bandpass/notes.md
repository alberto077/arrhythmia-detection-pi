# Experiment 05 — Bottleneck Fix, With Bandpass (8-Record Setup)

**Date:** 2026-02-28  
**Purpose:** Evaluate whether reintroducing Butterworth bandpass filtering improves the bottleneck-fixed model on the held-out patient.

---

## Setup

### Data / Split
- **Dataset:** MIT-BIH subset of 8 records
- **Records used:** `100, 102, 103, 105, 108, 109, 112, 113`
- **Train:** `100, 102, 103, 108, 112, 113`
- **Validation:** `105`
- **Test (held-out):** `109`

### Preprocessing
- **Bandpass filter:** Yes
- **Filter:** Butterworth bandpass, `0.5–50 Hz`, zero-phase (`sosfiltfilt`)
- **Window length:** `2.0 s` (`720` samples at `360 Hz`)
- **Normalization:** Per-record z-score after filtering
- **Oversampling:** None
- **Class weighting:** Natural-distribution class weights loaded from NPZ

### Loaded Data
- **Train shape:** `(10496, 720, 1)`
- **Validation shape:** `(2565, 720, 1)`
- **Test shape:** `(2526, 720, 1)`

### Class Weights
- `{0: 1.0, 1: 476.0909}`

---

## Architecture

### Key Change
Same bottleneck-fixed architecture as Experiment 04:
- added a fourth convolution + pooling block before `Flatten()`

### Architecture Summary
- `Conv1D(64, 7) -> MaxPool(2)`
- `Conv1D(128, 5) -> MaxPool(2)`
- `Conv1D(256, 3) -> MaxPool(2)`
- `Conv1D(128, 3) -> MaxPool(2)`  ← bottleneck fix
- `Flatten -> Dense(128) -> Dense(64) -> Sigmoid`

### Parameter Count
- **Total parameters:** `986,625`

---

## Training Summary

- **Best validation epoch:** `Epoch 1`
- **Best validation AUC:** `0.9933`
- Early stopping triggered at **Epoch 8**
- Weights restored from **Epoch 1**

### Training behavior
Validation quality peaked immediately and degraded afterward, indicating the model again became more unstable after the first epoch. However, the bottleneck-fixed architecture remained usable and produced a valid held-out result.

---

## Threshold Selection (Validation Record 105)

- **Optimal threshold:** `0.4833`
- **Validation F1 at threshold:** `0.7037`
- **Validation Precision:** `0.5672`
- **Validation Recall:** `0.9268`

This threshold favored high recall on validation, but the resulting test behavior was much more conservative.

---

## Held-Out Test Results (Record 109)

### Probability Summary
**Negative windows**
- Mean: `0.4573`
- P90: `0.4675`
- Max: `0.4784`

**Positive windows**
- Mean: `0.4823`
- P10: `0.4609`
- Min: `0.4253`

### AUC
- **Test AUC:** `0.8424`

### Confusion Matrix
```text
[[2488    0]
 [  22   16]]