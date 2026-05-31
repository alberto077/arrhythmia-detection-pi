# Experiment 04 — Bottleneck Fix, No Bandpass (8-Record Setup)

**Date:** 2026-02-28  
**Purpose:** Evaluate whether reducing the Flatten bottleneck improves generalization on the held-out patient while keeping preprocessing unchanged (no bandpass filter).

---

## Setup

### Data / Split
- **Dataset:** MIT-BIH subset of 8 records
- **Records used:** `100, 102, 103, 105, 108, 109, 112, 113`
- **Train:** `100, 102, 103, 108, 112, 113`
- **Validation:** `105`
- **Test (held-out):** `109`

### Preprocessing
- **Bandpass filter:** None
- **Window length:** `2.0 s` (`720` samples at `360 Hz`)
- **Normalization:** Per-record z-score
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
Added a fourth convolution + pooling block before `Flatten()` to reduce the feature map size before the dense layers.

### Rationale
The previous architecture flattened `90 x 256 = 23,040` features directly into `Dense(128)`, creating a very large weight matrix and increasing overfitting risk.

### New Architecture Summary
- `Conv1D(64, 7) -> MaxPool(2)`
- `Conv1D(128, 5) -> MaxPool(2)`
- `Conv1D(256, 3) -> MaxPool(2)`
- `Conv1D(128, 3) -> MaxPool(2)`  ← bottleneck fix
- `Flatten -> Dense(128) -> Dense(64) -> Sigmoid`

### Parameter Count
- **Total parameters:** `986,625`

This reduced model size significantly from the previous ~3.1M-parameter version.

---

## Training Summary

- **Best validation epoch:** `Epoch 2`
- **Best validation AUC:** `0.9971`
- Early stopping triggered at **Epoch 9**
- Weights restored from **Epoch 2**

### Training behavior
The model still showed some overfitting after the best epoch, but the collapse was much less severe than in earlier runs. Validation performance remained strong for multiple epochs instead of failing immediately.

---

## Threshold Selection (Validation Record 105)

- **Optimal threshold:** `0.6247`
- **Validation F1 at threshold:** `0.8108`
- **Validation Precision:** `0.9091`
- **Validation Recall:** `0.7317`

This threshold is much more reasonable than the extreme thresholds seen in earlier unstable runs.

---

## Held-Out Test Results (Record 109)

### Probability Summary
**Negative windows**
- Mean: `0.4495`
- P90: `0.5179`
- Max: `0.6382`

**Positive windows**
- Mean: `0.6152`
- P10: `0.5070`
- Min: `0.3458`

### AUC
- **Test AUC:** `0.9265`

### Confusion Matrix
```text
[[2484    4]
 [  17   21]]