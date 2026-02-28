# Experiment 03 — Bandpass Filter (8-Record Setup, Same Architecture)

**Date:** 2026-02-28  
**Branch:** `experiment/bandpass-prep-update`  
**Purpose:** Evaluate whether adding Butterworth bandpass preprocessing improves held-out performance without changing the baseline CNN architecture.

---

## Setup

### Data / Split
- **Dataset:** MIT-BIH subset of 8 records
- **Records used:** `100, 102, 103, 105, 108, 109, 112, 113`
- **Train:** `100, 102, 103, 108, 112, 113`
- **Validation:** `105`
- **Test (held-out):** `109`

### Preprocessing
- **New change in this experiment:** Added bandpass filtering in `prepare_mitbih.py`
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

### Model
- Same CNN architecture as prior baseline (no bottleneck fix yet)
- `Conv1D(64,7) -> Conv1D(128,5) -> Conv1D(256,3) -> Flatten -> Dense(128) -> Dense(64) -> Sigmoid`
- **Total parameters:** `3,099,521`

---

## Training Summary

- **Best validation epoch:** `Epoch 1`
- **Best validation AUC:** `0.9754`
- Early stopping triggered at **Epoch 8**
- Weights restored from **Epoch 1**

### Important observation
Training metrics kept improving, but validation performance collapsed after the first epoch. This suggests strong early overfitting / instability after the initial checkpoint.

---

## Threshold Selection (Validation Record 105)

- **Optimal threshold:** `0.9256`
- **Validation F1 at threshold:** `0.8675`
- **Validation Precision:** `0.8571`
- **Validation Recall:** `0.8780`

---

## Held-Out Test Results (Record 109)

### Probability Summary
**Negative windows**
- Mean: `0.8316`
- P90: `0.8924`
- Max: `0.9461`

**Positive windows**
- Mean: `0.9004`
- P10: `0.8274`
- Min: `0.7057`

### AUC
- **Test AUC:** `0.7899`

### Confusion Matrix
```text
[[2443   45]
 [  22   16]]