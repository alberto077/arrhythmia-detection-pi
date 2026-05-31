# Experiment 06 — Bottleneck Fix, Larger Training Set, No Bandpass

**Date:** 2026-02-28  
**Purpose:** Evaluate whether expanding the patient-wise training set improves generalization for the bottleneck-fixed model while keeping preprocessing unchanged (no bandpass filter).

---

## Setup

### Data / Split
- **Dataset:** Expanded MIT-BIH patient-wise split
- **Validation (held fixed):** `105`
- **Test (held fixed):** `109`
- **Training:** Expanded MIT-BIH pool of selected records excluding validation and test

### Preprocessing
- **Bandpass filter:** No
- **Window length:** `2.0 s` (`720` samples at `360 Hz`)
- **Normalization:** Per-record z-score
- **Oversampling:** None
- **Class weighting:** Natural-distribution class weights loaded from NPZ

### Architecture
- Same bottleneck-fixed architecture introduced in Experiment 04
- Added a fourth convolution + pooling block before `Flatten()`

### Parameter Count
- **Total parameters:** `986,625`

---

## Training Summary

- **Best validation epoch:** `Epoch 13`
- **Best validation AUC:** `0.9943`
- Early stopping triggered at **Epoch 20**
- Weights restored from **Epoch 13**

### Training behavior
Training was much more stable than the smaller 8-record setup:
- validation performance stayed strong across many epochs
- best validation did **not** collapse immediately
- this suggests the larger training pool reduced overfitting and improved patient-level generalization

---

## Threshold Selection (Validation Record 105)

- **Optimal threshold:** `0.8385`
- **Validation F1 at threshold:** `0.7600`
- **Validation Precision:** `0.6441`
- **Validation Recall:** `0.9268`

---

## Held-Out Test Results (Record 109)

### Probability Summary
**Negative windows**
- Mean: `0.0615`
- P90: `0.1584`
- Max: `0.9999`

**Positive windows**
- Mean: `0.9451`
- P10: `0.8728`
- Min: `0.0000`

### AUC
- **Test AUC:** `0.9716`

### Confusion Matrix

```text
[[2462   26]
 [   4   34]]