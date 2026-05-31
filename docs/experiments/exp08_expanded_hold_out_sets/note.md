# Experiment 08 — Expanded Hold-Out Sets & Cross-Patient Threshold Generalization

**Date:** March 2026  
**Model Architecture:** 1D CNN Bottleneck  
**Deployable Model:** `models/ecg_int8.tflite`  
**Dataset:** MIT-BIH Arrhythmia Database  

---

## 1. Goal

Evaluate whether the strong performance from the previous single-patient hold-out setup generalizes across a more diverse unseen patient population.

Earlier experiments achieved very strong performance on a single held-out patient (Record 109), but that raised the risk of a **single-patient illusion**. To test real cross-patient robustness, additional records were removed from training and reallocated into larger validation and test sets.

---

## 2. Data Splits

### Validation
- `105`
- `124`

### Test
- `109`
- `114`
- `205`

### Train
- `100`, `101`, `102`, `103`, `104`, `106`, `107`, `108`, `111`, `112`, `113`, `115`, `116`, `117`, `118`, `119`, `121`, `122`, `123`, `200`, `201`, `202`, `203`, `207`, `208`, `209`, `210`, `212`, `213`, `214`, `215`, `217`, `219`, `220`, `221`, `222`, `223`, `228`, `230`, `231`, `232`, `233`, `234`

**Training records:** 43  
**Validation records:** 2  
**Test records:** 3  

---

## 3. Global Results (Float32 Keras, Pooled Test Set)

These are the official pooled metrics from the **training script** (`train_and_export.py`) before TFLite export.

### Validation Threshold Selection
- **Optimal threshold:** `0.9775`
- **Validation F1:** `0.8308`
- **Validation Precision:** `0.7570`
- **Validation Recall:** `0.9205`

### Pooled Test Set Summary
- **Normal windows:** `6875`
- **Ventricular windows:** `152`
- **Total windows:** `7027`

### Pooled Test Metrics
- **Test AUC:** `0.9684`
- **Accuracy:** `0.9539`
- **Precision (ventricular):** `0.2943`
- **Recall (ventricular):** `0.8092`
- **F1 (ventricular):** `0.4316`

### Confusion Matrix
```text
[[6580  295]
 [  29  123]]