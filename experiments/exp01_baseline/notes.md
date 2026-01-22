# Experiment 01 — Baseline (Record 113)
**Date:** 2026-01-3  
**Device:** Raspberry Pi 4B (tflite_runtime)  
**Model:** INT8 quantized CNN  
**Dataset:** MIT-BIH patient-wise split  
**Record Evaluated:** 113  
**Ventricular Beats:** 0  

---

## Purpose
Evaluate the model on a *fully normal* patient record to measure:
- False positive rate  
- Threshold behavior  
- Real edge inference speed  

This test establishes the baseline for “alarm reliability.”  
If the model fires too often here, the threshold or model architecture must be adjusted.

---

## Results

### **Runtime Output (Pi):**
True label: 0 | Pred: 1 | Prob(ventricular)=0.461 | TH=0.457

=== TFLite Test Set Results ===
Confusion Matrix: [[TN=196, FP=1590], [FN=0, TP=0]]
Accuracy: 0.1097
Precision: 0.0000
Recall: 0.0000
F1: 0.0000
Avg inference time: 0.57 ms/window (N=1786)


---

## Interpretation (facts only)

- Record 113 **contains no ventricular beats**, so all predictions should be `0`
- Model predicted **1590 false positives**  
- This means threshold is too low or the model is biased toward predicting "ventricular"
- **Recall is undefined here** (no true positives available)
- **Precision = 0**, because every ventricular prediction was wrong
- Inference latency is **0.57 ms per window**, extremely fast and suitable for real-time IoT

---

## Conclusion
The model misclassifies normal beats as ventricular at a very high rate.  
This suggests threshold tuning alone is insufficient — rebalancing the training dataset or modifying loss weighting may be required.

This experiment acts as a baseline for later improvements.

