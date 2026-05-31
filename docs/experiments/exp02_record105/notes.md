# Experiment 02 — Record 105 (Ventricular-Containing Record)
**Date:** 2026-01-3  
**Device:** MacBook (tflite interpreter)  
**Model:** INT8 quantized CNN  
**Record Evaluated:** 105  
**Total Windows:** 2565  
**Ventricular Windows:** 41  

---

## Purpose
Evaluate the model on a record that actually contains ventricular beats to measure:
- Ability to detect arrhythmias (recall)
- Risk of false alarms (precision)
- Behavior of the threshold selected via validation F1

---

## Results

### **Runtime Output:**
Confusion Matrix: [[TN=0, FP=2524], [FN=0, TP=41]]
Precision: 0.0160
Recall: 1.0000
F1: 0.0315


---

## Interpretation (facts only)

- Model detected **all 41 ventricular beats** → **Recall = 1.0**
- It falsely flagged **every normal window as ventricular** → TN = 0
- This means:
  - The model is extremely sensitive  
  - But not specific  
  - It will fire an alert almost constantly in real deployment
- F1 score remains extremely low due to poor precision

This is *consistent* with Experiment 01:  
The model is heavily biased toward predicting “ventricular” for many windows.

---

## Conclusion
The model achieves perfect sensitivity but unusable specificity.  
This would create too many false alarms in a medical device context.

Next steps:  
- Rebalance training data  
- Adjust class weights  
- Improve patient-wise diversity  
- Potentially expand positive examples  
- Consider a secondary filtering algorithm (post-processing)

This experiment provides the first clear evidence that **the model needs architectural or data-level correction**.

