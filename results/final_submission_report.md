# Final Model Submission Report: Audio Deepfake Detection
**Date:** February 6, 2026
**Student:** Jason Holt (st199007)
**Model Architecture:** CNN2D (Robust Variant)

## 1. Executive Summary
This report documents the final "robust" training run for the Semester 1 Deep Learning project. Following insights from the final lecture regarding domain shift (calm/flat prosody in new bonafide audio), the model was trained with heavy regularization and calibration to prevent over-confidence and improve generalization to unseen attacks.

## 2. Training Configuration & Hyperparameters
The model was trained using a "Robust Training Recipe" designed to counteract overfitting to clean training conditions.

*   **Model:** CNN2D
*   **Augmentation Strategy (SpecAugment + Robustness):**
    *   `--spec-augment`: Time mask (0.20), Feature mask (0.10)
    *   `--time-shift`: 0.10 ratio
    *   `--channel-drop`: 0.05 probability
    *   `--gaussian-jitter`: 0.005 std dev
*   **Calibration & Optimization:**
    *   `--label-smoothing`: 0.05 (Crucial for preventing over-confidence)
    *   `--lr-scheduler`: Plateau (monitoring `dev_eer`)
    *   `--early-stop`: 8 epochs
    *   `--seed`: 2
*   **Checkpoint Used:** `checkpoints/final_robust/cnn2d_best.pt` (Epoch 16)

## 3. Performance Verification
The model was verified against the known `test1` set and monitored for score "hardness."

### Test 1 Results
*   **EER:** 0.000000
*   **Accuracy:** 100% (FAR: 0.0, FRR: 0.0)
*   **Distribution Analysis:**
    *   Min: 0.000841 | Max: 0.999979
    *   Median: 0.2705
    *   Fractions: <0.01: 5.80% | >0.99: 40.80%

### Final Test Set Distribution (Test Final)
Analysis of the 1000 samples in the final submission:
*   **Class 1 (Real) Estimate:** 418 samples (41.8%)
*   **Class 0 (Fake) Estimate:** 582 samples (58.2%)
*   **Score Calibration:** The presence of mid-range scores (e.g., 0.12, 0.30, 0.19) indicates the model is appropriately calibrated and expressing uncertainty on ambiguous samples rather than pinning everything to 0/1.

## 4. Addressing Professor's Feedback
*   **Alignment:** Utilized `pd.merge` on `uttid` during prediction generation to ensure zero risk of sample order misalignment.
*   **Over-confidence:** Label smoothing (0.05) was applied to ensure the model maintains a "soft" margin, protecting against the EER penalties associated with high-confidence misclassifications on new bonafide audio.
*   **Prosody Shift:** The use of Time Masking and Time Shifting specifically helps the model ignore global prosody patterns that might differ between the "vivid" training set and "flat" final test set.

## 5. Submission Artifacts
*   **Submission File:** `st199007-Jason-Holt-WhatAreLogits.pkl`
*   **Prediction Source:** `results/prediction_final_test.pkl`
*   **Features Source:** `data/test_final/final_test.pkl`

## 6. Submission Data Preview (Output of pred.py)
The following is a preview of the `predictions` dataframe contained within the final submission `.pkl` file:

```text
                uttid  predictions
0    a07b5c747ff85f0a     0.120770
1    ff3d1eb3679617ed     0.005989
2    f19e4db13954c0cf     0.001421
3    84aff282516742b6     0.026569
4    d2ec3c43eb61a552     0.005956
..                ...          ...
995  24f1554e04055f64     0.999866
996  81334c7d359ff889     0.303182
997  bfdf019091a55411     0.190218
998  137c7b57caac9684     0.990914
999  9fd041bdee4cc743     0.006506

[1000 rows x 2 columns]
Class 1 count: 418
Class 0 count: 582
```

---
**Status:** Ready for Upload.
