# Model Performance Report

## Experiment Summary
Comparison of three model architectures trained for 10 epochs on the Audio Deepfake Detection dataset.

**Hyperparameters:**
*   Epochs: 10
*   Batch Size: 32
*   Optimizer: Adam (lr=1e-3)
*   Loss: BCEWithLogitsLoss

## Results

| Model | Final Train Loss | Final Dev Loss | Best Dev EER | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **MeanPoolMLP** | 0.5307 | 0.6029 | **31.70%** | Baseline. Converged reasonably well but hit a performance ceiling. |
| **StatsPoolMLP** | 0.6856 | 0.6922 | **44.85%** | Failed experiment. Peformed worse than baseline, nearing random chance (50%). |
| **CNN1D** | 0.0096 | 0.1229 | **2.80%** | **Superior.** Drastic improvement. Temporal modeling is clearly essential. |

## Analysis

### 1. MeanPoolMLP (Baseline)
*   **Performance:** Decent for a simple baseline.
*   **Limitation:** Aggressive mean pooling destroys temporal information immediately. It can only detect if the "average spectral content" looks fake, missing subtle temporal artifacts.

### 2. StatsPoolMLP (Failed)
*   **Hypothesis:** Adding Std and Max pooling would capture "jitter" and "glitches".
*   **Reality:** The model failed to learn effective representations.
*   **Possible Causes:**
    *   **Noise Amplification:** `Max` pooling might be picking up non-deepfake outliers/noise, confusing the classifier.
    *   **Optimization Difficulty:** Tripling the input size (321 -> 963) while keeping the hidden layer fixed (128) created a severe bottleneck.
    *   **Lack of Locality:** Like MeanPool, it still treats the time dimension globally. It knows *that* there is high variance, but not *where* or *what pattern* caused it.

### 3. CNN1D (Winner)
*   **Performance:** Achieved <3% EER, which is excellent.
*   **Why it works:**
    *   **Temporal Awareness:** Convolutions scan the time axis, allowing the model to detect local inconsistencies (e.g., a 100ms glitch) that define deepfakes.
    *   **Hierarchical Features:** It builds complex features from simple ones over consecutive frames.

## Recommendations
1.  **Abandon MLP approaches:** Global pooling (even with stats) is insufficient for this task.
2.  **Focus on CNNs:** The `CNN1D` architecture is the correct direction.
3.  **Next Steps:**
    *   Implement `CNN2D` to treat the input as a spectrogram image (Time x Frequency).
    *   Tune `CNN1D` (increase receptive field, add residual connections).
