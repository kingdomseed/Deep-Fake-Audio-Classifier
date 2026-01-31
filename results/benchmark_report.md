# Model Comparison Report

## Experiment Setup
- Epochs: 30
- Batch size: 32
- Learning rate: 0.001
- Dropout (CNNs): 0.3
- Pool bins (CNN1D): 1
- Seeds: 0,1,2,3,4
- Optimizer policy:
  - CNNs: AdamW with weight decay 0.01 by default (unless --weight-decay overrides)
  - MLPs: Adam unless --weight-decay > 0

## Summary Table (mean EER, lower is better)

| Model | Mean EER | Std | Best EER | Best Epoch | Best Seed |
|---|---:|---:|---:|---:|---:|
| cnn2d | 0.0049 | 0.0012 | 0.0035 | 17 | 2 |

## Seed Stability (cnn2d)
- Mean best EER: 0.004889
- Std dev: 0.001349
- Range: 0.003482 to 0.006498
- Best epoch varies by seed (5, 8, 10, 11, 17), which is normal; early stopping should pick the right checkpoint.

## Overfitting Signals (heuristic)
First epoch where average train loss keeps decreasing while average dev loss rises for two consecutive steps.

- cnn2d: potential overfitting starts around epoch 10

## Plots
- combined: `results/plots/combined_losses.png`
- cnn2d: `results/plots/cnn2d_curves.png`

## Notes / Justifications
- Mean pooling vs max pooling: mean pooling preserves the average energy pattern across time and is less sensitive to single-frame outliers; max pooling can over-emphasize spiky artifacts and destabilize training for this dataset.
- CNNs preserve local temporal patterns that MLP pooling removes; this typically improves EER when deepfake artifacts are short and localized.
