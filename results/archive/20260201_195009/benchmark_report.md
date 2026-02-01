# Model Comparison Report

## Experiment Setup
- Epochs: 30
- Batch size: 32
- Learning rate: 0.001
- Dropout (CNNs): 0.3
- Pool bins (CNN1D): 1
- Seeds: 0
- Optimizer policy:
  - CNNs: AdamW with weight decay 0.01 by default (unless --weight-decay overrides)
  - MLPs: Adam unless --weight-decay > 0

## Summary Table (mean EER, lower is better)

| Model | Mean EER | Std | Best EER | Best Epoch | Best Seed | Avg Train Loss (<= best) | Avg Dev Loss (<= best) |
|---|---:|---:|---:|---:|---:|---:|---:|
| cnn2d | 0.0055 | 0.0000 | 0.0055 | 11 | 0 | 0.1227 | 0.1490 |
| cnn2d+specaug | 0.0055 | 0.0000 | 0.0055 | 11 | 0 | 0.1378 | 1.5973 |
| cnn2d_robust | 0.0140 | 0.0000 | 0.0140 | 10 | 0 | 0.1241 | 0.0746 |

## Overfitting Signals (heuristic)
First epoch where average train loss keeps decreasing while average dev loss rises for two consecutive steps.

- cnn2d: no clear overfitting signal in averaged curves
- cnn2d_robust: potential overfitting starts around epoch 4
- cnn2d+specaug: potential overfitting starts around epoch 4

## Plots
- combined: `results/plots/combined_losses.png`
- cnn2d: `results/plots/cnn2d_curves.png`
- cnn2d_robust: `results/plots/cnn2d_robust_curves.png`
- cnn2d+specaug: `results/plots/cnn2d+specaug_curves.png`

## Notes / Justifications
- Mean pooling vs max pooling: mean pooling preserves the average energy pattern across time and is less sensitive to single-frame outliers; max pooling can over-emphasize spiky artifacts and destabilize training for this dataset.
- CNNs preserve local temporal patterns that MLP pooling removes; this typically improves EER when deepfake artifacts are short and localized.