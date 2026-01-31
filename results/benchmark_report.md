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

| Model | Mean EER | Std | Best EER | Best Epoch | Best Seed |
|---|---:|---:|---:|---:|---:|
| cnn2d | 0.0055 | 0.0000 | 0.0055 | 11 | 0 |
| crnn2 | 0.0150 | 0.0000 | 0.0150 | 9 | 0 |
| crnn | 0.0195 | 0.0000 | 0.0195 | 3 | 0 |
| cnn1d | 0.0290 | 0.0000 | 0.0290 | 3 | 0 |
| cnn2d_spatial | 0.0290 | 0.0000 | 0.0290 | 9 | 0 |

## Overfitting Signals (heuristic)
First epoch where average train loss keeps decreasing while average dev loss rises for two consecutive steps.

- cnn1d: no clear overfitting signal in averaged curves
- cnn2d: no clear overfitting signal in averaged curves
- cnn2d_spatial: no clear overfitting signal in averaged curves
- crnn: no clear overfitting signal in averaged curves
- crnn2: potential overfitting starts around epoch 7

## Plots
- combined: `results/plots/combined_losses.png`
- cnn1d: `results/plots/cnn1d_curves.png`
- cnn2d: `results/plots/cnn2d_curves.png`
- cnn2d_spatial: `results/plots/cnn2d_spatial_curves.png`
- crnn: `results/plots/crnn_curves.png`
- crnn2: `results/plots/crnn2_curves.png`

## Notes / Justifications
- Mean pooling vs max pooling: mean pooling preserves the average energy pattern across time and is less sensitive to single-frame outliers; max pooling can over-emphasize spiky artifacts and destabilize training for this dataset.
- CNNs preserve local temporal patterns that MLP pooling removes; this typically improves EER when deepfake artifacts are short and localized.