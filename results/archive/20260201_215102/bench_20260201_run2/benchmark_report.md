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
| cnn2d | 0.0030 | 0.0000 | 0.0030 | 18 | 0 | 0.0943 | 0.1575 |
| cnn2d+specaug | 0.0035 | 0.0000 | 0.0035 | 18 | 0 | 0.1008 | 0.5770 |
| cnn2d_robust | 0.0120 | 0.0000 | 0.0120 | 15 | 0 | 0.1010 | 0.0572 |

## Overfitting Signals (heuristic)
First epoch where average train loss keeps decreasing while average dev loss rises for two consecutive steps.

- cnn2d: no clear overfitting signal in averaged curves
- cnn2d_robust: no clear overfitting signal in averaged curves
- cnn2d+specaug: no clear overfitting signal in averaged curves

## Plots
- combined: `results/bench_20260201_run2/plots/combined_losses.png`
- cnn2d: `results/bench_20260201_run2/plots/cnn2d_curves.png`
- cnn2d_robust: `results/bench_20260201_run2/plots/cnn2d_robust_curves.png`
- cnn2d+specaug: `results/bench_20260201_run2/plots/cnn2d+specaug_curves.png`

## Notes / Justifications
- Mean pooling vs max pooling: mean pooling preserves the average energy pattern across time and is less sensitive to single-frame outliers; max pooling can over-emphasize spiky artifacts and destabilize training for this dataset.
- CNNs preserve local temporal patterns that MLP pooling removes; this typically improves EER when deepfake artifacts are short and localized.