"""Evaluate a trained CAE via reconstruction error (anomaly scoring).

Computes per-sample MSE, uses negative MSE as the anomaly score
(higher = more bonafide), and reports EER against ground truth labels.
"""

import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_cae import FeatureNormalizer, FullLabeledDataset
from evaluation import calculate_eer
from model_cae import ConvAutoencoder


def resolve_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate_cae(model: ConvAutoencoder, dataloader: DataLoader,
                 device: str) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Compute reconstruction-error scores and EER.

    Returns:
        metrics: dict with avg_mse, eer, threshold_mse
        mse_scores: per-sample MSE (numpy array)
        labels: ground truth labels (numpy array)
    """
    model.eval()
    mse_criterion = nn.MSELoss(reduction="none")

    all_mse: List[float] = []
    labels: List[float] = []

    for features, batch_labels in dataloader:
        features = features.to(device)

        reconstruction, _ = model(features)

        # Per-sample MSE: mean over (T, F)
        mse = mse_criterion(reconstruction, features)
        mse_per_sample = mse.view(mse.size(0), -1).mean(dim=1)

        all_mse.extend(mse_per_sample.cpu().tolist())
        labels.extend(batch_labels.cpu().tolist())

    all_mse = np.array(all_mse)
    labels = np.array(labels)

    # Try both scoring conventions and pick the better one:
    # Convention A: -MSE (higher = bonafide, assumes fakes have MORE error)
    # Convention B: +MSE (higher = bonafide, assumes fakes have LESS error)
    scores_neg = -all_mse
    scores_pos = all_mse  # raw MSE

    eer_neg, thr_neg = calculate_eer(scores_neg.tolist(), labels.tolist())
    eer_pos, thr_pos = calculate_eer(scores_pos.tolist(), labels.tolist())

    if eer_neg <= eer_pos:
        eer, threshold_mse = eer_neg, -thr_neg
        convention = "standard (-MSE: fakes have higher error)"
    else:
        eer, threshold_mse = eer_pos, thr_pos
        convention = "inverted (+MSE: fakes have lower error)"

    metrics = {
        "avg_mse": float(np.mean(all_mse)),
        "avg_mse_bonafide": float(np.mean(all_mse[labels == 1])),
        "avg_mse_spoof": float(np.mean(all_mse[labels == 0])),
        "eer": eer,
        "eer_neg": eer_neg,
        "eer_pos": eer_pos,
        "threshold_mse": threshold_mse,
        "convention": convention,
    }
    return metrics, all_mse, labels


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CAE checkpoint.")
    p.add_argument("--features", required=True, help="Path to features.pkl")
    p.add_argument("--labels", required=True, help="Path to labels.pkl")
    p.add_argument("--checkpoint", required=True, help="Path to CAE checkpoint")
    p.add_argument("--normalizer", required=True, help="Path to normalizer.pt")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    # Load normalizer
    normalizer = FeatureNormalizer.load(args.normalizer)
    print(f"Loaded normalizer from {args.normalizer}")

    # Load model
    model = ConvAutoencoder(base_channels=args.base_channels).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Dataset (both classes, swap_tf + normalised)
    ds = FullLabeledDataset(args.features, args.labels,
                            normalizer=normalizer, swap_tf=True)
    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"Evaluating on {len(ds)} samples...")

    # Evaluate
    metrics, mse_scores, labels = evaluate_cae(model, dataloader, device)

    print(f"\n{'=' * 60}")
    print("CAE Anomaly Detection Results")
    print(f"  Avg MSE (all):      {metrics['avg_mse']:.6f}")
    print(f"  Avg MSE (bonafide): {metrics['avg_mse_bonafide']:.6f}")
    print(f"  Avg MSE (spoof):    {metrics['avg_mse_spoof']:.6f}")
    print(f"  MSE ratio (spoof/bonafide): "
          f"{metrics['avg_mse_spoof'] / metrics['avg_mse_bonafide']:.2f}x")
    print(f"  EER (-MSE):         {metrics['eer_neg']:.6f}")
    print(f"  EER (+MSE):         {metrics['eer_pos']:.6f}")
    print(f"  Best EER:           {metrics['eer']:.6f}  ({metrics['convention']})")
    print(f"  Threshold (MSE):    {metrics['threshold_mse']:.6f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
