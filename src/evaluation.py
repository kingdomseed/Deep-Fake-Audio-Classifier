import argparse
import numpy as np
import torch
import torch.nn as nn

from dataloaders import make_loader
from model import build_model


def calculate_eer(scores, labels):
    """
    Compute Equal Error Rate (EER) from scores and labels.
    Borrowed from scripts/evaluation.py so you can do quick model-level checks.
    """
    scores_np = np.array(scores)
    labels_np = np.array(labels)

    sorted_indices = np.argsort(scores_np)
    sorted_scores = scores_np[sorted_indices]
    sorted_labels = labels_np[sorted_indices]

    n_bonafide = np.sum(labels_np)
    n_spoof = len(labels_np) - n_bonafide

    if n_bonafide == 0 or n_spoof == 0:
        return 0.0, 0.0

    false_accept_rate = np.concatenate(
        [[1.0], (n_spoof - np.cumsum(sorted_labels == 0)) / n_spoof]
    )
    false_reject_rate = np.concatenate(
        [[0.0], np.cumsum(sorted_labels == 1) / n_bonafide]
    )

    eer_idx = np.argmin(np.abs(false_accept_rate - false_reject_rate))
    eer = (false_accept_rate[eer_idx] + false_reject_rate[eer_idx]) / 2.0

    threshold_epsilon = 1e-6
    if eer_idx == 0:
        threshold = sorted_scores[0] - threshold_epsilon
    elif eer_idx == len(sorted_scores):
        threshold = sorted_scores[-1] + threshold_epsilon
    else:
        threshold = sorted_scores[eer_idx - 1]

    return float(eer), float(threshold)


def evaluate(model, dataloader, criterion=None, device="cpu"):
    """
    Run model on a labeled dataloader and return metrics and raw outputs.

    Returns:
        metrics: dict with avg_loss (if criterion provided), eer, threshold
        scores: list of model scores (logits)
        labels: list of ground-truth labels
    """
    model.eval()
    scores = []
    labels = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for features, batch_labels in dataloader:
            features = features.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(features).squeeze(-1)

            if criterion is not None:
                loss = criterion(logits, batch_labels)
                total_loss += loss.item() * batch_labels.size(0)
                total_count += batch_labels.size(0)

            scores.extend(logits.detach().cpu().tolist())
            labels.extend(batch_labels.detach().cpu().tolist())

    avg_loss = (total_loss / total_count) if total_count > 0 else None
    eer, threshold = (None, None)
    if scores and labels:
        eer, threshold = calculate_eer(scores, labels)

    metrics = {
        "avg_loss": avg_loss,
        "eer": eer,
        "threshold": threshold,
    }
    return metrics, scores, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model checkpoint on a labeled dataset.")
    parser.add_argument("--features", required=True, help="Path to features.pkl")
    parser.add_argument("--labels", required=True, help="Path to labels.pkl")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--model",
        default="cnn1d",
        choices=["cnn1d", "cnn2d", "cnn2d_spatial", "crnn", "crnn2"],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu")
    parser.add_argument("--in-features", type=int, default=321)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pool-bins", type=int, default=1)
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    model_kwargs = {}
    if args.model in {"cnn1d"}:
        model_kwargs = {
            "in_channels": args.in_features,
            "dropout": args.dropout,
            "pool_bins": args.pool_bins,
        }
    elif args.model in {"cnn2d", "cnn2d_spatial"}:
        model_kwargs = {
            "in_features": args.in_features,
            "dropout": args.dropout,
        }
    elif args.model in {"crnn", "crnn2"}:
        model_kwargs = {
            "in_features": args.in_features,
            "dropout": args.dropout,
        }

    model = build_model(args.model, **model_kwargs).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    dataloader = make_loader(
        args.features,
        args.labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    criterion = nn.BCEWithLogitsLoss()
    metrics, _, _ = evaluate(model, dataloader, criterion=criterion, device=device)

    print(f"avg_loss={metrics['avg_loss']}")
    print(f"eer={metrics['eer']}")
    print(f"threshold={metrics['threshold']}")
