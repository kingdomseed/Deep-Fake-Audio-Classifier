"""Ensemble averaging of multiple model checkpoints.

Loads predictions from multiple trained models, averages their sigmoid
scores, and evaluates EER on a labeled dataset.  No new training needed.
"""

import argparse
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloaders import make_loader
from evaluation import calculate_eer
from model import CNN2D
from model_cnn1d import CNN1D


# ── helpers ─────────────────────────────────────────────────────────

def resolve_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(arch: str, checkpoint_path: str, device: str,
               in_features: int = 180, dropout: float = 0.2) -> torch.nn.Module:
    """Instantiate a model, load checkpoint weights, set to eval mode."""
    kwargs = {"in_features": in_features, "dropout": dropout}
    if arch == "cnn1d":
        model = CNN1D(**kwargs)
    else:
        model = CNN2D(**kwargs)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model


def collect_scores(model: torch.nn.Module, dataloader: DataLoader,
                   device: str, swap_tf: bool = True) -> List[float]:
    """Run inference, return per-sample sigmoid scores."""
    scores: List[float] = []
    with torch.no_grad():
        for features, _ in dataloader:
            features = features.to(device)
            if swap_tf:
                features = features.transpose(1, 2)
            logits = model(features).squeeze(-1)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    return scores


# ── main ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Ensemble-average sigmoid scores from multiple checkpoints."
    )
    p.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="arch:path pairs, e.g. cnn2d:checkpoints/final_robust/cnn2d_best.pt",
    )
    p.add_argument("--dev-features", default="data/dev/features.pkl")
    p.add_argument("--dev-labels", default="data/dev/labels.pkl")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--in-features", type=int, default=180)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--swap-tf", action="store_true", default=True)
    p.add_argument("--no-swap-tf", dest="swap_tf", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    dev_loader = make_loader(
        args.dev_features, args.dev_labels,
        batch_size=args.batch_size, shuffle=False,
    )

    # Collect ground-truth labels (same order for every model since shuffle=False)
    labels: List[float] = []
    for _, batch_labels in dev_loader:
        labels.extend(batch_labels.tolist())

    # Collect per-model scores
    all_scores: List[np.ndarray] = []
    model_names: List[str] = []

    for spec in args.checkpoints:
        arch, path = spec.split(":", 1)
        model = load_model(arch, path, device,
                           in_features=args.in_features,
                           dropout=args.dropout)
        scores = collect_scores(model, dev_loader, device,
                                swap_tf=args.swap_tf)
        scores_np = np.array(scores)
        all_scores.append(scores_np)
        model_names.append(f"{arch}:{path.split('/')[-1]}")

        eer, thr = calculate_eer(scores, labels)
        print(f"  {arch:6s}  {path}")
        print(f"         EER={eer:.6f}  threshold={thr:.6f}")

    # Ensemble: simple mean of sigmoid scores
    ensemble_scores = np.mean(all_scores, axis=0)
    eer, thr = calculate_eer(ensemble_scores.tolist(), labels)

    print(f"\n{'=' * 60}")
    print(f"Ensemble of {len(all_scores)} models")
    print(f"  EER      = {eer:.6f}")
    print(f"  threshold= {thr:.6f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
