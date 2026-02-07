"""Hybrid ensemble: combine supervised CNN scores with CAE anomaly scores.

Sweeps alpha in:  final_score = alpha * supervised + (1-alpha) * anomaly
Reports best alpha and EER on the dev set.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloaders import make_loader
from dataset_cae import FeatureNormalizer, FullLabeledDataset
from evaluation import calculate_eer
from model import CNN2D
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
def get_supervised_scores(model, dataloader, device, swap_tf=True):
    """Get sigmoid scores from supervised CNN2D."""
    model.eval()
    scores, labels = [], []
    for features, batch_labels in dataloader:
        features = features.to(device)
        if swap_tf:
            features = features.transpose(1, 2)
        logits = model(features).squeeze(-1)
        scores.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(batch_labels.tolist())
    return np.array(scores), np.array(labels)


@torch.no_grad()
def get_cae_scores(model, dataloader, device):
    """Get normalised anomaly scores from CAE (higher = more bonafide)."""
    model.eval()
    mse_criterion = nn.MSELoss(reduction="none")
    all_mse = []
    for features, _ in dataloader:
        features = features.to(device)
        recon, _ = model(features)
        mse = mse_criterion(recon, features).view(features.size(0), -1).mean(1)
        all_mse.extend(mse.cpu().tolist())
    all_mse = np.array(all_mse)

    # The CAE on this dataset reconstructs fakes BETTER (lower MSE).
    # So raw MSE is already: higher = bonafide.  Use it directly.
    return all_mse


def normalise_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def parse_args():
    p = argparse.ArgumentParser(
        description="Hybrid ensemble: supervised + CAE anomaly scoring."
    )
    # Supervised model
    p.add_argument("--sup-checkpoint", required=True,
                   help="Path to supervised CNN2D checkpoint")
    p.add_argument("--sup-arch", default="cnn2d", choices=["cnn2d"])
    # CAE model
    p.add_argument("--cae-checkpoint", required=True,
                   help="Path to CAE checkpoint")
    p.add_argument("--cae-normalizer", required=True,
                   help="Path to CAE normalizer.pt")
    # Data
    p.add_argument("--dev-features", default="data/dev/features.pkl")
    p.add_argument("--dev-labels", default="data/dev/labels.pkl")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    # Alpha sweep
    p.add_argument("--alpha-steps", type=int, default=21,
                   help="Number of alpha values to try from 0 to 1")
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    # ── Load supervised model ───────────────────────────────────────
    sup_model = CNN2D(in_features=180, dropout=0.2).to(device)
    ckpt = torch.load(args.sup_checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        sup_model.load_state_dict(ckpt["model_state"])
    else:
        sup_model.load_state_dict(ckpt)
    sup_model.eval()

    # ── Load CAE model ──────────────────────────────────────────────
    cae_normalizer = FeatureNormalizer.load(args.cae_normalizer)
    cae_model = ConvAutoencoder().to(device)
    cae_ckpt = torch.load(args.cae_checkpoint, map_location=device)
    if isinstance(cae_ckpt, dict) and "model_state" in cae_ckpt:
        cae_model.load_state_dict(cae_ckpt["model_state"])
    else:
        cae_model.load_state_dict(cae_ckpt)
    cae_model.eval()

    # ── Supervised scores (uses regular dataloader with swap in loop) ──
    dev_loader = make_loader(args.dev_features, args.dev_labels,
                             batch_size=args.batch_size, shuffle=False)
    sup_scores, labels = get_supervised_scores(sup_model, dev_loader, device)
    sup_eer, _ = calculate_eer(sup_scores.tolist(), labels.tolist())
    print(f"Supervised-only  EER = {sup_eer:.6f}")

    # ── CAE scores (uses normalised dataset) ────────────────────────
    cae_ds = FullLabeledDataset(args.dev_features, args.dev_labels,
                                normalizer=cae_normalizer, swap_tf=True)
    cae_loader = DataLoader(cae_ds, batch_size=args.batch_size, shuffle=False)
    cae_scores = get_cae_scores(cae_model, cae_loader, device)
    cae_eer, _ = calculate_eer(cae_scores.tolist(), labels.tolist())
    print(f"CAE-only         EER = {cae_eer:.6f}")

    # ── Normalise both to [0, 1] for fair combination ───────────────
    sup_norm = normalise_scores(sup_scores)
    cae_norm = normalise_scores(cae_scores)

    # ── Alpha sweep ─────────────────────────────────────────────────
    alphas = np.linspace(0.0, 1.0, args.alpha_steps)
    best_eer, best_alpha = 1.0, 0.0

    print(f"\n{'alpha':>6s}  {'EER':>10s}")
    print("-" * 20)
    for alpha in alphas:
        combined = alpha * sup_norm + (1 - alpha) * cae_norm
        eer, _ = calculate_eer(combined.tolist(), labels.tolist())
        marker = " *" if eer < best_eer else ""
        print(f"  {alpha:.2f}    {eer:.6f}{marker}")
        if eer < best_eer:
            best_eer = eer
            best_alpha = alpha

    print(f"\n{'=' * 60}")
    print("Hybrid Ensemble Results")
    print(f"  Supervised-only EER: {sup_eer:.6f}")
    print(f"  CAE-only EER:        {cae_eer:.6f}")
    print(f"  Best hybrid EER:     {best_eer:.6f}  (alpha={best_alpha:.2f})")
    print(f"  alpha=1.0 means 100% supervised, alpha=0.0 means 100% CAE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
