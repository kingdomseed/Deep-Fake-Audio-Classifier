"""Generate hybrid ensemble predictions on the final test set and compare
with the existing supervised-only submission."""

import argparse
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dataset_cae import FeatureNormalizer
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


class FeatureOnlyDataset(Dataset):
    """Raw features dataset (no labels, no transforms)."""
    def __init__(self, features_df):
        self.features = features_df["features"].reset_index(drop=True)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features.iloc[idx].float()


class NormedFeatureDataset(Dataset):
    """Features with swap_tf + normalisation for the CAE."""
    def __init__(self, features_df, normalizer: FeatureNormalizer):
        self.features = features_df["features"].reset_index(drop=True)
        self.normalizer = normalizer
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        feat = self.features.iloc[idx].float()
        feat = feat.transpose(0, 1)                 # (180,321) -> (321,180)
        feat = self.normalizer.transform(feat)
        return feat


@torch.no_grad()
def get_supervised_scores(model, features_df, device, batch_size=32):
    ds = FeatureOnlyDataset(features_df)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    scores = []
    model.eval()
    for features in loader:
        features = features.to(device)
        features = features.transpose(1, 2)          # swap_tf
        logits = model(features).squeeze(-1)
        scores.extend(torch.sigmoid(logits).cpu().tolist())
    return np.array(scores)


@torch.no_grad()
def get_cae_scores(model, features_df, normalizer, device, batch_size=32):
    ds = NormedFeatureDataset(features_df, normalizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    mse_criterion = nn.MSELoss(reduction="none")
    all_mse = []
    model.eval()
    for features in loader:
        features = features.to(device)
        recon, _ = model(features)
        mse = mse_criterion(recon, features).view(features.size(0), -1).mean(1)
        all_mse.extend(mse.cpu().tolist())
    return np.array(all_mse)


def normalise_01(scores):
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def print_distribution(name, scores):
    print(f"\n  {name}")
    print(f"    min={scores.min():.6f}  max={scores.max():.6f}")
    print(f"    mean={scores.mean():.6f}  median={np.median(scores):.6f}")
    print(f"    std={scores.std():.6f}")
    print(f"    est real (>0.5): {(scores > 0.5).sum()}  "
          f"est fake (<=0.5): {(scores <= 0.5).sum()}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sup-checkpoint", required=True)
    p.add_argument("--cae-checkpoint", required=True)
    p.add_argument("--cae-normalizer", required=True)
    p.add_argument("--test-features", required=True,
                   help="Path to final test features.pkl")
    p.add_argument("--existing-submission", default=None,
                   help="Path to existing .pkl submission for comparison")
    p.add_argument("--alpha", type=float, default=0.80,
                   help="Hybrid weight: alpha*supervised + (1-alpha)*cae")
    p.add_argument("--out", default="prediction_hybrid.pkl",
                   help="Output prediction pkl")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    # Load features
    features_df = pd.read_pickle(args.test_features)
    print(f"Test set: {len(features_df)} samples")

    # Load supervised model
    sup_model = CNN2D(in_features=180, dropout=0.2).to(device)
    ckpt = torch.load(args.sup_checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        sup_model.load_state_dict(ckpt["model_state"])
    else:
        sup_model.load_state_dict(ckpt)

    # Load CAE
    cae_norm = FeatureNormalizer.load(args.cae_normalizer)
    cae_model = ConvAutoencoder().to(device)
    cae_ckpt = torch.load(args.cae_checkpoint, map_location=device)
    if isinstance(cae_ckpt, dict) and "model_state" in cae_ckpt:
        cae_model.load_state_dict(cae_ckpt["model_state"])
    else:
        cae_model.load_state_dict(cae_ckpt)

    # Score
    print("Running supervised inference...")
    sup_scores = get_supervised_scores(sup_model, features_df, device,
                                       args.batch_size)
    print("Running CAE inference...")
    cae_scores = get_cae_scores(cae_model, features_df, cae_norm, device,
                                args.batch_size)

    # Combine
    sup_norm = normalise_01(sup_scores)
    cae_norm_scores = normalise_01(cae_scores)
    hybrid = args.alpha * sup_norm + (1 - args.alpha) * cae_norm_scores

    # Save prediction pkl
    pred_df = pd.DataFrame({
        "uttid": features_df["uttid"].values,
        "predictions": hybrid,
    })
    pred_df.to_pickle(args.out)
    print(f"\nSaved hybrid predictions to {args.out}")

    # ── Distribution comparison ─────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Distribution Comparison")
    print_distribution("Supervised-only (sigmoid)", sup_scores)
    print_distribution("CAE-only (raw MSE, higher=real)", cae_scores)
    print_distribution(f"Hybrid (alpha={args.alpha})", hybrid)

    if args.existing_submission:
        with open(args.existing_submission, "rb") as f:
            existing = pickle.load(f)
        if isinstance(existing, dict) and "predictions" in existing:
            old_df = existing["predictions"]
        else:
            old_df = existing

        old_preds = old_df["predictions"].values
        print_distribution("Existing submission", old_preds)

        # Per-sample difference
        # Align by uttid
        merged = pd.merge(
            pred_df, old_df, on="uttid", suffixes=("_new", "_old")
        )
        diff = merged["predictions_new"].values - merged["predictions_old"].values
        print(f"\n  Per-sample diff (new - old):")
        print(f"    mean={diff.mean():.6f}  std={diff.std():.6f}")
        print(f"    min={diff.min():.6f}  max={diff.max():.6f}")

        # Agreement on class
        new_class = (merged["predictions_new"].values > 0.5).astype(int)
        old_class = (merged["predictions_old"].values > 0.5).astype(int)
        agree = (new_class == old_class).sum()
        print(f"    class agreement: {agree}/{len(merged)} "
              f"({100 * agree / len(merged):.1f}%)")
        disagree_idx = np.where(new_class != old_class)[0]
        if len(disagree_idx) > 0 and len(disagree_idx) <= 20:
            print(f"    disagreements:")
            for i in disagree_idx:
                row = merged.iloc[i]
                print(f"      {row['uttid']}: old={row['predictions_old']:.4f} "
                      f"new={row['predictions_new']:.4f}")
        elif len(disagree_idx) > 20:
            print(f"    {len(disagree_idx)} disagreements (showing first 10):")
            for i in disagree_idx[:10]:
                row = merged.iloc[i]
                print(f"      {row['uttid']}: old={row['predictions_old']:.4f} "
                      f"new={row['predictions_new']:.4f}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
