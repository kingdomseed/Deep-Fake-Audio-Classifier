"""Dataset utilities for CAE anomaly detection training.

Provides:
 - BonafideDataset: returns only label==1 samples (for training the CAE).
 - FullDataset: returns (features, label) for both classes (for evaluation).
 - FeatureNormalizer: compute and apply per-feature-dim z-score normalisation
   (mean/std computed from bonafide training data only).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ── Normalisation ───────────────────────────────────────────────────

class FeatureNormalizer:
    """Per-feature-dim z-score normaliser.

    Compute stats from bonafide training data, then apply to any split.
    Works on (T, F) tensors — stats are computed over the T dimension.
    """

    def __init__(self):
        self.mean: torch.Tensor | None = None  # shape (F,)
        self.std: torch.Tensor | None = None   # shape (F,)

    def fit(self, features_list: list[torch.Tensor]) -> "FeatureNormalizer":
        """Compute mean/std from a list of (T, F) tensors."""
        # Stack everything along T to get (total_T, F)
        all_feats = torch.cat(features_list, dim=0)  # (sum_T, F)
        self.mean = all_feats.mean(dim=0)             # (F,)
        self.std = all_feats.std(dim=0).clamp(min=1e-8)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise a single (T, F) or (B, T, F) tensor."""
        if self.mean is None:
            raise RuntimeError("Call .fit() first")
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def save(self, path: str) -> None:
        torch.save({"mean": self.mean, "std": self.std}, path)

    @classmethod
    def load(cls, path: str) -> "FeatureNormalizer":
        obj = cls()
        data = torch.load(path, map_location="cpu")
        obj.mean = data["mean"]
        obj.std = data["std"]
        return obj


# ── Datasets ────────────────────────────────────────────────────────

class BonafideDataset(Dataset):
    """Training dataset: bonafide (label==1) samples only.

    Returns a single tensor per sample (no label needed for reconstruction).
    Optionally swaps to (T, F) and applies FeatureNormalizer.
    """

    def __init__(self, features_path: str, labels_path: str,
                 normalizer: FeatureNormalizer | None = None,
                 swap_tf: bool = True):
        features_df = pd.read_pickle(features_path)
        labels_df = pd.read_pickle(labels_path)
        merged = pd.merge(features_df, labels_df, on="uttid", how="inner")

        # Keep only bonafide
        bonafide = merged[merged["label"] == 1].reset_index(drop=True)
        self.features = bonafide["features"].tolist()
        self.normalizer = normalizer
        self.swap_tf = swap_tf

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx].float()          # (180, 321) stored
        if self.swap_tf:
            feat = feat.transpose(0, 1)             # -> (321, 180)
        if self.normalizer is not None:
            feat = self.normalizer.transform(feat)
        return feat


class FullLabeledDataset(Dataset):
    """Evaluation dataset: both classes, returns (features, label).

    Optionally swaps to (T, F) and applies FeatureNormalizer.
    """

    def __init__(self, features_path: str, labels_path: str,
                 normalizer: FeatureNormalizer | None = None,
                 swap_tf: bool = True):
        features_df = pd.read_pickle(features_path)
        labels_df = pd.read_pickle(labels_path)
        merged = pd.merge(features_df, labels_df, on="uttid", how="inner")
        merged = merged.reset_index(drop=True)
        self.features = merged["features"].tolist()
        self.labels = merged["label"].tolist()
        self.normalizer = normalizer
        self.swap_tf = swap_tf

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx].float()
        if self.swap_tf:
            feat = feat.transpose(0, 1)             # -> (321, 180)
        if self.normalizer is not None:
            feat = self.normalizer.transform(feat)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feat, label


def build_normalizer(features_path: str, labels_path: str,
                     swap_tf: bool = True) -> FeatureNormalizer:
    """Compute normalisation stats from bonafide training data.

    Args:
        swap_tf: if True (default), transpose features to (T, F) before
                 computing stats so the normaliser works per-feature-dim.
    """
    features_df = pd.read_pickle(features_path)
    labels_df = pd.read_pickle(labels_path)
    merged = pd.merge(features_df, labels_df, on="uttid", how="inner")
    bonafide = merged[merged["label"] == 1]

    feat_list = []
    for _, row in bonafide.iterrows():
        feat = row["features"].float()
        if swap_tf:
            feat = feat.transpose(0, 1)  # (180,321) -> (321,180)
        feat_list.append(feat)

    normalizer = FeatureNormalizer().fit(feat_list)
    return normalizer


if __name__ == "__main__":
    print("Building normalizer from bonafide training data...")
    norm = build_normalizer("data/train/features.pkl", "data/train/labels.pkl")
    print(f"  mean shape: {norm.mean.shape}, range: [{norm.mean.min():.2f}, {norm.mean.max():.2f}]")
    print(f"  std  shape: {norm.std.shape},  range: [{norm.std.min():.4f}, {norm.std.max():.2f}]")

    print("\nBonafideDataset (with normalizer)...")
    ds = BonafideDataset("data/train/features.pkl", "data/train/labels.pkl",
                         normalizer=norm)
    print(f"  size: {len(ds)}")
    sample = ds[0]
    print(f"  sample shape: {sample.shape}, mean: {sample.mean():.4f}, std: {sample.std():.4f}")

    print("\nFullLabeledDataset (with normalizer)...")
    ds_full = FullLabeledDataset("data/dev/features.pkl", "data/dev/labels.pkl",
                                 normalizer=norm)
    print(f"  size: {len(ds_full)}")
    feat, lbl = ds_full[0]
    print(f"  feat shape: {feat.shape}, label: {lbl}")
