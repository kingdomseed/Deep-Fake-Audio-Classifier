import random
import torch


def time_shift(
    features: torch.Tensor, max_shift_ratio: float = 0.1
) -> torch.Tensor:
    """Randomly circular-shift along the time dimension.

    Args:
        features: [B, T, F] tensor
        max_shift_ratio: max absolute shift as a ratio of T (default 0.1)

    Returns:
        Shifted features (same shape)
    """
    if max_shift_ratio <= 0:
        return features

    B, T, F = features.shape
    if T <= 1:
        return features

    max_shift = int(T * max_shift_ratio)
    if max_shift < 1:
        return features

    shift = random.randint(-max_shift, max_shift)
    if shift == 0:
        return features
    return torch.roll(features, shifts=shift, dims=1)


def channel_drop(
    features: torch.Tensor, drop_prob: float = 0.1
) -> torch.Tensor:
    """Randomly zero-out feature channels (a.k.a. input channel dropout).

    Args:
        features: [B, T, F] tensor
        drop_prob: probability of dropping each feature dim (default 0.1)

    Returns:
        Features with some channels zeroed (same shape)
    """
    if drop_prob <= 0:
        return features

    B, T, F = features.shape
    # mask shape [1, 1, F] broadcast across batch/time
    keep = (
        torch.rand((1, 1, F), device=features.device) >= drop_prob
    ).to(features.dtype)
    return features * keep


def gaussian_jitter(features: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    """Add small Gaussian noise in feature space.

    Args:
        features: [B, T, F] tensor
        std: noise stddev (default 0.01)

    Returns:
        Noisy features (same shape)
    """
    if std <= 0:
        return features
    noise = torch.randn_like(features) * std
    return features + noise


def compose(*fns):
    def _apply(x: torch.Tensor) -> torch.Tensor:
        for fn in fns:
            if fn is not None:
                x = fn(x)
        return x

    return _apply


def time_mask(features, max_mask_ratio=0.2, min_mask_ratio=0.05):
    """
    SpecAugment time masking: randomly zero out a contiguous segment of time steps.

    This forces the model to be robust to missing temporal information, preventing
    overfitting to specific time locations where artifacts appear.

    Args:
        features: [B, T, F] tensor (batch, time, features)
        max_mask_ratio: Maximum proportion of time to mask (default 0.2 = 20%)
        min_mask_ratio: Minimum proportion of time to mask (default 0.05 = 5%)

    Returns:
        Masked features (same shape as input)

    Example:
        For T=321, max_mask_ratio=0.2:
        - Might mask frames 70-134 (64 frames = 20%)
        - Or frames 110-119 (9 frames = 5%)
        - Different random segment each call
    """
    B, T, F = features.shape

    # Random mask length between min and max ratio
    mask_len = int(T * random.uniform(min_mask_ratio, max_mask_ratio))

    # Ensure mask_len is at least 1 and doesn't exceed sequence length
    mask_len = max(1, min(mask_len, T - 1))

    # Random start position
    start = random.randint(0, T - mask_len)

    # Clone to avoid modifying in-place
    features = features.clone()

    # Zero out the time segment for all features
    features[:, start:start+mask_len, :] = 0

    return features


def feature_mask(features, max_mask_ratio=0.1, min_mask_ratio=0.02):
    """
    SpecAugment feature masking: randomly zero out a contiguous segment of feature dimensions.

    This forces the model to be robust to missing frequency information, preventing
    overfitting to specific frequency bands.

    Args:
        features: [B, T, F] tensor (batch, time, features)
        max_mask_ratio: Maximum proportion of features to mask (default 0.1 = 10%)
        min_mask_ratio: Minimum proportion of features to mask (default 0.02 = 2%)

    Returns:
        Masked features (same shape as input)

    Example:
        For F=180, max_mask_ratio=0.1:
        - Might mask features 60-78 (18 features = 10%)
        - Or features 120-123 (3 features = 2%)
        - Different random segment each call
    """
    B, T, F = features.shape

    # Random mask length between min and max ratio
    mask_len = int(F * random.uniform(min_mask_ratio, max_mask_ratio))

    # Ensure mask_len is at least 1 and doesn't exceed feature dimension
    mask_len = max(1, min(mask_len, F - 1))

    # Random start position
    start = random.randint(0, F - mask_len)

    # Clone to avoid modifying in-place
    features = features.clone()

    # Zero out the feature segment for all time steps
    features[:, :, start:start+mask_len] = 0

    return features


def spec_augment(features, time_mask_ratio=0.2, feature_mask_ratio=0.1,
                 apply_time_mask=True, apply_feature_mask=False):
    """
    Combined SpecAugment with both time and feature masking.

    Args:
        features: [B, T, F] tensor
        time_mask_ratio: Max ratio for time masking
        feature_mask_ratio: Max ratio for feature masking
        apply_time_mask: Whether to apply time masking (default True)
        apply_feature_mask: Whether to apply feature masking (default False)

    Returns:
        Augmented features
    """
    if apply_time_mask:
        features = time_mask(features, max_mask_ratio=time_mask_ratio)

    if apply_feature_mask:
        features = feature_mask(features, max_mask_ratio=feature_mask_ratio)

    return features


if __name__ == "__main__":
    # Quick test
    print("Testing SpecAugment...")

    # Create dummy features [batch=4, time=321, features=180]
    x = torch.randn(4, 321, 180)

    print(f"\nOriginal shape: {x.shape}")
    print(f"Original mean: {x.mean().item():.4f}")
    print(f"Original std: {x.std().item():.4f}")

    # Test time masking
    x_time = time_mask(x.clone(), max_mask_ratio=0.2)
    print(f"\nAfter time masking:")
    print(f"  Shape: {x_time.shape}")
    print(f"  Mean: {x_time.mean().item():.4f} (should be closer to 0)")
    print(f"  Proportion of zeros: {(x_time == 0).float().mean().item():.4f}")

    # Test feature masking
    x_feat = feature_mask(x.clone(), max_mask_ratio=0.1)
    print(f"\nAfter feature masking:")
    print(f"  Shape: {x_feat.shape}")
    print(f"  Mean: {x_feat.mean().item():.4f} (should be closer to 0)")
    print(f"  Proportion of zeros: {(x_feat == 0).float().mean().item():.4f}")

    # Test combined
    x_both = spec_augment(x.clone(), apply_time_mask=True, apply_feature_mask=True)
    print(f"\nAfter both time and feature masking:")
    print(f"  Shape: {x_both.shape}")
    print(f"  Mean: {x_both.mean().item():.4f}")
    print(f"  Proportion of zeros: {(x_both == 0).float().mean().item():.4f}")

    print("\nSpecAugment tests passed!")
