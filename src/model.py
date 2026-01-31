import torch
import torch.nn as nn


class MeanPoolMLP(nn.Module):
    """
    Baseline: mean (and/or std) pool over time, then MLP.
    Input x: (batch, time, features) e.g. (B, 180, 321)
    Output: logits of shape (B, 1)
    """

    def __init__(self, in_features=321, hidden_dim=128, dropout=0.2):
        super().__init__()
        # TODO: define your MLP layers (Linear/ReLU/Dropout, etc.)
        # Example:
        # self.fc1 = nn.Linear(in_features, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, 1)
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # TODO: mean pool over time dimension -> (B, features)
        pooled = x.mean(dim=1)
        # TODO: feed pooled features through your MLP
        logits = self.feature_extractor(pooled)
        return logits


class StatsPoolMLP(nn.Module):
    """
    Enhanced MLP: Uses Mean + Std + Max pooling to capture temporal statistics.
    Input x: (B, T, F)
    """

    def __init__(self, in_features=321, hidden_dim=128, dropout=0.2):
        super().__init__()
        # Input dim is 3x because we concatenate Mean, Std, and Max
        input_dim = in_features * 3

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (B, T, F)

        # 1. Mean: Captures average energy/tone
        mean_feat = x.mean(dim=1)

        # 2. Std: Captures instability/jitter (common in deepfakes)
        # Unbiased=False to match numpy default, though not critical
        std_feat = x.std(dim=1, unbiased=False)

        # 3. Max: Captures strong artifacts/outliers in specific frames
        max_feat, _ = x.max(dim=1)

        # Concatenate all stats: (B, 3 * F)
        pooled = torch.cat([mean_feat, std_feat, max_feat], dim=1)

        logits = self.feature_extractor(pooled)
        return logits


class CNN1D(nn.Module):
    """
    Safe 1D CNN baseline over time.
    Input x: (B, T, F) -> transpose to (B, F, T)
    """

    def __init__(self, in_channels=321, num_classes=1, dropout=0.2, pool_bins=1):
        super().__init__()
        if pool_bins < 1:
            raise ValueError("pool_bins must be >= 1")
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(pool_bins)
        self.classifier = nn.Linear(256 * pool_bins, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        logits = self.classifier(x)
        return logits


class CNN1DSpatial(nn.Module):
    """
    1D CNN with spatial (channel) dropout over time.
    Input x: (B, T, F) -> transpose to (B, F, T)
    """

    def __init__(self, in_channels=321, num_classes=1, dropout=0.2, pool_bins=1):
        super().__init__()
        if pool_bins < 1:
            raise ValueError("pool_bins must be >= 1")
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(dropout),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(pool_bins)
        self.classifier = nn.Linear(256 * pool_bins, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        logits = self.classifier(x)
        return logits


class CNN2D(nn.Module):
    """
    2D CNN on time x feature "image".
    Input x: (B, T, F) -> add channel dim to (B, 1, T, F)
    """

    def __init__(self, in_features=321, base_channels=32, num_classes=1, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 1)),
            nn.Dropout(dropout),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 1)),
            nn.Dropout(dropout),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(base_channels * 4 * in_features, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        # Preserve feature (frequency) detail; average over time only.
        x = x.mean(dim=2)
        x = x.flatten(1)
        logits = self.classifier(x)
        return logits


class CNN2DSpatial(nn.Module):
    """
    2D CNN with spatial (channel) dropout.
    Input x: (B, T, F) -> add channel dim to (B, 1, T, F)
    """

    def __init__(self, in_features=321, base_channels=32, num_classes=1, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(dropout),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(dropout),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(base_channels * 4 * in_features, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.mean(dim=2)
        x = x.flatten(1)
        logits = self.classifier(x)
        return logits


def build_model(name: str, **kwargs):
    """
    Simple factory to pick an architecture.
    name: "mlp", "stats_mlp", "cnn1d", or "cnn2d"
    """
    name = name.lower()
    if name == "mlp":
        return MeanPoolMLP(**kwargs)
    if name == "stats_mlp":
        return StatsPoolMLP(**kwargs)
    if name == "cnn1d":
        return CNN1D(**kwargs)
    if name == "cnn1d_spatial":
        return CNN1DSpatial(**kwargs)
    if name == "cnn2d":
        return CNN2D(**kwargs)
    if name == "cnn2d_spatial":
        return CNN2DSpatial(**kwargs)
    raise ValueError(f"Unknown model name: {name}")


if __name__ == "__main__":
    model = MeanPoolMLP()
    x = torch.randn(4, 180, 321)
    logits = model(x)
    print(logits.shape)  # should be torch.Size([4, 1])
