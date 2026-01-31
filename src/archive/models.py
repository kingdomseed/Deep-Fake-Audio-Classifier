import torch
import torch.nn as nn


class MeanPoolMLP(nn.Module):
    """
    Archived baseline: mean pool over time, then MLP.
    """

    def __init__(self, in_features=321, hidden_dim=128, dropout=0.2):
        super().__init__()
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
        pooled = x.mean(dim=1)
        logits = self.feature_extractor(pooled)
        return logits


class StatsPoolMLP(nn.Module):
    """
    Archived baseline: mean + std + max pooling, then MLP.
    """

    def __init__(self, in_features=321, hidden_dim=128, dropout=0.2):
        super().__init__()
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
        mean_feat = x.mean(dim=1)
        std_feat = x.std(dim=1, unbiased=False)
        max_feat, _ = x.max(dim=1)
        pooled = torch.cat([mean_feat, std_feat, max_feat], dim=1)
        logits = self.feature_extractor(pooled)
        return logits


class CNN1DSpatial(nn.Module):
    """
    Archived: 1D CNN with spatial (channel) dropout over time.
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
