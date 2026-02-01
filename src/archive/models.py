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


class CNN1D(nn.Module):
    """
    Archived: 1D CNN baseline over time.
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


class CNN2DSpatial(nn.Module):
    """
    Archived: 2D CNN with spatial (channel) dropout.
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


class CRNN(nn.Module):
    """
    Archived: CRNN with CNN front-end + GRU back-end.
    Input x: (B, T, F)
    """

    def __init__(self, in_features=321, base_channels=32, rnn_hidden=128, num_classes=1, dropout=0.3):
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
        )
        self.rnn = nn.GRU(
            input_size=base_channels * 2 * in_features,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Linear(rnn_hidden, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, T, F)
        x = self.conv(x)    # (B, C, T', F)
        x = x.permute(0, 2, 1, 3)  # (B, T', C, F)
        x = x.flatten(2)          # (B, T', C*F)
        out, _ = self.rnn(x)      # (B, T', H)
        last = out[:, -1, :]      # (B, H)
        logits = self.classifier(last)
        return logits


class CRNN2(nn.Module):
    """
    Archived: CRNN with a 2-layer GRU back-end.
    Input x: (B, T, F)
    """

    def __init__(self, in_features=321, base_channels=32, rnn_hidden=128, num_classes=1, dropout=0.3):
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
        )
        self.rnn = nn.GRU(
            input_size=base_channels * 2 * in_features,
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.classifier = nn.Linear(rnn_hidden, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(0, 2, 1, 3)
        x = x.flatten(2)
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        logits = self.classifier(last)
        return logits
