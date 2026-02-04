import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    1D CNN over the time dimension.
    Treats Frequency as Channels.
    Input x: (B, T, F) -> transpose to (B, F, T)
    """

    def __init__(self, in_features=180, base_channels=32, num_classes=1, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            # Layer 1
            # Input channels = in_features (Frequency dim, e.g., 180)
            nn.Conv1d(in_features, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(),
        )
        # Global Average Pooling over time (reduces Time dim to 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x):
        # Input x is (Batch, Time, Freq)
        # We need (Batch, Freq, Time) for Conv1d to slide across Time
        x = x.transpose(1, 2)

        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    model = CNN1D()
    x = torch.randn(4, 321, 180)
    logits = model(x)
    print(f"CNN1D output shape: {logits.shape}")
