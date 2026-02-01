import torch
import torch.nn as nn


class CNN2D(nn.Module):
    """
    2D CNN over the time x feature grid.
    Input x: (B, T, F) -> add channel dim to (B, 1, T, F)
    Best-performing model so far on swapped orientation (T=321, F=180).
    """

    def __init__(self, in_features=180, base_channels=32, num_classes=1, dropout=0.2):
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


if __name__ == "__main__":
    model = CNN2D()
    x = torch.randn(4, 321, 180)
    logits = model(x)
    print(f"CNN2D output shape: {logits.shape}")  # should be torch.Size([4, 1])
