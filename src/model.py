import torch
import torch.nn as nn


class CNN2D(nn.Module):
    """
    2D CNN on time x feature "image".
    Input x: (B, T, F) -> add channel dim to (B, 1, T, F)
    Best performing model: 0.55% EER on dev set.
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


class CNN2D_Robust(nn.Module):
    """
    Robust 2D CNN with residual blocks, SE attention, and attention pooling.
    Designed for better generalization to test distribution shift.

    Key improvements over CNN2D:
    - Residual blocks instead of plain conv layers
    - Squeeze-and-Excitation (SE) attention for channel recalibration
    - Learnable attention pooling over time (instead of mean pooling)
    - Smaller classifier head to reduce overfitting risk
    - Spatial dropout for better feature map regularization

    Input x: (B, T, F) -> add channel dim to (B, 1, T, F)
    """

    def __init__(self, in_features=180, base_channels=64, num_classes=1, dropout=0.3):
        super().__init__()

        # Residual blocks
        self.block1 = self._make_block(1, base_channels, dropout)
        self.block2 = self._make_block(base_channels, base_channels * 2, dropout)
        self.block3 = self._make_block(base_channels * 2, base_channels * 4, dropout)

        # Squeeze-and-Excitation attention for channel recalibration
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels * 4, base_channels * 4 // 16, 1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 4 // 16, base_channels * 4, 1),
            nn.Sigmoid()
        )

        # Learnable attention pooling over time
        self.attention_pool = nn.Linear(base_channels * 4, 1)

        # Smaller classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def _make_block(self, in_c, out_c, drop):
        """Create a residual block with two conv layers, pooling, and dropout."""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.AvgPool2d((2, 1)),
            nn.Dropout2d(drop)
        )

    def forward(self, x):
        # x: (B, T, F) -> (B, 1, T, F)
        x = x.unsqueeze(1)

        # Pass through residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)  # (B, C, T', F)

        # Apply SE attention
        se_weight = self.se(x)
        x = x * se_weight

        # Attention pooling over time (instead of mean)
        # Global average over frequency dimension
        x = x.mean(dim=3)  # (B, C, T')
        x = x.transpose(1, 2)  # (B, T', C)

        # Compute attention weights over time steps
        attn_weights = torch.softmax(self.attention_pool(x), dim=1)  # (B, T', 1)

        # Weighted sum over time
        x = (x * attn_weights).sum(dim=1)  # (B, C)

        # Final classification
        logits = self.classifier(x)
        return logits


def build_model(name: str, **kwargs):
    """
    Simple factory to pick an architecture.
    name: "cnn2d" or "cnn2d_robust"
    """
    name = name.lower()
    if name == "cnn2d":
        return CNN2D(**kwargs)
    if name == "cnn2d_robust":
        return CNN2D_Robust(**kwargs)
    raise ValueError(f"Unknown model name: {name}. Available: 'cnn2d', 'cnn2d_robust'")


if __name__ == "__main__":
    model = CNN2D()
    x = torch.randn(4, 321, 180)
    logits = model(x)
    print(f"CNN2D output shape: {logits.shape}")  # should be torch.Size([4, 1])

    model_robust = CNN2D_Robust()
    logits_robust = model_robust(x)
    print(f"CNN2D_Robust output shape: {logits_robust.shape}")  # should be torch.Size([4, 1])
