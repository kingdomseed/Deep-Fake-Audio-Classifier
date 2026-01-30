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


class CNN1D(nn.Module):
    """
    1D CNN over time. Treat features as channels.
    Input x: (B, T, F) -> transpose to (B, F, T)
    """

    def __init__(self, in_channels=321, num_classes=1):
        super().__init__()
        # TODO: define 1D conv blocks here (Conv1d/BatchNorm/ReLU/Dropout)
        # self.conv = nn.Sequential(...)
        # TODO: define classifier head
        # self.classifier = nn.Linear(...)
        pass

    def forward(self, x):
        # TODO: transpose to (B, F, T)
        # x = x.transpose(1, 2)
        # TODO: apply conv blocks
        # x = self.conv(x)
        # TODO: global pooling over time (e.g., mean or max)
        # x = x.mean(dim=-1)
        # TODO: classifier to logits
        # logits = self.classifier(x)
        # return logits
        raise NotImplementedError


class CNN2D(nn.Module):
    """
    2D CNN on time x feature "image".
    Input x: (B, T, F) -> add channel dim to (B, 1, T, F)
    """

    def __init__(self, num_classes=1):
        super().__init__()
        # TODO: define 2D conv blocks here (Conv2d/BatchNorm/ReLU/Dropout)
        # self.conv = nn.Sequential(...)
        # TODO: define classifier head
        # self.classifier = nn.Linear(...)
        pass

    def forward(self, x):
        # TODO: add channel dimension (B, 1, T, F)
        # x = x.unsqueeze(1)
        # TODO: apply conv blocks
        # x = self.conv(x)
        # TODO: global pooling or flatten
        # x = ...
        # TODO: classifier to logits
        # logits = self.classifier(x)
        # return logits
        raise NotImplementedError


def build_model(name: str, **kwargs):
    """
    Simple factory to pick an architecture.
    name: "mlp", "cnn1d", or "cnn2d"
    """
    name = name.lower()
    if name == "mlp":
        return MeanPoolMLP(**kwargs)
    if name == "cnn1d":
        return CNN1D(**kwargs)
    if name == "cnn2d":
        return CNN2D(**kwargs)
    raise ValueError(f"Unknown model name: {name}")


if __name__ == "__main__":
    model = MeanPoolMLP()
    x = torch.randn(4, 180, 321)
    logits = model(x)
    print(logits.shape)  # should be torch.Size([4, 1])
