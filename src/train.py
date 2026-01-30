import torch
import torch.nn as nn

from dataloaders import create_dataloaders
from evaluation import evaluate
from model import build_model


def train_one_epoch(model, dataloader, criterion, optimizer, device="cpu"):
    """
    One training epoch loop.
    """
    model.train()
    total_loss = 0.0
    total_count = 0

    for features, labels in dataloader:
        # TODO: move to device
        # features = features.to(device)
        # labels = labels.to(device)

        # TODO: forward pass
        # logits = model(features).squeeze(-1)

        # TODO: compute loss
        # loss = criterion(logits, labels)

        # TODO: backprop and step
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # TODO: track loss
        # total_loss += loss.item() * labels.size(0)
        # total_count += labels.size(0)
        pass

    avg_loss = (total_loss / total_count) if total_count > 0 else None
    return avg_loss


def main():
    # TODO: set paths for your data
    train_features_path = "data/train/features.pkl"
    train_labels_path = "data/train/labels.pkl"
    dev_features_path = "data/dev/features.pkl"
    dev_labels_path = "data/dev/labels.pkl"
    test_features_path = "data/test1/features.pkl"

    # TODO: set hyperparameters
    batch_size = 32
    num_workers = 2
    num_epochs = 5
    lr = 1e-3
    model_name = "mlp"  # "mlp", "cnn1d", or "cnn2d"

    # TODO: pick device
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Data
    train_loader, dev_loader, _ = create_dataloaders(
        train_features_path=train_features_path,
        train_labels_path=train_labels_path,
        dev_features_path=dev_features_path,
        dev_labels_path=dev_labels_path,
        test_features_path=test_features_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Model
    model = build_model(model_name)
    model.to(device)

    # Loss + optimizer (BCEWithLogitsLoss expects raw logits)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # TODO: training loop
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader,
                                     criterion, optimizer, device=device)
        dev_metrics, _, _ = evaluate(model, dev_loader,
                                     criterion=criterion, device=device)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss} "
            f"dev_loss={dev_metrics['avg_loss']} "
            f"dev_eer={dev_metrics['eer']}"
        )

    # TODO: save model checkpoint
    # torch.save(model.state_dict(), "checkpoints/mlp_baseline.pt")


if __name__ == "__main__":
    main()
