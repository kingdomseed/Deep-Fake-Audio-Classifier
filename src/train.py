import os
import torch
import torch.nn as nn

from dataloaders import make_loader
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
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features).squeeze(-1)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_count += labels.size(0)

    avg_loss = (total_loss / total_count) if total_count > 0 else None
    return avg_loss


def main():
    train_features_path = "data/train/features.pkl"
    train_labels_path = "data/train/labels.pkl"
    dev_features_path = "data/dev/features.pkl"
    dev_labels_path = "data/dev/labels.pkl"

    batch_size = 32
    num_workers = 2
    num_epochs = 5
    lr = 1e-3
    model_name = "mlp"  # "mlp", "cnn1d", or "cnn2d"

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
    last_path = os.path.join(checkpoint_dir, f"{model_name}_last.pt")

    train_loader = make_loader(
        train_features_path,
        train_labels_path,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    dev_loader = make_loader(
        dev_features_path,
        dev_labels_path,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    # Model
    model = build_model(model_name)
    model.to(device)

    # Loss + optimizer (BCEWithLogitsLoss expects raw logits)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_eer = None
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

        if dev_metrics["eer"] is not None:
            if best_eer is None or dev_metrics["eer"] < best_eer:
                best_eer = dev_metrics["eer"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "config": {
                            "model_name": model_name,
                            "batch_size": batch_size,
                            "num_workers": num_workers,
                            "lr": lr,
                        },
                    },
                    best_path,
                )

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": num_epochs,
            "config": {
                "model_name": model_name,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "lr": lr,
            },
        },
        last_path,
    )


if __name__ == "__main__":
    main()
