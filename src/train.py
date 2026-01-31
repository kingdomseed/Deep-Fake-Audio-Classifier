import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn

from dataloaders import make_loader
from evaluation import evaluate
from model import build_model
from training import save_checkpoint
from visualizers import (
    BatchContext,
    BatchMetrics,
    EpochMetrics,
    TrainingConfig,
    create_visualizer,
)


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device: str = "cpu",
    batch_context: BatchContext | None = None,
):
    """
    Unified training epoch loop that works with any visualizer.

    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss criterion
        optimizer: Optimizer
        device: Device to train on
        batch_context: Optional BatchContext from visualizer.on_epoch_start()

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch_idx, (features, labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features).squeeze(-1)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_count += labels.size(0)

        # Update visualizer if batch_context provided
        if batch_context is not None and total_count > 0:
            metrics = BatchMetrics(
                batch_idx=batch_idx,
                running_loss=total_loss / total_count,
                batch_size=labels.size(0)
            )
            batch_context.update_batch(metrics)

    avg_loss = (total_loss / total_count) if total_count > 0 else None
    return avg_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for audio deepfake detection.")
    parser.add_argument("--train-features", default="data/train/features.pkl")
    parser.add_argument("--train-labels", default="data/train/labels.pkl")
    parser.add_argument("--dev-features", default="data/dev/features.pkl")
    parser.add_argument("--dev-labels", default="data/dev/labels.pkl")
    parser.add_argument(
        "--model",
        default="mlp",
        choices=["mlp", "stats_mlp", "cnn1d", "cnn1d_spatial", "cnn2d", "cnn2d_spatial"],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--early-stop", type=int, default=0, help="patience in epochs (0 disables)")
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu")
    parser.add_argument("--in-features", type=int, default=321)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pool-bins", type=int, default=1, help="cnn1d pooling bins (1 = global avg)")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--no-rich", action="store_true", help="disable rich visualization (use basic tqdm instead)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = args.device

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(args.checkpoint_dir, f"{args.model}_best.pt")
    last_path = os.path.join(args.checkpoint_dir, f"{args.model}_last.pt")

    train_loader = make_loader(
        args.train_features,
        args.train_labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    dev_loader = make_loader(
        args.dev_features,
        args.dev_labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # Model
    model_kwargs = {}
    if args.model in {"mlp", "stats_mlp"}:
        model_kwargs = {
            "in_features": args.in_features,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        }
    elif args.model in {"cnn1d", "cnn1d_spatial"}:
        model_kwargs = {
            "in_channels": args.in_features,
            "dropout": args.dropout,
            "pool_bins": args.pool_bins,
        }
    elif args.model in {"cnn2d", "cnn2d_spatial"}:
        model_kwargs = {
            "in_features": args.in_features,
            "dropout": args.dropout,
        }
    model = build_model(args.model, **model_kwargs)
    model.to(device)

    # Loss + optimizer (BCEWithLogitsLoss expects raw logits)
    criterion = nn.BCEWithLogitsLoss()
    use_adamw = args.model.startswith("cnn")
    weight_decay = args.weight_decay
    if use_adamw and weight_decay == 0.0:
        weight_decay = 0.01
    if weight_decay > 0:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create visualizer (Rich by default, tqdm if --no-rich)
    visualizer_type = "tqdm" if args.no_rich else "rich"
    visualizer = create_visualizer(visualizer_type)

    # Build training configuration
    config = TrainingConfig(
        device=device,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop,
        in_features=args.in_features,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    # Display training start
    visualizer.on_training_start(config)

    # Training loop
    best_eer = None
    prev_metrics = None
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        # Train with visualizer context
        with visualizer.on_epoch_start(epoch, len(train_loader)) as batch_ctx:
            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device=device,
                batch_context=batch_ctx
            )

        # Evaluate
        dev_metrics_dict, _, _ = evaluate(
            model,
            dev_loader,
            criterion=criterion,
            device=device
        )

        # Determine if this is the best EER
        is_best = False
        if dev_metrics_dict["eer"] is not None:
            if best_eer is None or dev_metrics_dict["eer"] < best_eer:
                is_best = True
                best_eer = dev_metrics_dict["eer"]
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        # Build epoch metrics
        improved = (
            prev_metrics is not None
            and prev_metrics.dev_eer is not None
            and dev_metrics_dict["eer"] is not None
            and dev_metrics_dict["eer"] < prev_metrics.dev_eer
        )
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            dev_loss=dev_metrics_dict["avg_loss"],
            dev_eer=dev_metrics_dict["eer"],
            is_best=is_best,
            improved=improved,
            epochs_no_improve=epochs_no_improve
        )

        # Display epoch end
        visualizer.on_epoch_end(metrics, prev_metrics)

        # Save checkpoint if this is the best model
        if is_best:
            save_checkpoint(model, optimizer, epoch, args, best_path)

        # Update history and prev_metrics
        history.append(metrics)
        prev_metrics = metrics

        # Check early stopping
        if args.early_stop and epochs_no_improve >= args.early_stop:
            print(
                f"\nEarly stopping at epoch {epoch} "
                f"(no improvement in {args.early_stop} epochs)"
            )
            break

    # Display training end
    visualizer.on_training_end(history)

    # Save final checkpoint at the last completed epoch
    last_epoch = history[-1].epoch if history else 0
    save_checkpoint(model, optimizer, last_epoch, args, last_path)


if __name__ == "__main__":
    main()
