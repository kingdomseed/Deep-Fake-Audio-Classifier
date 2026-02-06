import argparse
import os
import random
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from augmentation import (
    channel_drop,
    compose,
    gaussian_jitter,
    spec_augment,
    time_shift,
)
from dataloaders import make_loader
from evaluation import evaluate
from model import CNN2D
from model_cnn1d import CNN1D
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
    device: str = "mps",
    batch_context: BatchContext | None = None,
    augment_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    swap_tf: bool = False,
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
        augment_fn: Optional augmentation function to apply to features

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch_idx, (features, labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)

        if swap_tf:
            features = features.transpose(1, 2)

        # Apply augmentation if provided (training only)
        if augment_fn is not None:
            features = augment_fn(features)

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
        default="cnn2d",
        choices=["cnn2d", "cnn1d"],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--early-stop", type=int, default=0, help="patience in epochs (0 disables)")
    parser.add_argument(
        "--lr-scheduler",
        default="none",
        choices=["none", "plateau"],
        help="learning-rate scheduler to use (default: none)",
    )
    parser.add_argument(
        "--lr-scheduler-metric",
        default="dev_eer",
        choices=["dev_eer", "dev_loss"],
        help="metric to monitor for LR scheduling (default: dev_eer)",
    )
    parser.add_argument(
        "--lr-scheduler-factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau factor (default: 0.5)",
    )
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=2,
        help="ReduceLROnPlateau patience in epochs (default: 2)",
    )
    parser.add_argument(
        "--lr-scheduler-threshold",
        type=float,
        default=1e-4,
        help="ReduceLROnPlateau threshold (default: 1e-4)",
    )
    parser.add_argument(
        "--lr-scheduler-min-lr",
        type=float,
        default=1e-6,
        help="ReduceLROnPlateau minimum LR (default: 1e-6)",
    )
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu")
    parser.add_argument("--in-features", type=int, default=180)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional subfolder name under --checkpoint-dir for outputs (e.g. 20260206_cnn2d_mix).",
    )
    parser.add_argument("--no-rich", action="store_true", help="disable rich visualization (use basic tqdm instead)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--spec-augment",
        action="store_true",
        help="enable SpecAugment data augmentation during training",
    )
    parser.add_argument(
        "--time-mask-ratio",
        type=float,
        default=0.2,
        help="max ratio of time steps to mask (default: 0.2)",
    )
    parser.add_argument(
        "--feature-mask-ratio",
        type=float,
        default=0.1,
        help="max ratio of features to mask (default: 0.1)",
    )
    parser.add_argument(
        "--feature-mask",
        action="store_true",
        help="enable feature masking in addition to time masking",
    )

    # Additional robustness augmentations (training only)
    parser.add_argument(
        "--time-shift",
        action="store_true",
        help="enable random circular time shift on features during training",
    )
    parser.add_argument(
        "--time-shift-ratio",
        type=float,
        default=0.1,
        help="max time shift as ratio of T (default: 0.1)",
    )
    parser.add_argument(
        "--channel-drop",
        action="store_true",
        help="enable random channel/feature drop during training",
    )
    parser.add_argument(
        "--channel-drop-prob",
        type=float,
        default=0.1,
        help="drop prob per feature dim (default: 0.1)",
    )
    parser.add_argument(
        "--gaussian-jitter",
        action="store_true",
        help="enable small Gaussian feature noise during training",
    )
    parser.add_argument(
        "--gaussian-jitter-std",
        type=float,
        default=0.01,
        help="stddev of Gaussian feature noise (default: 0.01)",
    )

    # Calibration
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="label smoothing epsilon in [0, 0.5) (default: 0.0)",
    )

    parser.add_argument(
        "--debug-augment-stats",
        action="store_true",
        help=(
            "print feature stats before/after augmentation on the first batch"
        ),
    )
    swap_group = parser.add_mutually_exclusive_group()
    swap_group.add_argument(
        "--swap-tf",
        dest="swap_tf",
        action="store_true",
        help="swap time and feature dimensions (T <-> F) (default)",
    )
    swap_group.add_argument(
        "--no-swap-tf",
        dest="swap_tf",
        action="store_false",
        help="disable time/feature swap",
    )
    parser.set_defaults(swap_tf=True)
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

    checkpoint_root = args.checkpoint_dir
    if args.run_name:
        checkpoint_root = os.path.join(checkpoint_root, args.run_name)
    os.makedirs(checkpoint_root, exist_ok=True)
    best_path = os.path.join(checkpoint_root, f"{args.model}_best.pt")
    last_path = os.path.join(checkpoint_root, f"{args.model}_last.pt")

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
    if args.model == "cnn1d":
        model = CNN1D(
            in_features=args.in_features,
            dropout=args.dropout,
        )
    else:
        model = CNN2D(
            in_features=args.in_features,
            dropout=args.dropout,
        )
    model.to(device)

    # Loss + optimizer (BCEWithLogitsLoss expects raw logits)
    if not (0.0 <= args.label_smoothing < 0.5):
        raise ValueError("--label-smoothing must be in [0, 0.5)")

    def _maybe_smooth_labels(y: torch.Tensor) -> torch.Tensor:
        if args.label_smoothing <= 0:
            return y
        eps = float(args.label_smoothing)
        return y * (1.0 - eps) + 0.5 * eps

    bce = nn.BCEWithLogitsLoss()

    def criterion(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return bce(logits, _maybe_smooth_labels(y))
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

    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
            threshold=args.lr_scheduler_threshold,
            min_lr=args.lr_scheduler_min_lr,
        )

    # Setup augmentation if enabled
    augment_fns: list[Callable[[torch.Tensor], torch.Tensor]] = []
    if args.spec_augment:
        def _specaug_fn(features: torch.Tensor) -> torch.Tensor:
            return spec_augment(
                features,
                time_mask_ratio=args.time_mask_ratio,
                feature_mask_ratio=args.feature_mask_ratio,
                apply_time_mask=True,
                apply_feature_mask=args.feature_mask,
            )

        augment_fns.append(_specaug_fn)
        feature_mask_status = (
            f"{args.feature_mask_ratio:.2f}"
            if args.feature_mask
            else "disabled"
        )
        print(
            f"SpecAugment enabled: time_mask={args.time_mask_ratio:.2f}, "
            f"feature_mask={feature_mask_status}"
        )

    if args.time_shift:
        augment_fns.append(
            lambda x: time_shift(x, max_shift_ratio=args.time_shift_ratio)
        )
        print(
            f"Time shift enabled: max_shift_ratio={args.time_shift_ratio:.2f}"
        )

    if args.channel_drop:
        augment_fns.append(
            lambda x: channel_drop(x, drop_prob=args.channel_drop_prob)
        )
        print(f"Channel drop enabled: p={args.channel_drop_prob:.2f}")

    if args.gaussian_jitter:
        augment_fns.append(
            lambda x: gaussian_jitter(x, std=args.gaussian_jitter_std)
        )
        print(f"Gaussian jitter enabled: std={args.gaussian_jitter_std:.4f}")

    augment_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = (
        compose(*augment_fns) if augment_fns else None
    )

    if args.debug_augment_stats:
        def _stats(x: torch.Tensor) -> str:
            x_detached = x.detach()
            flat = x_detached.flatten()
            zeros = (flat == 0).float().mean().item()
            q01, q50, q99 = torch.quantile(
                flat,
                torch.tensor([0.01, 0.50, 0.99], device=flat.device),
            ).tolist()
            return (
                f"shape={tuple(x_detached.shape)} "
                f"min={x_detached.min().item():.4f} "
                f"q01={q01:.4f} "
                f"median={q50:.4f} "
                f"q99={q99:.4f} "
                f"max={x_detached.max().item():.4f} "
                f"mean={x_detached.mean().item():.4f} "
                f"std={x_detached.std().item():.4f} "
                f"zero%={zeros * 100:.4f}"
            )

        base_augment_fn = augment_fn

        printed = {"done": False}

        def _debug_augment_fn(features: torch.Tensor) -> torch.Tensor:
            # Note: by the time this hook runs, features are already in the
            # model-view orientation (swap_tf applied earlier in train loop).
            if not printed["done"]:
                print("[augment-stats] before:", _stats(features))
            out = (
                base_augment_fn(features)
                if base_augment_fn is not None
                else features
            )
            if not printed["done"]:
                print("[augment-stats] after: ", _stats(out))
                printed["done"] = True
            return out

        augment_fn = _debug_augment_fn

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
    best_train_loss = None
    best_dev_loss = None
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
                batch_context=batch_ctx,
                augment_fn=augment_fn,
                swap_tf=args.swap_tf,
            )

        # Evaluate
        dev_metrics_dict, _, _ = evaluate(
            model,
            dev_loader,
            criterion=criterion,
            device=device,
            swap_tf=args.swap_tf,
        )

        # Determine if this should become the new "best" checkpoint.
        #
        # Primary criterion: dev EER decreases.
        # Tie-breaker (edge case): dev EER is effectively unchanged, but both
        # train loss AND dev loss decrease vs the previous best checkpoint.
        #
        # Note: early stopping remains based on EER improvement only.
        is_best = False
        eer = dev_metrics_dict["eer"]
        dev_loss = dev_metrics_dict["avg_loss"]
        eer_tie_eps = 1e-4  # matches 4-decimal printing; treat smaller diffs as "no change"
        loss_improve_eps = 1e-6

        if eer is not None:
            if best_eer is None or eer < best_eer:
                is_best = True
                best_eer = eer
                best_train_loss = train_loss
                best_dev_loss = dev_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if (
                    best_eer is not None
                    and abs(eer - best_eer) <= eer_tie_eps
                    and train_loss is not None
                    and dev_loss is not None
                    and best_train_loss is not None
                    and best_dev_loss is not None
                    and train_loss < best_train_loss - loss_improve_eps
                    and dev_loss < best_dev_loss - loss_improve_eps
                ):
                    is_best = True
                    best_train_loss = train_loss
                    best_dev_loss = dev_loss

        if scheduler is not None:
            scheduler_metric = (
                dev_loss if args.lr_scheduler_metric == "dev_loss" else eer
            )
            if scheduler_metric is not None:
                scheduler.step(scheduler_metric)

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
            dev_loss=dev_loss,
            dev_eer=eer,
            is_best=is_best,
            improved=improved,
            epochs_no_improve=epochs_no_improve
        )

        # Display epoch end
        visualizer.on_epoch_end(metrics, prev_metrics)

        # Save checkpoint if this is the best model
        if is_best:
            save_checkpoint(model, optimizer, epoch, args, best_path, scheduler=scheduler)

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
    save_checkpoint(model, optimizer, last_epoch, args, last_path, scheduler=scheduler)


if __name__ == "__main__":
    main()
