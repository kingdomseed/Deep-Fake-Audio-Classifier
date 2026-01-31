"""Checkpoint utilities for saving and loading model state."""

import torch
from pathlib import Path
from typing import Any, Dict, Optional


def build_config_dict(args) -> Dict[str, Any]:
    """Build configuration dictionary from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Dictionary containing training configuration
    """
    return {
        "model_name": args.model,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "in_features": args.in_features,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "pool_bins": getattr(args, "pool_bins", None),
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args,
    path: str
) -> None:
    """Save model checkpoint with training state.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch number
        args: Command-line arguments containing configuration
        path: Path to save checkpoint
    """
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "config": build_config_dict(args),
    }

    # Ensure parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Load checkpoint from file.

    Args:
        path: Path to checkpoint file
        model: Optional model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load checkpoint to

    Returns:
        Dictionary containing checkpoint contents

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    if model is not None and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint
