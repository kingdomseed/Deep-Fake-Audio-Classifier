"""Training script for the Convolutional Autoencoder (anomaly detection).

Trains on bonafide-only data using MSE reconstruction loss.
Saves best checkpoint based on validation reconstruction loss.
Uses Rich live display with progress bars and epoch table.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_cae import BonafideDataset, FullLabeledDataset, build_normalizer
from model_cae import ConvAutoencoder
from training import save_checkpoint

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def resolve_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    batch_progress=None, batch_task_id=None):
    """One epoch of reconstruction training (bonafide-only)."""
    model.train()
    total_loss = 0.0
    total_count = 0

    for features in dataloader:
        features = features.to(device)
        reconstruction, _ = model(features)
        loss = criterion(reconstruction, features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        total_count += features.size(0)

        if batch_progress is not None and batch_task_id is not None:
            avg = total_loss / total_count if total_count > 0 else 0
            batch_progress.update(batch_task_id, advance=1,
                                  description=f"  [cyan]Train[/] MSE={avg:.6f}")

    return total_loss / total_count if total_count > 0 else None


@torch.no_grad()
def validate_reconstruction(model, dataloader, criterion, device,
                            batch_progress=None, batch_task_id=None):
    """Compute average reconstruction loss on bonafide validation data."""
    model.eval()
    total_loss = 0.0
    total_count = 0

    for features in dataloader:
        features = features.to(device)
        reconstruction, _ = model(features)
        loss = criterion(reconstruction, features)
        total_loss += loss.item() * features.size(0)
        total_count += features.size(0)

        if batch_progress is not None and batch_task_id is not None:
            avg = total_loss / total_count if total_count > 0 else 0
            batch_progress.update(batch_task_id, advance=1,
                                  description=f"  [yellow]Val[/]   MSE={avg:.6f}")

    return total_loss / total_count if total_count > 0 else None


def parse_args():
    p = argparse.ArgumentParser(description="Train CAE for anomaly detection.")
    p.add_argument("--train-features", default="data/train/features.pkl")
    p.add_argument("--train-labels", default="data/train/labels.pkl")
    p.add_argument("--dev-features", default="data/dev/features.pkl")
    p.add_argument("--dev-labels", default="data/dev/labels.pkl")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--early-stop", type=int, default=10,
                   help="patience in epochs (default: 10)")
    p.add_argument("--lr-scheduler-patience", type=int, default=7,
                   help="ReduceLROnPlateau patience in epochs (default: 7)")
    p.add_argument("--lr-scheduler-factor", type=float, default=0.5,
                   help="ReduceLROnPlateau factor (default: 0.5)")
    p.add_argument("--lr-scheduler-min-lr", type=float, default=1e-6,
                   help="ReduceLROnPlateau minimum LR (default: 1e-6)")
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--run-name", default="cae_anomaly",
                   help="Subfolder under --checkpoint-dir for outputs")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normalizer-path", default=None,
                   help="Path to saved normalizer (computed if not provided)")
    return p.parse_args()


def build_table(history_rows, early_stop_patience):
    table = Table(title="CAE Training Progress", show_lines=False)
    table.add_column("Epoch", justify="right", style="cyan", width=6)
    table.add_column("Train MSE", justify="right", width=12)
    table.add_column("Val MSE", justify="right", width=12)
    table.add_column("LR", justify="right", width=10)
    table.add_column("No Impr", justify="right", width=8)
    table.add_column("Best", justify="center", width=5)

    for row in history_rows[-20:]:
        best_marker = "[bold green]***[/]" if row["is_best"] else ""
        no_imp_style = (
            "[red]" if row["no_improve"] >= early_stop_patience - 2
            else "[yellow]" if row["no_improve"] >= 3
            else ""
        )
        table.add_row(
            str(row["epoch"]),
            f"{row['train_loss']:.6f}",
            f"{row['val_loss']:.6f}",
            f"{row['lr']:.2e}",
            f"{no_imp_style}{row['no_improve']}",
            best_marker,
        )
    return table


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    ckpt_dir = os.path.join(args.checkpoint_dir, args.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "cae_best.pt")
    last_path = os.path.join(ckpt_dir, "cae_last.pt")
    norm_path = os.path.join(ckpt_dir, "normalizer.pt")

    # ── Normalizer ──────────────────────────────────────────────────
    if args.normalizer_path and os.path.exists(args.normalizer_path):
        from dataset_cae import FeatureNormalizer
        normalizer = FeatureNormalizer.load(args.normalizer_path)
        norm_path = args.normalizer_path
    else:
        normalizer = build_normalizer(args.train_features, args.train_labels)
        normalizer.save(norm_path)

    # ── Data loaders ────────────────────────────────────────────────
    train_ds = BonafideDataset(args.train_features, args.train_labels,
                               normalizer=normalizer, swap_tf=True)
    val_ds = BonafideDataset(args.dev_features, args.dev_labels,
                             normalizer=normalizer, swap_tf=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    # ── Model ───────────────────────────────────────────────────────
    model = ConvAutoencoder(base_channels=args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        threshold=1e-4,
        min_lr=args.lr_scheduler_min_lr,
    )

    # ── Training loop ───────────────────────────────────────────────
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    console = Console() if HAS_RICH else None

    if console:
        console.print(Panel(
            f"[bold]CAE Training[/bold]\n"
            f"Device: {device}  |  Epochs: {args.epochs}  |  "
            f"Early stop: {args.early_stop}\n"
            f"LR: {args.lr}  |  Scheduler patience: {args.lr_scheduler_patience}  |  "
            f"Params: {n_params:,}\n"
            f"Train: {len(train_ds)} bonafide  |  Val: {len(val_ds)} bonafide",
            title="Config", border_style="blue",
        ))

        # Overall epoch progress bar
        epoch_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Epochs"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        epoch_task = epoch_progress.add_task("Epochs", total=args.epochs)

        # Per-epoch batch progress bar
        batch_progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
        )

        with Live(
            Group(epoch_progress, batch_progress, build_table([], args.early_stop)),
            console=console,
            refresh_per_second=4,
        ) as live:
            for epoch in range(1, args.epochs + 1):
                # Train
                train_task = batch_progress.add_task(
                    "  [cyan]Train[/]", total=n_train_batches)
                train_loss = train_one_epoch(
                    model, train_loader, criterion, optimizer, device,
                    batch_progress=batch_progress, batch_task_id=train_task)
                batch_progress.remove_task(train_task)

                # Validate
                val_task = batch_progress.add_task(
                    "  [yellow]Val[/]", total=n_val_batches)
                val_loss = validate_reconstruction(
                    model, val_loader, criterion, device,
                    batch_progress=batch_progress, batch_task_id=val_task)
                batch_progress.remove_task(val_task)

                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]["lr"]

                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    epochs_no_improve = 0
                    save_checkpoint(model, optimizer, epoch, args, best_path,
                                    scheduler=scheduler)
                else:
                    epochs_no_improve += 1

                history.append({
                    "epoch": epoch, "train_loss": train_loss,
                    "val_loss": val_loss, "lr": current_lr,
                    "no_improve": epochs_no_improve, "is_best": is_best,
                })

                epoch_progress.update(epoch_task, advance=1)
                live.update(Group(
                    epoch_progress, batch_progress,
                    build_table(history, args.early_stop),
                ))

                if args.early_stop and epochs_no_improve >= args.early_stop:
                    break

        if epochs_no_improve >= args.early_stop:
            console.print(
                f"\n[bold yellow]Early stopping at epoch {history[-1]['epoch']} "
                f"(no improvement in {args.early_stop} epochs)[/]"
            )
    else:
        # ── Plain fallback (no rich) ────────────────────────────────
        print(f"\nTraining on {device} for up to {args.epochs} epochs "
              f"(early stop patience={args.early_stop}, "
              f"scheduler patience={args.lr_scheduler_patience})")
        print("-" * 60)

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion,
                                         optimizer, device)
            val_loss = validate_reconstruction(model, val_loader, criterion,
                                               device)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                save_checkpoint(model, optimizer, epoch, args, best_path,
                                scheduler=scheduler)
            else:
                epochs_no_improve += 1

            history.append({
                "epoch": epoch, "train_loss": train_loss,
                "val_loss": val_loss, "lr": current_lr,
                "no_improve": epochs_no_improve, "is_best": is_best,
            })

            marker = " *" if is_best else ""
            print(f"  epoch {epoch:3d}  "
                  f"train_mse={train_loss:.6f}  "
                  f"val_mse={val_loss:.6f}  "
                  f"lr={current_lr:.2e}  "
                  f"no_improve={epochs_no_improve}{marker}")

            if args.early_stop and epochs_no_improve >= args.early_stop:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement in {args.early_stop} epochs)")
                break

    # ── Save final + summary ────────────────────────────────────────
    save_checkpoint(model, optimizer, history[-1]["epoch"], args, last_path,
                    scheduler=scheduler)

    if console:
        console.print(Panel(
            f"Best val MSE: [bold green]{best_val_loss:.6f}[/]  (epoch {best_epoch})\n"
            f"Checkpoints:  {best_path}\n"
            f"              {last_path}\n"
            f"Normalizer:   {norm_path}",
            title="Training Complete", border_style="green",
        ))
    else:
        print(f"\nBest val MSE: {best_val_loss:.6f} (epoch {best_epoch})")
        print(f"Checkpoints: {best_path}, {last_path}")
        print(f"Normalizer:  {norm_path}")


if __name__ == "__main__":
    main()
