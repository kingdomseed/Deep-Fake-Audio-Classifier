import argparse
import os
import torch
import torch.nn as nn

from dataloaders import make_loader
from evaluation import evaluate
from model import build_model
from tqdm import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, device="cpu", desc=None):
    """
    One training epoch loop.
    """
    model.train()
    total_loss = 0.0
    total_count = 0

    batch_iter = tqdm(dataloader, desc=desc, leave=False, ncols=80, mininterval=1.0)
    for features, labels in batch_iter:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features).squeeze(-1)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_count += labels.size(0)
        if total_count > 0:
            # Show running average loss in progress bar
            batch_iter.set_postfix(loss=f"{total_loss / total_count:.4f}")

    avg_loss = (total_loss / total_count) if total_count > 0 else None
    return avg_loss


def format_metrics(epoch, train_loss, dev_loss, dev_eer, prev_eer=None, is_best=False):
    """
    Format training metrics with improvement indicators.

    Args:
        epoch: Current epoch number
        train_loss: Training loss
        dev_loss: Development loss
        dev_eer: Development EER
        prev_eer: Previous epoch's EER (for comparison)
        is_best: Whether this is the best EER so far

    Returns:
        Formatted string with metrics and indicators
    """
    # Format basic metrics
    metrics_str = (
        f"Epoch {epoch}: "
        f"train={train_loss:.4f}  "
        f"dev={dev_loss:.4f}  "
        f"eer={dev_eer:.4f}"
    )

    # Add improvement indicator
    if prev_eer is not None:
        if dev_eer < prev_eer:
            metrics_str += " ↓"
        elif dev_eer > prev_eer:
            metrics_str += " ↑"
        else:
            metrics_str += " ="

    # Add BEST marker
    if is_best:
        metrics_str += " BEST"

    return metrics_str


def train_one_epoch_rich(model, dataloader, criterion, optimizer, device="cpu", progress=None, task_id=None):
    """
    One training epoch loop with Rich progress tracking.
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

        if progress is not None and task_id is not None:
            progress.update(task_id, advance=1, loss=f"{total_loss / total_count:.4f}")

    avg_loss = (total_loss / total_count) if total_count > 0 else None
    return avg_loss


def train_with_rich(model, train_loader, dev_loader, criterion, optimizer, device, args, best_path, last_path):
    """
    Training loop with Rich visualization.
    """
    try:
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        print("Rich library not found. Please install it with: pip install rich")
        print("Falling back to standard tqdm output...")
        return False

    console = Console()

    # Display training configuration
    console.print(f"[bold cyan]Training Configuration[/bold cyan]")
    console.print(f"Device: [yellow]{device}[/yellow]")
    console.print(f"Model: [yellow]{args.model}[/yellow]")
    console.print(f"Epochs: [yellow]{args.epochs}[/yellow]")
    console.print()

    best_eer = None
    prev_eer = None
    patience = args.early_stop
    epochs_no_improve = 0

    # Track training history for final summary
    history = []

    # Create progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:

        epoch_task = progress.add_task("[cyan]Training", total=args.epochs)

        for epoch in range(1, args.epochs + 1):
            # Train
            batch_task = progress.add_task(f"[green]Epoch {epoch}/{args.epochs}", total=len(train_loader))
            train_loss = train_one_epoch_rich(
                model, train_loader, criterion, optimizer, device=device,
                progress=progress, task_id=batch_task
            )
            progress.remove_task(batch_task)

            # Evaluate
            dev_metrics, _, _ = evaluate(model, dev_loader, criterion=criterion, device=device)

            # Determine if this is the best EER
            is_best = False
            if dev_metrics["eer"] is not None:
                if best_eer is None or dev_metrics["eer"] < best_eer:
                    is_best = True
                    best_eer = dev_metrics["eer"]
                    epochs_no_improve = 0

            # Format status with color
            status_text = Text()
            if is_best:
                status_text.append("↓ NEW BEST", style="bold green")
                if prev_eer is not None:
                    status_text.append(f" (prev: {prev_eer:.4f})", style="dim")
            elif prev_eer is not None:
                if dev_metrics["eer"] < prev_eer:
                    status_text.append("↓ Improved", style="green")
                elif dev_metrics["eer"] > prev_eer:
                    status_text.append("↑ Worse", style="red")
                else:
                    status_text.append("= Same", style="yellow")
            else:
                status_text.append("-", style="dim")

            # Create info panel with colored arrows
            info_table = Table.grid(padding=(0, 2))
            info_table.add_column(style="cyan", justify="right")
            info_table.add_column(style="magenta")

            # Train Loss with arrow
            train_loss_text = Text(f"{train_loss:.4f}")
            if len(history) > 0:
                prev_train_loss = history[-1]["train_loss"]
                if train_loss < prev_train_loss:
                    train_loss_text.append(" ↓", style="green")
                elif train_loss > prev_train_loss:
                    train_loss_text.append(" ↑", style="red")
            info_table.add_row("Train Loss:", train_loss_text)

            # Dev Loss with arrow
            dev_loss_text = Text(f"{dev_metrics['avg_loss']:.4f}")
            if len(history) > 0:
                prev_dev_loss = history[-1]["dev_loss"]
                if dev_metrics['avg_loss'] < prev_dev_loss:
                    dev_loss_text.append(" ↓", style="green")
                elif dev_metrics['avg_loss'] > prev_dev_loss:
                    dev_loss_text.append(" ↑", style="red")
            info_table.add_row("Dev Loss:", dev_loss_text)

            # Dev EER with arrow
            dev_eer_text = Text(f"{dev_metrics['eer']:.4f}")
            if len(history) > 0:
                prev_dev_eer = history[-1]["dev_eer"]
                if dev_metrics['eer'] < prev_dev_eer:
                    dev_eer_text.append(" ↓", style="green")
                elif dev_metrics['eer'] > prev_dev_eer:
                    dev_eer_text.append(" ↑", style="red")
            info_table.add_row("Dev EER:", dev_eer_text)

            info_table.add_row("Status:", status_text)
            if best_eer is not None:
                info_table.add_row("Best EER:", f"{best_eer:.4f}")

            panel = Panel(
                info_table,
                title=f"[bold]Epoch {epoch}/{args.epochs}[/bold]",
                border_style="blue"
            )
            console.print(panel)

            # Update history
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_loss": dev_metrics["avg_loss"],
                "dev_eer": dev_metrics["eer"],
                "is_best": is_best
            })

            # Update prev_eer
            if dev_metrics["eer"] is not None:
                prev_eer = dev_metrics["eer"]

            # Save checkpoint if this is the best model
            if is_best and dev_metrics["eer"] is not None:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "config": {
                            "model_name": args.model,
                            "batch_size": args.batch_size,
                            "num_workers": args.num_workers,
                            "lr": args.lr,
                            "weight_decay": args.weight_decay,
                            "in_features": args.in_features,
                            "hidden_dim": args.hidden_dim,
                            "dropout": args.dropout,
                        },
                    },
                    best_path,
                )
            else:
                epochs_no_improve += 1
                if patience and epochs_no_improve >= patience:
                    console.print(
                        f"\n[yellow]Early stopping at epoch {epoch} "
                        f"(no improvement in {patience} epochs)[/yellow]"
                    )
                    break

            progress.update(epoch_task, advance=1)

    # Save final checkpoint
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": args.epochs,
            "config": {
                "model_name": args.model,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "in_features": args.in_features,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
            },
        },
        last_path,
    )

    # Print final summary table
    console.print("\n[bold cyan]Training Summary[/bold cyan]")
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Epoch", justify="right")
    summary_table.add_column("Train Loss", justify="right")
    summary_table.add_column("Dev Loss", justify="right")
    summary_table.add_column("Dev EER", justify="right")
    summary_table.add_column("Status")

    # Find the epoch with the absolute best EER
    best_epoch_idx = min(range(len(history)), key=lambda i: history[i]["dev_eer"])

    for i, h in enumerate(history):
        # Add colored arrows for train loss
        train_loss_str = f"{h['train_loss']:.4f}"
        if i > 0:
            if h['train_loss'] < history[i-1]['train_loss']:
                train_loss_str += " [green]↓[/green]"
            elif h['train_loss'] > history[i-1]['train_loss']:
                train_loss_str += " [red]↑[/red]"

        # Add colored arrows for dev loss
        dev_loss_str = f"{h['dev_loss']:.4f}"
        if i > 0:
            if h['dev_loss'] < history[i-1]['dev_loss']:
                dev_loss_str += " [green]↓[/green]"
            elif h['dev_loss'] > history[i-1]['dev_loss']:
                dev_loss_str += " [red]↑[/red]"

        # Add colored arrows for dev EER
        dev_eer_str = f"{h['dev_eer']:.4f}"
        if i > 0:
            if h['dev_eer'] < history[i-1]['dev_eer']:
                dev_eer_str += " [green]↓[/green]"
            elif h['dev_eer'] > history[i-1]['dev_eer']:
                dev_eer_str += " [red]↑[/red]"

        # Only mark the absolute best epoch
        status = "[green]✓ BEST[/green]" if i == best_epoch_idx else ""

        summary_table.add_row(
            str(h["epoch"]),
            train_loss_str,
            dev_loss_str,
            dev_eer_str,
            status
        )

    console.print(summary_table)

    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for audio deepfake detection.")
    parser.add_argument("--train-features", default="data/train/features.pkl")
    parser.add_argument("--train-labels", default="data/train/labels.pkl")
    parser.add_argument("--dev-features", default="data/dev/features.pkl")
    parser.add_argument("--dev-labels", default="data/dev/labels.pkl")
    parser.add_argument("--model", default="mlp", choices=["mlp", "cnn1d", "cnn2d"])
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
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--no-rich", action="store_true", help="disable rich visualization (use basic tqdm instead)")
    return parser.parse_args()


def main():
    args = parse_args()

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
    if args.model == "mlp":
        model_kwargs = {
            "in_features": args.in_features,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        }
    model = build_model(args.model, **model_kwargs)
    model.to(device)

    # Loss + optimizer (BCEWithLogitsLoss expects raw logits)
    criterion = nn.BCEWithLogitsLoss()
    if args.weight_decay > 0:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Use Rich visualization by default, fall back to tqdm if disabled or unavailable
    use_rich = not args.no_rich

    if use_rich:
        rich_success = train_with_rich(
            model, train_loader, dev_loader, criterion, optimizer, device, args, best_path, last_path
        )
        if not rich_success:
            # Fall back to tqdm if Rich fails to import
            use_rich = False

    if not use_rich:
        # Display training configuration
        print("Training Configuration")
        print(f"Device: {device}")
        print(f"Model: {args.model}")
        print(f"Epochs: {args.epochs}")
        print()

        # Standard tqdm training loop
        best_eer = None
        prev_eer = None
        patience = args.early_stop
        epochs_no_improve = 0
        epoch_iter = tqdm(range(1, args.epochs + 1), desc="Epochs", ncols=100)
        for epoch in epoch_iter:
            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device=device,
                desc=f"Train {epoch}/{args.epochs}",
            )
            dev_metrics, _, _ = evaluate(model, dev_loader,
                                         criterion=criterion, device=device)

            # Determine if this is the best EER
            is_best = False
            if dev_metrics["eer"] is not None:
                if best_eer is None or dev_metrics["eer"] < best_eer:
                    is_best = True
                    best_eer = dev_metrics["eer"]
                    epochs_no_improve = 0

            # Print formatted metrics using tqdm.write to avoid interfering with progress bars
            metrics_output = format_metrics(
                epoch,
                train_loss,
                dev_metrics["avg_loss"],
                dev_metrics["eer"],
                prev_eer=prev_eer,
                is_best=is_best
            )
            tqdm.write(metrics_output)
            # Update the epoch progress bar postfix with best EER
            if best_eer is not None:
                epoch_iter.set_postfix({"Best EER": f"{best_eer:.4f}"})

            # Update prev_eer for next iteration
            if dev_metrics["eer"] is not None:
                prev_eer = dev_metrics["eer"]

            # Save checkpoint if this is the best model
            if is_best:
                if dev_metrics["eer"] is not None:
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "epoch": epoch,
                            "config": {
                                "model_name": args.model,
                                "batch_size": args.batch_size,
                                "num_workers": args.num_workers,
                                "lr": args.lr,
                                "weight_decay": args.weight_decay,
                                "in_features": args.in_features,
                                "hidden_dim": args.hidden_dim,
                                "dropout": args.dropout,
                            },
                        },
                        best_path,
                    )
                else:
                    epochs_no_improve += 1
                    if patience and epochs_no_improve >= patience:
                        print(
                            f"Early stopping at epoch {epoch} "
                            f"(no improvement in {patience} epochs)"
                        )
                        break

        # Save final checkpoint after training completes
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": args.epochs,
                "config": {
                    "model_name": args.model,
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "in_features": args.in_features,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                },
            },
            last_path,
        )


if __name__ == "__main__":
    main()
