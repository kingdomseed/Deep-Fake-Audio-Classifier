"""Rich-based visualizer with colored output and panels."""

from contextlib import contextmanager
from typing import List, Optional

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .base import (
    BatchContext,
    BatchMetrics,
    EpochMetrics,
    TrainingConfig,
    TrainingVisualizer,
)


class RichBatchContext(BatchContext):
    """Batch context using Rich progress tracking."""

    def __init__(self, progress: "Progress", task_id):
        """Initialize with Rich progress and task ID.

        Args:
            progress: Rich Progress object
            task_id: Task ID for this batch progress
        """
        self.progress = progress
        self.task_id = task_id

    def update_batch(self, metrics: BatchMetrics) -> None:
        """Update batch progress with current metrics.

        Args:
            metrics: Current batch metrics
        """
        self.progress.update(
            self.task_id,
            advance=1,
            loss=f"{metrics.running_loss:.4f}"
        )


class RichVisualizer(TrainingVisualizer):
    """Visualizer using Rich library for colored, styled output.

    Provides:
    - Colored training configuration display
    - Rich progress bars with spinners
    - Epoch panels with colored arrows for metric trends
    - Comprehensive summary table at the end

    Requires:
        pip install rich
    """

    def __init__(self):
        """Initialize RichVisualizer."""
        if not RICH_AVAILABLE:
            raise ImportError(
                "Rich library is required for RichVisualizer. "
                "Install it with: pip install rich"
            )
        self.console = Console()
        self.history: List[dict] = []
        self.progress: Optional[Progress] = None
        self.epoch_task = None
        self.batch_task = None
        self.config: Optional[TrainingConfig] = None

    def on_training_start(self, config: TrainingConfig) -> None:
        """Display training configuration with colors.

        Args:
            config: Training configuration to display
        """
        self.config = config
        self.console.print("[bold cyan]Training Configuration[/bold cyan]")
        self.console.print(f"Device: [yellow]{config.device}[/yellow]")
        self.console.print(f"Model: [yellow]{config.model}[/yellow]")
        self.console.print(f"Epochs: [yellow]{config.epochs}[/yellow]")
        self.console.print()

        # Create progress display
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self.progress.start()
        self.epoch_task = self.progress.add_task(
            "[cyan]Training",
            total=config.epochs
        )

    @contextmanager
    def on_epoch_start(self, epoch: int, total_batches: int):
        """Create Rich progress tracking for the epoch.

        Args:
            epoch: Current epoch number
            total_batches: Total number of batches

        Yields:
            RichBatchContext: Context for batch progress updates
        """
        if self.progress is None or self.config is None:
            raise RuntimeError("on_training_start must be called first")

        # Create batch task
        batch_task = self.progress.add_task(
            f"[green]Epoch {epoch}/{self.config.epochs}",
            total=total_batches
        )

        try:
            yield RichBatchContext(self.progress, batch_task)
        finally:
            # Remove batch task after epoch completes
            self.progress.remove_task(batch_task)

    def on_epoch_end(self, metrics: EpochMetrics,
                     prev_metrics: Optional[EpochMetrics] = None) -> None:
        """Display epoch summary panel with colored arrows.

        Args:
            metrics: Current epoch metrics
            prev_metrics: Previous epoch metrics for comparison
        """
        if self.config is None:
            raise RuntimeError("on_training_start must be called first")

        # Format status with color
        status_text = Text()
        if metrics.is_best:
            status_text.append("↓ NEW BEST", style="bold green")
            if prev_metrics is not None:
                status_text.append(
                    f" (prev: {prev_metrics.dev_eer:.4f})",
                    style="dim"
                )
        elif prev_metrics is not None:
            if metrics.dev_eer < prev_metrics.dev_eer:
                status_text.append("↓ Improved", style="green")
            elif metrics.dev_eer > prev_metrics.dev_eer:
                status_text.append("↑ Worse", style="red")
            else:
                status_text.append("= Same", style="yellow")
        else:
            status_text.append("-", style="dim")

        # Create info table with colored arrows
        info_table = Table.grid(padding=(0, 2))
        info_table.add_column(style="cyan", justify="right")
        info_table.add_column(style="magenta")

        # Train Loss with arrow
        if metrics.train_loss is not None:
            train_loss_text = Text(f"{metrics.train_loss:.4f}")
            if prev_metrics is not None and prev_metrics.train_loss is not None:
                if metrics.train_loss < prev_metrics.train_loss:
                    train_loss_text.append(" ↓", style="green")
                elif metrics.train_loss > prev_metrics.train_loss:
                    train_loss_text.append(" ↑", style="red")
        else:
            train_loss_text = Text("N/A", style="dim")
        info_table.add_row("Train Loss:", train_loss_text)

        # Dev Loss with arrow
        if metrics.dev_loss is not None:
            dev_loss_text = Text(f"{metrics.dev_loss:.4f}")
            if prev_metrics is not None and prev_metrics.dev_loss is not None:
                if metrics.dev_loss < prev_metrics.dev_loss:
                    dev_loss_text.append(" ↓", style="green")
                elif metrics.dev_loss > prev_metrics.dev_loss:
                    dev_loss_text.append(" ↑", style="red")
        else:
            dev_loss_text = Text("N/A", style="dim")
        info_table.add_row("Dev Loss:", dev_loss_text)

        # Dev EER with arrow
        if metrics.dev_eer is not None:
            dev_eer_text = Text(f"{metrics.dev_eer:.4f}")
            if prev_metrics is not None and prev_metrics.dev_eer is not None:
                if metrics.dev_eer < prev_metrics.dev_eer:
                    dev_eer_text.append(" ↓", style="green")
                elif metrics.dev_eer > prev_metrics.dev_eer:
                    dev_eer_text.append(" ↑", style="red")
        else:
            dev_eer_text = Text("N/A", style="dim")
        info_table.add_row("Dev EER:", dev_eer_text)

        info_table.add_row("Status:", status_text)

        # Find best EER from history
        best_eer = None
        for h in self.history:
            if h["dev_eer"] is not None:
                if best_eer is None or h["dev_eer"] < best_eer:
                    best_eer = h["dev_eer"]
        if metrics.dev_eer is not None and (best_eer is None or metrics.dev_eer < best_eer):
            best_eer = metrics.dev_eer

        if best_eer is not None:
            info_table.add_row("Best EER:", f"{best_eer:.4f}")

        # Create and print panel
        panel = Panel(
            info_table,
            title=f"[bold]Epoch {metrics.epoch}/{self.config.epochs}[/bold]",
            border_style="blue"
        )
        self.console.print(panel)

        # Update history
        self.history.append({
            "epoch": metrics.epoch,
            "train_loss": metrics.train_loss,
            "dev_loss": metrics.dev_loss,
            "dev_eer": metrics.dev_eer,
            "is_best": metrics.is_best
        })

        # Update epoch progress
        if self.progress is not None and self.epoch_task is not None:
            self.progress.update(self.epoch_task, advance=1)

    def on_training_end(self, history: List[EpochMetrics]) -> None:
        """Display final summary table.

        Args:
            history: Complete training history
        """
        # Stop progress display
        if self.progress is not None:
            self.progress.stop()

        # Print final summary table
        self.console.print("\n[bold cyan]Training Summary[/bold cyan]")
        summary_table = Table(show_header=True, header_style="bold magenta")
        summary_table.add_column("Epoch", justify="right")
        summary_table.add_column("Train Loss", justify="right")
        summary_table.add_column("Dev Loss", justify="right")
        summary_table.add_column("Dev EER", justify="right")
        summary_table.add_column("Status")

        # Find the epoch with the absolute best EER
        valid_eers = [(i, h.dev_eer) for i, h in enumerate(history) if h.dev_eer is not None]
        if valid_eers:
            best_epoch_idx = min(valid_eers, key=lambda x: x[1])[0]
        else:
            best_epoch_idx = -1

        for i, h in enumerate(history):
            # Add colored arrows for train loss
            if h.train_loss is not None:
                train_loss_str = f"{h.train_loss:.4f}"
                if i > 0 and history[i-1].train_loss is not None:
                    if h.train_loss < history[i-1].train_loss:
                        train_loss_str += " [green]↓[/green]"
                    elif h.train_loss > history[i-1].train_loss:
                        train_loss_str += " [red]↑[/red]"
            else:
                train_loss_str = "N/A"

            # Add colored arrows for dev loss
            if h.dev_loss is not None:
                dev_loss_str = f"{h.dev_loss:.4f}"
                if i > 0 and history[i-1].dev_loss is not None:
                    if h.dev_loss < history[i-1].dev_loss:
                        dev_loss_str += " [green]↓[/green]"
                    elif h.dev_loss > history[i-1].dev_loss:
                        dev_loss_str += " [red]↑[/red]"
            else:
                dev_loss_str = "N/A"

            # Add colored arrows for dev EER
            if h.dev_eer is not None:
                dev_eer_str = f"{h.dev_eer:.4f}"
                if i > 0 and history[i-1].dev_eer is not None:
                    if h.dev_eer < history[i-1].dev_eer:
                        dev_eer_str += " [green]↓[/green]"
                    elif h.dev_eer > history[i-1].dev_eer:
                        dev_eer_str += " [red]↑[/red]"
            else:
                dev_eer_str = "N/A"

            # Only mark the absolute best epoch
            status = "[green]✓ BEST[/green]" if i == best_epoch_idx else ""

            summary_table.add_row(
                str(h.epoch),
                train_loss_str,
                dev_loss_str,
                dev_eer_str,
                status
            )

        self.console.print(summary_table)
