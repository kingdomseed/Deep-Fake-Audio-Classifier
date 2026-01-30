"""Tqdm-based visualizer for simple progress bars and text output."""

from contextlib import contextmanager
from typing import List, Optional

from tqdm import tqdm

from .base import (
    BatchContext,
    BatchMetrics,
    EpochMetrics,
    TrainingConfig,
    TrainingVisualizer,
)


class TqdmBatchContext(BatchContext):
    """Batch context using tqdm progress bar."""

    def __init__(self, pbar: tqdm):
        """Initialize with tqdm progress bar.

        Args:
            pbar: tqdm progress bar instance
        """
        self.pbar = pbar

    def update_batch(self, metrics: BatchMetrics) -> None:
        """Update batch progress with current metrics.

        Args:
            metrics: Current batch metrics
        """
        self.pbar.update(1)
        self.pbar.set_postfix(loss=f"{metrics.running_loss:.4f}")


class TqdmVisualizer(TrainingVisualizer):
    """Simple visualizer using tqdm progress bars.

    Provides:
    - Training configuration display
    - Batch-level progress bars with loss
    - Epoch summary with formatted metrics and arrows
    - Best EER tracking in epoch progress bar
    """

    def __init__(self):
        """Initialize TqdmVisualizer."""
        self.epoch_pbar: Optional[tqdm] = None
        self.best_eer: Optional[float] = None

    def on_training_start(self, config: TrainingConfig) -> None:
        """Display training configuration.

        Args:
            config: Training configuration to display
        """
        print("Training Configuration")
        print(f"Device: {config.device}")
        print(f"Model: {config.model}")
        print(f"Epochs: {config.epochs}")
        print()

    @contextmanager
    def on_epoch_start(self, epoch: int, total_batches: int):
        """Create batch progress bar for the epoch.

        Args:
            epoch: Current epoch number
            total_batches: Total number of batches

        Yields:
            TqdmBatchContext: Context for batch progress updates
        """
        desc = f"Epoch {epoch}"
        batch_pbar = tqdm(
            total=total_batches,
            desc=desc,
            leave=False,
            ncols=80,
            mininterval=1.0
        )
        try:
            yield TqdmBatchContext(batch_pbar)
        finally:
            batch_pbar.close()

    def on_epoch_end(self, metrics: EpochMetrics,
                     prev_metrics: Optional[EpochMetrics] = None) -> None:
        """Display epoch summary with formatted metrics.

        Args:
            metrics: Current epoch metrics
            prev_metrics: Previous epoch metrics for comparison
        """
        # Format metrics string
        metrics_str = self._format_metrics(metrics, prev_metrics)

        # Write using tqdm.write to avoid interfering with progress bars
        tqdm.write(metrics_str)

        # Track best EER
        if metrics.is_best:
            self.best_eer = metrics.dev_eer

    def on_training_end(self, history: List[EpochMetrics]) -> None:
        """Training end (no summary for tqdm visualizer).

        Args:
            history: Complete training history
        """
        # Tqdm visualizer doesn't print a summary table
        pass

    def _format_metrics(self, metrics: EpochMetrics,
                       prev_metrics: Optional[EpochMetrics] = None) -> str:
        """Format training metrics with improvement indicators.

        Args:
            metrics: Current epoch metrics
            prev_metrics: Previous epoch metrics for comparison

        Returns:
            Formatted string with metrics and indicators
        """
        # Format basic metrics
        train_str = f"{metrics.train_loss:.4f}" if metrics.train_loss is not None else "N/A"
        dev_str = f"{metrics.dev_loss:.4f}" if metrics.dev_loss is not None else "N/A"
        eer_str = f"{metrics.dev_eer:.4f}" if metrics.dev_eer is not None else "N/A"

        metrics_str = (
            f"Epoch {metrics.epoch}: "
            f"train={train_str}  "
            f"dev={dev_str}  "
            f"eer={eer_str}"
        )

        # Add improvement indicator for EER
        if prev_metrics is not None and prev_metrics.dev_eer is not None and metrics.dev_eer is not None:
            if metrics.dev_eer < prev_metrics.dev_eer:
                metrics_str += " ↓"
            elif metrics.dev_eer > prev_metrics.dev_eer:
                metrics_str += " ↑"
            else:
                metrics_str += " ="

        # Add BEST marker
        if metrics.is_best:
            metrics_str += " BEST"

        return metrics_str
