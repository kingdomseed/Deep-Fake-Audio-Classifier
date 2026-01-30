"""Base classes and data structures for training visualization."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Configuration displayed at training start."""
    device: str
    model: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    early_stop_patience: int
    # Model-specific parameters
    in_features: int
    hidden_dim: int
    dropout: float


@dataclass
class BatchMetrics:
    """Metrics from a single training batch."""
    batch_idx: int
    running_loss: float
    batch_size: int


@dataclass
class EpochMetrics:
    """Complete metrics for an epoch."""
    epoch: int
    train_loss: Optional[float]
    dev_loss: Optional[float]
    dev_eer: Optional[float]
    is_best: bool
    improved: bool  # Better than previous epoch
    epochs_no_improve: int  # For early stopping tracking


class BatchContext(ABC):
    """Abstract context for batch progress updates."""

    @abstractmethod
    def update_batch(self, metrics: BatchMetrics) -> None:
        """Update batch progress with current metrics."""
        pass


class TrainingVisualizer(ABC):
    """Abstract base class for training visualization.

    This class defines the interface that all training visualizers must implement.
    Visualizers are responsible ONLY for display/formatting - they do not make
    training decisions (early stopping, checkpointing, best model tracking).

    The training loop maintains control over:
    - Early stopping logic
    - Checkpoint saving
    - Best model tracking
    - Training state (optimizer, model)

    The visualizer is responsible for:
    - Display/rendering
    - Progress indication
    - Formatting metrics
    - Communicating status visually
    """

    @abstractmethod
    def on_training_start(self, config: TrainingConfig) -> None:
        """Called once at the start of training.

        Args:
            config: Training configuration to display
        """
        pass

    @abstractmethod
    @contextmanager
    def on_epoch_start(self, epoch: int, total_batches: int):
        """Called at the start of each epoch.

        Returns a context manager for batch iteration. The context manager
        should yield a BatchContext object that can be used to update
        batch progress.

        Args:
            epoch: Current epoch number (1-indexed)
            total_batches: Total number of batches in the epoch

        Yields:
            BatchContext: Context object with update_batch() method

        Example:
            with visualizer.on_epoch_start(epoch, len(dataloader)) as ctx:
                for batch_idx, (features, labels) in enumerate(dataloader):
                    # ... training code ...
                    ctx.update_batch(BatchMetrics(...))
        """
        pass

    @abstractmethod
    def on_epoch_end(self, metrics: EpochMetrics,
                     prev_metrics: Optional[EpochMetrics] = None) -> None:
        """Called at the end of each epoch with complete metrics.

        Args:
            metrics: Metrics for the current epoch
            prev_metrics: Metrics from the previous epoch (for comparison)
        """
        pass

    @abstractmethod
    def on_training_end(self, history: List[EpochMetrics]) -> None:
        """Called once at the end of training with full history.

        Args:
            history: Complete list of epoch metrics for all epochs
        """
        pass
