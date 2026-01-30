"""No-operation visualizer for testing and CI environments."""

from contextlib import contextmanager
from typing import List, Optional

from .base import (
    BatchContext,
    BatchMetrics,
    EpochMetrics,
    TrainingConfig,
    TrainingVisualizer,
)


class NoOpBatchContext(BatchContext):
    """Batch context that does nothing."""

    def update_batch(self, metrics: BatchMetrics) -> None:
        """No-op batch update."""
        pass


class NoOpVisualizer(TrainingVisualizer):
    """Silent visualizer that produces no output.

    Useful for:
    - Automated testing
    - CI/CD pipelines
    - Headless training environments
    - Performance benchmarking (no I/O overhead)
    """

    def on_training_start(self, config: TrainingConfig) -> None:
        """No-op training start."""
        pass

    @contextmanager
    def on_epoch_start(self, epoch: int, total_batches: int):
        """No-op epoch start."""
        yield NoOpBatchContext()

    def on_epoch_end(self, metrics: EpochMetrics,
                     prev_metrics: Optional[EpochMetrics] = None) -> None:
        """No-op epoch end."""
        pass

    def on_training_end(self, history: List[EpochMetrics]) -> None:
        """No-op training end."""
        pass
