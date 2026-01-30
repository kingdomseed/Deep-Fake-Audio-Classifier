"""Training visualization package."""

from .base import (
    BatchContext,
    BatchMetrics,
    EpochMetrics,
    TrainingConfig,
    TrainingVisualizer,
)
from .noop_visualizer import NoOpVisualizer
from .tqdm_visualizer import TqdmVisualizer

__all__ = [
    "TrainingVisualizer",
    "TrainingConfig",
    "BatchMetrics",
    "EpochMetrics",
    "BatchContext",
    "NoOpVisualizer",
    "TqdmVisualizer",
    "create_visualizer",
]


def create_visualizer(visualizer_type: str = "rich") -> TrainingVisualizer:
    """Factory function to create appropriate visualizer.

    Args:
        visualizer_type: Type of visualizer to create.
            Options: "rich", "tqdm", "noop"

    Returns:
        TrainingVisualizer: Appropriate visualizer instance

    Raises:
        ValueError: If visualizer_type is not recognized

    Example:
        >>> viz = create_visualizer("rich")
        >>> viz.on_training_start(config)
    """
    if visualizer_type == "rich":
        try:
            from .rich_visualizer import RichVisualizer
            return RichVisualizer()
        except ImportError:
            print("Rich library not found. Please install it with: pip install rich")
            print("Falling back to tqdm visualizer...")
            visualizer_type = "tqdm"

    if visualizer_type == "tqdm":
        return TqdmVisualizer()

    if visualizer_type == "noop":
        return NoOpVisualizer()

    raise ValueError(
        f"Unknown visualizer type: {visualizer_type}. "
        f"Valid options: 'rich', 'tqdm', 'noop'"
    )
