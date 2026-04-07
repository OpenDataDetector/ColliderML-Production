"""
ColliderML benchmark tasks.

Public API:
    list_tasks() -> list[str]
    get_task(name: str) -> BenchmarkTask
    evaluate(task_name, predictions) -> dict[str, float]

Each task knows how to load its eval inputs, validate predictions,
and score them. Tasks are registered in TASKS below.

Systems tasks (latency, small-model, data loading) are wrappers around
the physics tasks that add constraints on top of the same scoring.
"""

from __future__ import annotations

from typing import Optional

from benchmarks._base import BenchmarkTask


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_TASKS: dict[str, type[BenchmarkTask]] = {}


def register(cls: type[BenchmarkTask]) -> type[BenchmarkTask]:
    _TASKS[cls.name] = cls
    return cls


# Import tasks to register them
from benchmarks.tracking.task import TrackingTask  # noqa: E402
from benchmarks.jets.task import JetClassificationTask  # noqa: E402
from benchmarks.anomaly.task import AnomalyDetectionTask  # noqa: E402
from benchmarks.systems.latency import TrackingLatencyTask  # noqa: E402
from benchmarks.systems.smallmodel import TrackingSmallModelTask  # noqa: E402
from benchmarks.systems.dataloading import DataLoadingTask  # noqa: E402

register(TrackingTask)
register(JetClassificationTask)
register(AnomalyDetectionTask)
register(TrackingLatencyTask)
register(TrackingSmallModelTask)
register(DataLoadingTask)


def list_tasks() -> list[str]:
    """Return the names of all registered benchmark tasks."""
    return sorted(_TASKS.keys())


def get_task(name: str) -> BenchmarkTask:
    """Return a task instance by name."""
    if name not in _TASKS:
        raise ValueError(
            f"Unknown task '{name}'. Available: {', '.join(list_tasks())}"
        )
    return _TASKS[name]()


def evaluate(task_name: str, predictions) -> dict[str, float]:
    """Score a prediction file/table against a task.

    Args:
        task_name: Registered task name, e.g. "tracking".
        predictions: Path to a Parquet file or a pyarrow Table.

    Returns:
        dict mapping metric name to float value.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if isinstance(predictions, (str,)):
        preds = pq.read_table(predictions)
    elif hasattr(predictions, "column_names"):
        preds = predictions
    elif hasattr(predictions, "to_parquet"):
        preds = pa.Table.from_pandas(predictions)
    else:
        raise TypeError("predictions must be a path, pyarrow Table, or pandas DataFrame")

    task = get_task(task_name)
    task.validate_predictions(preds)
    return task.score(preds)


__all__ = ["list_tasks", "get_task", "evaluate", "BenchmarkTask"]
