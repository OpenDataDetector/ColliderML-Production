"""Abstract base class for benchmark tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Optional

import pyarrow as pa


class BenchmarkTask(ABC):
    """Each task knows how to load eval inputs, validate predictions, and score them.

    Subclasses must set the following class attributes:
        name: Task identifier used in URLs, CLI, and the registry.
        dataset: HF dataset to draw eval events from (e.g. "ttbar_pu200").
        eval_event_range: (start, end) half-open on event_id. Events in this
            range are withheld from users for training; only predictions on
            these events count toward scoring.
        inputs: Table names to expose to the model.
        metrics: Metric names this task reports (ordered by primary importance).
        higher_is_better: Per-metric direction. True = bigger score wins.
    """

    name: ClassVar[str]
    dataset: ClassVar[str]
    eval_event_range: ClassVar[tuple[int, int]]
    inputs: ClassVar[list[str]]
    metrics: ClassVar[list[str]]
    higher_is_better: ClassVar[dict[str, bool]] = {}

    @abstractmethod
    def load_eval_inputs(self) -> dict[str, pa.Table]:
        """Return the eval inputs as a dict of pyarrow Tables.

        The server-side evaluator uses this to build the inputs for baseline
        runs. Users who want a fully local benchmark call the same method.
        """

    @abstractmethod
    def validate_predictions(self, preds: pa.Table) -> None:
        """Raise ValueError if preds don't match the expected schema/coverage."""

    @abstractmethod
    def score(self, preds: pa.Table) -> dict[str, float]:
        """Compute all metrics on preds. Returns {metric: value}."""

    def is_better(self, metric: str, new: float, current: float) -> bool:
        """Used by the leaderboard to decide whether to award credits."""
        higher = self.higher_is_better.get(metric, True)
        return new > current if higher else new < current
