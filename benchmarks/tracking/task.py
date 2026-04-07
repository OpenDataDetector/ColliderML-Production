"""Track reconstruction benchmark task."""

from __future__ import annotations

import pyarrow as pa

from benchmarks._base import BenchmarkTask
from benchmarks.tracking.metrics import (
    duplicate_rate,
    fake_rate,
    physics_eff_pt1,
    trackml_weighted_efficiency,
)


class TrackingTask(BenchmarkTask):
    name = "tracking"
    dataset = "ttbar_pu200"
    eval_event_range = (90_000, 100_000)
    inputs = ["tracker_hits"]
    metrics = ["trackml_eff", "fake_rate", "dup_rate", "physics_eff_pt1"]
    higher_is_better = {
        "trackml_eff": True,
        "fake_rate": False,
        "dup_rate": False,
        "physics_eff_pt1": True,
    }

    def load_eval_inputs(self):
        """Download the tracker_hits table for the eval split."""
        import colliderml
        data = colliderml.load(self.dataset, tables=["tracker_hits"])
        return {"tracker_hits": data}

    def _load_truth(self) -> tuple[pa.Table, pa.Table]:
        """Load truth hits and particles for the eval split."""
        import colliderml
        hits = colliderml.load(self.dataset, tables=["tracker_hits"])
        particles = colliderml.load(self.dataset, tables=["particles"])
        return hits, particles

    def validate_predictions(self, preds: pa.Table) -> None:
        required = {"event_id", "hit_id", "track_id"}
        have = set(preds.column_names)
        if not required.issubset(have):
            raise ValueError(
                f"Tracking predictions must have columns {sorted(required)}, got {sorted(have)}"
            )

        # Check coverage of the eval event range
        events = set(preds.column("event_id").to_pylist())
        expected = set(range(*self.eval_event_range))
        missing = expected - events
        if missing:
            # Be lenient: require at least 50% coverage (partial submissions allowed)
            if len(missing) > len(expected) * 0.5:
                raise ValueError(
                    f"Missing predictions for too many events: "
                    f"{len(missing)}/{len(expected)}"
                )

    def score(self, preds: pa.Table) -> dict[str, float]:
        hits, particles = self._load_truth()
        return {
            "trackml_eff": trackml_weighted_efficiency(preds, hits),
            "fake_rate": fake_rate(preds, hits),
            "dup_rate": duplicate_rate(preds, hits),
            "physics_eff_pt1": physics_eff_pt1(preds, particles),
        }
