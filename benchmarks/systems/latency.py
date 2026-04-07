"""Tracking inference latency benchmark.

Users submit a Python entry point that the server invokes repeatedly over
the eval split. We measure wall-clock time inside a standardised container.

For client-side self-scoring, users call this task locally and report the
time they observed. The server re-runs the same function in its own
environment for canonical numbers.
"""

from __future__ import annotations

import pyarrow as pa

from benchmarks._base import BenchmarkTask


class TrackingLatencyTask(BenchmarkTask):
    name = "tracking_latency"
    dataset = "ttbar_pu200"
    eval_event_range = (99_000, 100_000)  # 1000 events for timing
    inputs = ["tracker_hits"]
    metrics = ["wallclock_s", "events_per_sec"]
    higher_is_better = {"wallclock_s": False, "events_per_sec": True}

    def load_eval_inputs(self):
        import colliderml
        return {"tracker_hits": colliderml.load(self.dataset, tables=["tracker_hits"])}

    def validate_predictions(self, preds: pa.Table) -> None:
        required = {"wallclock_s", "n_events"}
        have = set(preds.column_names) if hasattr(preds, "column_names") else set()
        if not required.issubset(have):
            raise ValueError(
                "Latency submissions must be a 1-row table with columns "
                f"{sorted(required)}; got {sorted(have)}"
            )

    def score(self, preds: pa.Table) -> dict[str, float]:
        row = preds.to_pydict()
        wallclock = float(row["wallclock_s"][0])
        n_events = int(row["n_events"][0])
        if wallclock <= 0 or n_events <= 0:
            return {"wallclock_s": float("inf"), "events_per_sec": 0.0}
        return {
            "wallclock_s": round(wallclock, 3),
            "events_per_sec": round(n_events / wallclock, 3),
        }
