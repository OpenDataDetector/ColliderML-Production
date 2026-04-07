"""Data loading throughput benchmark.

Users submit (or report) the time it takes to load + iterate a fixed
eval split into numpy/torch tensors. Encourages contributions to
`colliderml.load()` and upstream (pyarrow, huggingface_hub).
"""

from __future__ import annotations

import pyarrow as pa

from benchmarks._base import BenchmarkTask


class DataLoadingTask(BenchmarkTask):
    name = "data_loading"
    dataset = "ttbar_pu200"
    eval_event_range = (0, 10_000)  # first 10k events
    inputs = ["tracker_hits"]
    metrics = ["events_per_sec_local", "events_per_sec_streaming"]
    higher_is_better = {
        "events_per_sec_local": True,
        "events_per_sec_streaming": True,
    }

    def load_eval_inputs(self):
        """This task's 'input' is the timing itself — the server-side scorer
        actually measures it with the user's code."""
        return {}

    def validate_predictions(self, preds: pa.Table) -> None:
        required = {"local_seconds", "streaming_seconds", "n_events"}
        have = set(preds.column_names) if hasattr(preds, "column_names") else set()
        if not required.issubset(have):
            raise ValueError(
                "data_loading submissions must be a 1-row table with columns "
                f"{sorted(required)}; got {sorted(have)}"
            )

    def score(self, preds: pa.Table) -> dict[str, float]:
        row = preds.to_pydict()
        n_events = int(row["n_events"][0])
        local = float(row["local_seconds"][0])
        streaming = float(row["streaming_seconds"][0])
        return {
            "events_per_sec_local": round(n_events / max(local, 1e-9), 3),
            "events_per_sec_streaming": round(n_events / max(streaming, 1e-9), 3),
        }
