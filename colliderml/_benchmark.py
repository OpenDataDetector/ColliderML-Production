"""
Public benchmark API — thin wrapper around the `benchmarks/` package.

Usage:
    import colliderml
    scores = colliderml.benchmark("tracking", "my_predictions.parquet")
    scores = colliderml.benchmark("tracking", "my_predictions.parquet", submit=True)
"""

from __future__ import annotations

import os
from typing import Any, Optional


def benchmark(
    task: str,
    predictions,
    submit: bool = False,
    backend_url: Optional[str] = None,
) -> dict:
    """Score a prediction file against a benchmark task.

    Args:
        task: Task name ("tracking", "jets", "anomaly", "tracking_latency",
              "tracking_small", "data_loading").
        predictions: Path to a Parquet file or a pyarrow Table / pandas DataFrame.
        submit: If True, also upload the predictions to the leaderboard
                backend, which re-scores them and awards credits for
                new bests.
        backend_url: Override the backend URL.

    Returns:
        dict of metric_name -> value.
    """
    # Lazy import so importing colliderml.benchmark doesn't cost anything
    # when the user only wants .load() or .simulate().
    try:
        from benchmarks import evaluate as _evaluate, list_tasks as _list_tasks
    except ImportError as e:
        raise RuntimeError(
            "The benchmarks/ package is not installed.\n"
            "If you installed colliderml via pip it should be bundled; if you "
            "installed from source, ensure you cloned the whole repository.\n"
            f"Original error: {e}"
        )

    scores = _evaluate(task, predictions)

    if submit:
        remote_scores = _submit_to_leaderboard(task, predictions, scores, backend_url)
        return remote_scores
    return scores


def list_benchmarks() -> list[str]:
    """Return the names of all registered benchmark tasks."""
    from benchmarks import list_tasks as _list_tasks
    return _list_tasks()


# ---------------------------------------------------------------------------
# Submission to the backend leaderboard
# ---------------------------------------------------------------------------
def _submit_to_leaderboard(
    task: str,
    predictions,
    local_scores: dict,
    backend_url: Optional[str],
) -> dict:
    import json
    from io import BytesIO

    import pyarrow as pa
    import pyarrow.parquet as pq
    import requests

    from colliderml._remote import (
        DEFAULT_BACKEND_URL,
        _auth_headers,
        _require_token,
    )

    token = _require_token()
    url = (backend_url or DEFAULT_BACKEND_URL).rstrip("/")

    # Serialise the predictions back to parquet bytes for upload
    if isinstance(predictions, (str,)):
        with open(predictions, "rb") as f:
            payload = f.read()
    elif hasattr(predictions, "column_names"):
        buf = BytesIO()
        pq.write_table(predictions, buf)
        payload = buf.getvalue()
    elif hasattr(predictions, "to_parquet"):
        buf = BytesIO()
        predictions.to_parquet(buf)
        payload = buf.getvalue()
    else:
        raise TypeError("Unsupported predictions type for submission")

    files = {"predictions": ("predictions.parquet", payload, "application/octet-stream")}
    data = {"local_scores": json.dumps(local_scores)}
    r = requests.post(
        f"{url}/v1/benchmark/{task}/submit",
        files=files,
        data=data,
        headers=_auth_headers(token),
        timeout=120,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Leaderboard submission failed: {r.status_code} {r.text}")
    result = r.json()
    print(
        f"Submitted to leaderboard for task '{task}':\n"
        f"  Scores: {result.get('scores')}\n"
        f"  Credits earned: {result.get('credits_earned', 0)}"
    )
    return result
