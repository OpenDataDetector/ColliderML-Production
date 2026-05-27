"""End-to-end test for the leaderboard submit → score → HF push path.

This is the test that *should* have existed before v0.4.0 was tagged. It
exercises the full HTTP → routing → ``colliderml.tasks`` → scoring →
fire-and-forget HF push chain in one shot, with two boundaries mocked:

* The HF data layer (``colliderml.tasks._loading.load_task_data``) is
  replaced by a synthetic list-per-event table, so the test doesn't
  pull GB of parquet shards in CI.
* The HF push (``app.leaderboard._push_result_to_hf``) is replaced by
  a recorder, so the test asserts the push *would* fire with the right
  args without needing real credentials or contacting hf.co.

What this test catches (any of which currently break in different ways):

* The ``from benchmarks import …`` regression in ``app/leaderboard.py``
  (fixed in this same commit window). The test imports the actual
  router, so a broken import surfaces immediately.
* The ``TrackingTask`` explode bug — list-per-event truth being handed
  straight to flat-format metrics. The score path runs end-to-end and
  asserts the metric values.
* The ``download_config`` over-fetch — wouldn't show up here because
  load_task_data is patched, but the *pattern* of patching matches
  what we'd want in CI.

Run with::

    DATABASE_URL=postgresql://postgres:postgres@localhost:5432/colliderml \\
        pytest backend/tests/test_leaderboard_e2e.py -v
"""

from __future__ import annotations

import io
import os
import time
from typing import Any, Dict, List

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


_HAS_DB = bool(os.environ.get("DATABASE_URL"))


# ---------------------------------------------------------------------------
# Synthetic data builders — match the shape colliderml.load() returns from
# the on-disk parquet (list-per-event), but with tiny per-event lists so the
# whole eval slice fits in a few hundred KB.
# ---------------------------------------------------------------------------
_N_EVENTS = 1000  # matches COLLIDERML_TRACKING_EVAL_RANGE=0:1000
_HITS_PER_EVENT = 10
_PARTS_PER_EVENT = 2


def _list_per_event_hits() -> pa.Table:
    """tracker_hits in list-per-event shape (one row per event)."""
    rows: List[Dict[str, Any]] = []
    for ev in range(_N_EVENTS):
        rows.append(
            {
                "event_id": ev,
                "x": [float(i) for i in range(_HITS_PER_EVENT)],
                "y": [0.0] * _HITS_PER_EVENT,
                "z": [0.0] * _HITS_PER_EVENT,
                # Each pair of hits belongs to one particle, so hits/track > 1
                # (avoids the perfect-oracle splitter being degenerate).
                "particle_id": [
                    ev * 100 + (i // 5) for i in range(_HITS_PER_EVENT)
                ],
            }
        )
    return pl.DataFrame(rows).to_arrow()


def _list_per_event_particles() -> pa.Table:
    """particles in list-per-event shape; physics_eff_pt1 reads px/py/primary."""
    rows: List[Dict[str, Any]] = []
    for ev in range(_N_EVENTS):
        rows.append(
            {
                "event_id": ev,
                "particle_id": [ev * 100 + i for i in range(_PARTS_PER_EVENT)],
                "px": [2.0] * _PARTS_PER_EVENT,
                "py": [0.5] * _PARTS_PER_EVENT,
                "primary": [True] * _PARTS_PER_EVENT,
            }
        )
    return pl.DataFrame(rows).to_arrow()


def _perfect_preds_parquet() -> bytes:
    """Flat (event_id, hit_id, track_id) predictions that match the truth."""
    event_ids: List[int] = []
    hit_ids: List[int] = []
    track_ids: List[int] = []
    for ev in range(_N_EVENTS):
        for h in range(_HITS_PER_EVENT):
            event_ids.append(ev)
            hit_ids.append(h)
            track_ids.append(ev * 100 + (h // 5))  # = the truth particle_id
    table = pa.table(
        {
            "event_id": event_ids,
            "hit_id": hit_ids,
            "track_id": track_ids,
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lib reads these in TrackingTask.__init__; the route freshly
    instantiates the task on every request, so the env vars are honored."""
    if not _HAS_DB:
        pytest.skip("DATABASE_URL not set")
    monkeypatch.setenv("COLLIDERML_TRACKING_DATASET", "ttbar_pu0")
    monkeypatch.setenv("COLLIDERML_TRACKING_EVAL_RANGE", f"0:{_N_EVENTS}")


@pytest.fixture
def patched_loader(env_overrides, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``load_task_data`` with an in-memory fake so scoring is
    fast and self-contained — no network, no parquet shards on disk."""
    hits = _list_per_event_hits()
    particles = _list_per_event_particles()

    def fake_load_task_data(
        dataset: str,
        *,
        tables,
        max_events=None,
        event_range=None,
        dataset_id=None,
    ):
        out: Dict[str, pa.Table] = {}
        if "tracker_hits" in tables:
            out["tracker_hits"] = hits
        if "particles" in tables:
            out["particles"] = particles
        return out

    from colliderml.tasks import _loading as _tasks_loading

    monkeypatch.setattr(_tasks_loading, "load_task_data", fake_load_task_data)
    from colliderml.tasks import _base as _tasks_base

    if hasattr(_tasks_base, "load_task_data"):
        monkeypatch.setattr(_tasks_base, "load_task_data", fake_load_task_data)


@pytest.fixture
def hf_push_recorder(monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, Any]]:
    """Replace _push_result_to_hf with an async recorder. The real
    implementation is fire-and-forget via asyncio.create_task, so we read
    back the recorded call after a brief poll."""
    calls: List[Dict[str, Any]] = []

    async def fake_push(**kwargs):
        calls.append(kwargs)

    from app import leaderboard

    monkeypatch.setattr(leaderboard, "_push_result_to_hf", fake_push)
    return calls


def _wait_for_push(calls: List[Dict[str, Any]], timeout_s: float = 5.0) -> None:
    """The HF push is fire-and-forget; poll briefly for the recorder to fill."""
    deadline = time.time() + timeout_s
    while not calls and time.time() < deadline:
        time.sleep(0.05)
    assert calls, f"HF push did not fire within {timeout_s}s"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_tracking_submit_scores_and_fires_hf_push(
    client, patched_loader, hf_push_recorder
) -> None:
    """The release gate: a tracking submission must score AND trigger the HF push."""
    blob = _perfect_preds_parquet()
    files = {"predictions": ("preds.parquet", blob, "application/octet-stream")}

    response = client.post("/v1/benchmark/tracking/submit", files=files)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["deduplicated"] is False
    assert "submission_id" in body and body["submission_id"]
    scores = body["scores"]
    assert scores["trackml_eff"] == pytest.approx(1.0)
    assert scores["fake_rate"] == pytest.approx(0.0)
    assert scores["dup_rate"] == pytest.approx(0.0)

    _wait_for_push(hf_push_recorder)
    pushed = hf_push_recorder[0]
    assert pushed["task_name"] == "tracking"
    assert pushed["username"] == "e2e_alice"
    assert pushed["scores"]["trackml_eff"] == pytest.approx(1.0)
    assert pushed["pred_hash"]  # non-empty sha
    assert pushed["model_repo_id"] == ""  # not supplied in this call
    assert pushed["credits_earned"] > 0  # perfect score on a fresh leaderboard


def test_tracking_submit_with_model_repo_id_propagates(
    client, patched_loader, hf_push_recorder
) -> None:
    """When model_repo_id is supplied, the HF push must receive it so it
    can land an eval_results yaml on the user's model repo."""
    blob = _perfect_preds_parquet()
    files = {"predictions": ("preds.parquet", blob, "application/octet-stream")}
    form = {"model_repo_id": "e2e_alice/my-tracker"}

    response = client.post(
        "/v1/benchmark/tracking/submit", files=files, data=form
    )
    assert response.status_code == 200, response.text

    _wait_for_push(hf_push_recorder)
    assert hf_push_recorder[0]["model_repo_id"] == "e2e_alice/my-tracker"


def test_tracking_submit_dedups_identical_blob(
    client, patched_loader, hf_push_recorder
) -> None:
    """Same parquet bytes from the same user, with no new model_repo_id,
    should hit dedup (deduplicated=True) and NOT re-fire the HF push.
    The dataset JSON is already there from the first call."""
    blob = _perfect_preds_parquet()
    files = {"predictions": ("preds.parquet", blob, "application/octet-stream")}

    first = client.post("/v1/benchmark/tracking/submit", files=files)
    assert first.status_code == 200
    _wait_for_push(hf_push_recorder)
    pushed_before = len(hf_push_recorder)

    # Re-post identical bytes — same predictions_sha256 → dedup
    files = {"predictions": ("preds.parquet", blob, "application/octet-stream")}
    second = client.post("/v1/benchmark/tracking/submit", files=files)
    assert second.status_code == 200, second.text
    body = second.json()
    assert body["deduplicated"] is True
    assert body["scores"] == first.json()["scores"]

    # Give the second call a chance to fire a (spurious) push before asserting
    time.sleep(0.5)
    assert len(hf_push_recorder) == pushed_before, (
        "dedup path must not re-fire the HF push without a new model_repo_id"
    )


def test_dedup_still_fires_push_when_new_model_repo_id_supplied(
    client, patched_loader, hf_push_recorder
) -> None:
    """Typical late-link flow: a user submitted predictions, then later
    created an HF model repo and wants to link the scores to its model
    card. Re-submitting the same parquet WITH a model_repo_id should
    fire the HF push again (eval_results yaml lands on the model repo)
    even though the dataset JSON dedupe still kicks in."""
    blob = _perfect_preds_parquet()

    # First submit: no model_repo_id, dataset JSON pushed.
    files = {"predictions": ("preds.parquet", blob, "application/octet-stream")}
    first = client.post("/v1/benchmark/tracking/submit", files=files)
    assert first.status_code == 200
    _wait_for_push(hf_push_recorder)
    assert hf_push_recorder[-1]["model_repo_id"] == ""

    # Second submit: identical predictions, now with model_repo_id.
    # Dedup tag = True, but the eval_results push must still fire.
    files = {"predictions": ("preds.parquet", blob, "application/octet-stream")}
    form = {"model_repo_id": "e2e_alice/my-tracker"}
    pushed_before = len(hf_push_recorder)
    second = client.post(
        "/v1/benchmark/tracking/submit", files=files, data=form
    )
    assert second.status_code == 200, second.text
    assert second.json()["deduplicated"] is True
    # Wait for the new push.
    deadline = time.time() + 5.0
    while len(hf_push_recorder) == pushed_before and time.time() < deadline:
        time.sleep(0.05)
    assert len(hf_push_recorder) == pushed_before + 1, (
        "dedup branch must still fire the HF push when a new model_repo_id is supplied"
    )
    assert hf_push_recorder[-1]["model_repo_id"] == "e2e_alice/my-tracker"


def test_invalid_parquet_returns_400(
    client, patched_loader, hf_push_recorder
) -> None:
    """A non-parquet upload is a user error, not a 500."""
    files = {"predictions": ("preds.txt", b"not a parquet", "text/plain")}
    response = client.post("/v1/benchmark/tracking/submit", files=files)
    assert response.status_code == 400
    assert "Parquet" in response.text


def test_unknown_task_returns_404(
    client, patched_loader, hf_push_recorder
) -> None:
    blob = _perfect_preds_parquet()
    files = {"predictions": ("preds.parquet", blob, "application/octet-stream")}
    response = client.post("/v1/benchmark/nope/submit", files=files)
    assert response.status_code == 404
