"""
Leaderboard routes for ColliderML benchmarks.

All scoring happens server-side — the truth data lives on the backend host
only. Users submit prediction parquet files; the backend re-scores them
using the `benchmarks/` package and awards credits for new bests.

Reproduction: any user can attempt to reproduce someone else's submission.
If they produce scores within 2% of the claimed values on every metric,
they earn 20 credits. This self-polices the leaderboard without requiring
manual review.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.auth import current_user
from app.config import get_settings
from app.db import db

# Make the top-level benchmarks package importable — it lives next to backend/
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["benchmarks"])


# Credit rewards per "column" beaten. Calibrated so earning a reward funds
# a follow-up experiment at comparable scale.
_REWARD_PER_METRIC = {
    "tracking": 50,
    "jets": 50,
    "anomaly": 50,
    "tracking_latency": 30,
    "tracking_small": 50,
    "data_loading": 30,
}
_REPRODUCE_REWARD = 20.0
_REPRODUCE_TOLERANCE = 0.02


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _load_task(task_name: str):
    try:
        from benchmarks import get_task
    except ImportError as e:
        raise HTTPException(503, f"benchmarks package not available: {e}")
    try:
        return get_task(task_name)
    except ValueError as e:
        raise HTTPException(404, str(e))


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


async def _score_table(task, preds: pa.Table) -> dict:
    task.validate_predictions(preds)
    return task.score(preds)


async def _push_result_to_hf(
    task_name: str,
    submission_id: str,
    username: str,
    scores: dict,
    credits_earned: float,
    pred_hash: str,
    model_repo_id: str,
) -> None:
    """Fire-and-forget: push result JSON to HF dataset + eval_results to model repo."""
    settings = get_settings()
    if not settings.hf_token:
        logger.debug("hf_token not set — skipping HF result push")
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=settings.hf_token)
        result_json = json.dumps(
            {
                "submission_id": submission_id,
                "task": task_name,
                "submitter": username,
                "model_repo_id": model_repo_id or None,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "scores": scores,
                "credits_earned": credits_earned,
                "is_baseline": False,
                "predictions_sha256": pred_hash,
            },
            indent=2,
        )
        api.upload_file(
            path_or_fileobj=result_json.encode(),
            path_in_repo=f"results/{task_name}/{username}/{pred_hash[:12]}.json",
            repo_id=settings.hf_results_dataset,
            repo_type="dataset",
            commit_message=f"score: {task_name} by {username}",
        )
        logger.info("pushed result to %s for %s/%s", settings.hf_results_dataset, task_name, username)

        if model_repo_id:
            metrics_yaml = _build_eval_results_yaml(task_name, scores, settings.hf_results_dataset)
            api.upload_file(
                path_or_fileobj=metrics_yaml.encode(),
                path_in_repo=f".eval_results/colliderml_{task_name}.yaml",
                repo_id=model_repo_id,
                commit_message=f"ColliderML {task_name} benchmark scores",
            )
            logger.info("pushed eval_results to model repo %s", model_repo_id)
    except Exception:
        logger.exception("HF result push failed (non-fatal)")


def _build_eval_results_yaml(task_name: str, scores: dict, dataset_id: str) -> str:
    import yaml

    metrics = [
        {"type": k, "value": round(float(v), 6), "name": k, "verified": False}
        for k, v in scores.items()
    ]
    doc = [
        {
            "config": f"colliderml_{task_name}",
            "dataset": {
                "type": dataset_id,
                "name": f"ColliderML {task_name.replace('_', ' ').title()} Benchmark",
            },
            "metrics": metrics,
            "source": {
                "url": f"https://api.colliderml.com/v1/leaderboard/{task_name}",
                "name": "ColliderML Backend",
            },
        }
    ]
    return yaml.dump(doc, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.get("/benchmark/tasks")
async def list_tasks_route() -> list[dict]:
    from benchmarks import list_tasks, get_task
    rows = []
    for name in list_tasks():
        t = get_task(name)
        rows.append(
            {
                "name": t.name,
                "dataset": t.dataset,
                "metrics": t.metrics,
                "higher_is_better": t.higher_is_better,
                "eval_event_range": list(t.eval_event_range),
            }
        )
    return rows


@router.get("/leaderboard/{task_name}")
async def get_leaderboard(task_name: str, limit: int = 100) -> list[dict]:
    """Public leaderboard for a task. Returns submissions sorted by primary metric."""
    await _load_task(task_name)  # validate name
    rows = await db.pool.fetch(
        """
        select id, hf_username, submitted_at, scores, is_baseline,
               n_params, code_url, credits_earned
        from benchmark_submissions
        where task = $1
        order by submitted_at desc
        limit $2
        """,
        task_name,
        limit,
    )
    return [dict(r) for r in rows]


@router.post("/benchmark/{task_name}/submit")
async def submit_benchmark(
    task_name: str,
    predictions: UploadFile = File(...),
    local_scores: str = Form(default="{}"),
    model_repo_id: str = Form(default=""),
    user: dict = Depends(current_user),
) -> dict:
    """Re-score a prediction file and award credits for new bests."""
    task = await _load_task(task_name)

    blob = await predictions.read()
    pred_hash = _sha256_bytes(blob)

    # Dedup: same user submitting the same file twice earns nothing extra.
    existing = await db.pool.fetchrow(
        """
        select id, scores, credits_earned from benchmark_submissions
        where task = $1 and hf_username = $2 and predictions_sha256 = $3
        """,
        task_name,
        user["hf_username"],
        pred_hash,
    )
    if existing is not None:
        return {
            "submission_id": str(existing["id"]),
            "scores": existing["scores"],
            "credits_earned": float(existing["credits_earned"]),
            "deduplicated": True,
        }

    try:
        table = pq.read_table(pa.BufferReader(blob))
    except Exception as e:
        raise HTTPException(400, f"Could not parse predictions as Parquet: {e}")

    scores = await _score_table(task, table)

    # Insert submission
    submission_row = await db.pool.fetchrow(
        """
        insert into benchmark_submissions
            (task, hf_username, predictions_sha256, scores, credits_earned)
        values ($1, $2, $3, $4::jsonb, 0)
        returning id
        """,
        task_name,
        user["hf_username"],
        pred_hash,
        json.dumps(scores),
    )
    submission_id = str(submission_row["id"])

    # Award credits for any metric that beats the current best
    credits_earned = 0.0
    reward = _REWARD_PER_METRIC.get(task_name, 50)
    for metric, value in scores.items():
        current = await db.pool.fetchrow(
            "select value from benchmark_bests where task=$1 and metric=$2",
            task_name, metric,
        )
        current_val = float(current["value"]) if current else None
        is_best = (
            current_val is None
            or task.is_better(metric, float(value), current_val)
        )
        if is_best:
            await db.pool.execute(
                """
                insert into benchmark_bests (task, metric, value, submission_id, hf_username)
                values ($1, $2, $3, $4, $5)
                on conflict (task, metric) do update set
                    value = excluded.value,
                    submission_id = excluded.submission_id,
                    hf_username = excluded.hf_username,
                    updated_at = now()
                """,
                task_name, metric, float(value), submission_row["id"], user["hf_username"],
            )
            credits_earned += reward

    if credits_earned > 0:
        await db.add_credit_transaction(
            user["hf_username"],
            credits_earned,
            reason=f"beat_{task_name}",
            metadata={"submission_id": submission_id, "scores": scores},
        )
        await db.pool.execute(
            "update benchmark_submissions set credits_earned = $1 where id = $2",
            credits_earned, submission_row["id"],
        )

    asyncio.create_task(_push_result_to_hf(
        task_name=task_name,
        submission_id=submission_id,
        username=user["hf_username"],
        scores=scores,
        credits_earned=credits_earned,
        pred_hash=pred_hash,
        model_repo_id=model_repo_id,
    ))

    return {
        "submission_id": submission_id,
        "scores": scores,
        "credits_earned": credits_earned,
        "deduplicated": False,
    }


@router.post("/benchmark/{task_name}/reproduce/{submission_id}")
async def reproduce_benchmark(
    task_name: str,
    submission_id: str,
    predictions: UploadFile = File(...),
    user: dict = Depends(current_user),
) -> dict:
    """Reproduce another user's submission. Earns 20 credits if within 2%."""
    task = await _load_task(task_name)

    original = await db.pool.fetchrow(
        "select id, scores, hf_username from benchmark_submissions where id = $1 and task = $2",
        submission_id, task_name,
    )
    if original is None:
        raise HTTPException(404, f"Submission {submission_id} not found for task {task_name}")
    if original["hf_username"] == user["hf_username"]:
        raise HTTPException(400, "You cannot reproduce your own submission")

    blob = await predictions.read()
    try:
        table = pq.read_table(pa.BufferReader(blob))
    except Exception as e:
        raise HTTPException(400, f"Could not parse predictions as Parquet: {e}")

    new_scores = await _score_table(task, table)
    original_scores = original["scores"] if isinstance(original["scores"], dict) else json.loads(original["scores"])

    within = True
    for metric in original_scores:
        orig = float(original_scores[metric])
        new = float(new_scores.get(metric, 0))
        if abs(orig) < 1e-9:
            delta = abs(new - orig)
        else:
            delta = abs(new - orig) / abs(orig)
        if delta > _REPRODUCE_TOLERANCE:
            within = False
            break

    credits = _REPRODUCE_REWARD if within else 0.0

    await db.pool.execute(
        """
        insert into benchmark_reproductions
            (submission_id, reproducer, reproduced_scores, within_tolerance, credits_earned)
        values ($1, $2, $3::jsonb, $4, $5)
        """,
        original["id"], user["hf_username"], json.dumps(new_scores), within, credits,
    )

    if within:
        await db.add_credit_transaction(
            user["hf_username"],
            credits,
            reason="reproduction",
            metadata={"submission_id": submission_id, "task": task_name},
        )

    return {
        "within_tolerance": within,
        "reproduced_scores": new_scores,
        "original_scores": original_scores,
        "credits_earned": credits,
    }
