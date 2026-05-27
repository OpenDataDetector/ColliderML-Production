"""
SFAPI runner: submit ColliderML jobs to Perlmutter and poll until they
terminate.

Runs as an async background task owned by the FastAPI app. One `SFAPIRunner`
instance is created at app startup and holds a long-lived SFAPI client that
is refreshed on token expiry.

Key responsibilities:
    - Render the sbatch template for each request
    - Upload it to NERSC scratch
    - Submit via sfapi_client
    - Poll status every `poll_interval_seconds` (configurable)
    - Update simulation_requests.state in the database
    - Reconcile credits on completion/failure
    - Send completion email
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from jinja2 import Template

from app import abuse
from app.cap import estimate_node_hours
from app.config import get_settings
from app.db import db
from app.schemas import SimulateRequest

logger = logging.getLogger(__name__)

_TEMPLATE_PATH = Path(__file__).parent / "sbatch_template.sh.j2"


class SFAPIRunner:
    """Submit and poll ColliderML jobs via the NERSC Superfacility API."""

    def __init__(self) -> None:
        self._client = None
        self._template: Optional[Template] = None
        self._background_tasks: set[asyncio.Task] = set()

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------
    async def start(self) -> None:
        self._template = Template(_TEMPLATE_PATH.read_text())
        settings = get_settings()
        if not settings.sfapi_client_id or not settings.sfapi_client_secret:
            logger.warning(
                "SFAPI credentials not set — runner is in mock mode. "
                "Jobs will be recorded but never submitted."
            )
            return

        try:
            from sfapi_client import Client
        except ImportError:
            logger.error("sfapi_client not installed; runner running in mock mode.")
            return

        # The real Client is synchronous; we wrap calls with asyncio.to_thread.
        self._client = Client(
            client_id=settings.sfapi_client_id,
            secret=settings.sfapi_client_secret,
        )
        logger.info("SFAPI runner started")

    async def stop(self) -> None:
        # Cancel any outstanding background polls on shutdown
        for task in list(self._background_tasks):
            task.cancel()
        self._background_tasks.clear()
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Submission
    # -----------------------------------------------------------------------
    def _split_request(self, req: SimulateRequest) -> dict:
        """Decide how to distribute a request across SLURM nodes/tasks.

        Strategy:
            - events_per_task: aim for 100 events per task for ttbar, 200 for
              lighter channels. Keep each task under ~2 hours wall time.
            - tasks_per_node: 32 on Perlmutter CPU (128 cores; DDSim is
              single-threaded but memory-bound, 32 is a safe default).
            - n_nodes: ceil(total_tasks / tasks_per_node), capped at 8 so we
              never accidentally ask for a huge allocation.
        """
        import math
        target_per_task = 50 if req.channel == "ttbar" else 100
        events_per_task = max(10, min(target_per_task, req.events))
        n_tasks = math.ceil(req.events / events_per_task)
        tasks_per_node = 32
        n_nodes = max(1, min(8, math.ceil(n_tasks / tasks_per_node)))
        if n_tasks < tasks_per_node:
            tasks_per_node = n_tasks
        return {
            "n_nodes": n_nodes,
            "tasks_per_node": tasks_per_node,
            "events_per_task": events_per_task,
            "n_tasks": n_tasks,
        }

    def _render_sbatch(
        self,
        request_id: str,
        req: SimulateRequest,
        user_email: Optional[str],
    ) -> str:
        settings = get_settings()
        letter = (settings.nersc_user or "x")[0]
        work_dir = f"/pscratch/sd/{letter}/{settings.nersc_user}/colliderml/{request_id}"
        cache_dir = f"/pscratch/sd/{letter}/{settings.nersc_user}/colliderml/.cache"
        output_hf_repo = (
            f"{settings.hf_dataset_org}/ColliderML-Service-{request_id}"
        )

        split = self._split_request(req)

        # Queue + wall-time heuristic
        est_hours = estimate_node_hours(req.channel, req.events, req.pileup)
        if est_hours * 3600 < 25 * 60 and split["n_nodes"] == 1:
            qos = "debug"
            time_limit = "00:30:00"
        elif est_hours < 2.0:
            qos = "regular"
            time_limit = "02:00:00"
        else:
            qos = "regular"
            time_limit = "04:00:00"

        return self._template.render(
            request_id=request_id,
            channel=req.channel,
            events=req.events,
            pileup=req.pileup,
            seed=req.seed,
            project=settings.nersc_project,
            user_email=user_email or "",
            image=settings.container_image,
            repo_branch=settings.colliderml_branch,
            qos=qos,
            time_limit=time_limit,
            n_nodes=split["n_nodes"],
            tasks_per_node=split["tasks_per_node"],
            events_per_task=split["events_per_task"],
            work_dir=work_dir,
            cache_dir=cache_dir,
            upload_to_hf=bool(settings.hf_token),
            output_hf_repo=output_hf_repo,
            hf_token=settings.hf_token,
        )

    async def submit(
        self,
        request_id: str,
        req: SimulateRequest,
        user: dict,
    ) -> Optional[str]:
        """Submit a job to Perlmutter. Returns the SLURM jobid or None (mock)."""
        sbatch = self._render_sbatch(
            request_id=request_id,
            req=req,
            user_email=user.get("email"),
        )

        if self._client is None:
            # Mock mode: just update state and schedule a pretend completion
            logger.info("Mock submission for request %s", request_id)
            await db.update_request(request_id, state="submitted", nersc_jobid="mock-0")
            task = asyncio.create_task(self._mock_poll(request_id, user))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            return "mock-0"

        settings = get_settings()
        letter = settings.nersc_user[0]
        scratch_path = (
            f"/pscratch/sd/{letter}/{settings.nersc_user}/colliderml/{request_id}.sh"
        )

        def _submit_sync():
            from sfapi_client.compute import Machine
            perlmutter = self._client.compute(Machine.perlmutter)
            perlmutter.upload(scratch_path, sbatch)
            return perlmutter.submit_job(scratch_path)

        job = await asyncio.to_thread(_submit_sync)
        jobid = str(job.jobid)
        await db.update_request(request_id, state="submitted", nersc_jobid=jobid)

        task = asyncio.create_task(self._poll_real(request_id, user, job))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return jobid

    # -----------------------------------------------------------------------
    # Polling
    # -----------------------------------------------------------------------
    async def _poll_real(self, request_id: str, user: dict, job) -> None:
        """Poll SFAPI until the job reaches a terminal state."""
        settings = get_settings()
        terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"}
        while True:
            await asyncio.sleep(settings.poll_interval_seconds)
            try:
                await asyncio.to_thread(job.update)
            except Exception as e:
                logger.warning("SFAPI poll failed for %s: %s", request_id, e)
                continue
            state = str(job.state)
            if state == "RUNNING":
                await db.update_request(request_id, state="running")
            if state in terminal:
                await self._finalise(request_id, user, state)
                return

    async def _mock_poll(self, request_id: str, user: dict) -> None:
        """Mock mode: mark the request as completed after a short delay."""
        await asyncio.sleep(2)
        await self._finalise(request_id, user, "COMPLETED")

    async def _finalise(self, request_id: str, user: dict, slurm_state: str) -> None:
        request = await db.get_request(request_id)
        if request is None:
            return
        estimated = float(request["estimated_node_hours"])

        if slurm_state == "COMPLETED":
            # In real deployment we'd parse sacct for the actual runtime.
            actual = estimated
            await db.update_request(
                request_id,
                state="completed",
                actual_node_hours=actual,
                output_hf_repo=(
                    f"{get_settings().hf_dataset_org}/ColliderML-Service-{request_id}"
                ),
            )
            await abuse.reconcile(request_id, user["hf_username"], estimated, actual)
            # Fire-and-forget email
            try:
                from app.email import send_completion
                await send_completion(request_id)
            except Exception as e:
                logger.warning("Failed to send completion email: %s", e)
        else:
            await db.update_request(
                request_id,
                state="failed",
                error_message=f"SLURM state: {slurm_state}",
            )
            # Full refund on failure
            await abuse.refund_full(
                request_id,
                user["hf_username"],
                estimated,
                reason="refund_job_failed",
            )
