"""
Remote simulation client — talks to the ColliderML backend service.

Used by `colliderml.simulate(remote=True)` to submit jobs to NERSC via the
FastAPI backend. Requires a HuggingFace token for authentication.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests

DEFAULT_BACKEND_URL = os.environ.get(
    "COLLIDERML_BACKEND",
    "https://api.colliderml.com",
)
POLL_INTERVAL_SECONDS = 30
SUBMIT_TIMEOUT_SECONDS = 60


def _get_hf_token() -> Optional[str]:
    """Find a HF token via env var or the huggingface_hub saved token."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token()
    except Exception:
        return None


def _require_token() -> str:
    token = _get_hf_token()
    if not token:
        raise RuntimeError(
            "Remote simulation requires a HuggingFace token.\n"
            "Run one of:\n"
            "  huggingface-cli login           (interactive)\n"
            "  export HF_TOKEN=<your-token>    (env var)\n\n"
            "If you don't have an HF account yet, create one at https://huggingface.co\n"
            "and generate a read token at https://huggingface.co/settings/tokens"
        )
    return token


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def submit_remote(
    channel: str,
    events: int,
    pileup: int,
    seed: int = 42,
    *,
    backend_url: Optional[str] = None,
    poll: bool = True,
) -> "RemoteSimulationResult":
    """Submit a simulation request to the ColliderML backend service.

    Args:
        channel: Physics channel name.
        events:  Number of events.
        pileup:  Pileup level.
        seed:    Random seed.
        backend_url: Override the backend URL (defaults to COLLIDERML_BACKEND env
                     var or https://api.colliderml.com).
        poll: If True, block until the job terminates and return a fully
              populated result. If False, return immediately after submission
              with state="submitted".

    Returns:
        RemoteSimulationResult with request_id, state, and (if poll=True) a
        downloaded SimulationResult-compatible object.
    """
    token = _require_token()
    url = (backend_url or DEFAULT_BACKEND_URL).rstrip("/")

    payload = {
        "channel": channel,
        "events": events,
        "pileup": pileup,
        "seed": seed,
    }

    try:
        r = requests.post(
            f"{url}/v1/simulate",
            json=payload,
            headers=_auth_headers(token),
            timeout=SUBMIT_TIMEOUT_SECONDS,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Could not reach ColliderML backend at {url}: {e}")

    if r.status_code == 401:
        raise RuntimeError(
            "Authentication failed. Your HF token may be invalid or expired.\n"
            "Run: huggingface-cli login"
        )
    if r.status_code == 402:
        # Insufficient credits — surface the friendly message from the server.
        raise RuntimeError(r.json().get("detail", "Insufficient credits"))
    if r.status_code == 409:
        detail = r.json().get("detail", {})
        if isinstance(detail, dict):
            existing = detail.get("output_hf_repo")
            print(
                f"This exact request already ran. Reusing existing dataset: {existing}",
                file=sys.stderr,
            )
            return RemoteSimulationResult(
                request_id=detail.get("existing_request_id", "(unknown)"),
                state="completed",
                output_hf_repo=existing,
                credits_charged=0.0,
                estimated_node_hours=0.0,
                estimated_completion_seconds=0,
                backend_url=url,
                token=token,
                channel=channel,
                events=events,
                pileup=pileup,
            )
        raise RuntimeError(f"Duplicate request: {detail}")
    if r.status_code >= 400:
        raise RuntimeError(f"Backend error {r.status_code}: {r.text}")

    data = r.json()
    request_id = data["request_id"]
    result = RemoteSimulationResult(
        request_id=request_id,
        state=data["state"],
        output_hf_repo=data.get("output_hf_repo"),
        credits_charged=float(data.get("credits_charged", 0)),
        estimated_node_hours=float(data.get("estimated_node_hours", 0)),
        estimated_completion_seconds=int(data.get("estimated_completion_seconds", 0)),
        backend_url=url,
        token=token,
        channel=channel,
        events=events,
        pileup=pileup,
    )

    if data.get("cached"):
        print(f"Reused cached dataset: {result.output_hf_repo}")
        return result

    print(
        f"Submitted request {request_id} "
        f"(est {result.estimated_node_hours:.2f} credits, "
        f"~{result.estimated_completion_seconds // 60} min)."
    )

    if poll:
        return result.wait()
    return result


def get_status(
    request_id: str,
    *,
    backend_url: Optional[str] = None,
) -> dict:
    token = _require_token()
    url = (backend_url or DEFAULT_BACKEND_URL).rstrip("/")
    r = requests.get(
        f"{url}/v1/requests/{request_id}",
        headers=_auth_headers(token),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def get_me(*, backend_url: Optional[str] = None) -> dict:
    token = _require_token()
    url = (backend_url or DEFAULT_BACKEND_URL).rstrip("/")
    r = requests.get(
        f"{url}/v1/me",
        headers=_auth_headers(token),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


class RemoteSimulationResult:
    """Handle for a simulation running on the backend service.

    Can be polled, waited on, or converted to a SimulationResult once the
    output parquet files have been downloaded locally.
    """

    def __init__(
        self,
        request_id: str,
        state: str,
        output_hf_repo: Optional[str],
        credits_charged: float,
        estimated_node_hours: float,
        estimated_completion_seconds: int,
        backend_url: str,
        token: str,
        channel: str,
        events: int,
        pileup: int,
    ) -> None:
        self.request_id = request_id
        self.state = state
        self.output_hf_repo = output_hf_repo
        self.credits_charged = credits_charged
        self.estimated_node_hours = estimated_node_hours
        self.estimated_completion_seconds = estimated_completion_seconds
        self.backend_url = backend_url
        self._token = token
        self.channel = channel
        self.events = events
        self.pileup = pileup
        self._downloaded = None  # cached SimulationResult

    def __repr__(self) -> str:
        return (
            f"RemoteSimulationResult(request_id={self.request_id!r}, "
            f"state={self.state!r}, channel={self.channel!r}, "
            f"events={self.events}, pileup={self.pileup})"
        )

    def update(self) -> None:
        """Refresh state from the backend."""
        data = get_status(self.request_id, backend_url=self.backend_url)
        self.state = data["state"]
        self.output_hf_repo = data.get("output_hf_repo")

    def wait(self, timeout: Optional[int] = None) -> "RemoteSimulationResult":
        """Block until the request terminates. Raises on failure."""
        start = time.time()
        terminal = {"completed", "failed", "cancelled"}
        while self.state not in terminal:
            if timeout is not None and (time.time() - start) > timeout:
                raise TimeoutError(f"Request {self.request_id} did not complete within {timeout}s")
            time.sleep(POLL_INTERVAL_SECONDS)
            try:
                self.update()
            except requests.RequestException as e:
                print(f"  warning: poll failed: {e}", file=sys.stderr)
                continue
            print(f"  state={self.state}")

        if self.state == "failed":
            raise RuntimeError(f"Request {self.request_id} failed.")
        if self.state == "cancelled":
            raise RuntimeError(f"Request {self.request_id} was cancelled.")
        return self

    def download(self, cache_dir: Optional[Path] = None):
        """Download the per-request HF dataset and return a SimulationResult."""
        if self._downloaded is not None:
            return self._downloaded
        if not self.output_hf_repo:
            raise RuntimeError("No output_hf_repo on this request — nothing to download.")

        from colliderml._loader import load as _load
        # Temporarily override the HF_REPO in the loader module.
        import colliderml._loader as loader
        saved_repo = loader.HF_REPO
        loader.HF_REPO = self.output_hf_repo
        try:
            tables = _load(
                f"{self.channel}_pu{self.pileup}",
                tables=["tracker_hits", "particles", "tracks"],
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        finally:
            loader.HF_REPO = saved_repo

        from colliderml._simulate import SimulationResult
        result = SimulationResult(
            output_dir=str(cache_dir or Path.home() / ".cache/colliderml"),
            run_dir=str(cache_dir or Path.home() / ".cache/colliderml"),
            channel=self.channel,
            events=self.events,
            pileup=self.pileup,
            stages=[{"stage": "remote", "returncode": 0}],
        )
        result._remote_tables = tables  # attach for lazy access
        self._downloaded = result
        return result

    # Convenience: make the remote result quack like a SimulationResult
    @property
    def particles(self):
        return self.download().particles if self.output_hf_repo else None

    @property
    def tracks(self):
        return self.download().tracks if self.output_hf_repo else None

    @property
    def tracker_hits(self):
        return self.download().tracker_hits if self.output_hf_repo else None
