#!/usr/bin/env python3
"""
SFAPI proof-of-concept: submit a single simulation job to Perlmutter.

This is a standalone script that validates the end-to-end NERSC submission
path before we build the FastAPI backend in Phase 2. It mirrors the logic
that `backend/app/sfapi_runner.py` will eventually implement, but with
synchronous polling and no database.

Environment variables required:
    SFAPI_CLIENT_ID      - NERSC IRIS service account client ID
    SFAPI_CLIENT_SECRET  - Corresponding private key (PEM-encoded)
    NERSC_PROJECT        - NERSC project (e.g. "m4958")
    NERSC_USER           - Your NERSC username (used for scratch paths)

Optional:
    COLLIDERML_BRANCH    - Git branch to clone (default: main)
    COLLIDERML_IMAGE     - Shifter image tag
                           (default: ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0)
    DRY_RUN=1            - Print the sbatch script without submitting

Usage:
    python scripts/sfapi/poc_submit.py --channel higgs_portal --events 10 --pileup 10
"""

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

try:
    from jinja2 import Template
except ImportError:
    print("ERROR: jinja2 not installed. Run: pip install jinja2", file=sys.stderr)
    sys.exit(1)

DEFAULT_IMAGE = os.environ.get(
    "COLLIDERML_IMAGE",
    "ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0",
)
DEFAULT_BRANCH = os.environ.get("COLLIDERML_BRANCH", "main")


def render_sbatch(
    request_id: str,
    channel: str,
    events: int,
    pileup: int,
    seed: int,
    project: str,
    user: str,
    *,
    image: str = DEFAULT_IMAGE,
    repo_branch: str = DEFAULT_BRANCH,
    qos: str = "debug",
    time_limit: str = "00:30:00",
    n_nodes: int = 1,
    user_email: str = "",
    upload_to_hf: bool = False,
    output_hf_repo: str = "",
) -> str:
    """Render the sbatch template for a single request."""
    template_path = Path(__file__).parent / "sbatch_template.sh.j2"
    tmpl = Template(template_path.read_text())

    work_dir = f"/pscratch/sd/{user[0]}/{user}/colliderml/{request_id}"
    cache_dir = f"/pscratch/sd/{user[0]}/{user}/colliderml/.cache"

    return tmpl.render(
        request_id=request_id,
        channel=channel,
        events=events,
        pileup=pileup,
        seed=seed,
        project=project,
        user=user,
        user_email=user_email,
        image=image,
        repo_branch=repo_branch,
        qos=qos,
        time_limit=time_limit,
        n_nodes=n_nodes,
        work_dir=work_dir,
        cache_dir=cache_dir,
        upload_to_hf=upload_to_hf,
        output_hf_repo=output_hf_repo,
    )


def submit_and_poll(
    channel: str,
    events: int,
    pileup: int,
    seed: int = 42,
    poll_interval: int = 30,
) -> dict:
    """Submit a job via SFAPI and poll until it terminates.

    Returns a dict with final state and output path.
    Raises on submission errors or non-zero exit.
    """
    project = os.environ["NERSC_PROJECT"]
    user = os.environ["NERSC_USER"]

    request_id = f"poc_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    sbatch = render_sbatch(
        request_id=request_id,
        channel=channel,
        events=events,
        pileup=pileup,
        seed=seed,
        project=project,
        user=user,
    )

    if os.environ.get("DRY_RUN") == "1":
        print("=== DRY RUN: sbatch script ===")
        print(sbatch)
        print("=== END ===")
        return {"request_id": request_id, "state": "dry-run"}

    try:
        from sfapi_client import Client
        from sfapi_client.compute import Machine
    except ImportError:
        raise RuntimeError(
            "sfapi_client not installed. Run: pip install sfapi-client"
        )

    with Client(
        client_id=os.environ["SFAPI_CLIENT_ID"],
        secret=os.environ["SFAPI_CLIENT_SECRET"],
    ) as client:
        perlmutter = client.compute(Machine.perlmutter)

        # Upload the sbatch script to scratch
        scratch_path = f"/pscratch/sd/{user[0]}/{user}/colliderml/{request_id}.sh"
        perlmutter.upload(scratch_path, sbatch)

        job = perlmutter.submit_job(scratch_path)
        print(f"Submitted job {job.jobid} (request {request_id})")
        print(f"  sbatch: {scratch_path}")
        print(f"  work:   /pscratch/sd/{user[0]}/{user}/colliderml/{request_id}")

        terminal_states = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"}
        while True:
            job.update()
            print(f"  state={job.state}")
            if job.state in terminal_states:
                break
            time.sleep(poll_interval)

        return {
            "request_id": request_id,
            "jobid": job.jobid,
            "state": str(job.state),
            "work_dir": f"/pscratch/sd/{user[0]}/{user}/colliderml/{request_id}",
        }


def main():
    parser = argparse.ArgumentParser(description="SFAPI proof-of-concept submitter")
    parser.add_argument("--channel", default="higgs_portal")
    parser.add_argument("--events", type=int, default=10)
    parser.add_argument("--pileup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poll-interval", type=int, default=30)
    args = parser.parse_args()

    try:
        result = submit_and_poll(
            channel=args.channel,
            events=args.events,
            pileup=args.pileup,
            seed=args.seed,
            poll_interval=args.poll_interval,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print()
    print("=== Result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    if result.get("state") not in ("COMPLETED", "dry-run"):
        sys.exit(2)


if __name__ == "__main__":
    main()
