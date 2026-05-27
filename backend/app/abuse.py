"""
Abuse detection and credit-charging for simulation requests.

No tiers. No per-user quotas. The credit balance is the only per-user gate.
Everything else here is global fraud prevention.

Call `check_and_charge(user, req)` before submitting to SFAPI. On success it
returns `(estimated_node_hours, config_hash)` and has already deducted the
credits from the user's balance. The SFAPI runner then reconciles the
difference (refund or extra charge) when the job terminates.
"""

from __future__ import annotations

import hashlib
import json

from fastapi import HTTPException

from app.cap import estimate_node_hours
from app.db import db
from app.schemas import SimulateRequest


def hash_request(req: SimulateRequest) -> str:
    """Stable SHA-256 of the normalised request payload."""
    payload = {
        "channel": req.channel,
        "events": req.events,
        "pileup": req.pileup,
        "seed": req.seed,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


async def check_and_charge(user: dict, req: SimulateRequest) -> tuple[float, str]:
    """Run all gates and atomically deduct credits if everything passes.

    Raises HTTPException on any failure. Returns (est_node_hours, config_hash).
    """
    if user.get("banned"):
        raise HTTPException(403, "This account has been revoked. Contact an admin.")

    cfg = await db.get_global_config()

    # 1. Kill switch
    if cfg["submissions_frozen"]:
        raise HTTPException(
            503, "Submissions are temporarily frozen by an admin. Please try again later."
        )

    # 2. Burst detection (rapid-fire protection)
    n_5min = await db.count_requests_recent(user["hf_username"], minutes=5)
    if n_5min >= 10:
        raise HTTPException(
            429,
            "Too many submissions in the last 5 minutes. Please wait a moment.",
        )

    # 3. Duplicate config dedup
    config_hash = hash_request(req)
    existing = await db.find_completed_by_hash(config_hash, since_days=7)
    if existing:
        raise HTTPException(
            409,
            {
                "detail": "This exact request already ran recently. Reuse the existing dataset.",
                "existing_request_id": str(existing["id"]),
                "output_hf_repo": existing.get("output_hf_repo"),
            },
        )

    # 4. Cost estimation
    est = estimate_node_hours(req.channel, req.events, req.pileup)

    # 5. Global monthly cap
    used = await db.monthly_usage_total()
    cap = float(cfg["monthly_node_hours_cap"])
    if used + est > cap:
        raise HTTPException(
            503,
            "Monthly compute budget reached. Try again next month, or run locally with Docker/Podman.",
        )

    # 6. User balance
    current = float(user["credits"])
    if current < est:
        raise HTTPException(
            402,
            f"Insufficient credits: this job needs {est:.2f} but you have {current:.2f}. "
            f"Earn more by submitting to the leaderboard or contributing baselines.",
        )

    # 7. Atomic deduction
    try:
        await db.deduct_credits(
            user["hf_username"],
            est,
            reason="spent_on_request",
            metadata={"config_hash": config_hash},
        )
    except ValueError as e:
        # Race condition: balance changed between step 6 and step 7.
        raise HTTPException(402, str(e))

    return est, config_hash


async def refund_full(request_id: str, hf_username: str, amount: float, reason: str) -> None:
    """Refund the full charged amount (used when a job fails)."""
    await db.add_credit_transaction(
        hf_username,
        amount,
        reason=reason,
        metadata={"request_id": str(request_id)},
    )


async def reconcile(
    request_id: str,
    hf_username: str,
    estimated: float,
    actual: float,
) -> None:
    """Apply the difference between estimated and actual cost.

    Positive delta (actual > estimated) would charge more, but we cap at the
    original estimate for user-friendliness: never charge more than quoted.
    Negative delta (actual < estimated) refunds the overage.
    """
    if actual < estimated:
        refund = estimated - actual
        await db.add_credit_transaction(
            hf_username,
            refund,
            reason="reconciliation_refund",
            metadata={"request_id": str(request_id), "actual": actual, "estimated": estimated},
        )
