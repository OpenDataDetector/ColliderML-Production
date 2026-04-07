"""
GitHub webhook handler for credit automation.

Configured on the ColliderML-Production repo to POST on pull_request.closed
events. On a merged PR with a known label, the author earns credits:

    type:baseline   -> 100 credits
    type:bugfix     ->  30 credits
    type:docs       ->  10 credits
    type:channel    -> 200 credits (flagged for manual review first)
    type:config-review -> 20 credits

The author is mapped from GitHub username to HF username via a table in the
database (gh_hf_mapping), which users populate by adding their GitHub
username to their HF profile bio or via a dedicated endpoint.

Signature verification: HMAC-SHA256 using GITHUB_WEBHOOK_SECRET.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Header, Request

from app.db import db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhooks", tags=["webhooks"])


REWARDS = {
    "type:baseline": 100,
    "type:bugfix": 30,
    "type:docs": 10,
    "type:channel": 200,
    "type:config-review": 20,
}


async def _verify_signature(payload: bytes, signature: str) -> bool:
    secret = os.environ.get("GITHUB_WEBHOOK_SECRET", "")
    if not secret:
        # If no secret is set, reject — don't silently accept untrusted payloads.
        return False
    if not signature or not signature.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    provided = signature.removeprefix("sha256=")
    return hmac.compare_digest(expected, provided)


async def _gh_to_hf(gh_username: str) -> str | None:
    """Map a GitHub username to a HF username via the database."""
    row = await db.pool.fetchrow(
        "select hf_username from gh_hf_mapping where gh_username = $1",
        gh_username,
    )
    return row["hf_username"] if row else None


@router.post("/github")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str = Header(default=""),
    x_github_event: str = Header(default=""),
) -> dict:
    body = await request.body()
    if not await _verify_signature(body, x_hub_signature_256):
        raise HTTPException(401, "Invalid GitHub webhook signature")

    if x_github_event != "pull_request":
        return {"ignored": True, "reason": f"event={x_github_event}"}

    payload = json.loads(body)
    if payload.get("action") != "closed":
        return {"ignored": True, "reason": f"action={payload.get('action')}"}

    pr = payload.get("pull_request", {})
    if not pr.get("merged"):
        return {"ignored": True, "reason": "pr was closed without merge"}

    gh_user = pr.get("user", {}).get("login")
    if not gh_user:
        return {"ignored": True, "reason": "no author"}

    hf_user = await _gh_to_hf(gh_user)
    if hf_user is None:
        return {"ignored": True, "reason": f"gh user {gh_user} not mapped"}

    labels = {l["name"] for l in pr.get("labels", [])}
    credits_total = 0.0
    matched = []
    for label, amount in REWARDS.items():
        if label in labels:
            credits_total += amount
            matched.append(label)

    if credits_total == 0:
        return {"ignored": True, "reason": "no rewarding label"}

    await db.add_credit_transaction(
        hf_user,
        credits_total,
        reason="github_pr_merged",
        metadata={
            "pr_url": pr.get("html_url"),
            "pr_number": pr.get("number"),
            "labels": matched,
            "gh_user": gh_user,
        },
    )

    return {
        "credited": True,
        "hf_user": hf_user,
        "amount": credits_total,
        "labels": matched,
    }
