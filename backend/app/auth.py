"""
HuggingFace token verification and user upsert.

Binary auth: you either present a valid HF token or you don't get in.
No tiers, no age gates. On first sign-in we grant seed credits. Banned
users are rejected with 403.
"""

from __future__ import annotations

import os

import httpx
from fastapi import Depends, Header, HTTPException

from app.db import db


async def _maybe_ci_test_user() -> dict | None:
    """CI/hermetic auth seam: when ALLOW_TEST_TOKEN is set, any bearer token
    resolves to a single seeded test user instead of being verified against
    HuggingFace. This lets an out-of-process client (the library) authenticate
    in CI with no real HF token. NEVER set ALLOW_TEST_TOKEN in production.
    """
    if not os.environ.get("ALLOW_TEST_TOKEN"):
        return None
    username = os.environ.get("AUTH_TEST_USER", "ci-pipeline")
    user = await db.get_user(username)
    if user is None:
        await db.create_user(username, None, credits=0)
        user = await db.get_user(username)
    return user


async def _fetch_hf_whoami(token: str) -> dict:
    """Call HF's whoami-v2 endpoint."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
        )
    if r.status_code != 200:
        raise HTTPException(401, "Invalid HuggingFace token")
    return r.json()


async def verify_hf_token(token: str) -> dict:
    """Verify a token and return the user row from our database.

    On first sign-in, creates the user row and grants seed credits
    (via a single credit_transactions entry).
    """
    info = await _fetch_hf_whoami(token)
    username = info.get("name")
    email = info.get("email")
    if not username:
        raise HTTPException(401, "HF whoami returned no username")

    user = await db.get_user(username)
    if user is None:
        cfg = await db.get_global_config()
        user = await db.create_user(username, email, credits=0)
        seed = float(cfg["seed_credits"])
        await db.add_credit_transaction(
            username,
            seed,
            reason="signup",
            metadata={"source": "first_signin"},
        )
        user = await db.get_user(username)
    else:
        await db.touch_last_seen(username)

    if user["banned"]:
        raise HTTPException(403, "This account has been revoked. Contact an admin.")

    return user


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------
async def current_user(
    authorization: str = Header(default=""),
) -> dict:
    """Extract and verify the HF token from the Authorization header."""
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing Bearer token in Authorization header")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(401, "Empty Bearer token")
    ci_user = await _maybe_ci_test_user()
    if ci_user is not None:
        return ci_user
    return await verify_hf_token(token)


async def admin_only(
    x_admin_token: str = Header(default=""),
) -> None:
    """Protect admin routes with a shared-secret header."""
    from app.config import get_settings
    settings = get_settings()
    if not x_admin_token or x_admin_token != settings.admin_token:
        raise HTTPException(401, "Invalid or missing X-Admin-Token")
