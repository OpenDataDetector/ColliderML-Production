"""Shared fixtures for the backend test suite.

The ``client`` fixture spins up the FastAPI app via Starlette's
``TestClient`` and overrides ``current_user`` so tests don't have to
juggle real HuggingFace tokens. The ``_reset_user`` autouse fixture
wipes the test user's state at the start of every test so cases stay
independent — credits, simulation_requests, benchmark_submissions, the
``submissions_frozen`` flag.

Both fixtures gate on ``DATABASE_URL`` being set; when it isn't, the
client fixture skips so unit tests in the same suite still run.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

import pytest


pytest_plugins = ["pytest_asyncio"]


_HAS_DB = bool(os.environ.get("DATABASE_URL"))


def _ensure_user(dsn: str) -> None:
    """Insert e2e_alice with 10 credits and a clean ledger (idempotent)."""
    import asyncpg

    async def _run() -> None:
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute(
                """
                insert into users (hf_username, email, credits)
                values ('e2e_alice', 'alice@test.example', 10)
                on conflict (hf_username) do update set credits = 10, banned = false
                """
            )
            # Clear prior runs so dedup branches and credit accounting reset
            # cleanly between tests.
            await conn.execute(
                "delete from simulation_requests where hf_username = 'e2e_alice'"
            )
            await conn.execute(
                "delete from credit_transactions where hf_username = 'e2e_alice'"
            )
            await conn.execute(
                "delete from benchmark_submissions where hf_username = 'e2e_alice'"
            )
            await conn.execute(
                "delete from benchmark_bests where hf_username = 'e2e_alice'"
            )
            await conn.execute(
                "insert into credit_transactions (hf_username, delta, reason)"
                " values ('e2e_alice', 10, 'signup')"
            )
            await conn.execute(
                "update global_config set submissions_frozen = false where id = 1"
            )
        finally:
            await conn.close()

    asyncio.get_event_loop().run_until_complete(_run())


@pytest.fixture(autouse=True)
def _reset_user() -> None:
    if _HAS_DB:
        _ensure_user(os.environ["DATABASE_URL"])


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """FastAPI TestClient with stubbed HF auth as ``e2e_alice``."""
    if not _HAS_DB:
        pytest.skip("DATABASE_URL not set")

    from fastapi.testclient import TestClient

    # Force the mock SFAPI runner — these tests never want to hit Perlmutter.
    monkeypatch.delenv("SFAPI_CLIENT_ID", raising=False)
    monkeypatch.delenv("SFAPI_CLIENT_SECRET", raising=False)

    from app.auth import current_user
    from app.main import app

    async def fake_user():
        # Pull live credits from the DB so the test sees deductions/grants
        # made by the route under test, not a snapshot.
        import asyncpg

        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            row = await conn.fetchrow(
                "select credits from users where hf_username = 'e2e_alice'"
            )
            credits = float(row["credits"]) if row else 10.0
        finally:
            await conn.close()
        now = datetime.now(timezone.utc)
        return {
            "hf_username": "e2e_alice",
            "email": "alice@test.example",
            "credits": credits,
            "banned": False,
            "created_at": now,
            "last_seen_at": now,
            "notes": None,
        }

    app.dependency_overrides[current_user] = fake_user
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(current_user, None)
