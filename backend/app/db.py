"""
Async database wrapper for the ColliderML backend.

All SQL lives here. The rest of the app talks to this module via the
`db` singleton. Uses asyncpg for direct Postgres connections (Supabase
exposes Postgres on :5432 or :6543).

For tests, you can swap this implementation out with a fake that keeps
state in memory — see tests/conftest.py.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg

from app.config import get_settings

logger = logging.getLogger(__name__)


class Database:
    """Thin wrapper around an asyncpg connection pool."""

    def __init__(self) -> None:
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        if self._pool is not None:
            return
        settings = get_settings()
        self._pool = await asyncpg.create_pool(
            settings.database_url,
            min_size=1,
            max_size=10,
            command_timeout=30,
        )
        logger.info("Database pool created")

    async def disconnect(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._pool

    # -----------------------------------------------------------------------
    # Users
    # -----------------------------------------------------------------------
    async def get_user(self, hf_username: str) -> Optional[dict]:
        row = await self.pool.fetchrow(
            "select * from users where hf_username = $1",
            hf_username,
        )
        return dict(row) if row else None

    async def create_user(
        self,
        hf_username: str,
        email: Optional[str],
        credits: float,
    ) -> dict:
        row = await self.pool.fetchrow(
            """
            insert into users (hf_username, email, credits)
            values ($1, $2, $3)
            on conflict (hf_username) do update set email = excluded.email
            returning *
            """,
            hf_username,
            email,
            credits,
        )
        return dict(row)

    async def touch_last_seen(self, hf_username: str) -> None:
        await self.pool.execute(
            "update users set last_seen_at = now() where hf_username = $1",
            hf_username,
        )

    async def set_banned(self, hf_username: str, banned: bool) -> None:
        await self.pool.execute(
            "update users set banned = $1 where hf_username = $2",
            banned,
            hf_username,
        )

    # -----------------------------------------------------------------------
    # Credits
    # -----------------------------------------------------------------------
    async def add_credit_transaction(
        self,
        hf_username: str,
        delta: float,
        reason: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Record a credit change in the ledger AND update the user balance.

        Must be called inside the same transaction as any simulation_requests
        state change that depends on it, to keep the ledger consistent.
        """
        import json as _json
        meta_json = _json.dumps(metadata) if metadata is not None else None
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    insert into credit_transactions (hf_username, delta, reason, metadata)
                    values ($1, $2, $3, $4::jsonb)
                    """,
                    hf_username,
                    delta,
                    reason,
                    meta_json,
                )
                await conn.execute(
                    "update users set credits = credits + $1 where hf_username = $2",
                    delta,
                    hf_username,
                )

    async def deduct_credits(
        self,
        hf_username: str,
        amount: float,
        reason: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Atomically deduct credits. Raises if balance would go negative."""
        import json as _json
        meta_json = _json.dumps(metadata) if metadata is not None else None
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                current = await conn.fetchval(
                    "select credits from users where hf_username = $1 for update",
                    hf_username,
                )
                if current is None:
                    raise ValueError(f"Unknown user: {hf_username}")
                if float(current) < amount:
                    raise ValueError(
                        f"Insufficient credits: have {current}, need {amount}"
                    )
                await conn.execute(
                    "update users set credits = credits - $1 where hf_username = $2",
                    amount,
                    hf_username,
                )
                await conn.execute(
                    """
                    insert into credit_transactions (hf_username, delta, reason, metadata)
                    values ($1, $2, $3, $4::jsonb)
                    """,
                    hf_username,
                    -amount,
                    reason,
                    meta_json,
                )

    async def list_transactions(
        self,
        hf_username: str,
        limit: int = 50,
    ) -> list[dict]:
        rows = await self.pool.fetch(
            """
            select id, delta, reason, metadata, created_at
            from credit_transactions
            where hf_username = $1
            order by created_at desc
            limit $2
            """,
            hf_username,
            limit,
        )
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # Simulation requests
    # -----------------------------------------------------------------------
    async def create_request(
        self,
        hf_username: str,
        channel: str,
        events: int,
        pileup: int,
        seed: int,
        config_hash: str,
        estimated_node_hours: float,
        credits_charged: float,
    ) -> dict:
        row = await self.pool.fetchrow(
            """
            insert into simulation_requests
                (hf_username, channel, events, pileup, seed, config_hash,
                 estimated_node_hours, credits_charged, state)
            values ($1, $2, $3, $4, $5, $6, $7, $8, 'queued')
            returning *
            """,
            hf_username,
            channel,
            events,
            pileup,
            seed,
            config_hash,
            estimated_node_hours,
            credits_charged,
        )
        return dict(row)

    async def update_request(self, request_id: str, **fields: Any) -> None:
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ${i + 2}" for i, k in enumerate(fields))
        await self.pool.execute(
            f"update simulation_requests set {set_clause} where id = $1",
            request_id,
            *fields.values(),
        )

    async def get_request(self, request_id: str) -> Optional[dict]:
        row = await self.pool.fetchrow(
            "select * from simulation_requests where id = $1", request_id
        )
        return dict(row) if row else None

    async def list_requests(self, hf_username: str, limit: int = 50) -> list[dict]:
        rows = await self.pool.fetch(
            """
            select * from simulation_requests
            where hf_username = $1
            order by created_at desc
            limit $2
            """,
            hf_username,
            limit,
        )
        return [dict(r) for r in rows]

    async def find_completed_by_hash(
        self, config_hash: str, since_days: int = 7
    ) -> Optional[dict]:
        row = await self.pool.fetchrow(
            """
            select * from simulation_requests
            where config_hash = $1
              and state in ('submitted', 'running', 'completed')
              and created_at > now() - make_interval(days => $2)
            order by created_at desc
            limit 1
            """,
            config_hash,
            int(since_days),
        )
        return dict(row) if row else None

    async def count_requests_recent(
        self, hf_username: str, minutes: int
    ) -> int:
        return await self.pool.fetchval(
            """
            select count(*) from simulation_requests
            where hf_username = $1
              and created_at > now() - make_interval(mins => $2)
            """,
            hf_username,
            int(minutes),
        ) or 0

    async def monthly_usage_total(self) -> float:
        val = await self.pool.fetchval("select coalesce(sum(node_hours), 0) from monthly_usage")
        return float(val or 0)

    async def monthly_usage_by_user(self, limit: int = 20) -> list[dict]:
        rows = await self.pool.fetch(
            """
            select hf_username, node_hours, n_requests
            from monthly_usage
            order by node_hours desc
            limit $1
            """,
            limit,
        )
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # Global config
    # -----------------------------------------------------------------------
    async def get_global_config(self) -> dict:
        row = await self.pool.fetchrow("select * from global_config where id = 1")
        if row is None:
            raise RuntimeError("global_config row missing - did migrations run?")
        return dict(row)

    async def set_submissions_frozen(self, frozen: bool) -> None:
        await self.pool.execute(
            "update global_config set submissions_frozen = $1 where id = 1",
            frozen,
        )


# Module-level singleton
db = Database()
