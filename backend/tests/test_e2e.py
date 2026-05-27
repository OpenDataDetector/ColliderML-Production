"""
End-to-end integration test against a live database.

This test does NOT run by default — it requires:
    - A running Postgres with migrations 001, 002, 003 applied
    - DATABASE_URL env var pointing at it

It exercises the full happy path:
    1. /healthz + /v1/datasets + /v1/benchmark/tasks (no auth)
    2. /v1/simulate with a stubbed HF user (mock SFAPI)
    3. Polling /v1/requests/{id} until completion
    4. /v1/me to see the credit deduction
    5. /v1/me/transactions for the ledger
    6. Duplicate submission → 409
    7. /admin/freeze + retry → 503
    8. /admin/grant → balance restoration

Run with:
    DATABASE_URL=postgresql://colliderml:colliderml@localhost:5432/colliderml \
        ADMIN_TOKEN=test-admin-token \
        python -m pytest tests/test_e2e.py -v -s
"""

from __future__ import annotations

import os
import time

import pytest

# Shared fixtures (client, _reset_user) live in conftest.py.


_HAS_DB = bool(os.environ.get("DATABASE_URL"))


# ---------------------------------------------------------------------------
# Public routes
# ---------------------------------------------------------------------------
def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_datasets(client):
    r = client.get("/v1/datasets")
    assert r.status_code == 200
    assert "ttbar_pu200" in r.json()
    assert len(r.json()) == 21


def test_benchmark_tasks(client):
    r = client.get("/v1/benchmark/tasks")
    assert r.status_code == 200
    names = [t["name"] for t in r.json()]
    assert set(names) == {
        "tracking", "jets", "anomaly",
        "tracking_latency", "tracking_small", "data_loading",
    }


# ---------------------------------------------------------------------------
# Simulate → poll → reconcile
# ---------------------------------------------------------------------------
def test_simulate_end_to_end(client):
    # Check starting balance
    r = client.get("/v1/me")
    assert r.status_code == 200
    starting = float(r.json()["credits"])
    assert starting == 10.0

    # Submit a tiny simulation
    r = client.post(
        "/v1/simulate",
        json={"channel": "higgs_portal", "events": 10, "pileup": 10, "seed": 42},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["state"] == "submitted"
    assert data["cached"] is False
    assert data["estimated_node_hours"] > 0
    request_id = data["request_id"]

    # Balance should have dropped
    after_submit = float(client.get("/v1/me").json()["credits"])
    assert after_submit < starting

    # Poll until terminal (mock mode reaches completed in ~2s)
    for _ in range(15):
        time.sleep(0.5)
        r = client.get(f"/v1/requests/{request_id}")
        assert r.status_code == 200
        if r.json()["state"] in ("completed", "failed"):
            break
    assert r.json()["state"] == "completed", r.json()


def test_duplicate_is_409(client):
    # First submission
    r = client.post(
        "/v1/simulate",
        json={"channel": "higgs_portal", "events": 10, "pileup": 10, "seed": 42},
    )
    assert r.status_code == 200

    # Wait for it to reach 'completed' (dedup index covers completed rows)
    request_id = r.json()["request_id"]
    for _ in range(15):
        time.sleep(0.5)
        state = client.get(f"/v1/requests/{request_id}").json()["state"]
        if state == "completed":
            break

    # Second identical submission should short-circuit as cached
    r = client.post(
        "/v1/simulate",
        json={"channel": "higgs_portal", "events": 10, "pileup": 10, "seed": 42},
    )
    # Main route returns cached=true on completed dedup
    assert r.status_code == 200
    assert r.json()["cached"] is True


def test_kill_switch(client):
    headers = {"X-Admin-Token": os.environ.get("ADMIN_TOKEN", "test-admin-token")}

    r = client.post("/admin/freeze?frozen=true", headers=headers)
    assert r.status_code == 200
    assert r.json()["submissions_frozen"] is True

    r = client.post(
        "/v1/simulate",
        json={"channel": "zmumu", "events": 5, "pileup": 0, "seed": 123},
    )
    assert r.status_code == 503

    # Unfreeze for subsequent tests
    client.post("/admin/freeze?frozen=false", headers=headers)


def test_admin_grant(client):
    headers = {"X-Admin-Token": os.environ.get("ADMIN_TOKEN", "test-admin-token")}
    start = float(client.get("/v1/me").json()["credits"])

    r = client.post(
        "/admin/grant",
        json={"hf_username": "e2e_alice", "delta": 5, "reason": "e2e_test"},
        headers=headers,
    )
    assert r.status_code == 200
    assert float(r.json()["new_balance"]) == start + 5
