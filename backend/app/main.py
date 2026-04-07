"""
ColliderML FastAPI backend.

Routes:
    POST  /v1/simulate               submit a simulation request
    GET   /v1/requests/{id}          status + output
    GET   /v1/requests               list current user's requests
    GET   /v1/me                     user profile
    GET   /v1/me/transactions        full credit ledger for the user
    GET   /v1/datasets               public dataset catalogue (cached)
    GET   /healthz                   liveness

    POST  /admin/freeze              kill switch (freeze/unfreeze submissions)
    POST  /admin/grant               grant credits to a user
    POST  /admin/ban                 ban/unban a user
    GET   /admin/usage               top users by node-hours this month

All v1 routes require `Authorization: Bearer <HF-token>`.
All admin routes require `X-Admin-Token: <shared-secret>`.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import abuse
from app.auth import admin_only, current_user
from app.cap import estimate_completion_seconds, estimate_node_hours
from app.config import get_settings
from app.db import db
from app.schemas import (
    AdminBan,
    AdminGrant,
    CreditTransaction,
    RequestStatus,
    SimulateRequest,
    SimulateResponse,
    UsageRow,
    UserMe,
)
from app.sfapi_runner import SFAPIRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module-level runner (lifespan creates/destroys it)
runner = SFAPIRunner()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    await runner.start()
    logger.info("ColliderML backend started")
    try:
        yield
    finally:
        await runner.stop()
        await db.disconnect()


app = FastAPI(
    title="ColliderML Backend",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Leaderboard routes (Phase 3)
from app.leaderboard import router as leaderboard_router  # noqa: E402
app.include_router(leaderboard_router)


# ===========================================================================
# Public / user routes
# ===========================================================================

@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/v1/me", response_model=UserMe)
async def me(user: dict = Depends(current_user)) -> UserMe:
    return UserMe(**{k: user[k] for k in UserMe.model_fields.keys() if k in user})


@app.get("/v1/me/transactions", response_model=list[CreditTransaction])
async def my_transactions(
    user: dict = Depends(current_user),
    limit: int = 100,
) -> list[CreditTransaction]:
    rows = await db.list_transactions(user["hf_username"], limit=limit)
    return [CreditTransaction(**r) for r in rows]


@app.post("/v1/simulate", response_model=SimulateResponse)
async def simulate(
    req: SimulateRequest,
    user: dict = Depends(current_user),
) -> SimulateResponse:
    # Dedup cache hit check before we charge anything
    config_hash = abuse.hash_request(req)
    existing = await db.find_completed_by_hash(config_hash, since_days=7)
    if existing and existing["state"] == "completed":
        return SimulateResponse(
            request_id=existing["id"],
            state="completed",
            estimated_node_hours=float(existing["estimated_node_hours"]),
            estimated_completion_seconds=0,
            credits_charged=0.0,
            output_hf_repo=existing.get("output_hf_repo"),
            cached=True,
        )

    # Run gates and atomically charge
    estimated, _ = await abuse.check_and_charge(user, req)

    # Create request row
    row = await db.create_request(
        hf_username=user["hf_username"],
        channel=req.channel,
        events=req.events,
        pileup=req.pileup,
        seed=req.seed,
        config_hash=config_hash,
        estimated_node_hours=estimated,
        credits_charged=estimated,
    )
    request_id = str(row["id"])

    # Submit to NERSC (or mock)
    try:
        await runner.submit(request_id, req, user)
    except Exception as e:
        logger.exception("SFAPI submission failed for %s", request_id)
        await db.update_request(
            request_id,
            state="failed",
            error_message=f"submission error: {e}",
        )
        await abuse.refund_full(
            request_id, user["hf_username"], estimated, reason="refund_submit_failed"
        )
        raise HTTPException(502, f"Backend failed to submit job: {e}")

    return SimulateResponse(
        request_id=row["id"],
        state="submitted",
        estimated_node_hours=estimated,
        estimated_completion_seconds=estimate_completion_seconds(
            req.channel, req.events, req.pileup
        ),
        credits_charged=estimated,
        output_hf_repo=None,
        cached=False,
    )


@app.get("/v1/requests/{request_id}", response_model=RequestStatus)
async def get_request(
    request_id: str,
    user: dict = Depends(current_user),
) -> RequestStatus:
    row = await db.get_request(request_id)
    if row is None:
        raise HTTPException(404, "Request not found")
    if row["hf_username"] != user["hf_username"]:
        raise HTTPException(403, "Not your request")
    return RequestStatus(**{k: row[k] for k in RequestStatus.model_fields.keys() if k in row})


@app.get("/v1/requests", response_model=list[RequestStatus])
async def list_my_requests(
    user: dict = Depends(current_user),
    limit: int = 50,
) -> list[RequestStatus]:
    rows = await db.list_requests(user["hf_username"], limit=limit)
    return [
        RequestStatus(**{k: r[k] for k in RequestStatus.model_fields.keys() if k in r})
        for r in rows
    ]


@app.get("/v1/datasets")
async def list_datasets() -> list[str]:
    """Static list mirroring colliderml._loader.KNOWN_DATASETS."""
    return [
        "ttbar_pu0", "ttbar_pu40", "ttbar_pu200",
        "higgs_portal_pu0", "higgs_portal_pu10", "higgs_portal_pu200",
        "zmumu_pu0", "zmumu_pu200",
        "zee_pu0", "zee_pu200",
        "diphoton_pu0", "diphoton_pu200",
        "jets_pu0", "jets_pu200",
        "susy_gmsb_pu0", "susy_gmsb_pu200",
        "hidden_valley_pu0", "hidden_valley_pu200",
        "zprime_pu0", "zprime_pu200",
        "single_muon_pu0",
    ]


# ===========================================================================
# Admin routes
# ===========================================================================

@app.post("/admin/freeze", dependencies=[Depends(admin_only)])
async def admin_freeze(frozen: bool = True) -> dict:
    await db.set_submissions_frozen(frozen)
    return {"submissions_frozen": frozen}


@app.post("/admin/grant", dependencies=[Depends(admin_only)])
async def admin_grant(grant: AdminGrant) -> dict:
    await db.add_credit_transaction(
        grant.hf_username,
        grant.delta,
        reason=grant.reason,
        metadata=grant.metadata,
    )
    user = await db.get_user(grant.hf_username)
    return {"hf_username": grant.hf_username, "new_balance": float(user["credits"]) if user else None}


@app.post("/admin/ban", dependencies=[Depends(admin_only)])
async def admin_ban(ban: AdminBan) -> dict:
    await db.set_banned(ban.hf_username, ban.banned)
    return {"hf_username": ban.hf_username, "banned": ban.banned}


@app.get("/admin/usage", dependencies=[Depends(admin_only)], response_model=list[UsageRow])
async def admin_usage(limit: int = 20) -> list[UsageRow]:
    rows = await db.monthly_usage_by_user(limit=limit)
    return [UsageRow(**r) for r in rows]
