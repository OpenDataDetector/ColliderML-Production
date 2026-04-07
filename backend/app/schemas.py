"""Pydantic request/response schemas for the HTTP API."""

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# -- Channels the backend accepts. Keep in sync with the pipeline.
Channel = Literal[
    "higgs_portal",
    "ttbar",
    "zmumu",
    "zee",
    "diphoton",
    "jets",
    "susy_gmsb",
    "hidden_valley",
    "zprime",
    "single_muon",
]

RequestState = Literal["queued", "submitted", "running", "completed", "failed", "cancelled"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class SimulateRequest(BaseModel):
    channel: Channel
    events: int = Field(ge=1, le=100_000)
    pileup: int = Field(ge=0, le=200, default=0)
    seed: int = Field(default=42)


class SimulateResponse(BaseModel):
    request_id: UUID
    state: RequestState
    estimated_node_hours: float
    estimated_completion_seconds: int
    credits_charged: float
    output_hf_repo: Optional[str] = None
    cached: bool = False   # true if this was deduped to an existing result


class RequestStatus(BaseModel):
    id: UUID
    hf_username: str
    channel: str
    events: int
    pileup: int
    seed: int
    state: RequestState
    nersc_jobid: Optional[str] = None
    estimated_node_hours: float
    actual_node_hours: Optional[float] = None
    credits_charged: float
    output_hf_repo: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class UserMe(BaseModel):
    hf_username: str
    email: Optional[str] = None
    credits: float
    created_at: datetime
    last_seen_at: datetime


class CreditTransaction(BaseModel):
    id: int
    delta: float
    reason: str
    metadata: Optional[dict] = None
    created_at: datetime


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------
class AdminGrant(BaseModel):
    hf_username: str
    delta: float
    reason: str = "admin_grant"
    metadata: Optional[dict] = None


class AdminBan(BaseModel):
    hf_username: str
    banned: bool = True


class UsageRow(BaseModel):
    hf_username: str
    node_hours: float
    n_requests: int
