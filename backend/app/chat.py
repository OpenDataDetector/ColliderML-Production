"""POST /v1/chat — the ColliderML copilot proxy.

A deliberately thin LLM proxy that keeps the provider API key server-side. It
serves BOTH agents the public site needs, distinguished only by ``mode``:

  * ``"docs"``    — documentation assistant. System prompt = the full public
                    docs bundle. No tools. Answers questions about the library,
                    the SaaS API, simulation, tasks, etc.
  * ``"builder"`` — workflow-builder copilot. Same docs context PLUS builder
                    instructions and the *current* workflow state, and a set of
                    tools the **frontend** executes against the builder UI
                    (read / add / remove / update / reorder stages, validate,
                    submit). The model's ``tool_use`` blocks are returned
                    verbatim; the browser runs them against the reactive Vue
                    state and calls back with a ``tool_result``. UI mutation
                    stays client-side; the key stays here.

Provider is auto-detected from the environment: ANTHROPIC_API_KEY (preferred)
or OPENAI_API_KEY. With no key the endpoint still returns 200 with a friendly
"copilot not configured" message so the UI degrades gracefully.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional, Union

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()

_DOCS_BUNDLE = Path(__file__).with_name("docs_bundle.md")
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_MAX_TOKENS = 1500


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    # Either a plain string or a list of content blocks (Anthropic format),
    # so the frontend can send tool_result follow-ups.
    content: Union[str, list[dict[str, Any]]]


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    mode: Literal["docs", "builder"] = "docs"
    workflow: Optional[dict[str, Any]] = Field(
        default=None,
        description="Current builder pipeline state (builder mode only).",
    )


class ContentBlock(BaseModel):
    type: str
    text: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[dict[str, Any]] = None


class ChatResponse(BaseModel):
    provider: str
    model: Optional[str] = None
    stop_reason: Optional[str] = None
    content: list[ContentBlock]


# ---------------------------------------------------------------------------
# Docs context + system prompts
# ---------------------------------------------------------------------------

@lru_cache
def _docs_context() -> str:
    if _DOCS_BUNDLE.exists():
        return _DOCS_BUNDLE.read_text(encoding="utf-8")
    return "(documentation bundle not found on the server)"


_BUILDER_GUIDE = """
You are the ColliderML **workflow-builder copilot**, embedded next to a visual
pipeline builder on the public ColliderML site. The user assembles a simulation
workflow as an ordered list of *stage cards*; you can read and edit that
pipeline with the tools provided, and you must judge whether the assembled
workflow is sound.

## The pipeline model
A workflow is an ordered list of stages. Valid stage `type` values and their
`params`:

- `geometry`        — params: {"detector": "ODD"}  (ODD is the only detector
                      available on this deployment; CLD/LHCb are not wired up.)
- `generation`      — params: {"generator": "madgraph"|"pythia",
                      "channel": <one of the channels below>,
                      "events": int 1..100000, "seed": int}
- `simulation`      — params: {"pileup": int 0..200}   (Geant4 / ddsim)
- `digitization`    — params: {}   (ACTS digitization)
- `reconstruction`  — params: {}   (ACTS seeding + CKF track finding)

Channels: higgs_portal, ttbar, zmumu, zee, diphoton, jets, susy_gmsb,
hidden_valley, zprime.

## Validity rules to enforce (flag issues, offer to fix)
1. **Ordering**: stages must appear in the canonical order
   geometry → generation → simulation → digitization → reconstruction.
   A later stage may not precede an earlier one.
2. **Dependencies**: simulation requires a preceding generation stage;
   digitization requires simulation; reconstruction requires digitization.
   A geometry stage should come first if present.
3. **Generator/channel sanity**: hard-process channels (ttbar, susy_gmsb,
   zprime, hidden_valley, jets) are MadGraph→Pythia, so `generator` should be
   "madgraph". Soft/portal channels (higgs_portal, zmumu, zee, diphoton) are
   Pythia-only, so `generator` should be "pythia".
4. **Parameter coherence**: events in 1..100000; pileup in 0..200; a pileup
   above ~100 is very expensive — warn the user.
5. **Completeness**: a runnable workflow needs at least generation +
   simulation; digitization+reconstruction produce the tracks/hits tables.

## How to work
- When the user asks you to build or change the pipeline, USE THE TOOLS — call
  add_stage / update_stage / remove_stage / reorder_pipeline rather than only
  describing changes. Then briefly tell the user what you did.
- Call validate_workflow after structural edits and relay any issues.
- Only call submit_workflow when the user explicitly confirms; it spends
  credits. State the channel/events/pileup you're about to submit first.
- Be concise. You are a copilot, not a lecturer.
""".strip()


_DOCS_GUIDE = """
You are the ColliderML **documentation assistant**, available on every page of
the public site. Answer questions about ColliderML — the Python library, the
CLI, data loading, local and remote (SaaS) simulation, the benchmark tasks, and
the HuggingFace Spaces — using ONLY the documentation provided below. If the
answer is not in the docs, say so plainly and point to the closest relevant
section rather than inventing details. Be concise and include short code
snippets when helpful.
""".strip()


def _system_prompt(req: ChatRequest) -> str:
    docs = _docs_context()
    if req.mode == "builder":
        wf = json.dumps(req.workflow or {"stages": []}, indent=2)
        return (
            f"{_BUILDER_GUIDE}\n\n"
            f"## Current workflow (live state)\n```json\n{wf}\n```\n\n"
            f"## ColliderML documentation\n{docs}"
        )
    return f"{_DOCS_GUIDE}\n\n## ColliderML documentation\n{docs}"


# ---------------------------------------------------------------------------
# Builder tool schemas (executed client-side)
# ---------------------------------------------------------------------------

_STAGE_TYPES = ["geometry", "generation", "simulation", "digitization", "reconstruction"]

_BUILDER_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_workflow",
        "description": "Read the current pipeline (ordered list of stages with their params).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "add_stage",
        "description": "Append or insert a stage into the pipeline.",
        "input_schema": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": _STAGE_TYPES},
                "params": {"type": "object", "description": "Stage parameters (see system prompt)."},
                "position": {"type": "integer", "description": "0-based index to insert at; omit to append."},
            },
            "required": ["type"],
        },
    },
    {
        "name": "update_stage",
        "description": "Update params on an existing stage, identified by its id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "params": {"type": "object"},
            },
            "required": ["id", "params"],
        },
    },
    {
        "name": "remove_stage",
        "description": "Remove a stage by its id.",
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    },
    {
        "name": "reorder_pipeline",
        "description": "Set the full stage order by listing stage ids in the desired order.",
        "input_schema": {
            "type": "object",
            "properties": {"order": {"type": "array", "items": {"type": "string"}}},
            "required": ["order"],
        },
    },
    {
        "name": "validate_workflow",
        "description": "Run the deterministic validity checks and return any issues.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "submit_workflow",
        "description": "Submit the assembled workflow to the SaaS backend (spends credits). Only after explicit user confirmation.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------

def _to_anthropic_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    return [{"role": m.role, "content": m.content} for m in messages]


async def _call_anthropic(req: ChatRequest, key: str) -> ChatResponse:
    model = os.environ.get("CHAT_MODEL", "claude-sonnet-4-6")
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": _MAX_TOKENS,
        "system": _system_prompt(req),
        "messages": _to_anthropic_messages(req.messages),
    }
    if req.mode == "builder":
        body["tools"] = _BUILDER_TOOLS
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(_ANTHROPIC_URL, json=body, headers=headers)
    if r.status_code != 200:
        return ChatResponse(
            provider="anthropic",
            model=model,
            stop_reason="error",
            content=[ContentBlock(type="text", text=f"[copilot error {r.status_code}] {r.text[:300]}")],
        )
    data = r.json()
    content = [ContentBlock(**b) for b in data.get("content", [])]
    return ChatResponse(
        provider="anthropic",
        model=data.get("model", model),
        stop_reason=data.get("stop_reason"),
        content=content or [ContentBlock(type="text", text="(no content)")],
    )


def _to_openai_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]},
        }
        for t in _BUILDER_TOOLS
    ]


def _flatten_openai_content(content: Union[str, list[dict[str, Any]]]) -> str:
    """OpenAI wants string content; collapse Anthropic block lists to text/tool_result."""
    if isinstance(content, str):
        return content
    out = []
    for b in content:
        if b.get("type") == "text":
            out.append(b.get("text", ""))
        elif b.get("type") == "tool_result":
            out.append(f"[tool_result {b.get('tool_use_id','')}] {json.dumps(b.get('content'))}")
    return "\n".join(out) or "(empty)"


async def _call_openai(req: ChatRequest, key: str) -> ChatResponse:
    model = os.environ.get("CHAT_MODEL", "gpt-4o")
    msgs = [{"role": "system", "content": _system_prompt(req)}]
    for m in req.messages:
        msgs.append({"role": m.role, "content": _flatten_openai_content(m.content)})
    body: dict[str, Any] = {"model": model, "max_tokens": _MAX_TOKENS, "messages": msgs}
    if req.mode == "builder":
        body["tools"] = _to_openai_tools()
    headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(_OPENAI_URL, json=body, headers=headers)
    if r.status_code != 200:
        return ChatResponse(
            provider="openai",
            model=model,
            stop_reason="error",
            content=[ContentBlock(type="text", text=f"[copilot error {r.status_code}] {r.text[:300]}")],
        )
    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    blocks: list[ContentBlock] = []
    if msg.get("content"):
        blocks.append(ContentBlock(type="text", text=msg["content"]))
    for tc in msg.get("tool_calls", []) or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        blocks.append(ContentBlock(type="tool_use", id=tc.get("id"), name=fn.get("name"), input=args))
    stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else choice.get("finish_reason")
    return ChatResponse(
        provider="openai",
        model=data.get("model", model),
        stop_reason=stop,
        content=blocks or [ContentBlock(type="text", text="(no content)")],
    )


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
# /v1/chat is public and unauthenticated, and each turn spends Anthropic tokens
# on the server's key (the frontend can fire up to ~6 calls per user message).
# Bound that with an in-memory per-IP sliding window plus a global daily cap.
# In-memory is fine: the backend runs as a single Render instance.

_RATE_WINDOW_S = 60
_RATE_MAX_PER_IP = int(os.environ.get("CHAT_RATE_PER_MIN", "15") or "15")
_DAILY_CAP = int(os.environ.get("CHAT_DAILY_CAP", "2000") or "2000")
_ip_hits: dict[str, deque] = defaultdict(deque)
_daily = {"day": "", "n": 0}


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(request: Request) -> None:
    """Raise 429 when the per-IP window or the global daily cap is exceeded."""
    now = time.time()
    today = datetime.now(timezone.utc).date().isoformat()
    if _daily["day"] != today:
        _daily["day"], _daily["n"] = today, 0
    if _daily["n"] >= _DAILY_CAP:
        raise HTTPException(429, "The copilot has hit its daily limit. Please try again tomorrow.")
    dq = _ip_hits[_client_ip(request)]
    cutoff = now - _RATE_WINDOW_S
    while dq and dq[0] < cutoff:
        dq.popleft()
    if len(dq) >= _RATE_MAX_PER_IP:
        raise HTTPException(429, "You're sending messages too fast — give it a few seconds.")
    dq.append(now)
    _daily["n"] += 1


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    _enforce_rate_limit(request)
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if anthropic_key:
        return await _call_anthropic(req, anthropic_key)
    if openai_key:
        return await _call_openai(req, openai_key)
    return ChatResponse(
        provider="none",
        model=None,
        stop_reason="not_configured",
        content=[
            ContentBlock(
                type="text",
                text=(
                    "The copilot isn't configured yet — set ANTHROPIC_API_KEY (or "
                    "OPENAI_API_KEY) in the backend environment and reload. "
                    "Meanwhile you can build the pipeline manually with the cards."
                ),
            )
        ],
    )


@router.get("/v1/chat/health")
async def chat_health() -> dict:
    """Lets the UI show whether the copilot is wired up, without leaking the key."""
    provider = (
        "anthropic" if os.environ.get("ANTHROPIC_API_KEY", "").strip()
        else "openai" if os.environ.get("OPENAI_API_KEY", "").strip()
        else "none"
    )
    return {
        "configured": provider != "none",
        "provider": provider,
        "model": os.environ.get("CHAT_MODEL", "claude-sonnet-4-6" if provider != "openai" else "gpt-4o"),
        "docs_tokens_approx": len(_docs_context()) // 4,
    }
