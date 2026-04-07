# ColliderML Testing & Deployment Guide

End-to-end test plan and deployment instructions for the ColliderML platform.

## Table of contents

1. [Overview](#overview)
2. [Layered test plan](#layered-test-plan)
3. [Researcher walkthrough](#researcher-walkthrough)
4. [Deployment (Render + Supabase + HF Spaces)](#deployment)
5. [Secret inventory](#secret-inventory)
6. [Troubleshooting](#troubleshooting)

---

## Overview

The platform has four deployable pieces:

| Piece | Runs on | Source |
|---|---|---|
| **Backend API** | Render (Docker) | `backend/` |
| **Database** | Supabase (Postgres) | `backend/migrations/*.sql` |
| **5x HF Spaces** | HuggingFace Spaces (Gradio) | `spaces/<name>/` |
| **`colliderml` pip package** | User's machine | `colliderml/`, `benchmarks/` |

The backend runs in **mock SFAPI mode** for the test deployment: it accepts
simulation requests, records them in Postgres, and marks them completed
after ~2 seconds without ever touching NERSC. Every other code path is
real.

All test resources live under the `staging` branch.

---

## Layered test plan

Work upward from Layer 0. Don't skip layers — each one catches different bugs.

### Layer 0: Unit tests (no services)

```bash
# From the repo root
cd backend
pip install -e ".[dev]"
pytest tests/ -v
```

Expected: `test_cap.py` and `test_abuse.py` pass (9 tests total).

```bash
cd ..
python - <<'PY'
from benchmarks import list_tasks, get_task
for t in list_tasks():
    task = get_task(t)
    print(f"  {t}: dataset={task.dataset}, metrics={task.metrics}")
assert len(list_tasks()) == 6
PY
```

Expected: All 6 tasks registered. No ImportError.

```bash
python -c "import colliderml; print(colliderml.__version__, colliderml.list_benchmarks())"
```

Expected: `0.4.0 ['anomaly', 'data_loading', 'jets', 'tracking', 'tracking_latency', 'tracking_small']`.

---

### Layer 1: Local backend stack (mock SFAPI, real Postgres)

```bash
cd backend
cp .env.example .env         # leave SFAPI_* empty for mock mode
docker compose up --build
```

Wait until you see `ColliderML backend started` in the logs. In another terminal:

```bash
# Liveness
curl -s http://localhost:8000/healthz
# {"status":"ok"}

# Public catalogue (no auth)
curl -s http://localhost:8000/v1/datasets | python -m json.tool

# Benchmark tasks (no auth)
curl -s http://localhost:8000/v1/benchmark/tasks | python -m json.tool
```

Expected: 21 datasets listed, 6 benchmark tasks.

```bash
# Auth-required route without auth → 401
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/v1/me
# 401
```

---

### Layer 2: pip package against local backend

```bash
pip install -e .
huggingface-cli login       # get a token from https://huggingface.co/settings/tokens

export COLLIDERML_BACKEND=http://localhost:8000
export HF_TOKEN=$(huggingface-cli whoami --token 2>/dev/null || cat ~/.cache/huggingface/token)

# First touch creates the user and grants 10 seed credits
colliderml balance
# User:    <your-hf-username>
# Credits: 10.00

# Submit a tiny mock simulation
python - <<'PY'
import colliderml
result = colliderml.simulate(
    channel="higgs_portal",
    events=10,
    pileup=10,
    remote=True,
)
print(result)
PY
```

Expected sequence:
1. `Submitted request <uuid> (est 0.17 credits, ~5 min).`
2. Polling: `state=submitted` → `state=completed` (within ~2 seconds in mock mode).
3. Balance drops to ~9.83 credits.

```bash
colliderml balance
# Credits should now be ~9.83

# Resubmit same parameters → 409 dedup
python -c "import colliderml; colliderml.simulate(channel='higgs_portal', events=10, pileup=10, remote=True)"
# Expected: "This exact request already ran recently. Reuse the existing dataset."
```

---

### Layer 3: Local Spaces against local backend

Each Space is a standalone Gradio app. Run each in turn:

```bash
# Event display
cd spaces/event-display
pip install -r requirements.txt
python app.py                # http://localhost:7860
```

Test: pick a dataset + event, verify a 3D plot renders. (The Space falls back gracefully if no data is cached.)

```bash
# Simulation form
cd ../simulation-form
pip install -r requirements.txt
COLLIDERML_BACKEND=http://localhost:8000 python app.py
```

Test:
- Click "Sign in with HuggingFace" → OAuth flow.
- Submit a small simulation → see request ID appear.
- Click "Refresh status" → see "completed" within ~2 seconds.
- (Chat tab only works if `ANTHROPIC_API_KEY` is set.)

```bash
# Leaderboard
cd ../leaderboard
pip install -r requirements.txt
COLLIDERML_BACKEND=http://localhost:8000 python app.py
```

Test: open each task tab; expect the "No submissions yet" message.

```bash
# Model zoo
cd ../model-zoo
pip install -r requirements.txt
python app.py
```

Test: the table loads (may be empty if no HF models are tagged `colliderml` yet).

```bash
# Admin dashboard
cd ../colliderml-admin
pip install -r requirements.txt
COLLIDERML_BACKEND=http://localhost:8000 python app.py
```

Test: enter admin token (value of `ADMIN_TOKEN` from `backend/.env`), click
"Refresh" on the Usage tab → see your one submitted request aggregated.

---

### Layer 4: Deployed stack (Render + Supabase + HF Spaces)

This is the full integration test on live infrastructure. See
[Deployment](#deployment) below for the setup, then repeat **Layer 2** and
**Layer 3** against the deployed backend URL.

```bash
export COLLIDERML_BACKEND=https://colliderml-backend-staging.onrender.com
colliderml balance
python -c "import colliderml; colliderml.simulate(channel='higgs_portal', events=10, pileup=10, remote=True)"
```

Visit each deployed Space:

- `https://huggingface.co/spaces/<HF_USER>/colliderml-event-display`
- `https://huggingface.co/spaces/<HF_USER>/colliderml-simulation-form`
- `https://huggingface.co/spaces/<HF_USER>/colliderml-leaderboard`
- `https://huggingface.co/spaces/<HF_USER>/colliderml-model-zoo`
- `https://huggingface.co/spaces/<HF_USER>/colliderml-colliderml-admin`

---

## Researcher walkthrough

Imagine a new researcher ("Alice") installs the package for the first time.
Every pathway is exercised here.

### 1. Install and load public data (no auth, no Docker)

```bash
pip install colliderml
python - <<'PY'
import colliderml
# Load pre-generated data from HuggingFace
data = colliderml.load("ttbar_pu0", tables=["tracks"], max_events=50)
print(type(data), data.num_rows if hasattr(data, 'num_rows') else len(data))
PY
```

Expected: streams from `CERN/ColliderML-Release-1`. No credit cost.

### 2. Browse available datasets and benchmarks from the CLI

```bash
colliderml list-datasets
colliderml list-presets
python -c "import colliderml; print(colliderml.list_benchmarks())"
```

### 3. Try local simulation (Docker or Podman required)

```bash
colliderml simulate --preset higgs-portal-quick
# Or Podman:
COLLIDERML_RUNTIME=podman colliderml simulate --preset higgs-portal-quick
```

Expected: pipeline runs in container, output in `colliderml_output/`.
(Skip if Docker/Podman are unavailable — the remote path covers the same thing.)

### 4. Sign in for remote simulation

```bash
huggingface-cli login
colliderml balance
```

Expected on first run: user row created, 10 seed credits granted, `balance`
prints the amount.

### 5. Submit a remote simulation request

```bash
python - <<'PY'
import colliderml
result = colliderml.simulate(
    channel="higgs_portal",
    events=100,
    pileup=0,
    remote=True,
)
print("Done:", result.output_hf_repo)
PY
```

Expected: submission, polling, completion, balance deducted.

### 6. Submit a second request (different seed) and watch it queue

```bash
python -c "import colliderml; colliderml.simulate(channel='higgs_portal', events=100, pileup=0, seed=43, remote=True)"
colliderml balance   # should show two deductions in the ledger
```

### 7. Attempt an oversized request → hits credit gate

```bash
python -c "import colliderml; colliderml.simulate(channel='ttbar', events=100000, pileup=200, remote=True)"
# Expected: RuntimeError: Insufficient credits
```

### 8. Local benchmark scoring + submission to the leaderboard

```bash
# Build some synthetic tracking predictions for the demo
python - <<'PY'
import pyarrow as pa
import pyarrow.parquet as pq
table = pa.table({
    "event_id": list(range(90000, 90100)) * 5,
    "hit_id": list(range(500)),
    "track_id": [i // 5 for i in range(500)],
})
pq.write_table(table, "fake_preds.parquet")
print(f"Wrote {table.num_rows} rows")
PY

# Score locally (no network)
python -c "import colliderml; print(colliderml.benchmark('tracking', 'fake_preds.parquet'))"

# Submit to the leaderboard
python -c "import colliderml; print(colliderml.benchmark('tracking', 'fake_preds.parquet', submit=True))"
```

### 9. Browse the HF Spaces

Open each in a browser and click around:

- **Event display** → pick a dataset + event, confirm 3D plot.
- **Simulation form** → sign in, submit a job via the form, check status.
- **Simulation form → Chat tab** → type "Estimate 1000 ttbar events with pileup 200". Expect the agent to call `estimate_compute` and return a cost.
- **Leaderboard** → click each task tab, try uploading `fake_preds.parquet` in the tracking tab.
- **Model zoo** → confirm the refresh loads models.
- **Admin** → enter the admin token, view usage, grant yourself 5 credits, check `colliderml balance` increased.

### 10. Verify kill switch + admin actions

In the admin Space:
- Click **FREEZE submissions** → then try submitting via the pip package. Expect 503 "temporarily frozen".
- Click **Unfreeze** → retry, expect success.
- Ban yourself → try `colliderml balance` → expect 403.
- Unban → works again.

---

## Deployment

### Prerequisites

- GitHub repo with push access to the `staging` branch
- Supabase account (free tier is enough)
- Render account (free tier is enough)
- HuggingFace account with a **write token** at https://huggingface.co/settings/tokens

### Step 1: Push the `staging` branch

```bash
git checkout staging
git push -u origin staging
```

### Step 2: Supabase setup

1. Go to https://supabase.com/dashboard → **New project**.
2. Name it `colliderml-staging`. Choose a region close to your Render region (Oregon if you're in the US).
3. Save the database password.
4. Once provisioned, go to **Project Settings → Database → Connection pooling**.
5. Copy the **Session mode** connection string. Replace `[YOUR-PASSWORD]` with the password you saved.
6. Open **SQL Editor → New query** and run each migration in order:
   - Paste `backend/migrations/001_initial.sql` → **Run**.
   - Paste `backend/migrations/002_benchmarks.sql` → **Run**.
   - Paste `backend/migrations/003_webhooks.sql` → **Run**.
7. Verify: go to **Table Editor**. You should see `users`, `credit_transactions`, `simulation_requests`, `global_config`, `benchmark_submissions`, `benchmark_bests`, `benchmark_reproductions`, `gh_hf_mapping`.

### Step 3: Render setup

1. Go to https://dashboard.render.com → **New → Blueprint**.
2. Connect your GitHub repo, select the `staging` branch.
3. Render detects `backend/render.yaml` automatically. Click **Apply**.
4. When prompted, paste the `DATABASE_URL` from Supabase (Step 2.5). Leave `HF_TOKEN`, `SMTP_PASSWORD`, and `GITHUB_WEBHOOK_SECRET` blank for now — you can add them later.
5. Render builds the Docker image (takes ~3-5 min on the first build). Watch the logs until you see `ColliderML backend started`.
6. Copy the service URL from the Render dashboard, e.g. `https://colliderml-backend-<hash>.onrender.com`.
7. Test it: `curl https://<your-render-url>/healthz` → `{"status":"ok"}`.

### Step 4: HF Spaces setup (auto-sync via GitHub Actions)

1. In the GitHub repo, go to **Settings → Secrets and variables → Actions → New repository secret**.
2. Add:
   - `HF_TOKEN` → the HF write token
   - `HF_USER` → your HF username (e.g. `dmurnane`)
3. Push any change to `spaces/**` on the `staging` branch (or trigger manually from the Actions tab → "Sync HuggingFace Spaces" → **Run workflow**).
4. The workflow creates 5 Spaces under `<HF_USER>/colliderml-<name>` and uploads each directory.
5. For each Space, go to **Settings → Variables and secrets** and add:
   - **All Spaces**: `COLLIDERML_BACKEND` = your Render URL from Step 3.
   - **simulation-form only**: `ANTHROPIC_API_KEY` (optional, enables the chat tab).
   - **colliderml-admin only**: `ADMIN_TOKEN` (copy from the Render dashboard — Render generated it for you).

### Step 5: Run the Layer 4 verification

See [Layer 4](#layer-4-deployed-stack-render--supabase--hf-spaces) above.

---

## Secret inventory

Every secret and where it lives. Nothing is in git.

| Secret | Where it comes from | Stored in |
|---|---|---|
| `DATABASE_URL` | Supabase → Connection pooling (session mode) | Render env var |
| `ADMIN_TOKEN` | Render auto-generates on first deploy | Render env var + admin Space secret |
| `HF_TOKEN` (service account, for output uploads) | HF settings → tokens (write scope) | Render env var (optional in mock mode) |
| `HF_TOKEN` (user, for the workflow) | HF settings → tokens (write scope) | GitHub repo secret |
| `HF_USER` | Your HF username | GitHub repo secret |
| `ANTHROPIC_API_KEY` | console.anthropic.com | HF Space secret (simulation-form only) |
| `GITHUB_WEBHOOK_SECRET` | Generate with `openssl rand -hex 32` | Render env var + GitHub webhook config |
| `SMTP_*` | Any SMTP provider (optional) | Render env vars |
| `SFAPI_*`, `NERSC_*` | NERSC IRIS (skip for staging — mock mode) | Render env vars (leave empty) |
| `COLLIDERML_BACKEND` | Render service URL | Each HF Space secret + user's local env |

---

## Troubleshooting

**Render build fails on `pip install .`:** The `backend/pyproject.toml` uses
`asyncpg` which needs `libpq`. The `python:3.12-slim` base image doesn't
bundle it but the pip wheel for asyncpg is self-contained, so this should
Just Work. If it doesn't, check the Render build logs for which package
actually failed.

**`/v1/me` returns 500 with `asyncpg` connection errors:** Your Supabase
`DATABASE_URL` is wrong. Make sure you used the **Session mode** pooler
URL, not the direct connection URL or transaction mode pooler. asyncpg
requires persistent connections.

**Mock mode never completes:** Check the backend logs — if you see
`SFAPI credentials not set — runner is in mock mode`, that's correct. Jobs
should transition `submitted → completed` in ~2 seconds. If they're stuck
in `submitted`, the background task died; restart the backend.

**Spaces sync workflow fails with 403:** Your `HF_TOKEN` doesn't have write
scope, or the target org doesn't exist. Check that `HF_USER` matches a real
HF account/org.

**HF Space fails on load with `oauth_token` errors:** The simulation-form
Space requires HF OAuth, which is configured via the `hf_oauth: true` field
in its `README.md` frontmatter. HF must approve this on first sync — check
the Space settings for an OAuth consent screen.

**"Insufficient credits" on a brand-new account:** The signup grant is 10
credits, which is enough for ~100 pu0 events of ttbar or ~1000 events of
`single_muon`. Bigger jobs will hit this gate. Use the admin Space to grant
yourself more.

**Leaderboard shows "no submissions yet":** Normal for a fresh deploy. The
three seeded baselines (CKF, BDT, IsoForest) have submission **scripts**
under `benchmarks/*/baselines/` but nobody has run them on the staging
deploy yet. Run them manually or wait for first users.
