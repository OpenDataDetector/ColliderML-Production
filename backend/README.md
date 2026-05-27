# ColliderML Backend

FastAPI service that brokers simulation requests between the `colliderml` pip
package (or the HF Space web form) and NERSC Perlmutter via SFAPI. Also owns
the credit ledger and abuse detection.

## Architecture

```
colliderml pip package  \
                         \
HF Space simulation form ---> backend (FastAPI) ---> Supabase (Postgres)
                         /                       \
HF Space admin          /                         \-> NERSC Perlmutter (SFAPI)
                                                      Container: shifter
                                                      Output: HF datasets
```

## Routes

All `/v1/*` routes require `Authorization: Bearer <HF-token>`.
All `/admin/*` routes require `X-Admin-Token: <shared-secret>`.

| Method | Path | Purpose |
|--------|------|---------|
| GET | /healthz | Liveness probe |
| POST | /v1/simulate | Submit a new simulation |
| GET | /v1/requests/{id} | Status + output for one request |
| GET | /v1/requests | List current user's requests |
| GET | /v1/me | User profile (username, credits) |
| GET | /v1/me/transactions | Credit ledger |
| GET | /v1/datasets | Public dataset catalogue |
| POST | /admin/freeze | Kill switch |
| POST | /admin/grant | Grant credits |
| POST | /admin/ban | Ban/unban a user |
| GET | /admin/usage | Top users by node-hours this month |

## Local development

```bash
cd backend
docker compose up            # starts Postgres + backend
curl http://localhost:8000/healthz
```

### Without Docker

```bash
# 1. Run Postgres however you like, set DATABASE_URL
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/colliderml"

# 2. Apply migrations
psql "$DATABASE_URL" -f migrations/001_initial.sql

# 3. Install and run
pip install -e .
uvicorn app.main:app --reload
```

### Mock SFAPI mode

If `SFAPI_CLIENT_ID` / `SFAPI_CLIENT_SECRET` are unset, the runner goes into
**mock mode**: submissions are recorded in the database and marked completed
after 2 seconds, but never reach NERSC. This is the default in `docker compose`
and is useful for end-to-end client testing.

## Environment variables

See `app/config.py` for the full list. Key ones:

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | Postgres DSN |
| `ADMIN_TOKEN` | Shared secret for /admin/* |
| `SFAPI_CLIENT_ID` | NERSC IRIS client ID |
| `SFAPI_CLIENT_SECRET` | NERSC IRIS PEM |
| `NERSC_PROJECT` | e.g. m4958 |
| `NERSC_USER` | Service account username |
| `HF_TOKEN` | HF service account (for output uploads) |
| `HF_DATASET_ORG` | Org for per-request datasets (default: CERN) |
| `SMTP_*` | Email notifications (optional) |

## Credit economy

The canonical unit:

```
1 credit = 1 node-hour on Perlmutter cpu queue
         ≈ 100 hard-scatter events @ pu=0
         ≈ 20 hard-scatter events @ pu=200
```

Users get **10 seed credits** on first sign-in. That's ~1000 pu0 events, enough
for one solid experiment. Earning credits on the leaderboard funds follow-up
experiments at comparable scale.

## Deployment

Railway (via `railway.json`), Fly.io, or any Docker host. Use a managed
Postgres (Supabase free tier is sufficient for early users).

## Testing

```bash
pip install -e ".[dev]"
pytest
```

Tests use an in-memory fake for the database layer — they do not need a
running Postgres.
