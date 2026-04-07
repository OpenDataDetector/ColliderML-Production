-- ColliderML backend schema v1
-- Applies cleanly to a fresh Supabase (or any Postgres 14+) database.
--
-- Run with: psql "$DATABASE_URL" -f migrations/001_initial.sql
--
-- Key design notes:
--   - HF username is the primary key for users. No passwords.
--   - Credits are stored as numeric(10,3) to allow fractional refunds.
--   - simulation_requests.config_hash allows dedup of identical submissions.
--   - Banned users are rejected at auth time via the banned flag.
--   - monthly_usage view aggregates over the current calendar month.

create extension if not exists "pgcrypto";  -- for gen_random_uuid()

-- ---------------------------------------------------------------------------
-- users
-- ---------------------------------------------------------------------------
create table if not exists users (
    hf_username text primary key,
    email text,
    credits numeric(10, 3) not null default 0,
    banned boolean not null default false,
    created_at timestamptz not null default now(),
    last_seen_at timestamptz not null default now(),
    notes text
);

create index if not exists idx_users_last_seen on users(last_seen_at);

-- ---------------------------------------------------------------------------
-- credit_transactions (append-only ledger)
-- ---------------------------------------------------------------------------
create table if not exists credit_transactions (
    id bigserial primary key,
    hf_username text not null references users(hf_username) on delete cascade,
    delta numeric(10, 3) not null,
    reason text not null,
    metadata jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_credit_tx_user on credit_transactions(hf_username, created_at desc);
create index if not exists idx_credit_tx_reason on credit_transactions(reason);

-- ---------------------------------------------------------------------------
-- simulation_requests
-- ---------------------------------------------------------------------------
create table if not exists simulation_requests (
    id uuid primary key default gen_random_uuid(),
    hf_username text not null references users(hf_username) on delete cascade,
    channel text not null,
    events int not null check (events > 0),
    pileup int not null default 0 check (pileup >= 0),
    seed int not null default 42,
    config_hash text not null,
    state text not null default 'queued'
        check (state in ('queued','submitted','running','completed','failed','cancelled')),
    nersc_jobid text,
    estimated_node_hours numeric(10, 3) not null,
    actual_node_hours numeric(10, 3),
    credits_charged numeric(10, 3) not null,
    output_hf_repo text,
    error_message text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_sim_req_user on simulation_requests(hf_username, created_at desc);
create index if not exists idx_sim_req_state on simulation_requests(state);
-- Dedup: no two active or completed requests with the same config hash
create unique index if not exists idx_sim_req_hash
    on simulation_requests(config_hash)
    where state in ('queued','submitted','running','completed');

-- ---------------------------------------------------------------------------
-- global_config (single-row table)
-- ---------------------------------------------------------------------------
create table if not exists global_config (
    id int primary key default 1 check (id = 1),
    monthly_node_hours_cap numeric(10, 1) not null default 500,
    submissions_frozen boolean not null default false,
    seed_credits numeric(10, 3) not null default 10,
    notes text
);
insert into global_config (id) values (1) on conflict do nothing;

-- ---------------------------------------------------------------------------
-- monthly_usage view
-- ---------------------------------------------------------------------------
create or replace view monthly_usage as
select
    hf_username,
    sum(coalesce(actual_node_hours, estimated_node_hours)) as node_hours,
    count(*) as n_requests
from simulation_requests
where created_at >= date_trunc('month', now())
  and state in ('submitted', 'running', 'completed')
group by hf_username;

-- ---------------------------------------------------------------------------
-- trigger: keep simulation_requests.updated_at fresh
-- ---------------------------------------------------------------------------
create or replace function touch_updated_at() returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

drop trigger if exists trg_sim_req_updated_at on simulation_requests;
create trigger trg_sim_req_updated_at
    before update on simulation_requests
    for each row execute function touch_updated_at();
