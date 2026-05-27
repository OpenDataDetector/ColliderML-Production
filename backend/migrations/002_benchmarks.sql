-- Benchmark submissions and reproduction tracking.
-- Adds the leaderboard layer on top of the Phase 2 schema.

create extension if not exists "pgcrypto";

-- ---------------------------------------------------------------------------
-- benchmark_submissions
-- ---------------------------------------------------------------------------
create table if not exists benchmark_submissions (
    id uuid primary key default gen_random_uuid(),
    task text not null,
    hf_username text not null references users(hf_username) on delete cascade,
    submitted_at timestamptz not null default now(),
    predictions_sha256 text not null,         -- dedup key; identical preds = same row
    scores jsonb not null,                    -- {metric_name: value}
    is_baseline boolean not null default false,
    n_params bigint,                          -- for tracking_small
    code_url text,                            -- optional link to notebook/model card
    credits_earned numeric(10, 3) not null default 0,
    notes text
);

create index if not exists idx_bench_task on benchmark_submissions(task, submitted_at desc);
create index if not exists idx_bench_user on benchmark_submissions(hf_username);
create unique index if not exists idx_bench_dedup
    on benchmark_submissions(task, hf_username, predictions_sha256);

-- ---------------------------------------------------------------------------
-- benchmark_bests
-- Tracks the current best score for each (task, metric) pair. Updated on
-- every submission — if the new score beats the current best for a metric,
-- the submitter earns credits and this row is replaced.
-- ---------------------------------------------------------------------------
create table if not exists benchmark_bests (
    task text not null,
    metric text not null,
    value numeric(14, 6) not null,
    submission_id uuid not null references benchmark_submissions(id) on delete cascade,
    hf_username text not null references users(hf_username) on delete cascade,
    updated_at timestamptz not null default now(),
    primary key (task, metric)
);

-- ---------------------------------------------------------------------------
-- benchmark_reproductions
-- When one user reproduces another user's result (within 2% tolerance),
-- the reproducer earns 20 credits. This self-polices the leaderboard.
-- ---------------------------------------------------------------------------
create table if not exists benchmark_reproductions (
    id bigserial primary key,
    submission_id uuid not null references benchmark_submissions(id) on delete cascade,
    reproducer text not null references users(hf_username) on delete cascade,
    reproduced_scores jsonb not null,
    within_tolerance boolean not null,
    credits_earned numeric(10, 3) not null default 0,
    created_at timestamptz not null default now()
);

create index if not exists idx_repro_submission on benchmark_reproductions(submission_id);
