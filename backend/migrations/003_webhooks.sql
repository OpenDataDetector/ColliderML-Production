-- GitHub -> HF username mapping for credit automation on merged PRs.
-- Users populate this by linking their GitHub handle in the frontend, or an
-- admin can seed it manually.

create table if not exists gh_hf_mapping (
    gh_username text primary key,
    hf_username text not null references users(hf_username) on delete cascade,
    verified boolean not null default false,
    created_at timestamptz not null default now()
);

create index if not exists idx_gh_hf_hfuser on gh_hf_mapping(hf_username);
