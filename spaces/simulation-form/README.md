---
title: ColliderML Simulation
emoji: ⚛️
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: apache-2.0
hf_oauth: true
hf_oauth_scopes:
  - openid
  - profile
  - email
---

# ColliderML Simulation Service

Submit custom ColliderML simulation requests without installing anything.
This Space is a thin frontend for the ColliderML backend — it sends the same
payloads that the `colliderml` pip package sends and returns the same results.

## Usage

1. Click **Sign in with HuggingFace** (you'll get 10 free credits on your first sign-in).
2. Choose a physics channel, number of events, and pileup.
3. Click **Submit**. You'll be given a request ID and an email when the job
   completes.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `COLLIDERML_BACKEND` | Backend URL (default: `https://api.colliderml.com`) |

## See also

- [Event display](https://huggingface.co/spaces/CERN/colliderml-event-display) — visualise existing datasets
- [Leaderboard](https://huggingface.co/spaces/CERN/colliderml-leaderboard) — benchmarks and credits
- [Pip package](https://pypi.org/project/colliderml/) — same capabilities from the CLI
