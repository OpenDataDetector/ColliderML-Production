---
title: ColliderML Admin
emoji: 🔧
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# ColliderML Admin Dashboard

Internal dashboard for monitoring usage, managing user credits, and
triggering the kill switch. Not meant for public use.

## Authentication

Protected by a shared-secret **admin token** (not HF OAuth). The same token
is required by the backend's `/admin/*` routes.

Set the token in the Space's secrets:
- `COLLIDERML_BACKEND` — backend URL
- `ADMIN_TOKEN` — admin shared secret

Users of the dashboard also need to enter the admin token in the login form
(it is forwarded as `X-Admin-Token` on each API call).

## Tabs

1. **Usage** — top users by node-hours this month, running monthly total vs cap.
2. **User management** — search by HF username, grant credits, ban/unban.
3. **Kill switch** — freeze/unfreeze all submissions.
