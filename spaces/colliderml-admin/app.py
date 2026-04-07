"""
ColliderML Admin Dashboard — Gradio HuggingFace Space.

Protected by a shared-secret admin token (not HF OAuth). The token is
entered in a login box at the top of the app and forwarded as the
`X-Admin-Token` header on each call to the backend's /admin/* routes.

Tabs:
    1. Usage — monthly node-hours by user + running total vs cap
    2. User management — grant credits, ban/unban
    3. Kill switch — freeze/unfreeze all submissions
"""

from __future__ import annotations

import os

import gradio as gr
import pandas as pd
import plotly.express as px
import requests

BACKEND_URL = os.environ.get("COLLIDERML_BACKEND", "https://api.colliderml.com").rstrip("/")


def _admin_headers(token: str) -> dict:
    return {"X-Admin-Token": token}


def fetch_usage(token: str) -> pd.DataFrame:
    r = requests.get(
        f"{BACKEND_URL}/admin/usage?limit=50",
        headers=_admin_headers(token),
        timeout=20,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Backend error: {r.status_code} {r.text}")
    return pd.DataFrame(r.json())


def fetch_channels(token: str) -> pd.DataFrame:
    r = requests.get(
        f"{BACKEND_URL}/admin/analytics/channels",
        headers=_admin_headers(token),
        timeout=20,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Backend error: {r.status_code} {r.text}")
    return pd.DataFrame(r.json())


def fetch_daily(token: str, days: int = 30) -> pd.DataFrame:
    r = requests.get(
        f"{BACKEND_URL}/admin/analytics/daily?days={days}",
        headers=_admin_headers(token),
        timeout=20,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Backend error: {r.status_code} {r.text}")
    return pd.DataFrame(r.json())


def fetch_failures(token: str) -> dict:
    r = requests.get(
        f"{BACKEND_URL}/admin/analytics/failures",
        headers=_admin_headers(token),
        timeout=20,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Backend error: {r.status_code} {r.text}")
    return r.json()


def freeze_toggle(token: str, frozen: bool) -> str:
    r = requests.post(
        f"{BACKEND_URL}/admin/freeze?frozen={'true' if frozen else 'false'}",
        headers=_admin_headers(token),
        timeout=20,
    )
    if r.status_code != 200:
        return f"Error: {r.status_code} {r.text}"
    return f"OK — submissions frozen={r.json()['submissions_frozen']}"


def grant_credits(token: str, username: str, delta: float, reason: str) -> str:
    r = requests.post(
        f"{BACKEND_URL}/admin/grant",
        json={"hf_username": username, "delta": delta, "reason": reason or "admin_grant"},
        headers=_admin_headers(token),
        timeout=20,
    )
    if r.status_code != 200:
        return f"Error: {r.status_code} {r.text}"
    return f"OK — new balance for `{username}`: {r.json().get('new_balance')}"


def ban_user(token: str, username: str, banned: bool) -> str:
    r = requests.post(
        f"{BACKEND_URL}/admin/ban",
        json={"hf_username": username, "banned": banned},
        headers=_admin_headers(token),
        timeout=20,
    )
    if r.status_code != 200:
        return f"Error: {r.status_code} {r.text}"
    return f"OK — {username} banned={banned}"


# ---------------------------------------------------------------------------
# UI handlers
# ---------------------------------------------------------------------------
def on_load_usage(token: str):
    if not token:
        return None, "Enter admin token to load usage."
    try:
        df = fetch_usage(token)
    except Exception as e:
        return None, f"Error: {e}"
    if df.empty:
        return None, "No usage this month yet."

    fig = px.bar(
        df,
        x="hf_username",
        y="node_hours",
        hover_data=["n_requests"],
        title=f"Top users — monthly node-hours (total: {df['node_hours'].sum():.1f})",
    )
    fig.update_layout(xaxis_title="HF user", yaxis_title="node-hours")
    return fig, f"{len(df)} users active this month. Total: {df['node_hours'].sum():.1f} node-hours."


def on_freeze(token: str):
    if not token:
        return "Enter admin token."
    return freeze_toggle(token, frozen=True)


def on_unfreeze(token: str):
    if not token:
        return "Enter admin token."
    return freeze_toggle(token, frozen=False)


def on_grant(token: str, username: str, delta_str: str, reason: str):
    if not token or not username or not delta_str:
        return "Enter token, username, and delta."
    try:
        delta = float(delta_str)
    except ValueError:
        return "Delta must be a number."
    return grant_credits(token, username, delta, reason)


def on_ban(token: str, username: str):
    return ban_user(token, username, banned=True) if username else "Enter username."


def on_unban(token: str, username: str):
    return ban_user(token, username, banned=False) if username else "Enter username."


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="ColliderML Admin",
    theme=gr.themes.Soft(primary_hue="red"),
) as demo:
    gr.Markdown(
        """
        # ColliderML Admin Dashboard

        Internal use only. Authenticate with the admin shared secret.
        """
    )

    admin_token = gr.Textbox(
        label="Admin token",
        type="password",
        placeholder="X-Admin-Token shared secret",
    )

    with gr.Tab("Usage"):
        load_usage_btn = gr.Button("Refresh", variant="primary")
        usage_summary = gr.Markdown()
        usage_plot = gr.Plot()
        load_usage_btn.click(
            on_load_usage,
            inputs=admin_token,
            outputs=[usage_plot, usage_summary],
        )

    with gr.Tab("Analytics"):
        load_analytics_btn = gr.Button("Refresh analytics", variant="primary")
        with gr.Row():
            channel_plot = gr.Plot(label="Requests per channel")
            daily_plot = gr.Plot(label="Daily node-hours (30d)")
        failure_md = gr.Markdown()

        def on_load_analytics(token: str):
            if not token:
                return None, None, "Enter admin token."
            try:
                channels_df = fetch_channels(token)
                daily_df = fetch_daily(token)
                fail = fetch_failures(token)
            except Exception as e:
                return None, None, f"Error: {e}"

            ch_fig = None
            if not channels_df.empty:
                ch_fig = px.bar(
                    channels_df, x="channel", y="n",
                    hover_data=["node_hours"],
                    title="Requests per channel this month",
                )
            daily_fig = None
            if not daily_df.empty:
                daily_fig = px.line(
                    daily_df, x="day", y="node_hours",
                    markers=True,
                    title="Daily node-hours (last 30 days)",
                )
            md = (
                f"**Failure rate**: {fail['failure_rate']*100:.1f}% "
                f"({fail['failed']} failed / {fail['total']} total this month)"
            )
            return ch_fig, daily_fig, md

        load_analytics_btn.click(
            on_load_analytics,
            inputs=admin_token,
            outputs=[channel_plot, daily_plot, failure_md],
        )

    with gr.Tab("User management"):
        with gr.Row():
            username_in = gr.Textbox(label="HF username")
            delta_in = gr.Textbox(label="Credit delta (+/-)", value="10")
            reason_in = gr.Textbox(label="Reason", value="admin_grant")
        with gr.Row():
            grant_btn = gr.Button("Grant credits", variant="primary")
            ban_btn = gr.Button("Ban user", variant="stop")
            unban_btn = gr.Button("Unban user")
        user_mgmt_output = gr.Markdown()
        grant_btn.click(
            on_grant,
            inputs=[admin_token, username_in, delta_in, reason_in],
            outputs=user_mgmt_output,
        )
        ban_btn.click(on_ban, inputs=[admin_token, username_in], outputs=user_mgmt_output)
        unban_btn.click(on_unban, inputs=[admin_token, username_in], outputs=user_mgmt_output)

    with gr.Tab("Kill switch"):
        gr.Markdown(
            """
            ### Freeze all submissions

            This stops new simulation requests from entering the queue.
            Existing running jobs are **not** cancelled — they finish normally.
            """
        )
        with gr.Row():
            freeze_btn = gr.Button("FREEZE all submissions", variant="stop")
            unfreeze_btn = gr.Button("Unfreeze submissions")
        freeze_output = gr.Markdown()
        freeze_btn.click(on_freeze, inputs=admin_token, outputs=freeze_output)
        unfreeze_btn.click(on_unfreeze, inputs=admin_token, outputs=freeze_output)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
