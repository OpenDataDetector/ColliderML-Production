"""
Email notifications via SMTP (aiosmtplib).

Only sends on completion or failure of a simulation request. No newsletters,
no marketing. If SMTP is not configured, the calls are no-ops.
"""

from __future__ import annotations

import logging

from app.config import get_settings
from app.db import db

logger = logging.getLogger(__name__)


async def send_email(to: str, subject: str, body: str) -> None:
    """Send a plaintext email. Silent no-op if SMTP is not configured."""
    settings = get_settings()
    if not settings.smtp_host or not to:
        logger.info("SMTP not configured or no recipient; skipping email to %s", to)
        return

    try:
        import aiosmtplib
        from email.message import EmailMessage
    except ImportError:
        logger.warning("aiosmtplib not installed; skipping email")
        return

    msg = EmailMessage()
    msg["From"] = settings.smtp_from
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        await aiosmtplib.send(
            msg,
            hostname=settings.smtp_host,
            port=settings.smtp_port,
            username=settings.smtp_user or None,
            password=settings.smtp_password or None,
            start_tls=True,
        )
    except Exception as e:
        logger.error("Failed to send email to %s: %s", to, e)


async def send_completion(request_id: str) -> None:
    request = await db.get_request(request_id)
    if request is None:
        return
    user = await db.get_user(request["hf_username"])
    if user is None or not user.get("email"):
        return

    repo = request.get("output_hf_repo") or "(not uploaded)"
    body = f"""\
Your ColliderML simulation request is complete.

Request ID: {request_id}
Channel:    {request['channel']}
Events:     {request['events']}
Pileup:     {request['pileup']}
Output:     https://huggingface.co/datasets/{repo}

Load it in Python:

    import colliderml
    data = colliderml.load(
        "{request['channel']}_pu{request['pileup']}",
        repo="{repo}",
    )

Thank you for using ColliderML.
"""
    await send_email(user["email"], "ColliderML simulation complete", body)


async def send_failure(request_id: str, error_message: str) -> None:
    request = await db.get_request(request_id)
    if request is None:
        return
    user = await db.get_user(request["hf_username"])
    if user is None or not user.get("email"):
        return

    body = f"""\
Your ColliderML simulation request failed.

Request ID: {request_id}
Channel:    {request['channel']}
Events:     {request['events']}
Pileup:     {request['pileup']}

Error: {error_message}

Your credits ({request['credits_charged']}) have been refunded.
"""
    await send_email(user["email"], "ColliderML simulation failed", body)
