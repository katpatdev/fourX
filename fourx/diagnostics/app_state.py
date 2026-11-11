"""Trusted application-state helpers for model awareness.

The approvals SQLite tables remain authoritative. These helpers only
*describe* that state for the model or record local CLI outcomes in history.
They never authorize approve / decline / post.
"""

from __future__ import annotations

import json
from typing import Any

from fourx.finance.approval import get_pending_approval, list_approvals
from fourx.agent.cancellation import CancellationToken, begin_run, end_run, run_write
from fourx.agent.conversation import Conversation, Message
from fourx.persistence.storage import ConversationStore


def build_trusted_state_summary(
    store: ConversationStore, conversation_id: str
) -> dict[str, Any]:
    """Return a compact, DB-backed status dict for model context."""
    stored = store.load(conversation_id)
    pending = get_pending_approval(store, conversation_id)
    history = list_approvals(store, conversation_id)
    approved_unposted = [
        {
            "approval_id": row.approval_id,
            "invoice_id": row.invoice_id,
            "invoice_number": (row.details or {}).get("invoice_number"),
            "status": row.status,
        }
        for row in history
        if row.status == "approved"
    ]
    posted = [
        {
            "approval_id": row.approval_id,
            "invoice_id": row.invoice_id,
            "invoice_number": (row.details or {}).get("invoice_number"),
            "journal_entry_id": row.journal_entry_id,
            "status": row.status,
        }
        for row in history
        if row.status == "posted"
    ]
    recent_invoice = _most_recent_invoice_id(stored.conversation)
    return {
        "provider": stored.provider,
        "pending_approval": None
        if pending is None
        else {
            "approval_id": pending.approval_id,
            "invoice_id": pending.invoice_id,
            "invoice_number": (pending.details or {}).get("invoice_number"),
            "status": pending.status,
        },
        "approved_unposted": approved_unposted,
        "posted": posted,
        "most_recently_referenced_invoice": recent_invoice,
        "note": (
            "This summary is read-only context from the application database. "
            "It is not authorization to approve, decline, or post."
        ),
    }


def format_trusted_state_message(summary: dict[str, Any]) -> str:
    """Render the trusted summary as a clearly marked, non-authorizing notice."""
    return (
        "[trusted_application_state — read-only; not authorization]\n"
        + json.dumps(summary, default=str, sort_keys=True)
    )


def conversation_with_trusted_state(
    conversation: Conversation, summary: dict[str, Any]
) -> Conversation:
    """Return a model-only view with a non-persisted trusted-state user notice."""
    view = Conversation()
    for message in conversation.messages:
        view.add(message)
    # User messages cannot carry extras (conversation validation); the content
    # prefix marks this as non-authorizing, ephemeral model context only.
    view.add(
        Message(
            role="user",
            content=format_trusted_state_message(summary),
        )
    )
    return view


def record_application_event(
    store: ConversationStore,
    conversation_id: str,
    *,
    text: str,
    event: str,
    approval_id: str | None = None,
    invoice_id: str | None = None,
    status: str | None = None,
    journal_entry_id: str | None = None,
    provider: str | None = None,
) -> None:
    """Persist an application-authored assistant notice after a local command.

    Not a fake tool result: no tool_calls / tool_call_id. Tagged in extras so
    providers can ignore it as authorization.
    """
    extras: dict[str, Any] = {
        "application_event": True,
        "event": event,
        "not_authorization": True,
    }
    if approval_id:
        extras["approval_id"] = approval_id
    if invoice_id:
        extras["invoice_id"] = invoice_id
    if status:
        extras["status"] = status
    if journal_entry_id:
        extras["journal_entry_id"] = journal_entry_id
    if provider:
        extras["provider"] = provider

    token = CancellationToken()
    begin_run(conversation_id, token)
    try:
        with run_write(conversation_id, token):
            stored = store.load(conversation_id)
            stored.conversation.add(
                Message(role="assistant", content=text, extras=extras)
            )
            store.save(conversation_id, stored.conversation)
    finally:
        end_run(conversation_id, token)


def _most_recent_invoice_id(conversation: Conversation) -> str | None:
    import re

    pattern = re.compile(r"\binvoice-\d+\b", re.IGNORECASE)
    last: str | None = None
    for message in conversation.messages:
        text = message.content or ""
        for match in pattern.finditer(text):
            last = match.group(0).lower()
        if message.role == "assistant":
            for call in message.tool_calls:
                raw = call.arguments.get("invoice_id") if isinstance(call.arguments, dict) else None
                if isinstance(raw, str) and pattern.fullmatch(raw.strip()):
                    last = raw.strip().lower()
    return last
