"""Durable human approval for invoice posting (no ledger side effects).

The model may request ``post_invoice``; only explicit application APIs
``approve`` / ``decline`` record a human decision. Nothing here calls
``ledger.post()`` — use ``posting.post_approved`` for that.

Each request has a durable ``approval_id``. Completed decisions are never
overwritten. While an unresolved approved/posting intent exists for a
conversation, new pending requests are refused so that intent cannot be lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from invoice_tools import prepare_post
from storage import (
    ApprovalDecisionConflict,
    ApprovalPendingExists,
    ApprovalRowMissing,
    ApprovalUnprocessedExists,
    ConversationStore,
    StoredApproval,
)

ApprovalStatus = Literal[
    "pending",
    "approved",
    "declined",
    "posting",
    "posted",
    "posting_uncertain",
]

# Fields the clerk must see before deciding (from prepare_post only).
DISPLAY_KEYS = (
    "invoice_id",
    "vendor_id",
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "currency",
    "amount",
)


class ApprovalNotFound(LookupError):
    """No matching approval row."""


class ApprovalAlreadyPending(ValueError):
    """Conversation already has an unresolved pending approval."""


class ApprovalBlockedByApproved(ValueError):
    """An approved-but-unposted intent still occupies this conversation."""


class ApprovalAlreadyDecided(ValueError):
    """Approve/decline was attempted on a non-pending request."""


class InvalidDeclineReason(ValueError):
    """Decline requires a non-empty human reason."""


@dataclass(frozen=True)
class ApprovalRequest:
    """One posting authorization request (journal_entry_id only when posted)."""

    approval_id: str
    conversation_id: str
    invoice_id: str
    details: dict[str, Any]
    status: ApprovalStatus
    decline_reason: str | None = None
    journal_entry_id: str | None = None

    def display(self) -> dict[str, Any]:
        """Exact clerk-facing fields from deterministic eligibility details."""
        return {key: self.details[key] for key in DISPLAY_KEYS if key in self.details}


def _from_stored(row: StoredApproval) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=row.approval_id,
        conversation_id=row.conversation_id,
        invoice_id=row.invoice_id,
        details=dict(row.details),
        status=row.status,
        decline_reason=row.decline_reason,
        journal_entry_id=row.journal_entry_id,
    )


def request_approval(
    store: ConversationStore,
    conversation_id: str,
    invoice_id: str,
) -> ApprovalRequest:
    """Run ``prepare_post`` and persist a new pending approval. Never posts."""
    details = prepare_post(invoice_id)
    try:
        stored = store.create_pending_approval(conversation_id, details)
    except ApprovalPendingExists as exc:
        raise ApprovalAlreadyPending(str(exc)) from exc
    except ApprovalUnprocessedExists as exc:
        raise ApprovalBlockedByApproved(str(exc)) from exc
    return _from_stored(stored)


def get_approval(
    store: ConversationStore, approval_id: str
) -> ApprovalRequest | None:
    """Load one approval by durable ``approval_id``."""
    row = store.get_approval_by_id(approval_id)
    return _from_stored(row) if row is not None else None


def list_approvals(
    store: ConversationStore, conversation_id: str
) -> list[ApprovalRequest]:
    """Return the full approval history for a conversation."""
    return [_from_stored(row) for row in store.list_approvals(conversation_id)]


def get_pending_approval(
    store: ConversationStore, conversation_id: str
) -> ApprovalRequest | None:
    row = store.get_pending_approval(conversation_id)
    return _from_stored(row) if row is not None else None


def approve(
    store: ConversationStore,
    conversation_id: str,
    approval_id: str | None = None,
) -> ApprovalRequest:
    """Record an explicit human approval for the pending request. Does not post.

    If ``approval_id`` is omitted, the conversation's single pending request is
    used. If provided, it must match that pending request.
    """
    pending = store.get_pending_approval(conversation_id)
    if pending is None:
        raise ApprovalNotFound(
            f"no pending approval for conversation {conversation_id!r}"
        )
    if approval_id is not None and approval_id != pending.approval_id:
        raise ApprovalNotFound(
            f"approval_id {approval_id!r} is not the pending request "
            f"for conversation {conversation_id!r}"
        )
    try:
        stored = store.set_approval_decision(pending.approval_id, status="approved")
    except ApprovalRowMissing as exc:
        raise ApprovalNotFound(str(exc)) from exc
    except ApprovalDecisionConflict as exc:
        raise ApprovalAlreadyDecided(str(exc)) from exc
    return _from_stored(stored)


def decline(
    store: ConversationStore,
    conversation_id: str,
    reason: str,
    approval_id: str | None = None,
) -> ApprovalRequest:
    """Record an explicit human decline with a required reason. Does not post."""
    if not isinstance(reason, str) or not reason.strip():
        raise InvalidDeclineReason("decline reason must be a non-empty string")
    pending = store.get_pending_approval(conversation_id)
    if pending is None:
        raise ApprovalNotFound(
            f"no pending approval for conversation {conversation_id!r}"
        )
    if approval_id is not None and approval_id != pending.approval_id:
        raise ApprovalNotFound(
            f"approval_id {approval_id!r} is not the pending request "
            f"for conversation {conversation_id!r}"
        )
    try:
        stored = store.set_approval_decision(
            pending.approval_id,
            status="declined",
            decline_reason=reason.strip(),
        )
    except ApprovalRowMissing as exc:
        raise ApprovalNotFound(str(exc)) from exc
    except ApprovalDecisionConflict as exc:
        raise ApprovalAlreadyDecided(str(exc)) from exc
    return _from_stored(stored)
