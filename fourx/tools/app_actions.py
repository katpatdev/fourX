"""Execute validated application-control tools (Mode C).

Called from the harness with the current user message. Financial mutations go
only through ``approval`` / ``posting`` APIs after local authorization checks.
"""

from __future__ import annotations

import json
from typing import Any

from fourx.finance.approval import (
    ApprovalAlreadyDecided,
    ApprovalNotFound,
    DISPLAY_KEYS,
    InvalidDeclineReason,
    decline,
    get_pending_approval,
    list_approvals,
)
from fourx.tools.invoice_tools import BlockedVendor, DuplicateInvoice
from fourx.diagnostics.app_state import build_trusted_state_summary
from fourx.tools.nl_routing import (
    AmbiguousTarget,
    ResolvedTarget,
    extract_decline_reason,
    normalize_provider,
    resolve_approval_target,
    user_message_authorizes_approve,
    user_message_authorizes_post,
    user_message_authorizes_provider_switch,
)
from fourx.finance.posting import (
    ApprovalNotPostable,
    InvoiceReservedByOther,
    LedgerInvoiceConflict,
    PostingDetailsMismatch,
    PostingError,
    PostingUncertain,
    approve_and_post,
    post_approved,
)
from fourx.persistence.storage import ConversationStore, InvalidProvider, StoredApproval
from fourx.agent.conversation import Message, ToolCall


def _ok(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def _err(message: str, error_type: str = "ActionRejected") -> dict[str, Any]:
    return {"ok": False, "error": message, "error_type": error_type}


def _approval_summary(row: Any) -> dict[str, Any]:
    """Clerk-facing fields from ApprovalRequest or StoredApproval (no display())."""
    details = getattr(row, "details", None)
    if not isinstance(details, dict):
        details = {}
    summary = {key: details[key] for key in DISPLAY_KEYS if key in details}
    if "invoice_id" not in summary and getattr(row, "invoice_id", None):
        summary["invoice_id"] = row.invoice_id
    return summary


def execute_app_control_tool(
    *,
    store: ConversationStore,
    conversation_id: str,
    user_message: str,
    call: ToolCall,
    turn_state: dict[str, Any] | None = None,
) -> Message:
    """Validate and run one app-control tool; return a tool Message."""
    args = call.arguments if isinstance(call.arguments, dict) else {}
    state = turn_state if turn_state is not None else {}
    try:
        payload = _dispatch(
            store, conversation_id, user_message, call.name, args, state
        )
    except (
        ApprovalNotFound,
        ApprovalAlreadyDecided,
        InvalidDeclineReason,
        ApprovalNotPostable,
        PostingError,
        PostingUncertain,
        BlockedVendor,
        DuplicateInvoice,
        InvoiceReservedByOther,
        LedgerInvoiceConflict,
        PostingDetailsMismatch,
        InvalidProvider,
        TypeError,
        ValueError,
        KeyError,
    ) as exc:
        payload = _err(str(exc), type(exc).__name__)
        if isinstance(exc, PostingUncertain):
            payload["approval_id"] = exc.approval.approval_id
            payload["status"] = exc.approval.status
    except Exception as exc:  # noqa: BLE001
        payload = _err(str(exc), type(exc).__name__)
    return Message(
        role="tool",
        tool_call_id=call.id,
        content=json.dumps(payload, default=str),
    )


def _dispatch(
    store: ConversationStore,
    conversation_id: str,
    user_message: str,
    name: str,
    args: dict[str, Any],
    turn_state: dict[str, Any],
) -> dict[str, Any]:
    if name == "get_current_provider":
        return _ok(provider=store.load(conversation_id).provider)

    if name == "switch_provider":
        provider = normalize_provider(args.get("provider"))
        if provider is None:
            return _err("provider must be 'gemini' or 'groq'", "InvalidProvider")
        if not user_message_authorizes_provider_switch(user_message, provider):
            return _err(
                "current user message does not clearly request a provider switch",
                "AuthorizationRequired",
            )
        store.set_provider(conversation_id, provider)
        return _ok(provider=provider, message=f"switched to {provider}")

    if name == "get_pending_approval":
        pending = get_pending_approval(store, conversation_id)
        if pending is None:
            return _ok(pending=None, message="no pending approval")
        return _ok(
            pending=pending.display(),
            approval_id=pending.approval_id,
            status=pending.status,
        )

    if name == "get_application_status":
        summary = build_trusted_state_summary(store, conversation_id)
        stored = store.load(conversation_id)
        history = list_approvals(store, conversation_id)
        return _ok(
            provider=summary["provider"],
            messages=len(stored.conversation.messages),
            pending=summary["pending_approval"] is not None,
            pending_approval=summary["pending_approval"],
            approved_unposted=summary["approved_unposted"],
            posted=summary["posted"],
            most_recently_referenced_invoice=summary[
                "most_recently_referenced_invoice"
            ],
            approvals=[
                {
                    "approval_id": a.approval_id,
                    "status": a.status,
                    "invoice_id": a.invoice_id,
                    "journal_entry_id": a.journal_entry_id,
                }
                for a in history
            ],
        )

    if name == "inspect_posting_status":
        rows = store.list_unprocessed_approved(conversation_id)
        history = list_approvals(store, conversation_id)
        return _ok(
            unresolved=[
                {
                    "approval_id": r.approval_id,
                    "status": r.status,
                    "invoice_id": r.invoice_id,
                }
                for r in rows
            ],
            recent=[
                {
                    "approval_id": a.approval_id,
                    "status": a.status,
                    "journal_entry_id": a.journal_entry_id,
                }
                for a in history[-5:]
            ],
        )

    if name == "approve_pending_approval":
        if not user_message_authorizes_approve(user_message):
            return _err(
                "current user message does not clearly authorize approval; "
                "ask the clerk to say e.g. 'I want to approve this' or "
                "'approve this invoice'",
                "AuthorizationRequired",
            )
        resolved = resolve_approval_target(
            user_message,
            store=store,
            conversation_id=conversation_id,
            intent="approve",
        )
        if resolved is None:
            return _err("no pending approval to approve", "ApprovalNotFound")
        if isinstance(resolved, AmbiguousTarget):
            return _err(resolved.message, "AmbiguousApprovalTarget")
        pending = resolved.approval
        requested = args.get("approval_id")
        if requested and requested != pending.approval_id:
            return _err(
                f"approval_id {requested!r} is not the current pending approval "
                f"{pending.approval_id!r}",
                "ApprovalMismatch",
            )
        # Assignment semantics: human approve → safe post_approved workflow.
        try:
            result = approve_and_post(
                store, conversation_id, pending.approval_id
            )
        except PostingUncertain as exc:
            turn_state["approved_this_turn"] = True
            row = exc.approval
            return {
                "ok": False,
                "error": str(exc),
                "error_type": "PostingUncertain",
                "approval_id": row.approval_id,
                "invoice_id": row.invoice_id,
                "status": row.status,
                "journal_entry_id": getattr(row, "journal_entry_id", None),
                "message": (
                    "approval recorded but posting is uncertain; "
                    "use /reconcile — no automatic retry"
                ),
            }
        turn_state["approved_this_turn"] = True
        approved_ids = turn_state.setdefault("approved_ids_this_turn", set())
        if isinstance(approved_ids, set):
            approved_ids.add(result.approval_id)
        if not isinstance(result, StoredApproval):
            return _err(
                f"unexpected approve/post result type {type(result).__name__}",
                "PostingError",
            )
        if result.status != "posted" or not result.journal_entry_id:
            return {
                "ok": False,
                "error": (
                    "approval recorded but posting did not confirm a journal entry "
                    f"(status={result.status!r})"
                ),
                "error_type": "PostingNotConfirmed",
                "approval_id": result.approval_id,
                "invoice_id": result.invoice_id,
                "status": result.status,
                "journal_entry_id": result.journal_entry_id,
            }
        return _ok(
            status=result.status,
            approval_id=result.approval_id,
            invoice_id=result.invoice_id,
            journal_entry_id=result.journal_entry_id,
            message=f"approved and posted as {result.journal_entry_id}",
            approval=_approval_summary(result),
        )

    if name == "decline_pending_approval":
        reason = args.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            return _err("decline requires a non-empty reason", "InvalidDeclineReason")
        # Reason must appear in (or match) the current user message — no invention.
        extracted = extract_decline_reason(user_message)
        if extracted is None:
            return _err(
                "current user message does not include a decline reason "
                "(e.g. 'decline because PO missing')",
                "AuthorizationRequired",
            )
        # Prefer the reason from the user message over a model-invented one.
        reason = extracted
        pending = get_pending_approval(store, conversation_id)
        if pending is None:
            return _err("no pending approval to decline", "ApprovalNotFound")
        requested = args.get("approval_id")
        if requested and requested != pending.approval_id:
            return _err(
                f"approval_id {requested!r} is not the current pending approval",
                "ApprovalMismatch",
            )
        result = decline(store, conversation_id, reason)
        return _ok(
            status=result.status,
            approval_id=result.approval_id,
            decline_reason=result.decline_reason,
            approval=result.display(),
        )

    if name == "post_approved_invoice":
        if not user_message_authorizes_post(user_message):
            return _err(
                "current user message does not explicitly request posting; "
                "say e.g. 'post this invoice' or 'post the approved invoice'",
                "AuthorizationRequired",
            )
        # Approval and posting require separate human turns.
        if user_message_authorizes_approve(user_message) or turn_state.get(
            "approved_this_turn"
        ):
            return _err(
                "approval and posting must be separate turns; "
                "approve first, then post in a later message",
                "SeparateTurnRequired",
            )
        approval_id = args.get("approval_id")
        if approval_id is not None and not isinstance(approval_id, str):
            return _err("approval_id must be a string", "TypeError")
        if not approval_id:
            resolved = resolve_approval_target(
                user_message,
                store=store,
                conversation_id=conversation_id,
                intent="post",
            )
            if resolved is None:
                return _err(
                    "no approved request ready to post",
                    "ApprovalNotFound",
                )
            if isinstance(resolved, AmbiguousTarget):
                return _err(resolved.message, "AmbiguousPostTarget")
            approval_id = resolved.approval.approval_id
        else:
            # Model-supplied id must still match a durable approved row and
            # any explicit invoice reference in the current user message.
            resolved = resolve_approval_target(
                user_message,
                store=store,
                conversation_id=conversation_id,
                intent="post",
            )
            if isinstance(resolved, AmbiguousTarget):
                return _err(resolved.message, "AmbiguousPostTarget")
            if isinstance(resolved, ResolvedTarget):
                if resolved.approval.approval_id != approval_id:
                    return _err(
                        f"approval_id {approval_id!r} does not match resolved "
                        f"target {resolved.approval.approval_id!r}",
                        "ApprovalMismatch",
                    )
        try:
            result = post_approved(store, conversation_id, approval_id)
        except PostingUncertain as exc:
            row = exc.approval
            return {
                "ok": False,
                "error": str(exc),
                "error_type": "PostingUncertain",
                "approval_id": row.approval_id,
                "status": row.status,
                "journal_entry_id": getattr(row, "journal_entry_id", None),
            }
        if not isinstance(result, StoredApproval):
            return _err(
                f"unexpected posting result type {type(result).__name__}",
                "PostingError",
            )
        # Success only with verified posted status and journal evidence.
        if result.status != "posted" or not result.journal_entry_id:
            return {
                "ok": False,
                "error": (
                    "posting did not confirm a verified journal entry "
                    f"(status={result.status!r})"
                ),
                "error_type": "PostingNotConfirmed",
                "approval_id": result.approval_id,
                "status": result.status,
                "journal_entry_id": result.journal_entry_id,
            }
        return _ok(
            status=result.status,
            approval_id=result.approval_id,
            journal_entry_id=result.journal_entry_id,
            approval=_approval_summary(result),
        )

    return _err(f"unknown application tool: {name!r}", "UnknownTool")
