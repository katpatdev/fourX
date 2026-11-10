"""Provider-independent agent loop: model ↔ tools ↔ persistence ↔ approval.

Ownership: the future CLI (or test harness) appends the user message and
persists it *before* calling ``run_agent_turn``. This module only appends and
saves assistant messages and tool results during the model/tool loop.

``post_invoice`` runs deterministic eligibility via ``prepare_post`` and may
create a durable pending approval. It never calls ``ledger.post()``. When a
pending approval is created, the loop stops and raises ``ApprovalRequired``.

Batch behavior: tool calls are processed in order. Read-only tools before a
successful ``post_invoice`` run normally. Once an eligible post creates an
approval, later tools are *not* executed; each receives a ``SkippedForApproval``
tool result so every ``tool_call_id`` has a matching result. Pending approval
and those tool results are committed in one SQLite transaction.

Cancellation is cooperative (``CancellationToken`` / ``threading.Event``).
It is checked before each model call, after each model response, and before
each tool. Unexecuted tools in the interrupted batch receive
``SkippedForCancellation`` results so the next user turn can proceed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from app_actions import execute_app_control_tool
from approval import (
    ApprovalAlreadyPending,
    ApprovalBlockedByApproved,
    ApprovalRequest,
    get_approval,
    get_pending_approval,
)
from cancellation import (
    CancellationToken,
    RunCancelled,
    StaleRunError,
    assert_run_active,
    begin_run,
    end_run,
    run_write,
)
from conversation import Conversation, Message, ToolCall
from invoice_tools import (
    BlockedVendor,
    DuplicateInvoice,
    InvalidInvoiceId,
    InvalidInvoiceTotal,
    InvoiceNotFound,
    VendorNotFound,
    list_invoices,
    lookup_vendor,
    prepare_post,
    read_invoice,
)
from nl_routing import APP_CONTROL_TOOL_NAMES, app_control_declarations
from run_logging import NullRunLogger, RunLogger
from storage import (
    ApprovalPendingExists,
    ApprovalUnprocessedExists,
    ConversationStore,
)

# Smallest practical model surface: history + tool schemas → assistant Message.
ModelFn = Callable[[Conversation, list[dict[str, Any]]], Message]

DEFAULT_MAX_MODEL_CALLS = 8

# Expected tool-level errors are returned to the model; they do not abort the run.
_TOOL_ERRORS = (
    InvalidInvoiceId,
    InvoiceNotFound,
    VendorNotFound,
    BlockedVendor,
    DuplicateInvoice,
    InvalidInvoiceTotal,
    TypeError,
    ValueError,
    KeyError,
    ApprovalAlreadyPending,
    ApprovalBlockedByApproved,
)


class InvalidModelResponse(ValueError):
    """Raised when the model returns something other than a valid assistant Message."""


class IterationLimitReached(RuntimeError):
    """Raised when the model keeps requesting tools past the configured limit."""


class PersistenceError(RuntimeError):
    """Raised when conversation state cannot be saved; the loop must stop."""


class ApprovalRequired(Exception):
    """Human approval boundary: a pending posting request awaits an explicit decision.

    ``approval`` carries the durable request. This is not proof of posting.
    """

    def __init__(self, approval: ApprovalRequest) -> None:
        self.approval = approval
        super().__init__(
            f"approval pending for {approval.invoice_id} "
            f"({approval.approval_id}) in conversation {approval.conversation_id!r}"
        )


class RecoveryRequired(RuntimeError):
    """History is incomplete; do not call the model until an explicit recovery.

    Typical cause: assistant tool requests were saved, then the atomic
    approval + tool-result transaction failed. Full recovery is deferred.
    """


def tool_declarations() -> list[dict[str, Any]]:
    """Invoice tool schemas (approval-gated ``post_invoice`` included).

    Application-control schemas are merged at turn time via
    ``all_tool_declarations`` so provider adapters stay compatible with the
    original four invoice tools.
    """
    return [
        {
            "name": "list_invoices",
            "description": "List available invoices with number, vendor, currency and total.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "read_invoice",
            "description": "Read one invoice JSON document by invoice_id (e.g. invoice-06).",
            "parameters": {
                "type": "object",
                "properties": {"invoice_id": {"type": "string"}},
                "required": ["invoice_id"],
            },
        },
        {
            "name": "lookup_vendor",
            "description": "Look up a vendor by exact name or vendor_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "vendor_id": {"type": "string"},
                },
                "required": [],
            },
        },
        {
            "name": "post_invoice",
            "description": (
                "Request human approval to post an invoice to the ledger. "
                "Does not post immediately; eligibility is checked in application code."
            ),
            "parameters": {
                "type": "object",
                "properties": {"invoice_id": {"type": "string"}},
                "required": ["invoice_id"],
            },
        },
    ]


def all_tool_declarations() -> list[dict[str, Any]]:
    """Invoice tools plus Mode-C application-control tools for the agent loop."""
    return tool_declarations() + app_control_declarations()


def _persist(store: ConversationStore, conversation_id: str, conversation: Conversation) -> None:
    try:
        store.save(conversation_id, conversation)
    except Exception as exc:  # noqa: BLE001 — any save failure must stop the loop
        raise PersistenceError(f"failed to persist conversation {conversation_id!r}: {exc}") from exc


def _validate_assistant(message: Any) -> Message:
    if not isinstance(message, Message):
        raise InvalidModelResponse(
            f"model must return a Message, got {type(message).__name__}"
        )
    if message.role != "assistant":
        raise InvalidModelResponse(f"model message role must be 'assistant', got {message.role!r}")
    return message


def unmatched_tool_call_ids(conversation: Conversation) -> list[str]:
    """Return tool_call ids from assistant batches that lack a tool result."""
    unmatched: list[str] = []
    index = 0
    messages = conversation.messages
    while index < len(messages):
        message = messages[index]
        if message.role != "assistant" or not message.tool_calls:
            index += 1
            continue
        expected = {call.id for call in message.tool_calls}
        index += 1
        seen: set[str] = set()
        while index < len(messages) and messages[index].role == "tool":
            tool_id = messages[index].tool_call_id
            if tool_id:
                seen.add(tool_id)
            index += 1
        missing = [call_id for call_id in expected if call_id not in seen]
        unmatched.extend(missing)
    return unmatched


def apply_skipped_for_cancellation(conversation: Conversation) -> int:
    """Append ``SkippedForCancellation`` results for a trailing incomplete tool batch.

    Safe only when the incomplete assistant tool batch is at the end of history
    (no later user/assistant messages). Returns the number of skip results added.
    Does not execute tools, create approvals, or claim success.

    Raises ``RecoveryRequired`` when unmatched calls exist in a non-trailing /
    ambiguous position that cannot be repaired by appending at the end.
    """
    missing = unmatched_tool_call_ids(conversation)
    if not missing:
        return 0

    messages = conversation.messages
    # Locate the last assistant message that still has unmatched tool calls.
    trail_index = None
    trail_calls: list[ToolCall] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role != "assistant" or not message.tool_calls:
            index += 1
            continue
        assistant_index = index
        expected = {call.id: call for call in message.tool_calls}
        index += 1
        seen: set[str] = set()
        while index < len(messages) and messages[index].role == "tool":
            tool_id = messages[index].tool_call_id
            if tool_id:
                seen.add(tool_id)
            index += 1
        unanswered = [expected[cid] for cid in expected if cid not in seen]
        if unanswered:
            trail_index = assistant_index
            trail_calls = unanswered

    if trail_index is None:
        raise RecoveryRequired(
            "conversation has unmatched tool_call ids that cannot be located"
        )

    # Anything after the tool-result region for this assistant makes append unsafe.
    scan = trail_index + 1
    while scan < len(messages) and messages[scan].role == "tool":
        scan += 1
    if scan != len(messages):
        raise RecoveryRequired(
            "incomplete tool batch is not at the end of history; "
            "manual recovery is required before calling the model"
        )

    for call in trail_calls:
        conversation.add(_skipped_for_cancellation_message(call))
    return len(trail_calls)


def _answered_tool_ids_after(conversation: Conversation, assistant_index: int) -> set[str]:
    answered: set[str] = set()
    index = assistant_index + 1
    while index < len(conversation.messages) and conversation.messages[index].role == "tool":
        tool_id = conversation.messages[index].tool_call_id
        if tool_id:
            answered.add(tool_id)
        index += 1
    return answered


def _execute_readonly_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run one read-only tool. Never calls ledger.post or creates approvals."""
    if name == "post_invoice":
        return {
            "ok": False,
            "error": "post_invoice must be handled by the harness approval path",
            "error_type": "UnsupportedTool",
        }
    if name == "list_invoices":
        if arguments:
            raise TypeError("list_invoices takes no arguments")
        return {"ok": True, "result": list_invoices()}
    if name == "read_invoice":
        if set(arguments.keys()) != {"invoice_id"}:
            raise TypeError("read_invoice requires only invoice_id")
        return {"ok": True, "result": read_invoice(arguments["invoice_id"])}
    if name == "lookup_vendor":
        return {"ok": True, "result": lookup_vendor(**arguments)}
    return {
        "ok": False,
        "error": f"unknown tool: {name!r}",
        "error_type": "UnknownTool",
    }


def execute_tool_call(
    call: ToolCall,
    *,
    store: ConversationStore | None = None,
    conversation_id: str | None = None,
    user_message: str = "",
    turn_state: dict[str, Any] | None = None,
) -> Message:
    """Execute one tool call and return a tool Message linked by tool_call_id.

    Application-control tools require ``store``, ``conversation_id``, and the
    current ``user_message`` for authorization checks. ``turn_state`` tracks
    same-turn approval so posting cannot follow in the same human turn.
    """
    if call.name in APP_CONTROL_TOOL_NAMES:
        if store is None or conversation_id is None:
            return Message(
                role="tool",
                tool_call_id=call.id,
                content=json.dumps(
                    {
                        "ok": False,
                        "error": "application tool missing turn context",
                        "error_type": "MissingContext",
                    }
                ),
            )
        return execute_app_control_tool(
            store=store,
            conversation_id=conversation_id,
            user_message=user_message,
            call=call,
            turn_state=turn_state,
        )
    try:
        if not isinstance(call.arguments, dict):
            raise TypeError("tool arguments must be an object")
        payload = _execute_readonly_tool(call.name, call.arguments)
    except _TOOL_ERRORS as exc:
        payload = {
            "ok": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
    return Message(
        role="tool",
        tool_call_id=call.id,
        content=json.dumps(payload, default=str),
    )


def _skipped_for_approval_message(call: ToolCall) -> Message:
    payload = {
        "ok": False,
        "error": "tool skipped because the run is awaiting human approval",
        "error_type": "SkippedForApproval",
        "tool_name": call.name,
    }
    return Message(
        role="tool",
        tool_call_id=call.id,
        content=json.dumps(payload, default=str),
    )


def _skipped_for_cancellation_message(call: ToolCall) -> Message:
    payload = {
        "ok": False,
        "error": "tool skipped because the run was cancelled",
        "error_type": "SkippedForCancellation",
        "tool_name": call.name,
    }
    return Message(
        role="tool",
        tool_call_id=call.id,
        content=json.dumps(payload, default=str),
    )


def _post_invoice_error_message(call: ToolCall, exc: Exception) -> Message:
    payload = {
        "ok": False,
        "error": str(exc),
        "error_type": type(exc).__name__,
    }
    return Message(
        role="tool",
        tool_call_id=call.id,
        content=json.dumps(payload, default=str),
    )


def _prepare_eligible_post(call: ToolCall) -> dict[str, Any]:
    """Validate post_invoice args and run prepare_post. Never creates approvals."""
    if not isinstance(call.arguments, dict):
        raise TypeError("tool arguments must be an object")
    extra = set(call.arguments) - {"invoice_id"}
    if extra:
        raise TypeError(f"post_invoice does not accept fields: {sorted(extra)}")
    if "invoice_id" not in call.arguments:
        raise TypeError("post_invoice requires invoice_id")
    invoice_id = call.arguments["invoice_id"]
    if not isinstance(invoice_id, str):
        raise TypeError("invoice_id must be a string")
    return prepare_post(invoice_id)


def _raise_startup_guards(store: ConversationStore, conversation_id: str) -> None:
    """Fail closed before any model call when approval or history is unresolved."""
    pending = get_pending_approval(store, conversation_id)
    if pending is not None:
        raise ApprovalRequired(pending)

    conversation = store.load(conversation_id).conversation
    missing = unmatched_tool_call_ids(conversation)
    if missing:
        raise RecoveryRequired(
            f"conversation {conversation_id!r} has incomplete tool results for "
            f"tool_call_id(s) {missing}; do not call the model until recovered"
        )


def _checkpoint(
    store: ConversationStore,
    conversation_id: str,
    conversation: Conversation,
    token: CancellationToken,
) -> None:
    """Persist under the run-ownership lock (check + SQLite save are one section)."""
    with run_write(conversation_id, token):
        _persist(store, conversation_id, conversation)


def _commit_pending_approval(
    store: ConversationStore,
    conversation_id: str,
    details: dict[str, Any],
    conversation: Conversation,
    token: CancellationToken,
) -> Any:
    """Atomic approval+history commit under the same ownership lock as checkpoints."""
    with run_write(conversation_id, token):
        return store.commit_pending_approval_with_conversation(
            conversation_id, details, conversation
        )


def _finish_cancelled_batch(
    *,
    store: ConversationStore,
    conversation_id: str,
    conversation: Conversation,
    assistant: Message,
    assistant_index: int,
    token: CancellationToken,
) -> None:
    """Write SkippedForCancellation for unanswered tool calls, then stop."""
    answered = _answered_tool_ids_after(conversation, assistant_index)
    for call in assistant.tool_calls:
        if call.id in answered:
            continue
        conversation.add(_skipped_for_cancellation_message(call))
    _checkpoint(store, conversation_id, conversation, token)
    raise RunCancelled("agent turn cancelled")


def run_agent_turn(
    *,
    store: ConversationStore,
    conversation_id: str,
    model: ModelFn,
    max_model_calls: int = DEFAULT_MAX_MODEL_CALLS,
    cancel: CancellationToken | None = None,
    user_message: str = "",
    allow_pending_resolution: bool = False,
    run_logger: RunLogger | None = None,
) -> Message:
    """Run the model/tool loop until a final text answer, limit, or approval boundary.

    Raises ``ApprovalRequired`` after atomically persisting a pending approval
    and a complete tool-result batch. Raises ``RunCancelled`` when ``cancel``
    is signaled (after reconciling any open tool batch). Does not append user
    messages.

    ``user_message`` is the current clerk turn (for authorizing approve/decline/post
    tools). When ``allow_pending_resolution`` is True, an existing pending approval
    does not block the model call so the clerk can approve/decline conversationally.

    Cancellation is cooperative: an in-flight blocking SDK call is not aborted
    by the Event alone; after it returns, the harness stops further work.
    """
    if max_model_calls < 1:
        raise ValueError("max_model_calls must be at least 1")

    log = run_logger if run_logger is not None else NullRunLogger(conversation_id)
    token = cancel if cancel is not None else CancellationToken()
    # begin_run shares _active_lock with run_write: waits until any in-flight
    # protected SQLite write finishes, then claims ownership before load.
    begin_run(conversation_id, token)
    try:
        if allow_pending_resolution:
            # Still refuse incomplete tool history; pending approval may be resolved.
            conversation_pre = store.load(conversation_id).conversation
            missing = unmatched_tool_call_ids(conversation_pre)
            if missing:
                raise RecoveryRequired(
                    f"conversation {conversation_id!r} has incomplete tool results for "
                    f"tool_call_id(s) {missing}; do not call the model until recovered"
                )
        else:
            # Pending approval outranks a stop request so the clerk still sees it.
            _raise_startup_guards(store, conversation_id)
        token.check()

        # Load after ownership so we see commits that finished before begin_run.
        # Future CLI must append user messages similarly (do not save a stale
        # Conversation snapshot that omits another turn's protected write).
        stored = store.load(conversation_id)
        conversation = stored.conversation
        declarations = all_tool_declarations()
        # Per human turn: blocks post_approved_invoice after approve in this turn
        # (including later model iterations in the same run_agent_turn).
        turn_state: dict[str, Any] = {
            "approved_this_turn": False,
            "approved_ids_this_turn": set(),
        }
        final: Message | None = None

        for _ in range(max_model_calls):
            assert_run_active(conversation_id, token)
            token.check()

            log.emit("model_call_started", message_count=len(conversation.messages))
            try:
                # SDK / model calls stay outside run_write (must not hold the lock).
                raw = model(conversation, declarations)
            except InvalidModelResponse as exc:
                log.emit("model_call_failed", error=str(exc), error_type="InvalidModelResponse")
                raise
            except Exception as exc:
                log.emit(
                    "model_call_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                raise

            # Soft pre-check only; durable writes re-validate inside run_write.
            assert_run_active(conversation_id, token)
            # Observe cancellation after the (possibly blocking) model returns.
            if token.cancelled:
                assistant = _validate_assistant(raw)
                conversation.add(assistant)
                assistant_index = len(conversation.messages) - 1
                _checkpoint(store, conversation_id, conversation, token)
                if assistant.tool_calls:
                    _finish_cancelled_batch(
                        store=store,
                        conversation_id=conversation_id,
                        conversation=conversation,
                        assistant=assistant,
                        assistant_index=assistant_index,
                        token=token,
                    )
                raise RunCancelled("agent turn cancelled")

            assistant = _validate_assistant(raw)
            log.emit(
                "model_call_completed",
                has_tool_calls=bool(assistant.tool_calls),
                tool_count=len(assistant.tool_calls or ()),
            )
            conversation.add(assistant)
            assistant_index = len(conversation.messages) - 1
            _checkpoint(store, conversation_id, conversation, token)

            if not assistant.tool_calls:
                final = assistant
                break

            call_index = 0
            while call_index < len(assistant.tool_calls):
                assert_run_active(conversation_id, token)
                if token.cancelled:
                    _finish_cancelled_batch(
                        store=store,
                        conversation_id=conversation_id,
                        conversation=conversation,
                        assistant=assistant,
                        assistant_index=assistant_index,
                        token=token,
                    )

                call = assistant.tool_calls[call_index]
                invoice_arg = None
                if isinstance(call.arguments, dict):
                    invoice_arg = call.arguments.get("invoice_id")
                log.emit(
                    "tool_requested",
                    tool_name=call.name,
                    tool_call_id=call.id,
                    invoice_id=invoice_arg if isinstance(invoice_arg, str) else None,
                )
                if call.name != "post_invoice":
                    tool_message = execute_tool_call(
                        call,
                        store=store,
                        conversation_id=conversation_id,
                        user_message=user_message,
                        turn_state=turn_state,
                    )
                    try:
                        payload = json.loads(tool_message.content or "{}")
                    except json.JSONDecodeError:
                        payload = {}
                    if isinstance(payload, dict) and payload.get("ok") is False:
                        log.emit(
                            "tool_failed",
                            tool_name=call.name,
                            tool_call_id=call.id,
                            error_type=payload.get("error_type"),
                        )
                    else:
                        log.emit(
                            "tool_completed",
                            tool_name=call.name,
                            tool_call_id=call.id,
                            ok=True,
                        )
                    conversation.add(tool_message)
                    _checkpoint(store, conversation_id, conversation, token)
                    call_index += 1
                    continue

                # Eligibility only — persistence of pending is deferred to the atomic commit.
                try:
                    details = _prepare_eligible_post(call)
                except _TOOL_ERRORS as exc:
                    conversation.add(_post_invoice_error_message(call, exc))
                    _checkpoint(store, conversation_id, conversation, token)
                    log.emit(
                        "tool_failed",
                        tool_name=call.name,
                        tool_call_id=call.id,
                        error_type=type(exc).__name__,
                    )
                    call_index += 1
                    continue

                if token.cancelled:
                    _finish_cancelled_batch(
                        store=store,
                        conversation_id=conversation_id,
                        conversation=conversation,
                        assistant=assistant,
                        assistant_index=assistant_index,
                        token=token,
                    )

                remaining = list(assistant.tool_calls[call_index + 1 :])
                pending_payload = {
                    "ok": True,
                    "status": "pending",
                    "message": "human approval required before posting",
                    "approval": {
                        key: details[key]
                        for key in (
                            "invoice_id",
                            "vendor_id",
                            "vendor_name",
                            "invoice_number",
                            "invoice_date",
                            "currency",
                            "amount",
                        )
                        if key in details
                    },
                }
                conversation.add(
                    Message(
                        role="tool",
                        tool_call_id=call.id,
                        content=json.dumps(pending_payload, default=str),
                    )
                )
                for skipped in remaining:
                    conversation.add(_skipped_for_approval_message(skipped))

                try:
                    stored_approval = _commit_pending_approval(
                        store, conversation_id, details, conversation, token
                    )
                except StaleRunError:
                    raise
                except (ApprovalPendingExists, ApprovalUnprocessedExists) as exc:
                    drop = 1 + len(remaining)
                    del conversation.messages[-drop:]
                    mapped: Exception
                    if isinstance(exc, ApprovalPendingExists):
                        mapped = ApprovalAlreadyPending(str(exc))
                    else:
                        mapped = ApprovalBlockedByApproved(str(exc))
                    conversation.add(_post_invoice_error_message(call, mapped))
                    _checkpoint(store, conversation_id, conversation, token)
                    call_index += 1
                    continue
                except Exception as exc:  # noqa: BLE001
                    raise PersistenceError(
                        f"failed to commit pending approval with tool results for "
                        f"{conversation_id!r}: {exc}"
                    ) from exc

                loaded = get_approval(store, stored_approval.approval_id)
                assert loaded is not None
                log.emit(
                    "approval_created",
                    approval_id=loaded.approval_id,
                    invoice_id=loaded.invoice_id,
                    status=loaded.status,
                )
                log.emit(
                    "tool_completed",
                    tool_name="post_invoice",
                    tool_call_id=call.id,
                    ok=True,
                    status="pending",
                )
                raise ApprovalRequired(loaded)

        else:
            raise IterationLimitReached(
                f"stopped after {max_model_calls} model call(s) without a final answer"
            )

        assert final is not None
        return final
    except RunCancelled:
        log.emit("cancellation_completed", phase="run_agent_turn")
        raise
    finally:
        end_run(conversation_id, token)
