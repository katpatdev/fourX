"""Natural-language routing helpers for hybrid CLI interaction.

Three layers (used by ``cli.py`` / ``harness.py``):

1. Narrow deterministic phrase matches (no model call).
2. Authorization validators for the *current* user turn (approve / decline / post).
3. Application-control tool schemas executed only after Python validation.

Never treat free-form assistant text as an executable command.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from fourx.persistence.storage import ALLOWED_PROVIDERS, ConversationStore, StoredApproval

ActionName = Literal[
    "stop",
    "quit",
    "help",
    "status",
    "pending",
    "provider",
    "approve",
    "decline",
    "post",
]


@dataclass(frozen=True)
class RoutedAction:
    """A locally resolved application action (Mode B)."""

    action: ActionName
    provider: str | None = None
    reason: str | None = None
    approval_id: str | None = None
    invoice_id: str | None = None


@dataclass(frozen=True)
class ResolvedTarget:
    """A single durable approval/invoice target resolved from context."""

    approval: StoredApproval
    invoice_id: str


@dataclass(frozen=True)
class AmbiguousTarget:
    """More than one candidate; clerk must clarify."""

    message: str
    candidates: tuple[str, ...]


# --- Mode B: narrow whole-utterance matches ---------------------------------

_STOP = re.compile(
    r"^\s*(stop|cancel(?:\s+the\s+current\s+task)?|stop\s+processing)\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_QUIT = re.compile(
    r"^\s*(quit|exit(?:\s+the\s+application)?|goodbye)\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_HELP = re.compile(r"^\s*(help|show\s+help)\s*[.!?]?\s*$", re.IGNORECASE)
_STATUS = re.compile(
    r"^\s*(show(?:\s+current)?\s+status|status|what(?:'s|\s+is)\s+the\s+status)\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_PENDING = re.compile(
    r"^\s*(show(?:\s+my)?\s+pending\s+approval|pending\s+approval|"
    r"what(?:'s|\s+is)\s+pending)\s*[.!?]?\s*$",
    re.IGNORECASE,
)

_PROVIDER = re.compile(
    r"^\s*(?:"
    r"(?:please\s+)?(?:can\s+we\s+|could\s+we\s+|let'?s\s+)?"
    r"(?:switch\s+to|use|continue\s+with|change(?:\s+(?:over\s+)?to)?|"
    r"change\s+the\s+model\s+to|i\s+want\s+to\s+use)\s+"
    r"(?:the\s+)?(gemini|groq)(?:\s+model)?|"
    r"(?:please\s+)?(?:switch|change)\s+to\s+(?:the\s+)?(gemini|groq)(?:\s+model)?"
    r")\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_PROVIDER_OTHER_UTTERANCE = re.compile(
    r"^\s*(?:please\s+)?(?:can\s+we\s+|could\s+we\s+|let'?s\s+)?"
    r"(?:switch\s+to\s+(?:the\s+)?other\s+provider|"
    r"use\s+(?:the\s+)?other\s+provider|"
    r"switch\s+providers?|"
    r"change\s+(?:the\s+)?providers?)\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_PROVIDER_INQUIRY = re.compile(
    r"(?i)^\s*(?:"
    r"what\s+is\s+(?:gemini|groq)|"
    r"who\s+is\s+(?:gemini|groq)|"
    r"are\s+we\s+(?:already\s+)?using\s+(?:gemini|groq)|"
    r"which\s+(?:provider|model)\s+are\s+we\s+using|"
    r"what\s+(?:provider|model)\s+are\s+we\s+(?:on|using)|"
    r"current\s+provider\??|"
    r"do\s+we\s+use\s+(?:gemini|groq)"
    r").*$"
)

_APPROVE = re.compile(
    r"^\s*(?:please\s+)?"
    r"(?:"
    r"(?:yes[,\s]+)?(?:i\s+(?:want\s+to|would\s+like\s+to)\s+)?"
    r"approve(?:\s+(?:this|it|that|the))?(?:\s+(?:invoice|approval|request))?"
    r"(?:\s+(?:now|please))?"
    r"|"
    r"(?:yes[,\s]+)?i\s+approve(?:\s+(?:this|it|that))?"
    r"(?:\s+(?:invoice|approval|request))?"
    r"(?:\s+(?:invoice-\d+|the\s+(?:first|second|third|\d+(?:st|nd|rd|th))\s+invoice))?"
    r"|"
    r"(?:yes[,\s]+)?i(?:'m|\s+am)\s+happy\s+to\s+approve(?:\s+(?:it|this)(?:\s+invoice)?)?"
    r"|"
    r"please\s+approve(?:\s+(?:it|this|that)(?:\s+(?:invoice|approval|request))?)?"
    r")"
    r"\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_DECLINE = re.compile(
    r"^\s*(?:please\s+)?(?:decline|reject)\s+"
    r"(?:this\s+)?(?:invoice|approval|request)?\s*"
    r"(?:because|as|since)\s+(.+?)\s*$",
    re.IGNORECASE,
)
_POST = re.compile(
    r"^\s*(?:(?:okay|ok|sure|please|go\s+ahead(?:\s+and)?),?\s+)*"
    r"post(?:\s+(?:this|that|it|the))?(?:\s+approved)?(?:\s+invoice)?"
    r"(?:\s+(?:to\s+(?:the\s+)?ledger|now|i\s+approved|the\s+approved\s+invoice))?"
    r"(?:\s+now)?"
    r"\s*[.!?]?\s*$",
    re.IGNORECASE,
)

_AMBIGUOUS_APPROVAL = re.compile(
    r"^\s*(looks\s+good|okay|ok|sure|what\s+do\s+you\s+think|"
    r"should\s+i\s+approve|the\s+model\s+recommends)\s*[!?.]*\s*$",
    re.IGNORECASE,
)
_HYPOTHETICAL = re.compile(
    r"\b(what\s+happens\s+if|what\s+happens\s+when|should\s+i|"
    r"can\s+i\s+approve|do\s+you\s+think|would\s+it\s+be)\b",
    re.IGNORECASE,
)
_QUOTED_APPROVE = re.compile(
    r"""['"].*\bapprove\b.*['"]""",
    re.IGNORECASE,
)

_INVOICE_ID = re.compile(r"\binvoice-\d+\b", re.IGNORECASE)
_ORDINAL = re.compile(
    r"\b(?:the\s+)?(first|second|third|fourth|fifth|sixth|1st|2nd|3rd|4th|5th|6th)"
    r"\s+invoice\b",
    re.IGNORECASE,
)
_ORDINAL_MAP = {
    "first": 1,
    "1st": 1,
    "second": 2,
    "2nd": 2,
    "third": 3,
    "3rd": 3,
    "fourth": 4,
    "4th": 4,
    "fifth": 5,
    "5th": 5,
    "sixth": 6,
    "6th": 6,
}


def match_deterministic_action(
    text: str, *, current_provider: str | None = None
) -> RoutedAction | None:
    """Return a Mode-B action when the whole utterance is an unambiguous shortcut."""
    raw = text.strip()
    if not raw or raw.startswith("/"):
        return None
    if _STOP.match(raw):
        return RoutedAction("stop")
    if _QUIT.match(raw):
        return RoutedAction("quit")
    if _HELP.match(raw):
        return RoutedAction("help")
    if _STATUS.match(raw):
        return RoutedAction("status")
    if _PENDING.match(raw):
        return RoutedAction("pending")
    if _PROVIDER_INQUIRY.match(raw):
        return None
    m = _PROVIDER.match(raw)
    if m:
        provider = (m.group(1) or m.group(2) or "").lower()
        if provider in ALLOWED_PROVIDERS:
            return RoutedAction("provider", provider=provider)
    if _PROVIDER_OTHER_UTTERANCE.match(raw):
        other = _other_provider(current_provider)
        if other is not None:
            return RoutedAction("provider", provider=other)
    if _APPROVE.match(raw) and not _HYPOTHETICAL.search(raw):
        return RoutedAction("approve", invoice_id=extract_invoice_reference(raw))
    m = _DECLINE.match(raw)
    if m:
        reason = m.group(1).strip().rstrip(".!?")
        if reason:
            return RoutedAction("decline", reason=reason)
    if _POST.match(raw) and not _HYPOTHETICAL.search(raw):
        return RoutedAction("post", invoice_id=extract_invoice_reference(raw))
    return None


def _other_provider(current: str | None) -> str | None:
    if current == "gemini":
        return "groq"
    if current == "groq":
        return "gemini"
    return None


def extract_invoice_reference(text: str) -> str | None:
    """Return an explicit invoice-id or ordinal mapping when present."""
    m = _INVOICE_ID.search(text)
    if m:
        return m.group(0).lower()
    ord_m = _ORDINAL.search(text)
    if ord_m:
        n = _ORDINAL_MAP.get(ord_m.group(1).lower())
        if n is not None:
            return f"invoice-{n:02d}"
    return None


# --- Mode C: authorize from the *current* user message only -----------------

def user_message_authorizes_approve(user_message: str) -> bool:
    """True only when the current user turn clearly authorizes approval."""
    text = user_message.strip()
    if not text or _AMBIGUOUS_APPROVAL.match(text) or _HYPOTHETICAL.search(text):
        return False
    if _QUOTED_APPROVE.search(text) and not _APPROVE.match(text):
        if re.search(r"\b(says?|said|quote|invoice\s+says)\b", text, re.I):
            return False
    if _APPROVE.match(text):
        return True
    if re.search(
        r"\b("
        r"i\s+(?:want\s+to|would\s+like\s+to|explicitly\s+)?approve|"
        r"please\s+approve|"
        r"i(?:'m|\s+am)\s+happy\s+to\s+approve|"
        r"yes[,\s]+i\s+approve|"
        r"approve\s+(?:this|it|that|the)\b|"
        r"approve\s+(?:the\s+)?(?:current\s+)?(?:pending\s+)?"
        r"(?:invoice|approval|request)"
        r")\b",
        text,
        re.I,
    ) and not _HYPOTHETICAL.search(text):
        return True
    return False


def extract_decline_reason(user_message: str) -> str | None:
    """Return decline reason from the current user message, or None."""
    text = user_message.strip()
    m = _DECLINE.match(text)
    if m:
        reason = m.group(1).strip().rstrip(".!?")
        return reason or None
    m = re.search(
        r"\b(?:decline|reject)\b.{0,40}?\b(?:because|as|since)\s+(.+)$",
        text,
        re.I | re.S,
    )
    if m:
        reason = m.group(1).strip().rstrip(".!?")
        return reason or None
    return None


def user_message_authorizes_post(user_message: str) -> bool:
    """True only for an explicit posting instruction in the current turn."""
    text = user_message.strip()
    if not text or _HYPOTHETICAL.search(text):
        return False
    if _POST.match(text):
        return True
    if re.search(
        r"\b(please\s+|go\s+ahead(?:\s+and)\s+|okay[,\s]+|ok[,\s]+)?"
        r"post\s+(?:this|that|it|the)\b",
        text,
        re.I,
    ) and not re.search(r"\b(should|would|could|what\s+if)\b", text, re.I):
        return True
    if re.search(
        r"\b(please\s+)?post\s+(?:the\s+)?(?:approved\s+)?(?:invoice|approval|request)\b",
        text,
        re.I,
    ) and not re.search(r"\b(should|would|could|what\s+if)\b", text, re.I):
        return True
    if re.search(
        r"\bpost\s+it\s+to\s+(?:the\s+)?ledger\b",
        text,
        re.I,
    ):
        return True
    return False


_PROVIDER_NAMED = re.compile(
    r"\b(?:switch\s+to|use|continue\s+with|change\s+(?:over\s+)?to|"
    r"change\s+the\s+model\s+to|go\s+(?:back\s+)?to|i\s+want\s+to\s+use|"
    r"can\s+we\s+switch\s+to|could\s+we\s+switch\s+to|let'?s\s+continue\s+with)\s+"
    r"(?:the\s+)?(gemini|groq)(?:\s+model)?\b",
    re.IGNORECASE,
)
_PROVIDER_OTHER = re.compile(
    r"\b(?:(?:the\s+)?other\s+provider|switch\s+providers?|"
    r"change\s+(?:the\s+)?providers?|different\s+provider|"
    r"switch\s+to\s+(?:the\s+)?other\s+provider)\b",
    re.IGNORECASE,
)


def user_message_authorizes_provider_switch(
    user_message: str, target_provider: str
) -> bool:
    """True when the current turn clearly asks to change providers.

    Named targets must match. Phrases like "the other provider" authorize a
    switch to either allowed provider. Incidental mentions alone do not.
    """
    text = user_message.strip()
    if not text or target_provider not in ALLOWED_PROVIDERS:
        return False
    if _PROVIDER_INQUIRY.match(text):
        return False
    named = _PROVIDER_NAMED.search(text)
    if named:
        return named.group(1).lower() == target_provider
    whole = _PROVIDER.match(text)
    if whole:
        provider = (whole.group(1) or whole.group(2) or "").lower()
        return provider == target_provider
    if _PROVIDER_OTHER.search(text):
        return True
    return False


def normalize_provider(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    name = value.strip().lower()
    if name in ALLOWED_PROVIDERS:
        return name
    return None


def resolve_approval_target(
    user_message: str,
    *,
    store: ConversationStore,
    conversation_id: str,
    intent: Literal["approve", "post"],
) -> ResolvedTarget | AmbiguousTarget | None:
    """Resolve 'this' / ordinals / invoice ids against durable approval rows.

    For approve: candidates are pending. For post: candidates are approved.
    Returns None when there is no candidate at all.
    """
    from fourx.finance.approval import get_pending_approval, list_approvals

    if intent == "approve":
        pending = get_pending_approval(store, conversation_id)
        candidates = [pending] if pending is not None else []
    else:
        candidates = [
            row
            for row in list_approvals(store, conversation_id)
            if row.status == "approved"
        ]

    if not candidates:
        return None

    wanted = extract_invoice_reference(user_message)
    if wanted:
        matched = [c for c in candidates if c.invoice_id == wanted]
        if len(matched) == 1:
            return ResolvedTarget(approval=matched[0], invoice_id=matched[0].invoice_id)
        if not matched:
            labels = tuple(sorted({c.invoice_id for c in candidates}))
            return AmbiguousTarget(
                message=(
                    f"No {intent}-eligible approval for {wanted}. "
                    f"Candidates: {', '.join(labels) or 'none'}."
                ),
                candidates=labels,
            )

    if len(candidates) == 1:
        row = candidates[0]
        return ResolvedTarget(approval=row, invoice_id=row.invoice_id)

    labels = tuple(sorted({c.invoice_id for c in candidates}))
    return AmbiguousTarget(
        message=(
            f"Ambiguous {intent} target. "
            f"Please name one of: {', '.join(labels)}."
        ),
        candidates=labels,
    )


# --- Application-control tool schemas ---------------------------------------

APP_CONTROL_TOOL_NAMES = frozenset(
    {
        "get_current_provider",
        "switch_provider",
        "get_pending_approval",
        "get_application_status",
        "approve_pending_approval",
        "decline_pending_approval",
        "post_approved_invoice",
        "inspect_posting_status",
    }
)


def app_control_declarations() -> list[dict[str, Any]]:
    """Tool schemas for Mode-C application control (manual orchestration)."""
    return [
        {
            "name": "get_current_provider",
            "description": "Return the conversation's current model provider (gemini or groq).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "switch_provider",
            "description": (
                "Switch the conversation provider when the user asked to change providers. "
                "Only gemini and groq are allowed. Do not call this merely because a "
                "provider name was mentioned, or when the user only asked a question "
                "about the current provider."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "enum": ["gemini", "groq"],
                    }
                },
                "required": ["provider"],
            },
        },
        {
            "name": "get_pending_approval",
            "description": "Show the current pending approval awaiting human decision, if any.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_application_status",
            "description": (
                "Summarize provider, message count, pending approval, approved-but-unposted "
                "invoices, and posted journal IDs from the application database."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "approve_pending_approval",
            "description": (
                "When the CURRENT user message clearly authorizes approval, record "
                "that approval and post the invoice through the safe application "
                "posting workflow. Never invent approval; Python validates the message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "approval_id": {
                        "type": "string",
                        "description": "Optional specific approval_id; defaults to the pending one.",
                    }
                },
                "required": [],
            },
        },
        {
            "name": "decline_pending_approval",
            "description": (
                "Decline the pending approval using a reason taken from the CURRENT "
                "user message. Does not post."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Decline reason copied from the user's message.",
                    },
                    "approval_id": {"type": "string"},
                },
                "required": ["reason"],
            },
        },
        {
            "name": "post_approved_invoice",
            "description": (
                "Post an already-approved invoice when the CURRENT user message "
                "explicitly requests posting. Never combines approve+post."
            ),
            "parameters": {
                "type": "object",
                "properties": {"approval_id": {"type": "string"}},
                "required": [],
            },
        },
        {
            "name": "inspect_posting_status",
            "description": "List recent approvals and any posting/posting_uncertain rows.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    ]
