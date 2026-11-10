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

from storage import ALLOWED_PROVIDERS

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
    r"^\s*(?:switch\s+to|use|continue\s+with)\s+(gemini|groq)\s*[.!?]?\s*$",
    re.IGNORECASE,
)
# Explicit approve — whole utterance, not "looks good" / questions.
_APPROVE = re.compile(
    r"^\s*(?:please\s+)?(?:yes[,\s]+)?(?:i\s+(?:have\s+checked\s+(?:it|this)\s+and\s+)?)?"
    r"(?:approve(?:\s+this(?:\s+(?:invoice|approval|request))?)?|"
    r"i(?:'m|\s+am)\s+happy\s+to\s+approve(?:\s+it|this(?:\s+invoice)?)?|"
    r"i\s+approve(?:\s+this(?:\s+(?:invoice|approval|request))?)?)"
    r"(?:\s+now)?\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_DECLINE = re.compile(
    r"^\s*(?:please\s+)?(?:decline|reject)\s+"
    r"(?:this\s+)?(?:invoice|approval|request)?\s*"
    r"(?:because|as|since)\s+(.+?)\s*$",
    re.IGNORECASE,
)
_POST = re.compile(
    r"^\s*(?:please\s+)?post(?:\s+the)?(?:\s+approved)?(?:\s+invoice)?"
    r"(?:\s+i\s+approved)?(?:\s+now)?\s*[.!?]?\s*$",
    re.IGNORECASE,
)

# Phrases that must NOT authorize (used for Mode C validation too).
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


def match_deterministic_action(text: str) -> RoutedAction | None:
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
    m = _PROVIDER.match(raw)
    if m:
        return RoutedAction("provider", provider=m.group(1).lower())
    if _APPROVE.match(raw) and not _HYPOTHETICAL.search(raw):
        return RoutedAction("approve")
    m = _DECLINE.match(raw)
    if m:
        reason = m.group(1).strip().rstrip(".!?")
        if reason:
            return RoutedAction("decline", reason=reason)
    if _POST.match(raw) and not _HYPOTHETICAL.search(raw):
        return RoutedAction("post")
    return None


# --- Mode C: authorize from the *current* user message only -----------------

def user_message_authorizes_approve(user_message: str) -> bool:
    """True only when the current user turn clearly authorizes approval."""
    text = user_message.strip()
    if not text or _AMBIGUOUS_APPROVAL.match(text) or _HYPOTHETICAL.search(text):
        return False
    if _QUOTED_APPROVE.search(text) and not _APPROVE.match(text):
        # Quoted instruction alone is not authorization.
        if re.search(r"\b(says?|said|quote|invoice\s+says)\b", text, re.I):
            return False
    if _APPROVE.match(text):
        return True
    # Slightly broader but still explicit for LLM-assisted phrasing.
    if re.search(
        r"\b("
        r"i\s+(?:explicitly\s+)?approve|"
        r"please\s+approve|"
        r"i(?:'m|\s+am)\s+happy\s+to\s+approve|"
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
        r"\b(please\s+)?post\s+(?:the\s+)?(?:approved\s+)?(?:invoice|approval|request)\b",
        text,
        re.I,
    ) and not re.search(r"\b(should|would|could|what\s+if)\b", text, re.I):
        return True
    return False


_PROVIDER_NAMED = re.compile(
    r"\b(?:switch\s+to|use|continue\s+with|change\s+(?:over\s+)?to|"
    r"go\s+(?:back\s+)?to)\s+(gemini|groq)\b",
    re.IGNORECASE,
)
_PROVIDER_OTHER = re.compile(
    r"\b(?:(?:the\s+)?other\s+provider|switch\s+providers?|"
    r"change\s+(?:the\s+)?providers?|different\s+provider)\b",
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
    named = _PROVIDER_NAMED.search(text)
    if named:
        return named.group(1).lower() == target_provider
    # Whole-utterance Mode-B style ("Switch to Groq").
    whole = _PROVIDER.match(text)
    if whole:
        return whole.group(1).lower() == target_provider
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
                "Only gemini and groq are allowed."
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
            "description": "Summarize provider, message count, and approval statuses.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "approve_pending_approval",
            "description": (
                "Record human approval for the pending approval when the CURRENT user "
                "message clearly authorizes approval. Does not post to the ledger."
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
