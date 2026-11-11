"""Canonical in-memory conversation history for the FourX agent harness.

Provider SDKs each use different message shapes. We keep one simple, ordered
list of messages here and convert to/from Gemini or Groq only at the edges
(later iterations). Tool calls and tool results stay linked by ``tool_call_id``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant", "tool"]

_ALLOWED_ROLES = frozenset({"user", "assistant", "tool"})


@dataclass(frozen=True)
class ToolCall:
    """One model-requested tool invocation, identified by a stable ``id``."""

    id: str
    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        # Same rules for direct construction and from_jsonable reconstruction.
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("tool call requires a non-empty string id")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("tool call requires a non-empty string name")
        if not isinstance(self.arguments, dict):
            raise ValueError("tool call arguments must be an object")
        try:
            json.dumps(self.arguments)
        except (TypeError, ValueError) as exc:
            raise ValueError("tool call arguments must be JSON-compatible") from exc

    def to_jsonable(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": dict(self.arguments)}

    @classmethod
    def from_jsonable(cls, data: Any) -> ToolCall:
        if not isinstance(data, dict):
            raise ValueError(f"tool call must be an object, got {type(data).__name__}")
        # Reject unknown keys so persistence round-trips do not silently drop fields.
        unknown = set(data) - {"id", "name", "arguments"}
        if unknown:
            raise ValueError(f"unsupported tool call field(s): {sorted(unknown)}")
        arguments = data.get("arguments", {})
        return cls(id=data.get("id"), name=data.get("name"), arguments=arguments)  # type: ignore[arg-type]

@dataclass(frozen=True)
class Message:
    """One turn in the conversation.

    - ``user``: ordinary text (``content`` required).
    - ``assistant``: optional text plus zero or more ``tool_calls``.
    - ``tool``: a tool result tied to an earlier call via ``tool_call_id``.

    ``extras`` holds optional JSON-compatible provider metadata (for example
    Gemini thought signatures / native content parts) so Gemini↔Groq switching
    can keep canonical tool IDs while still round-tripping Gemini-native data.
    """

    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    extras: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _validate_message(self)

    def to_jsonable(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            payload["content"] = self.content
        if self.tool_calls:
            payload["tool_calls"] = [call.to_jsonable() for call in self.tool_calls]
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.extras is not None:
            payload["extras"] = dict(self.extras)
        return payload

    @classmethod
    def from_jsonable(cls, data: Any) -> Message:
        if not isinstance(data, dict):
            raise ValueError(f"message must be an object, got {type(data).__name__}")
        unknown = set(data) - {"role", "content", "tool_calls", "tool_call_id", "extras"}
        if unknown:
            raise ValueError(f"unsupported message field(s): {sorted(unknown)}")

        role = data.get("role")
        if role not in _ALLOWED_ROLES:
            raise ValueError(f"unsupported role: {role!r}")

        content = data.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError("content must be a string when present")

        raw_calls = data.get("tool_calls", [])
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, list):
            raise ValueError("tool_calls must be a list when present")
        tool_calls = tuple(ToolCall.from_jsonable(item) for item in raw_calls)

        tool_call_id = data.get("tool_call_id")
        if tool_call_id is not None and not isinstance(tool_call_id, str):
            raise ValueError("tool_call_id must be a string when present")

        extras = data.get("extras")
        if extras is not None and not isinstance(extras, dict):
            raise ValueError("extras must be an object when present")

        return cls(
            role=role,  # type: ignore[arg-type]
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            extras=dict(extras) if extras is not None else None,
        )


def _validate_message(message: Message) -> None:
    if message.role not in _ALLOWED_ROLES:
        raise ValueError(f"unsupported role: {message.role!r}")

    if message.extras is not None:
        if not isinstance(message.extras, dict):
            raise ValueError("extras must be a dict when present")
        try:
            json.dumps(message.extras)
        except (TypeError, ValueError) as exc:
            raise ValueError("extras must be JSON-compatible") from exc

    if message.role == "user":
        if not isinstance(message.content, str) or message.content == "":
            raise ValueError("user messages require non-empty content")
        if message.tool_calls:
            raise ValueError("user messages cannot include tool_calls")
        if message.tool_call_id is not None:
            raise ValueError("user messages cannot include tool_call_id")
        if message.extras is not None:
            raise ValueError("user messages cannot include extras")
        return

    if message.role == "assistant":
        if message.tool_call_id is not None:
            raise ValueError("assistant messages cannot include tool_call_id")
        if message.content is None and not message.tool_calls:
            raise ValueError("assistant messages need content and/or tool_calls")
        if message.content is not None and not isinstance(message.content, str):
            raise ValueError("assistant content must be a string when present")
        seen_ids: set[str] = set()
        for call in message.tool_calls:
            if call.id in seen_ids:
                raise ValueError(f"duplicate tool_call id in one assistant message: {call.id!r}")
            seen_ids.add(call.id)
        return

    # role == "tool"
    if not isinstance(message.tool_call_id, str) or not message.tool_call_id:
        raise ValueError("tool messages require a non-empty tool_call_id")
    if message.tool_calls:
        raise ValueError("tool messages cannot include tool_calls")
    if message.content is None:
        raise ValueError("tool messages require content (the tool result payload)")
    if not isinstance(message.content, str):
        raise ValueError("tool message content must be a string")
    if message.extras is not None:
        raise ValueError("tool messages cannot include extras")


@dataclass
class Conversation:
    """Ordered conversation history. Append-only list preserves event order."""

    messages: list[Message] = field(default_factory=list)

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def to_jsonable(self) -> list[dict[str, Any]]:
        return [message.to_jsonable() for message in self.messages]

    @classmethod
    def from_jsonable(cls, data: Any) -> Conversation:
        if not isinstance(data, list):
            raise ValueError(f"conversation must be a list of messages, got {type(data).__name__}")
        return cls(messages=[Message.from_jsonable(item) for item in data])
