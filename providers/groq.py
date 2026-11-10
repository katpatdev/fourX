"""Groq adapter using the official ``groq`` SDK (local tool calling only).

Converts canonical Conversation history ↔ OpenAI-style chat messages with
``tool_calls`` / ``role=tool``. Application tools are never executed by the
SDK — only by ``harness.py``.
"""

from __future__ import annotations

import json
import os
from typing import Any

from conversation import Conversation, Message, ToolCall

# Tool-calling model verified against the Groq API for this take-home.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
# Per-request timeout (seconds). Does not cooperatively cancel mid-flight; it
# only bounds how long the blocking SDK call may wait.
DEFAULT_GROQ_TIMEOUT_S = 60.0


class MissingAPIKeyError(ValueError):
    """Raised when GROQ_API_KEY is missing."""


class ProviderResponseError(ValueError):
    """Raised when the Groq response cannot be converted."""


def groq_tools_from_declarations(declarations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert harness tool declarations into Groq function-tool schemas."""
    tools: list[dict[str, Any]] = []
    for decl in declarations:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": decl["name"],
                    "description": decl.get("description") or "",
                    "parameters": decl.get("parameters")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return tools


def conversation_to_groq_messages(conversation: Conversation) -> list[dict[str, Any]]:
    """Build Groq chat messages from canonical history."""
    messages: list[dict[str, Any]] = []
    for message in conversation.messages:
        if message.role == "user":
            messages.append({"role": "user", "content": message.content})
            continue
        if message.role == "assistant":
            item: dict[str, Any] = {
                "role": "assistant",
                "content": message.content if message.content is not None else None,
            }
            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ]
            messages.append(item)
            continue
        if message.role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.content,
                }
            )
            continue
        raise ProviderResponseError(f"unexpected message role in Groq history: {message.role!r}")
    return messages


def groq_response_to_message(response: Any) -> Message:
    """Convert a Groq chat.completions response into a canonical Message."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ProviderResponseError("Groq response has no choices")
    message = choices[0].message
    if message is None:
        raise ProviderResponseError("Groq response message is empty")

    text = getattr(message, "content", None)
    raw_calls = getattr(message, "tool_calls", None) or []
    tool_calls: list[ToolCall] = []
    for raw in raw_calls:
        function = getattr(raw, "function", None)
        if function is None:
            raise ProviderResponseError("Groq tool_call missing function")
        name = getattr(function, "name", None)
        call_id = getattr(raw, "id", None)
        arguments_text = getattr(function, "arguments", None) or "{}"
        if not name or not call_id:
            raise ProviderResponseError("Groq tool_call missing id or name")
        try:
            arguments = json.loads(arguments_text)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(
                f"invalid JSON in Groq function arguments: {arguments_text!r}"
            ) from exc
        if not isinstance(arguments, dict):
            raise ProviderResponseError("Groq function arguments must be an object")
        tool_calls.append(ToolCall(id=str(call_id), name=str(name), arguments=arguments))

    if (text is None or text == "") and not tool_calls:
        raise ProviderResponseError("Groq response contained neither text nor tool calls")

    return Message(
        role="assistant",
        content=text if text else None,
        tool_calls=tuple(tool_calls),
    )


class GroqModel:
    """ModelFn-compatible Groq client. Does not execute tools."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_GROQ_MODEL,
        client: Any | None = None,
    ) -> None:
        self.model = model
        key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY")
        if client is None:
            if not key:
                raise MissingAPIKeyError("GROQ_API_KEY is not set")
            from groq import Groq

            self._client = Groq(api_key=key)
        else:
            self._client = client

    def __call__(
        self,
        conversation: Conversation,
        declarations: list[dict[str, Any]],
    ) -> Message:
        messages = conversation_to_groq_messages(conversation)
        tools = groq_tools_from_declarations(declarations)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                timeout=DEFAULT_GROQ_TIMEOUT_S,
            )
        except MissingAPIKeyError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ProviderResponseError(f"Groq request failed: {exc}") from exc
        return groq_response_to_message(response)
