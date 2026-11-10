"""Google Gemini adapter using the official ``google-genai`` SDK (manual tools).

Converts canonical Conversation history ↔ Gemini Contents, declares tools with
FunctionDeclaration (no Python callables), and returns a canonical assistant
Message. Tool execution stays in ``harness.py``.

Thought signatures / native parts: when Gemini returns content parts, we store
``extras["gemini_content"]`` on the assistant Message so later Gemini turns can
round-trip signatures. History produced by Groq lacks that metadata; we rebuild
FunctionCall parts from canonical ToolCalls (documented limitation for Gemini 3
strict validators).
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from conversation import Conversation, Message, ToolCall

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
# google-genai HttpOptions.timeout is milliseconds; cooperative cancel cannot
# abort an in-flight call, so bound how long we wait for the SDK to return.
# Bound how long we wait for the SDK; does not interrupt an in-flight call early
# for cooperative /stop. 180s was tried during live diagnostics and did not fix
# provider DEADLINE_EXCEEDED / ReadTimeout; keep 60s for cancel UX.
DEFAULT_GEMINI_TIMEOUT_MS = 60_000


class MissingAPIKeyError(ValueError):
    """Raised when GEMINI_API_KEY is missing."""


class ProviderResponseError(ValueError):
    """Raised when the Gemini response cannot be converted."""


def gemini_tools_from_declarations(declarations: list[dict[str, Any]]) -> list[Any]:
    """Convert harness tool declarations into google-genai Tool objects."""
    from google.genai import types

    functions = []
    for decl in declarations:
        name = decl["name"]
        params = decl.get("parameters") or {"type": "object", "properties": {}}
        functions.append(
            types.FunctionDeclaration(
                name=name,
                description=decl.get("description") or "",
                parameters_json_schema=params,
            )
        )
    return [types.Tool(function_declarations=functions)]


def _tool_name_for_id(conversation: Conversation, call_id: str) -> str:
    for message in conversation.messages:
        if message.role != "assistant":
            continue
        for call in message.tool_calls:
            if call.id == call_id:
                return call.name
    raise ProviderResponseError(f"no assistant tool call found for id {call_id!r}")


def conversation_to_gemini_contents(conversation: Conversation) -> list[Any]:
    """Build Gemini ``contents`` from canonical history (no tool execution)."""
    from google.genai import types

    contents: list[Any] = []
    messages = conversation.messages
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role == "user":
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=message.content or "")],
                )
            )
            index += 1
            continue

        if message.role == "assistant":
            native = (message.extras or {}).get("gemini_content")
            if isinstance(native, dict):
                contents.append(types.Content.model_validate(native))
            else:
                parts: list[Any] = []
                if message.content:
                    parts.append(types.Part.from_text(text=message.content))
                for call in message.tool_calls:
                    parts.append(
                        types.Part.from_function_call(
                            name=call.name,
                            args=call.arguments,
                        )
                    )
                    # Attach id when the SDK Part supports it via FunctionCall.
                    if parts[-1].function_call is not None and call.id:
                        try:
                            parts[-1].function_call.id = call.id
                        except Exception:  # noqa: BLE001 — best-effort id round-trip
                            pass
                if not parts:
                    raise ProviderResponseError("assistant message produced no Gemini parts")
                contents.append(types.Content(role="model", parts=parts))
            index += 1

            tool_parts: list[Any] = []
            while index < len(messages) and messages[index].role == "tool":
                tool_message = messages[index]
                try:
                    payload = json.loads(tool_message.content or "")
                except json.JSONDecodeError:
                    payload = {"raw": tool_message.content}
                if not isinstance(payload, dict):
                    payload = {"result": payload}
                name = _tool_name_for_id(conversation, tool_message.tool_call_id or "")
                response_part = types.Part.from_function_response(
                    name=name,
                    response=payload,
                )
                if (
                    response_part.function_response is not None
                    and tool_message.tool_call_id
                ):
                    try:
                        response_part.function_response.id = tool_message.tool_call_id
                    except Exception:  # noqa: BLE001
                        pass
                tool_parts.append(response_part)
                index += 1
            if tool_parts:
                # gemini-3.8-flash rejects role="tool"; function responses use USER.
                contents.append(types.Content(role="user", parts=tool_parts))
            continue

        raise ProviderResponseError(f"unexpected message role in Gemini history: {message.role!r}")

    return contents


def _synthetic_tool_call_id() -> str:
    """Unique id for Gemini function calls that omit ``id``."""
    return f"gemini-{uuid.uuid4().hex}"


def gemini_response_to_message(response: Any) -> Message:
    """Convert a Gemini generate_content response into a canonical Message."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise ProviderResponseError("Gemini response has no candidates")
    content = candidates[0].content
    if content is None or not getattr(content, "parts", None):
        raise ProviderResponseError("Gemini response has empty content")

    text_chunks: list[str] = []
    tool_calls: list[ToolCall] = []
    for part in content.parts:
        function_call = getattr(part, "function_call", None)
        if function_call is not None and getattr(function_call, "name", None):
            raw_args = getattr(function_call, "args", None) or {}
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError as exc:
                    raise ProviderResponseError(
                        f"invalid JSON in Gemini function arguments: {raw_args!r}"
                    ) from exc
            if not isinstance(raw_args, dict):
                raise ProviderResponseError("Gemini function arguments must be an object")
            provided_id = getattr(function_call, "id", None)
            if provided_id:
                call_id = str(provided_id)
            else:
                # SDK FunctionCall.id is optional; assign locally so canonical + native match.
                call_id = _synthetic_tool_call_id()
                try:
                    function_call.id = call_id
                except Exception as exc:  # noqa: BLE001
                    raise ProviderResponseError(
                        "could not assign synthetic id on Gemini function_call"
                    ) from exc
            tool_calls.append(
                ToolCall(id=call_id, name=str(function_call.name), arguments=dict(raw_args))
            )
            continue
        text = getattr(part, "text", None)
        if text:
            text_chunks.append(text)

    if not text_chunks and not tool_calls:
        raise ProviderResponseError("Gemini response contained neither text nor tool calls")

    # Native content (including thought signatures) is required for later Gemini turns.
    try:
        extras = {
            "gemini_content": content.model_dump(mode="json", exclude_none=True),
        }
        json.dumps(extras)
    except Exception as exc:  # noqa: BLE001 — surface serialization failure
        raise ProviderResponseError(
            "failed to serialize Gemini native content for conversation extras"
        ) from exc

    return Message(
        role="assistant",
        content="\n".join(text_chunks) if text_chunks else None,
        tool_calls=tuple(tool_calls),
        extras=extras,
    )


class GeminiModel:
    """ModelFn-compatible Gemini client. Does not execute tools."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
        client: Any | None = None,
    ) -> None:
        self.model = model
        key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        if client is None:
            if not key:
                raise MissingAPIKeyError("GEMINI_API_KEY is not set")
            from google import genai

            self._client = genai.Client(api_key=key)
        else:
            self._client = client

    def __call__(
        self,
        conversation: Conversation,
        declarations: list[dict[str, Any]],
    ) -> Message:
        from google.genai import types

        contents = conversation_to_gemini_contents(conversation)
        tools = gemini_tools_from_declarations(declarations)
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=tools,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                    http_options=types.HttpOptions(timeout=DEFAULT_GEMINI_TIMEOUT_MS),
                ),
            )
        except MissingAPIKeyError:
            raise
        except Exception as exc:  # noqa: BLE001 — wrap SDK failures
            raise ProviderResponseError(f"Gemini request failed: {exc}") from exc
        return gemini_response_to_message(response)
