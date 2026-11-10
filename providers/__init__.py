"""Official provider adapters (Gemini / Groq) for the FourX harness."""

from __future__ import annotations

import os
from typing import Any

from conversation import Conversation, Message
from storage import ALLOWED_PROVIDERS, InvalidProvider

from providers.gemini import DEFAULT_GEMINI_MODEL, GeminiModel, gemini_tools_from_declarations
from providers.groq import DEFAULT_GROQ_MODEL, GroqModel, groq_tools_from_declarations

__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_GROQ_MODEL",
    "GeminiModel",
    "GroqModel",
    "gemini_tools_from_declarations",
    "groq_tools_from_declarations",
    "make_model",
]


def make_model(
    provider: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    client: Any | None = None,
) -> Any:
    """Return a ModelFn-compatible adapter for ``gemini`` or ``groq``.

    API keys default to ``GEMINI_API_KEY`` / ``GROQ_API_KEY``. Pass ``client``
    in tests to inject a mock SDK client without network access.
    """
    if provider not in ALLOWED_PROVIDERS:
        raise InvalidProvider(
            f"unsupported provider {provider!r}; expected one of {sorted(ALLOWED_PROVIDERS)}"
        )
    if provider == "gemini":
        key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        return GeminiModel(api_key=key, model=model or DEFAULT_GEMINI_MODEL, client=client)
    key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY")
    return GroqModel(api_key=key, model=model or DEFAULT_GROQ_MODEL, client=client)
