"""Per-run cooperative cancellation for the FourX agent harness.

Uses ``threading.Event`` and one process-local lock. This does **not** interrupt
an already-blocking provider SDK network call; the harness observes the signal
between steps (before/after model calls and before each tool). Pair with SDK
request timeouts so a hung call eventually returns and the stop can be honored.

Run ownership and durable writes share ``_active_lock`` so a superseded run
cannot pass an ownership check and then overwrite a newer SQLite snapshot.
**This lock is single-process only** — it does not protect concurrent processes
writing the same conversation database file.
"""

from __future__ import annotations

import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


class RunCancelled(Exception):
    """The active agent turn stopped because cancellation was requested."""


class StaleRunError(RuntimeError):
    """An older cancelled (or superseded) run tried to continue writing state."""


# conversation_id -> run_id of the currently authorized turn
_active_runs: dict[str, str] = {}
# Serializes begin_run / end_run / ownership-checked durable writes.
# Never hold across Gemini/Groq network calls.
_active_lock = threading.Lock()


@dataclass
class CancellationToken:
    """One-shot stop signal for a single ``run_agent_turn`` invocation."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _event: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        """Request cooperative stop. Idempotent; does not clear itself."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        """Raise ``RunCancelled`` if stop was requested."""
        if self.cancelled:
            raise RunCancelled("agent turn cancelled")


def begin_run(conversation_id: str, token: CancellationToken) -> None:
    """Mark ``token`` as the only run allowed to mutate this conversation.

    Blocks while another run holds ``run_write`` so a newly authorized run does
    not load or write until an in-flight protected SQLite transaction finishes.
    """
    with _active_lock:
        _active_runs[conversation_id] = token.run_id


def end_run(conversation_id: str, token: CancellationToken) -> None:
    """Clear the active run if ``token`` still owns it."""
    with _active_lock:
        if _active_runs.get(conversation_id) == token.run_id:
            del _active_runs[conversation_id]


def assert_run_active(conversation_id: str, token: CancellationToken) -> None:
    """Fail if a newer run superseded this one (soft pre-check; not a write fence)."""
    with _active_lock:
        current = _active_runs.get(conversation_id)
        if current != token.run_id:
            raise StaleRunError(
                f"run {token.run_id!r} is no longer active for conversation "
                f"{conversation_id!r} (active={current!r})"
            )


@contextmanager
def run_write(conversation_id: str, token: CancellationToken) -> Iterator[None]:
    """Hold ownership across verify + durable SQLite write.

    Enter only for short local DB work. Do **not** invoke provider SDKs inside
    this context — that would block ``begin_run`` for other turns.
    """
    with _active_lock:
        current = _active_runs.get(conversation_id)
        if current != token.run_id:
            raise StaleRunError(
                f"run {token.run_id!r} is no longer active for conversation "
                f"{conversation_id!r} (active={current!r})"
            )
        yield
