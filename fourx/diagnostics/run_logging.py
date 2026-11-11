"""Per-session diagnostic logging for FourX CLI (stdlib only).

Each CLI launch gets a unique ``run_id`` and an append-only JSONL file under
``logs/``. Events carry safe metadata only — never API keys, env dumps, or
full invoice / SDK payloads.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fourx.finance.approval import get_pending_approval, list_approvals
from fourx.persistence.ledger_guard import using_ledger
from fourx.persistence.storage import ConversationStore

# Event names required by the diagnostic design.
EVENT_NAMES = frozenset(
    {
        "session_started",
        "session_ended",
        "user_input",
        "model_call_started",
        "model_call_completed",
        "model_call_failed",
        "tool_requested",
        "tool_completed",
        "tool_failed",
        "approval_created",
        "approval_approved",
        "approval_declined",
        "posting_started",
        "posting_confirmed",
        "posting_uncertain",
        "provider_changed",
        "cancellation_requested",
        "cancellation_completed",
        "reconciliation_started",
        "reconciliation_completed",
        "state_snapshot",
        "reset_started",
        "reset_completed",
        "nl_intent",
        "authorization_rejected",
        "authorization_ambiguous",
        "stale_response_suppressed",
        "ledger_summary_executed",
        "state_transition",
    }
)

# Prefer allowlisting diagnostic fields over logging arbitrary blobs.
_ALLOWED_FIELD_KEYS = frozenset(
    {
        "approval_id",
        "after_reset",
        "conv_db",
        "error",
        "error_type",
        "has_tool_calls",
        "invoice_id",
        "journal_entry_id",
        "ledger_db",
        "message_count",
        "new_conv_db",
        "new_conversation_id",
        "new_ledger_db",
        "note",
        "ok",
        "old_conv_db",
        "old_conversation_id",
        "old_ledger_db",
        "old_run_id",
        "phase",
        "provider",
        "reason_present",
        "reason_len",
        "refused",
        "resolved",
        "snapshot",
        "source",
        "status",
        "tool_call_id",
        "tool_count",
        "tool_name",
        "unlisted_event",
        "user_text_len",
        "user_text_sha12",
        "user_text_redacted",
        "intent_category",
        "resolved_action",
        "state_transition",
        "entry_count",
        "matched_count",
        "candidates",
    }
)

_SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|authorization|bearer)\s*[=:]\s*\S+"
)
_KEY_LIKE_RE = re.compile(
    r"\b(AIza[0-9A-Za-z\-_]{10,}|gsk_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9]{16,}|"
    r"ghp_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})\b"
)
_ENV_ASSIGN_RE = re.compile(
    r"(?i)\b(GEMINI_API_KEY|GROQ_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY)\s*=\s*\S+"
)


def new_run_id() -> str:
    return uuid.uuid4().hex


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_text(value: Any, *, max_len: int = 240) -> str:
    """Redact credential-like substrings and truncate for safe logging."""
    text = "" if value is None else str(value)
    text = _ENV_ASSIGN_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
    text = _SECRET_RE.sub(r"\1=[REDACTED]", text)
    text = _KEY_LIKE_RE.sub("[REDACTED]", text)
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def sanitize_value(value: Any, *, depth: int = 0) -> Any:
    """Recursively sanitize nested dict/list structures for logging."""
    if depth > 6:
        return "[TRUNCATED_DEPTH]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, Path):
        return sanitize_text(str(value))
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_s = str(key)
            # Drop obviously sensitive containers entirely.
            if key_s.lower() in {
                "authorization",
                "headers",
                "env",
                "environ",
                "api_key",
                "apikey",
                "payload",
                "body",
                "request",
                "response",
                "invoice",
                "lines",
                "taxes",
                "content",
                "arguments",
                "raw",
            }:
                out[key_s] = "[REDACTED_FIELD]"
                continue
            out[key_s] = sanitize_value(item, depth=depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [sanitize_value(item, depth=depth + 1) for item in value[:50]]
    return sanitize_text(value)


def redact_user_text(text: str) -> dict[str, Any]:
    """Never store free-form user text verbatim — metadata summary only."""
    import hashlib

    raw = text if isinstance(text, str) else str(text)
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]
    return {
        "user_text_redacted": True,
        "user_text_len": len(raw),
        "user_text_sha12": digest,
    }


def _safe_chmod(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


class RunLogger:
    """Append-only JSONL logger bound to one CLI session / run_id."""

    def __init__(
        self,
        *,
        log_dir: str | Path,
        conversation_id: str,
        run_id: str | None = None,
        previous_run_id: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.conversation_id = conversation_id
        self.run_id = run_id or new_run_id()
        self.previous_run_id = previous_run_id
        self.log_dir = Path(log_dir)
        self._lock = threading.Lock()
        self.path: Path | None = None
        if enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.path = self.log_dir / f"run-{self.run_id}.jsonl"
            if not self.path.exists():
                self.path.touch()
                _safe_chmod(self.path)

    def emit(self, event: str, **fields: Any) -> dict[str, Any] | None:
        """Append one event with allowlisted, recursively sanitized metadata."""
        if not self.enabled or self.path is None:
            return None
        record: dict[str, Any] = {
            "timestamp": _utc_now(),
            "run_id": self.run_id,
            "event": event,
            "conversation_id": self.conversation_id,
        }
        if self.previous_run_id:
            record["previous_run_id"] = self.previous_run_id
        if event not in EVENT_NAMES:
            fields = {**fields, "unlisted_event": True}

        # Free-form user text → redacted summary only.
        if "user_text" in fields:
            record.update(redact_user_text(str(fields.pop("user_text"))))
        if "reason" in fields:
            reason = fields.pop("reason")
            record["reason_present"] = bool(reason)
            record["reason_len"] = len(str(reason)) if reason is not None else 0

        for key, value in fields.items():
            if value is None:
                continue
            if key not in _ALLOWED_FIELD_KEYS:
                continue
            record[key] = sanitize_value(value)

        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            _safe_chmod(self.path)
        return record

    def read_recent(self, limit: int = 30) -> list[dict[str, Any]]:
        if not self.enabled or self.path is None or not self.path.is_file():
            return []
        lines: list[str]
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        out: list[dict[str, Any]] = []
        for raw in lines[-max(1, limit) :]:
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                out.append(
                    {
                        "event": "parse_error",
                        "raw": sanitize_text(raw, max_len=80),
                    }
                )
        return out

    def collect_snapshot(
        self,
        store: ConversationStore,
        *,
        ledger_db: Path | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Read-only conversation / approval / ledger summary (no invoice bodies)."""
        snapshot: dict[str, Any] = {"note": note} if note else {}
        try:
            stored = store.load(self.conversation_id)
            snapshot["conversation"] = {
                "conversation_id": self.conversation_id,
                "provider": stored.provider,
                "message_count": len(stored.conversation.messages),
            }
        except Exception as exc:  # noqa: BLE001
            snapshot["conversation_error"] = sanitize_text(exc)

        try:
            history = list_approvals(store, self.conversation_id)
            counts = Counter(a.status for a in history)
            pending = get_pending_approval(store, self.conversation_id)
            reservations = 0
            for row in history:
                try:
                    if store.get_reservation_for_approval(row.approval_id) is not None:
                        reservations += 1
                except Exception:  # noqa: BLE001
                    continue
            snapshot["approvals"] = {
                "total": len(history),
                "by_status": dict(counts),
                "pending_approval_id": None if pending is None else pending.approval_id,
                "reservation_count": reservations,
            }
        except Exception as exc:  # noqa: BLE001
            snapshot["approvals_error"] = sanitize_text(exc)

        try:
            import ledger

            target = Path(ledger_db) if ledger_db is not None else None
            with using_ledger(target):
                entries = ledger.entries()
            snapshot["ledger"] = {
                "entry_count": len(entries),
                "entry_ids": [e.get("entry_id") for e in entries if isinstance(e, dict)],
            }
        except Exception as exc:  # noqa: BLE001
            snapshot["ledger_error"] = sanitize_text(exc)

        return sanitize_value(snapshot)

    def log_snapshot(
        self,
        store: ConversationStore,
        *,
        ledger_db: Path | None = None,
        note: str | None = None,
    ) -> dict[str, Any] | None:
        try:
            snapshot = self.collect_snapshot(store, ledger_db=ledger_db, note=note)
        except Exception as exc:  # noqa: BLE001
            return self.emit("state_snapshot", error=sanitize_text(exc), ok=False)
        return self.emit("state_snapshot", ok=True, snapshot=snapshot)


class NullRunLogger(RunLogger):
    """Disabled logger for tests that do not need diagnostics."""

    def __init__(self, conversation_id: str = "unused") -> None:
        super().__init__(
            log_dir=Path("."),
            conversation_id=conversation_id,
            enabled=False,
        )
