"""SQLite persistence for canonical conversations, provider, and approvals.

This module only stores and loads state. The future agent loop should call
``save`` after each meaningful step so a restart can resume mid-work:

- after receiving a user message
- after an assistant response (including tool calls)
- after each tool result
- after a final assistant text reply
- after a provider switch (``set_provider``)
- after creating or deciding a human approval

Uses a dedicated database path — never ``ledger.db``.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from conversation import Conversation

Provider = Literal["gemini", "groq"]
ALLOWED_PROVIDERS = frozenset({"gemini", "groq"})
ApprovalStatus = Literal[
    "pending",
    "approved",
    "declined",
    "posting",
    "posted",
    "posting_uncertain",
]
_ALLOWED_APPROVAL_STATUSES = frozenset(
    {
        "pending",
        "approved",
        "declined",
        "posting",
        "posted",
        "posting_uncertain",
    }
)
# Blocks a new pending request until confirmed posted or declined.
_UNRESOLVED_POSTING_STATUSES = frozenset({"approved", "posting", "posting_uncertain"})

_APPROVAL_SELECT = (
    "approval_id, conversation_id, invoice_id, details_json, status, "
    "decline_reason, journal_entry_id"
)


class ConversationNotFound(LookupError):
    """Raised when no row exists for the requested conversation ID."""


class ConversationExists(ValueError):
    """Raised when create() would overwrite an existing conversation."""


class CorruptConversationData(ValueError):
    """Raised when stored JSON cannot be reconstructed into a Conversation."""


class InvalidProvider(ValueError):
    """Raised when the provider name is not gemini or groq."""


class ApprovalPendingExists(ValueError):
    """Raised when a conversation already has a pending approval."""


class ApprovalUnprocessedExists(ValueError):
    """Raised when an unresolved approved/posting intent still blocks new requests."""


class ApprovalRowMissing(LookupError):
    """Raised when an approval decision targets a missing row."""


class ApprovalDecisionConflict(ValueError):
    """Raised when a decision cannot be applied (wrong status / duplicate)."""


class ApprovalPostingClaimConflict(ValueError):
    """Raised when approved → posting cannot be claimed (lost race or wrong status)."""


class InvoiceAlreadyReserved(ValueError):
    """Another approval already holds the durable (vendor_id, invoice_number) reservation."""


class CorruptApprovalData(ValueError):
    """Raised when stored approval JSON is invalid."""


@dataclass(frozen=True)
class PostingReservation:
    """Durable cross-conversation lock on invoice identifiers within one app DB."""

    vendor_id: str
    invoice_number: str
    approval_id: str
    conversation_id: str


@dataclass(frozen=True)
class StoredConversation:
    """Provider + history loaded from disk."""

    conversation_id: str
    provider: Provider
    conversation: Conversation


@dataclass(frozen=True)
class StoredApproval:
    """Durable approval row (authorization intent; journal_entry_id only when posted)."""

    approval_id: str
    conversation_id: str
    invoice_id: str
    details: dict[str, Any]
    status: ApprovalStatus
    decline_reason: str | None = None
    journal_entry_id: str | None = None


def _require_conversation_id(conversation_id: str) -> str:
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ValueError("conversation_id must be a non-empty string")
    return conversation_id


def _require_provider(provider: str) -> Provider:
    if provider not in ALLOWED_PROVIDERS:
        raise InvalidProvider(
            f"unsupported provider {provider!r}; expected one of {sorted(ALLOWED_PROVIDERS)}"
        )
    return provider  # type: ignore[return-value]


def _new_approval_id() -> str:
    return f"apr-{uuid.uuid4().hex}"


def _approval_from_row(row: sqlite3.Row) -> StoredApproval:
    try:
        details = json.loads(row["details_json"])
    except (json.JSONDecodeError, TypeError) as exc:
        raise CorruptApprovalData(
            f"corrupt approval details for {row['approval_id']!r}: {exc}"
        ) from exc
    if not isinstance(details, dict):
        raise CorruptApprovalData(
            f"approval details must be an object for {row['approval_id']!r}"
        )
    status = row["status"]
    if status not in _ALLOWED_APPROVAL_STATUSES:
        raise CorruptApprovalData(
            f"invalid approval status {status!r} for {row['approval_id']!r}"
        )
    return StoredApproval(
        approval_id=row["approval_id"],
        conversation_id=row["conversation_id"],
        invoice_id=row["invoice_id"],
        details=details,
        status=status,  # type: ignore[arg-type]
        decline_reason=row["decline_reason"],
        journal_entry_id=row["journal_entry_id"],
    )


class ConversationStore:
    """One SQLite file holding many named conversations and approvals."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    messages_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    invoice_id TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    decline_reason TEXT,
                    journal_entry_id TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                )
                """
            )
            # Older conversation DBs created before Iteration 9 lack this column.
            columns = {
                info[1] for info in conn.execute("PRAGMA table_info(approvals)").fetchall()
            }
            if "journal_entry_id" not in columns:
                conn.execute("ALTER TABLE approvals ADD COLUMN journal_entry_id TEXT")
            # At most one pending approval per conversation.
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_approvals_one_pending
                ON approvals(conversation_id)
                WHERE status = 'pending'
                """
            )
            # One posting attempt per vendor_id+invoice_number across conversations
            # that share this application database. Survives restart; not deleted
            # after post so a second approval cannot reclaim the same invoice.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS posting_reservations (
                    vendor_id TEXT NOT NULL,
                    invoice_number TEXT NOT NULL,
                    approval_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    PRIMARY KEY (vendor_id, invoice_number),
                    FOREIGN KEY (approval_id) REFERENCES approvals(approval_id)
                )
                """
            )

    def create(self, conversation_id: str, provider: str) -> StoredConversation:
        """Create an empty conversation. Fails if the ID already exists."""
        conversation_id = _require_conversation_id(conversation_id)
        provider = _require_provider(provider)
        empty = Conversation()
        payload = json.dumps(empty.to_jsonable())
        try:
            with closing(self._connect()) as conn, conn:
                conn.execute(
                    "INSERT INTO conversations (conversation_id, provider, messages_json) "
                    "VALUES (?, ?, ?)",
                    (conversation_id, provider, payload),
                )
        except sqlite3.IntegrityError as exc:
            raise ConversationExists(
                f"conversation already exists: {conversation_id!r}"
            ) from exc
        return StoredConversation(conversation_id, provider, empty)

    def load(self, conversation_id: str) -> StoredConversation:
        """Load provider and history. Missing vs corrupt are distinct errors."""
        conversation_id = _require_conversation_id(conversation_id)
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT conversation_id, provider, messages_json "
                "FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            raise ConversationNotFound(f"conversation not found: {conversation_id!r}")

        provider = _require_provider(row["provider"])
        try:
            raw = json.loads(row["messages_json"])
            conversation = Conversation.from_jsonable(raw)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise CorruptConversationData(
                f"corrupt conversation data for {conversation_id!r}: {exc}"
            ) from exc
        return StoredConversation(conversation_id, provider, conversation)

    def save(self, conversation_id: str, conversation: Conversation) -> None:
        """Persist the latest history. Does not create a missing conversation."""
        conversation_id = _require_conversation_id(conversation_id)
        if not isinstance(conversation, Conversation):
            raise TypeError("conversation must be a Conversation instance")
        payload = json.dumps(conversation.to_jsonable())
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "UPDATE conversations SET messages_json = ? WHERE conversation_id = ?",
                (payload, conversation_id),
            )
            if cursor.rowcount == 0:
                raise ConversationNotFound(f"conversation not found: {conversation_id!r}")

    def set_provider(self, conversation_id: str, provider: str) -> None:
        """Persist the selected provider so it survives process restart."""
        conversation_id = _require_conversation_id(conversation_id)
        provider = _require_provider(provider)
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "UPDATE conversations SET provider = ? WHERE conversation_id = ?",
                (provider, conversation_id),
            )
            if cursor.rowcount == 0:
                raise ConversationNotFound(f"conversation not found: {conversation_id!r}")

    def get_approval_by_id(self, approval_id: str) -> StoredApproval | None:
        if not isinstance(approval_id, str) or not approval_id.strip():
            raise ValueError("approval_id must be a non-empty string")
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"SELECT {_APPROVAL_SELECT} FROM approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
        if row is None:
            return None
        return _approval_from_row(row)

    def list_approvals(self, conversation_id: str) -> list[StoredApproval]:
        """Return all approval rows for a conversation (oldest first)."""
        conversation_id = _require_conversation_id(conversation_id)
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT {_APPROVAL_SELECT} FROM approvals WHERE conversation_id = ? "
                "ORDER BY rowid ASC",
                (conversation_id,),
            ).fetchall()
        return [_approval_from_row(row) for row in rows]

    def get_pending_approval(self, conversation_id: str) -> StoredApproval | None:
        """Return the single pending approval for a conversation, if any."""
        conversation_id = _require_conversation_id(conversation_id)
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"SELECT {_APPROVAL_SELECT} FROM approvals "
                "WHERE conversation_id = ? AND status = 'pending'",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        return _approval_from_row(row)

    def list_unprocessed_approved(self, conversation_id: str) -> list[StoredApproval]:
        """Unresolved authorized intents (approved / posting / posting_uncertain)."""
        conversation_id = _require_conversation_id(conversation_id)
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT {_APPROVAL_SELECT} FROM approvals "
                "WHERE conversation_id = ? AND status IN "
                "('approved', 'posting', 'posting_uncertain') "
                "ORDER BY rowid ASC",
                (conversation_id,),
            ).fetchall()
        return [_approval_from_row(row) for row in rows]

    def _validate_details(self, details: dict[str, Any]) -> tuple[str, str]:
        if not isinstance(details, dict):
            raise TypeError("approval details must be a dict")
        invoice_id = details.get("invoice_id")
        if not isinstance(invoice_id, str) or not invoice_id:
            raise ValueError("approval details require a non-empty invoice_id")
        try:
            details_json = json.dumps(details)
        except (TypeError, ValueError) as exc:
            raise ValueError("approval details must be JSON-compatible") from exc
        return invoice_id, details_json

    def _assert_can_create_pending(self, conn: sqlite3.Connection, conversation_id: str) -> None:
        exists = conn.execute(
            "SELECT 1 FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if exists is None:
            raise ConversationNotFound(f"conversation not found: {conversation_id!r}")

        pending = conn.execute(
            "SELECT approval_id FROM approvals "
            "WHERE conversation_id = ? AND status = 'pending'",
            (conversation_id,),
        ).fetchone()
        if pending is not None:
            raise ApprovalPendingExists(
                f"pending approval already exists for {conversation_id!r}"
            )

        unresolved = conn.execute(
            "SELECT approval_id, status FROM approvals "
            "WHERE conversation_id = ? AND status IN "
            "('approved', 'posting', 'posting_uncertain') LIMIT 1",
            (conversation_id,),
        ).fetchone()
        if unresolved is not None:
            raise ApprovalUnprocessedExists(
                f"conversation {conversation_id!r} has unresolved request "
                f"{unresolved['approval_id']!r} (status={unresolved['status']!r})"
            )

    def create_pending_approval(
        self, conversation_id: str, details: dict[str, Any]
    ) -> StoredApproval:
        """Insert a new pending approval row. Never overwrites prior decisions."""
        conversation_id = _require_conversation_id(conversation_id)
        invoice_id, details_json = self._validate_details(details)
        approval_id = _new_approval_id()

        with closing(self._connect()) as conn, conn:
            self._assert_can_create_pending(conn, conversation_id)
            try:
                conn.execute(
                    """
                    INSERT INTO approvals (
                        approval_id, conversation_id, invoice_id, details_json,
                        status, decline_reason
                    ) VALUES (?, ?, ?, ?, 'pending', NULL)
                    """,
                    (approval_id, conversation_id, invoice_id, details_json),
                )
            except sqlite3.IntegrityError as exc:
                raise ApprovalPendingExists(
                    f"pending approval already exists for {conversation_id!r}"
                ) from exc

        stored = self.get_approval_by_id(approval_id)
        if stored is None or stored.status != "pending":
            raise RuntimeError(f"failed to persist pending approval {approval_id!r}")
        return stored

    def commit_pending_approval_with_conversation(
        self,
        conversation_id: str,
        details: dict[str, Any],
        conversation: Conversation,
    ) -> StoredApproval:
        """Atomically insert a pending approval and persist conversation history.

        If this transaction fails, neither the new approval row nor the updated
        messages_json is committed.
        """
        conversation_id = _require_conversation_id(conversation_id)
        if not isinstance(conversation, Conversation):
            raise TypeError("conversation must be a Conversation instance")
        invoice_id, details_json = self._validate_details(details)
        approval_id = _new_approval_id()
        payload = json.dumps(conversation.to_jsonable())

        with closing(self._connect()) as conn, conn:
            self._assert_can_create_pending(conn, conversation_id)
            try:
                conn.execute(
                    """
                    INSERT INTO approvals (
                        approval_id, conversation_id, invoice_id, details_json,
                        status, decline_reason
                    ) VALUES (?, ?, ?, ?, 'pending', NULL)
                    """,
                    (approval_id, conversation_id, invoice_id, details_json),
                )
            except sqlite3.IntegrityError as exc:
                raise ApprovalPendingExists(
                    f"pending approval already exists for {conversation_id!r}"
                ) from exc
            cursor = conn.execute(
                "UPDATE conversations SET messages_json = ? WHERE conversation_id = ?",
                (payload, conversation_id),
            )
            if cursor.rowcount == 0:
                raise ConversationNotFound(f"conversation not found: {conversation_id!r}")

        stored = self.get_approval_by_id(approval_id)
        if stored is None or stored.status != "pending":
            raise RuntimeError(f"failed to persist pending approval {approval_id!r}")
        return stored

    def set_approval_decision(
        self,
        approval_id: str,
        *,
        status: ApprovalStatus,
        decline_reason: str | None = None,
    ) -> StoredApproval:
        """Apply approve/decline once to a specific pending approval_id."""
        if not isinstance(approval_id, str) or not approval_id.strip():
            raise ValueError("approval_id must be a non-empty string")
        if status not in {"approved", "declined"}:
            raise ValueError(f"decision status must be approved or declined, got {status!r}")
        if status == "declined":
            if not isinstance(decline_reason, str) or not decline_reason.strip():
                raise ValueError("declined approvals require a non-empty decline_reason")
            decline_reason = decline_reason.strip()
        else:
            decline_reason = None

        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT status FROM approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            if row is None:
                raise ApprovalRowMissing(f"no approval with id {approval_id!r}")
            if row["status"] != "pending":
                raise ApprovalDecisionConflict(
                    f"approval {approval_id!r} is already {row['status']}"
                )
            cursor = conn.execute(
                """
                UPDATE approvals
                SET status = ?, decline_reason = ?
                WHERE approval_id = ? AND status = 'pending'
                """,
                (status, decline_reason, approval_id),
            )
            if cursor.rowcount != 1:
                raise ApprovalDecisionConflict(
                    f"could not apply {status} decision for {approval_id!r}"
                )

        stored = self.get_approval_by_id(approval_id)
        if stored is None or stored.status != status:
            raise RuntimeError(f"failed to persist {status} decision for {approval_id!r}")
        return stored

    def get_posting_reservation(
        self, *, vendor_id: str, invoice_number: str
    ) -> PostingReservation | None:
        """Return the durable reservation for these invoice identifiers, if any."""
        if not isinstance(vendor_id, str) or not vendor_id.strip():
            raise ValueError("vendor_id must be a non-empty string")
        if not isinstance(invoice_number, str) or not invoice_number.strip():
            raise ValueError("invoice_number must be a non-empty string")
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT vendor_id, invoice_number, approval_id, conversation_id
                FROM posting_reservations
                WHERE vendor_id = ? AND invoice_number = ?
                """,
                (vendor_id, invoice_number),
            ).fetchone()
        if row is None:
            return None
        return PostingReservation(
            vendor_id=row["vendor_id"],
            invoice_number=row["invoice_number"],
            approval_id=row["approval_id"],
            conversation_id=row["conversation_id"],
        )

    def get_reservation_for_approval(
        self, approval_id: str
    ) -> PostingReservation | None:
        if not isinstance(approval_id, str) or not approval_id.strip():
            raise ValueError("approval_id must be a non-empty string")
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT vendor_id, invoice_number, approval_id, conversation_id
                FROM posting_reservations
                WHERE approval_id = ?
                """,
                (approval_id,),
            ).fetchone()
        if row is None:
            return None
        return PostingReservation(
            vendor_id=row["vendor_id"],
            invoice_number=row["invoice_number"],
            approval_id=row["approval_id"],
            conversation_id=row["conversation_id"],
        )

    def claim_for_posting(
        self, approval_id: str, conversation_id: str
    ) -> StoredApproval:
        """Atomically claim approved → posting and reserve vendor_id+invoice_number.

        The approval transition and ``posting_reservations`` insert share one
        SQLite transaction. A UNIQUE conflict on the invoice identifiers rolls
        back both changes so the loser never reaches ``ledger.post``.

        Reservations are not deleted after ``posted`` / ``posting_uncertain`` so
        another approval in this application DB cannot reclaim the same invoice.
        This coordinates processes sharing this DB only — not independent app
        databases, and not a ledger-level uniqueness guarantee.
        """
        if not isinstance(approval_id, str) or not approval_id.strip():
            raise ValueError("approval_id must be a non-empty string")
        conversation_id = _require_conversation_id(conversation_id)

        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                f"SELECT {_APPROVAL_SELECT} FROM approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            if row is None:
                raise ApprovalRowMissing(f"no approval with id {approval_id!r}")
            if row["conversation_id"] != conversation_id:
                raise ApprovalDecisionConflict(
                    f"approval {approval_id!r} belongs to conversation "
                    f"{row['conversation_id']!r}, not {conversation_id!r}"
                )
            if row["status"] != "approved":
                raise ApprovalPostingClaimConflict(
                    f"approval {approval_id!r} cannot be claimed for posting "
                    f"(status={row['status']!r})"
                )
            try:
                details = json.loads(row["details_json"])
            except (json.JSONDecodeError, TypeError) as exc:
                raise CorruptApprovalData(
                    f"corrupt approval details for {approval_id!r}: {exc}"
                ) from exc
            vendor_id = details.get("vendor_id") if isinstance(details, dict) else None
            invoice_number = (
                details.get("invoice_number") if isinstance(details, dict) else None
            )
            if not isinstance(vendor_id, str) or not vendor_id.strip():
                raise ApprovalDecisionConflict(
                    f"approval {approval_id!r} details lack vendor_id"
                )
            if not isinstance(invoice_number, str) or not invoice_number.strip():
                raise ApprovalDecisionConflict(
                    f"approval {approval_id!r} details lack invoice_number"
                )

            existing = conn.execute(
                """
                SELECT approval_id, conversation_id FROM posting_reservations
                WHERE vendor_id = ? AND invoice_number = ?
                """,
                (vendor_id, invoice_number),
            ).fetchone()
            if existing is not None:
                if existing["approval_id"] == approval_id:
                    # Same approval already reserved (should not be 'approved').
                    raise ApprovalPostingClaimConflict(
                        f"approval {approval_id!r} already holds the reservation"
                    )
                raise InvoiceAlreadyReserved(
                    f"invoice {invoice_number!r} for {vendor_id} is reserved by "
                    f"approval {existing['approval_id']!r} "
                    f"(conversation {existing['conversation_id']!r})"
                )

            cursor = conn.execute(
                """
                UPDATE approvals
                SET status = 'posting'
                WHERE approval_id = ? AND conversation_id = ? AND status = 'approved'
                """,
                (approval_id, conversation_id),
            )
            if cursor.rowcount != 1:
                raise ApprovalPostingClaimConflict(
                    f"could not claim approval {approval_id!r} for posting"
                )
            try:
                conn.execute(
                    """
                    INSERT INTO posting_reservations (
                        vendor_id, invoice_number, approval_id, conversation_id
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (vendor_id, invoice_number, approval_id, conversation_id),
                )
            except sqlite3.IntegrityError as exc:
                # UNIQUE(vendor_id, invoice_number) lost the race — roll back
                # the approved→posting update with this transaction.
                raise InvoiceAlreadyReserved(
                    f"invoice {invoice_number!r} for {vendor_id} was reserved "
                    f"by another approval"
                ) from exc

        stored = self.get_approval_by_id(approval_id)
        if stored is None or stored.status != "posting":
            raise RuntimeError(f"failed to persist posting claim for {approval_id!r}")
        reservation = self.get_reservation_for_approval(approval_id)
        if reservation is None:
            raise RuntimeError(f"failed to persist reservation for {approval_id!r}")
        return stored

    def mark_approval_posted(
        self, approval_id: str, journal_entry_id: str
    ) -> StoredApproval:
        """Record a verified journal entry: posting/posting_uncertain → posted."""
        if not isinstance(approval_id, str) or not approval_id.strip():
            raise ValueError("approval_id must be a non-empty string")
        if not isinstance(journal_entry_id, str) or not journal_entry_id.strip():
            raise ValueError("journal_entry_id must be a non-empty string")
        journal_entry_id = journal_entry_id.strip()

        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT status, journal_entry_id FROM approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            if row is None:
                raise ApprovalRowMissing(f"no approval with id {approval_id!r}")
            if row["status"] == "posted":
                if row["journal_entry_id"] == journal_entry_id:
                    return self.get_approval_by_id(approval_id)  # type: ignore[return-value]
                raise ApprovalDecisionConflict(
                    f"approval {approval_id!r} already posted as "
                    f"{row['journal_entry_id']!r}"
                )
            if row["status"] not in {"posting", "posting_uncertain"}:
                raise ApprovalDecisionConflict(
                    f"approval {approval_id!r} cannot be marked posted "
                    f"(status={row['status']!r})"
                )
            cursor = conn.execute(
                """
                UPDATE approvals
                SET status = 'posted', journal_entry_id = ?
                WHERE approval_id = ? AND status IN ('posting', 'posting_uncertain')
                """,
                (journal_entry_id, approval_id),
            )
            if cursor.rowcount != 1:
                raise ApprovalDecisionConflict(
                    f"could not mark approval {approval_id!r} as posted"
                )

        stored = self.get_approval_by_id(approval_id)
        if stored is None or stored.status != "posted":
            raise RuntimeError(f"failed to persist posted status for {approval_id!r}")
        return stored

    def mark_approval_posting_uncertain(self, approval_id: str) -> StoredApproval:
        """Mark a posting attempt as uncertain (no automatic repost)."""
        if not isinstance(approval_id, str) or not approval_id.strip():
            raise ValueError("approval_id must be a non-empty string")

        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT status FROM approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            if row is None:
                raise ApprovalRowMissing(f"no approval with id {approval_id!r}")
            if row["status"] == "posting_uncertain":
                return self.get_approval_by_id(approval_id)  # type: ignore[return-value]
            if row["status"] == "posted":
                raise ApprovalDecisionConflict(
                    f"approval {approval_id!r} is already posted"
                )
            if row["status"] != "posting":
                raise ApprovalDecisionConflict(
                    f"approval {approval_id!r} cannot become posting_uncertain "
                    f"(status={row['status']!r})"
                )
            cursor = conn.execute(
                """
                UPDATE approvals
                SET status = 'posting_uncertain'
                WHERE approval_id = ? AND status = 'posting'
                """,
                (approval_id,),
            )
            if cursor.rowcount != 1:
                raise ApprovalDecisionConflict(
                    f"could not mark approval {approval_id!r} as posting_uncertain"
                )

        stored = self.get_approval_by_id(approval_id)
        if stored is None or stored.status != "posting_uncertain":
            raise RuntimeError(
                f"failed to persist posting_uncertain for {approval_id!r}"
            )
        return stored
