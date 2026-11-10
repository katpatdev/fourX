"""Explicit approved-invoice posting with durable attempt + ledger reconciliation.

Only ``post_approved`` may call ``ledger.post``. Human ``approve`` / ``decline``
and model ``post_invoice`` never post.

Two separate ledger checks:

- **Identifier block (fresh post):** any ``ledger.entries`` row for the same
  ``vendor_id`` + ``invoice_number`` refuses a new ``ledger.post``.
- **Full-detail match (reconcile):** date, currency, and Decimal amount must
  match before marking ``posted``. Optional ``reference=approval_id`` strengthens
  ownership when present; missing references are not invented.

Cross-conversation coordination uses a durable ``posting_reservations`` row in
the application SQLite DB, inserted in the same transaction as
``approved → posting``. That coordinates processes sharing one app database
only. Independent installations with separate DBs still cannot get global
exactly-once posting from the original ledger (no uniqueness there).
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

import ledger
from cancellation import CancellationToken, RunCancelled
from invoice_tools import BlockedVendor, DuplicateInvoice
from storage import (
    ApprovalDecisionConflict,
    ApprovalPostingClaimConflict,
    ApprovalRowMissing,
    ConversationStore,
    InvoiceAlreadyReserved,
    StoredApproval,
)

# Fields that must still match the human-approved snapshot before posting.
_DETAIL_KEYS = (
    "invoice_id",
    "vendor_id",
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "currency",
    "amount",
)


class PostingError(ValueError):
    """Base class for refused posting attempts (no ledger side effect intended)."""


class ApprovalNotPostable(PostingError):
    """Approval is missing, wrong conversation, or not in an actionable state."""


class PostingDetailsMismatch(PostingError):
    """Current invoice/eligibility details no longer match the approved snapshot."""


class LedgerInvoiceConflict(PostingError):
    """Ledger already has vendor_id+invoice_number with conflicting or ambiguous rows."""


class InvoiceReservedByOther(PostingError):
    """Another approval in this application DB already reserved the invoice."""


class PostingUncertain(RuntimeError):
    """A posting attempt exists but the ledger outcome cannot be verified.

    Automatic ``ledger.post`` retries are forbidden. Call ``reconcile_posting``
    again after manual investigation; never invent a journal id.
    """

    def __init__(self, approval: StoredApproval, message: str) -> None:
        super().__init__(message)
        self.approval = approval


class LedgerAmbiguity(PostingUncertain):
    """Multiple ledger rows fully match the approved invoice details."""


def _money_equal(left: Any, right: Any) -> bool:
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except (InvalidOperation, TypeError, ValueError):
        return False


def _details_match(approved: dict[str, Any], prepared: dict[str, Any]) -> bool:
    for key in _DETAIL_KEYS:
        if key not in approved or key not in prepared:
            return False
        if key == "amount":
            if not _money_equal(approved[key], prepared[key]):
                return False
        elif approved[key] != prepared[key]:
            return False
    return True


def _entry_matches_details(entry: dict[str, Any], details: dict[str, Any]) -> bool:
    """Verify a ledger row represents the approved invoice (exact money-safe)."""
    try:
        if entry.get("vendor_id") != details["vendor_id"]:
            return False
        if entry.get("invoice_number") != details["invoice_number"]:
            return False
        if entry.get("invoice_date") != details["invoice_date"]:
            return False
        if entry.get("currency") != details["currency"]:
            return False
        if not _money_equal(entry.get("amount"), details["amount"]):
            return False
    except KeyError:
        return False
    return isinstance(entry.get("entry_id"), str) and bool(entry["entry_id"])


def list_identifier_ledger_entries(details: dict[str, Any]) -> list[dict[str, Any]]:
    """All ledger rows for vendor_id + invoice_number (no detail filter)."""
    from ledger_guard import ledger_lock

    with ledger_lock():
        return ledger.entries(
            vendor_id=details["vendor_id"],
            invoice_number=details["invoice_number"],
        )


def find_matching_ledger_entries(details: dict[str, Any]) -> list[dict[str, Any]]:
    """Full-detail matches for reconciliation (date, currency, Decimal amount)."""
    rows = list_identifier_ledger_entries(details)
    return [row for row in rows if _entry_matches_details(row, details)]


def find_reconcile_matches(
    details: dict[str, Any], approval_id: str
) -> list[dict[str, Any]]:
    """Full-detail matches owned by this approval when reference is present.

    Entries with ``reference`` equal to ``approval_id`` win. Entries with a
    different non-empty reference are never claimed. Entries with missing
    reference remain eligible for a single full-detail match (historical /
    opening-ledger style rows) so legitimate reconcile still works.
    """
    full = find_matching_ledger_entries(details)
    owned = [row for row in full if row.get("reference") == approval_id]
    if owned:
        return owned
    foreign = [
        row
        for row in full
        if isinstance(row.get("reference"), str)
        and row["reference"].strip()
        and row["reference"] != approval_id
    ]
    return [row for row in full if row not in foreign]


def refuse_fresh_post_for_existing_identifiers(details: dict[str, Any]) -> None:
    """Block a new ledger.post if any row already uses these invoice identifiers.

    Exact full-detail match → ``DuplicateInvoice``.
    Conflicting date/currency/amount → ``LedgerInvoiceConflict``.
    Multiple identifier rows → ``LedgerInvoiceConflict`` (ambiguity).
    Lookup errors propagate (fail closed).
    """
    try:
        rows = list_identifier_ledger_entries(details)
    except Exception as exc:
        raise PostingError(
            f"ledger lookup failed while checking identifiers for "
            f"{details.get('invoice_number')!r}: {exc}"
        ) from exc

    if not rows:
        return
    if len(rows) > 1:
        raise LedgerInvoiceConflict(
            f"ambiguous ledger rows for {details['vendor_id']} / "
            f"{details['invoice_number']}: {[r.get('entry_id') for r in rows]}; "
            f"refusing fresh post"
        )
    entry = rows[0]
    if _entry_matches_details(entry, details):
        raise DuplicateInvoice(
            f"invoice already present in ledger as "
            f"{entry.get('entry_id', 'unknown')}; refusing to post"
        )
    raise LedgerInvoiceConflict(
        f"ledger already has {details['vendor_id']} / {details['invoice_number']} "
        f"as {entry.get('entry_id', 'unknown')} with conflicting details "
        f"(date={entry.get('invoice_date')!r}, currency={entry.get('currency')!r}, "
        f"amount={entry.get('amount')!r}); refusing fresh post"
    )


def _load_for_conversation(
    store: ConversationStore, conversation_id: str, approval_id: str
) -> StoredApproval:
    row = store.get_approval_by_id(approval_id)
    if row is None:
        raise ApprovalNotPostable(f"unknown approval_id {approval_id!r}")
    if row.conversation_id != conversation_id:
        raise ApprovalNotPostable(
            f"approval {approval_id!r} belongs to conversation "
            f"{row.conversation_id!r}, not {conversation_id!r}"
        )
    return row


def _recheck_prepared_details(row: StoredApproval) -> dict[str, Any]:
    """Rebuild eligibility from fixtures; require exact match to approved snapshot."""
    from invoice_tools import (
        lookup_vendor,
        read_invoice,
        validate_posting_amount,
    )

    invoice = read_invoice(row.invoice_id)
    vendor_name = invoice.get("vendor_name")
    if not isinstance(vendor_name, str) or not vendor_name:
        raise PostingDetailsMismatch(
            f"invoice {row.invoice_id!r} has no vendor_name"
        )
    vendor = lookup_vendor(name=vendor_name)
    status = vendor.get("status")
    if status == "blocked":
        reason = vendor.get("blocked_reason") or "no reason given"
        raise BlockedVendor(
            f"vendor {vendor['vendor_id']} ({vendor['name']}) is blocked: {reason}"
        )
    if status != "active":
        raise BlockedVendor(
            f"vendor {vendor['vendor_id']} ({vendor['name']}) is not eligible "
            f"(status={status!r}; only 'active' vendors may proceed)"
        )
    amount = validate_posting_amount(invoice.get("total"))
    prepared = {
        "invoice_id": row.invoice_id,
        "vendor_id": vendor["vendor_id"],
        "vendor_name": vendor["name"],
        "invoice_number": invoice["invoice_number"],
        "invoice_date": invoice["invoice_date"],
        "currency": invoice["currency"],
        "amount": amount,
    }
    if not _details_match(row.details, prepared):
        raise PostingDetailsMismatch(
            f"approved details for {row.approval_id!r} no longer match "
            f"current invoice/eligibility data"
        )
    return prepared


def reconcile_posting(
    store: ConversationStore,
    conversation_id: str,
    approval_id: str,
) -> StoredApproval:
    """Resolve a ``posting`` / ``posting_uncertain`` row from ledger evidence only.

    Never calls ``ledger.post``. Reservation for the owning approval is left
    intact so competitors cannot reclaim the invoice identifiers.
    """
    row = _load_for_conversation(store, conversation_id, approval_id)
    if row.status == "posted":
        return row
    if row.status not in {"posting", "posting_uncertain"}:
        raise ApprovalNotPostable(
            f"approval {approval_id!r} is not awaiting reconciliation "
            f"(status={row.status!r})"
        )

    try:
        prepared = _recheck_prepared_details(row)
        matches = find_reconcile_matches(prepared, approval_id)
    except BlockedVendor:
        if row.status == "posting":
            row = store.mark_approval_posting_uncertain(approval_id)
        raise PostingUncertain(
            row,
            f"vendor no longer eligible while reconciling {approval_id!r}; "
            f"manual ledger inspection required",
        )
    except Exception as exc:
        if row.status == "posting":
            try:
                row = store.mark_approval_posting_uncertain(approval_id)
            except ApprovalDecisionConflict:
                row = _load_for_conversation(store, conversation_id, approval_id)
        raise PostingUncertain(
            row,
            f"ledger/eligibility lookup failed while reconciling {approval_id!r}: {exc}",
        ) from exc

    if len(matches) > 1:
        if row.status == "posting":
            row = store.mark_approval_posting_uncertain(approval_id)
        raise LedgerAmbiguity(
            row,
            f"ambiguous ledger matches for {approval_id!r}: "
            f"{[m.get('entry_id') for m in matches]}",
        )
    if len(matches) == 1:
        return store.mark_approval_posted(approval_id, matches[0]["entry_id"])

    if row.status == "posting":
        row = store.mark_approval_posting_uncertain(approval_id)
    raise PostingUncertain(
        row,
        f"no verified ledger entry for {approval_id!r}; automatic repost forbidden",
    )


def post_approved(
    store: ConversationStore,
    conversation_id: str,
    approval_id: str,
    *,
    cancel_token: CancellationToken | None = None,
) -> StoredApproval:
    """Post one previously approved request after durable attempt recording.

    Steps:
    1. Load exact approval_id for this conversation.
    2. If already ``posted``, return (idempotent; no second ledger call).
    3. If ``posting`` / ``posting_uncertain``, reconcile only (never repost).
    4. Require ``approved``; re-check eligibility and detail match.
    5. Refuse if any ledger row already uses these invoice identifiers.
    6. Atomically claim ``approved`` → ``posting`` and insert reservation.
    7. Call ``ledger.post`` once.
    8. Reconcile and mark ``posted`` with the verified journal id.
    """
    if cancel_token is not None:
        cancel_token.check()

    row = _load_for_conversation(store, conversation_id, approval_id)

    if row.status == "posted":
        return row

    if row.status in {"posting", "posting_uncertain"}:
        return reconcile_posting(store, conversation_id, approval_id)

    if row.status == "pending":
        raise ApprovalNotPostable(
            f"approval {approval_id!r} is still pending; human approve() first"
        )
    if row.status == "declined":
        raise ApprovalNotPostable(f"approval {approval_id!r} was declined")
    if row.status != "approved":
        raise ApprovalNotPostable(
            f"approval {approval_id!r} cannot be posted (status={row.status!r})"
        )

    try:
        prepared = _recheck_prepared_details(row)
    except Exception as exc:
        if isinstance(exc, (BlockedVendor, PostingDetailsMismatch, PostingError)):
            raise
        raise PostingError(
            f"eligibility lookup failed for {approval_id!r}: {exc}"
        ) from exc

    # Identifier-level block: any existing vendor_id+invoice_number refuses post.
    refuse_fresh_post_for_existing_identifiers(prepared)

    if cancel_token is not None:
        cancel_token.check()

    try:
        store.claim_for_posting(approval_id, conversation_id)
    except InvoiceAlreadyReserved as exc:
        raise InvoiceReservedByOther(str(exc)) from exc
    except ApprovalPostingClaimConflict as exc:
        current = _load_for_conversation(store, conversation_id, approval_id)
        if current.status == "posted":
            return current
        if current.status in {"posting", "posting_uncertain"}:
            return reconcile_posting(store, conversation_id, approval_id)
        raise ApprovalNotPostable(str(exc)) from exc
    except ApprovalRowMissing as exc:
        raise ApprovalNotPostable(str(exc)) from exc
    except ApprovalDecisionConflict as exc:
        raise ApprovalNotPostable(str(exc)) from exc

    # Durable posting marker + reservation committed. Do not hold the SQLite
    # transaction open across ledger.post.
    post_kwargs = {
        "vendor_id": prepared["vendor_id"],
        "invoice_number": prepared["invoice_number"],
        "invoice_date": prepared["invoice_date"],
        "currency": prepared["currency"],
        "amount": prepared["amount"],
        "reference": approval_id,
    }

    try:
        if cancel_token is not None and cancel_token.cancelled:
            return reconcile_posting(store, conversation_id, approval_id)
        from ledger_guard import ledger_lock

        # Hold the process ledger lock only around the ERP call — never across
        # provider SDK waits. Path must already be the intended session ledger.
        with ledger_lock():
            returned_id = ledger.post(**post_kwargs)
    except RunCancelled:
        return reconcile_posting(store, conversation_id, approval_id)
    except Exception:
        return reconcile_posting(store, conversation_id, approval_id)

    try:
        matches = find_reconcile_matches(prepared, approval_id)
    except Exception as exc:
        uncertain = store.mark_approval_posting_uncertain(approval_id)
        raise PostingUncertain(
            uncertain,
            f"ledger lookup failed after post for {approval_id!r}: {exc}",
        ) from exc

    if len(matches) > 1:
        uncertain = store.mark_approval_posting_uncertain(approval_id)
        raise LedgerAmbiguity(
            uncertain,
            f"ambiguous ledger matches after post for {approval_id!r}",
        )

    if len(matches) == 1:
        entry_id = matches[0]["entry_id"]
        if returned_id and entry_id != returned_id:
            pass
        return store.mark_approval_posted(approval_id, entry_id)

    if isinstance(returned_id, str) and returned_id.strip():
        uncertain = store.mark_approval_posting_uncertain(approval_id)
        raise PostingUncertain(
            uncertain,
            f"ledger.post returned {returned_id!r} but no verified entries() "
            f"match for {approval_id!r}",
        )

    uncertain = store.mark_approval_posting_uncertain(approval_id)
    raise PostingUncertain(
        uncertain,
        f"no verified ledger entry after post for {approval_id!r}",
    )
