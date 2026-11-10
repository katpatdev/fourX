"""Deterministic invoice/vendor helpers for the FourX agent harness.

These functions run in application code (not in the model). They list and read
invoices, resolve vendors, and decide whether an invoice is eligible for the
human approval step. They never call ``ledger.post()``.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import ledger

ROOT = Path(__file__).resolve().parent
INVOICES_DIR = ROOT / "invoices"
VENDORS_PATH = ROOT / "vendors.json"

# Invoice IDs are only ever the basename stem, e.g. "invoice-01".
_INVOICE_ID_RE = re.compile(r"^invoice-\d+$")


class InvalidInvoiceId(ValueError):
    """Raised when an invoice_id is malformed or unsafe."""


class InvoiceNotFound(LookupError):
    """Raised when a well-formed invoice_id is not on disk."""


class VendorNotFound(LookupError):
    """Raised when no vendor matches the lookup."""


class BlockedVendor(ValueError):
    """Raised when the invoice's vendor is blocked and must not be posted."""


class DuplicateInvoice(ValueError):
    """Raised when vendor_id + invoice_number already exist in the ledger."""


class InvalidInvoiceTotal(ValueError):
    """Raised when the invoice total is not a valid positive monetary amount."""


def _available_invoice_ids() -> list[str]:
    """Whitelist of invoice IDs discovered under invoices/ (sorted)."""
    if not INVOICES_DIR.is_dir():
        return []
    root = INVOICES_DIR.resolve()
    ids: list[str] = []
    for path in sorted(INVOICES_DIR.glob("invoice-*.json")):
        # Only accept plain files whose resolved parent is the invoices directory.
        resolved = path.resolve()
        if not resolved.is_file() or resolved.parent != root:
            continue
        invoice_id = resolved.stem
        if _INVOICE_ID_RE.fullmatch(invoice_id):
            ids.append(invoice_id)
    return ids


def _resolve_invoice_path(invoice_id: str) -> Path:
    """Map a validated invoice_id to a file inside invoices/ only."""
    if not isinstance(invoice_id, str) or not _INVOICE_ID_RE.fullmatch(invoice_id):
        raise InvalidInvoiceId(f"invalid invoice id: {invoice_id!r}")
    allowed = _available_invoice_ids()
    if invoice_id not in allowed:
        raise InvoiceNotFound(f"invoice not found: {invoice_id!r}")
    path = (INVOICES_DIR / f"{invoice_id}.json").resolve()
    if path.parent != INVOICES_DIR.resolve():
        raise InvalidInvoiceId(f"invalid invoice id: {invoice_id!r}")
    return path


def _load_vendors() -> list[dict[str, Any]]:
    return json.loads(VENDORS_PATH.read_text(encoding="utf-8"))


def validate_posting_amount(amount: Any) -> str:
    """Require a finite, positive amount that fits exactly in two decimal places.

    Values that would need rounding to match ``ledger.post``'s ``0.01`` quantize
    are rejected. Valid original strings are returned unchanged.
    """
    if isinstance(amount, bool) or amount is None:
        raise InvalidInvoiceTotal(f"invalid invoice total: {amount!r}")
    text = amount if isinstance(amount, str) else str(amount)
    try:
        value = Decimal(text)
    except (InvalidOperation, TypeError) as exc:
        raise InvalidInvoiceTotal(f"invalid invoice total: {amount!r}") from exc
    if not value.is_finite() or value <= 0:
        raise InvalidInvoiceTotal(f"invoice total must be a positive amount: {amount!r}")
    # Reject amounts that cannot be represented exactly at two decimal places
    # (same scale ledger.post uses), without relying on float.
    two_places = value.quantize(Decimal("0.01"))
    if two_places != value:
        raise InvalidInvoiceTotal(
            f"invoice total must have at most two decimal places: {amount!r}"
        )
    # Preserve the caller's decimal text when it was already a string.
    return text if isinstance(amount, str) else format(value, "f")


def list_invoices() -> list[dict[str, Any]]:
    """Return a concise summary of every invoice, ordered by invoice_id."""
    summaries: list[dict[str, Any]] = []
    for invoice_id in _available_invoice_ids():
        invoice = read_invoice(invoice_id)
        summaries.append(
            {
                "invoice_id": invoice_id,
                "invoice_number": invoice["invoice_number"],
                "vendor_name": invoice["vendor_name"],
                "currency": invoice["currency"],
                "total": invoice["total"],
            }
        )
    return summaries


def read_invoice(invoice_id: str) -> dict[str, Any]:
    """Load one invoice JSON document by safe invoice_id."""
    path = _resolve_invoice_path(invoice_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invoice file is not an object: {invoice_id!r}")
    return data


def lookup_vendor(
    *,
    name: str | None = None,
    vendor_id: str | None = None,
) -> dict[str, Any]:
    """Return one vendor record by exact name or vendor_id.

    Unknown vendors are errors — never treated as active by default.
    """
    if (name is None) == (vendor_id is None):
        raise ValueError("provide exactly one of name or vendor_id")

    for vendor in _load_vendors():
        if name is not None and vendor.get("name") == name:
            return dict(vendor)
        if vendor_id is not None and vendor.get("vendor_id") == vendor_id:
            return dict(vendor)

    if name is not None:
        raise VendorNotFound(f"vendor not found for name: {name!r}")
    raise VendorNotFound(f"vendor not found for vendor_id: {vendor_id!r}")


def prepare_post(invoice_id: str) -> dict[str, Any]:
    """Return approval-ready posting details, or raise if the invoice is ineligible.

    Checks (in order): invoice exists, vendor exists, vendor status is exactly
    ``active`` (blocked/other/missing fail closed), amount is a valid two-place
    positive Decimal, and ledger has no matching (vendor_id, invoice_number).
    Does not call ``ledger.post()``.
    """
    invoice = read_invoice(invoice_id)
    vendor_name = invoice.get("vendor_name")
    if not isinstance(vendor_name, str) or not vendor_name:
        raise VendorNotFound(f"invoice {invoice_id!r} has no vendor_name")

    vendor = lookup_vendor(name=vendor_name)
    status = vendor.get("status")
    if status == "blocked":
        reason = vendor.get("blocked_reason") or "no reason given"
        raise BlockedVendor(
            f"vendor {vendor['vendor_id']} ({vendor['name']}) is blocked: {reason}"
        )
    # Fail closed: only an explicit "active" status may proceed to approval.
    if status != "active":
        raise BlockedVendor(
            f"vendor {vendor['vendor_id']} ({vendor['name']}) is not eligible to post "
            f"(status={status!r}; only 'active' vendors may proceed)"
        )

    amount = validate_posting_amount(invoice.get("total"))

    vendor_id = vendor["vendor_id"]
    invoice_number = invoice["invoice_number"]
    # Fail closed: any ledger lookup error must not look like "no duplicate".
    # Serialize against CLI snapshot/reset path swaps (process-local).
    from ledger_guard import ledger_lock

    with ledger_lock():
        existing = ledger.entries(vendor_id=vendor_id, invoice_number=invoice_number)
    if existing:
        entry_id = existing[0].get("entry_id", "unknown")
        raise DuplicateInvoice(
            f"invoice {invoice_number!r} for {vendor_id} already posted as {entry_id}"
        )

    return {
        "invoice_id": invoice_id,
        "vendor_id": vendor_id,
        "vendor_name": vendor["name"],
        "invoice_number": invoice_number,
        "invoice_date": invoice["invoice_date"],
        "currency": invoice["currency"],
        "amount": amount,
    }


def ledger_summary(
    *,
    invoice_number: str | None = None,
    vendor_id: str | None = None,
) -> dict[str, Any]:
    """Read-only ledger overview from ``ledger.entries`` (never posts).

    Counts journal entries only — approvals are not journal rows.
    """
    from ledger_guard import ledger_lock

    filters: dict[str, Any] = {}
    if invoice_number is not None:
        filters["invoice_number"] = invoice_number
    if vendor_id is not None:
        filters["vendor_id"] = vendor_id

    with ledger_lock():
        if filters:
            matched = ledger.entries(**filters)
            all_entries = ledger.entries()
        else:
            all_entries = ledger.entries()
            matched = all_entries

    compact = [
        {
            "entry_id": row.get("entry_id"),
            "vendor_id": row.get("vendor_id"),
            "invoice_number": row.get("invoice_number"),
            "currency": row.get("currency"),
            "amount": row.get("amount"),
        }
        for row in matched
    ]
    return {
        "entry_count": len(all_entries),
        "matched_count": len(matched),
        "entry_ids": [row.get("entry_id") for row in all_entries],
        "invoice_numbers": [row.get("invoice_number") for row in all_entries],
        "contains_invoice_number": (
            None
            if invoice_number is None
            else any(row.get("invoice_number") == invoice_number for row in all_entries)
        ),
        "entries": compact,
    }
