"""A fake ERP ledger for the take-home. Use it as it is: don't change this file.

Posting takes a couple of seconds, like a call to a real ERP. Like many real ERPs,
nothing here stops the same invoice being posted twice, and the entry is saved partway
through the call: if your process dies before `post` returns, the entry is there but
you never got its id.

    python ledger.py          # show what's posted
    python ledger.py reset    # start again from the opening ledger
"""

from __future__ import annotations

import sqlite3
import sys
import time
import uuid
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

DB_PATH = Path(__file__).with_name("ledger.db")

_COLUMNS = (
    "entry_id",
    "vendor_id",
    "invoice_number",
    "invoice_date",
    "currency",
    "amount",
    "reference",
    "posted_at",
)

# Already posted before you start.
_OPENING_ENTRIES = [
    {
        "entry_id": "JE-000001",
        "vendor_id": "V-002",
        "invoice_number": "BL-88190",
        "invoice_date": "2026-09-28",
        "currency": "INR",
        "amount": "25960.00",
        "reference": None,
        "posted_at": "2026-10-01T10:15:00+00:00",
    }
]


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS entries ("
            " entry_id TEXT PRIMARY KEY, vendor_id TEXT NOT NULL,"
            " invoice_number TEXT NOT NULL, invoice_date TEXT NOT NULL,"
            " currency TEXT NOT NULL, amount TEXT NOT NULL, reference TEXT,"
            " posted_at TEXT NOT NULL)"
        )
        conn.executemany(
            f"INSERT OR IGNORE INTO entries ({', '.join(_COLUMNS)})"
            f" VALUES ({', '.join('?' for _ in _COLUMNS)})",
            [tuple(entry[column] for column in _COLUMNS) for entry in _OPENING_ENTRIES],
        )
    return conn


def post(
    *,
    vendor_id: str,
    invoice_number: str,
    invoice_date: str,
    currency: str,
    amount: str | int | float | Decimal,
    reference: str | None = None,
) -> str:
    """Post an invoice to the ledger and return the new entry's id (``JE-...``).

    ``amount`` is the invoice total. ``reference`` is free text: use it however you like.
    """
    try:
        total = Decimal(str(amount)).quantize(Decimal("0.01"))
    except InvalidOperation:
        raise ValueError(f"amount is not a number: {amount!r}") from None
    if not total.is_finite() or total <= 0:
        raise ValueError(f"amount must be a positive number: {amount!r}")

    entry = {
        "entry_id": f"JE-{uuid.uuid4().hex[:8].upper()}",
        "vendor_id": vendor_id,
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "currency": currency,
        "amount": str(total),
        "reference": reference,
        "posted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    time.sleep(1)
    with closing(_connect()) as conn, conn:
        conn.execute(
            f"INSERT INTO entries ({', '.join(_COLUMNS)})"
            f" VALUES ({', '.join('?' for _ in _COLUMNS)})",
            tuple(entry[column] for column in _COLUMNS),
        )
    time.sleep(1)
    return entry["entry_id"]


def entries(**filters: str) -> list[dict]:
    """Every posted entry, oldest first. Filter on any column: ``entries(vendor_id="V-002")``."""
    unknown = sorted(set(filters) - set(_COLUMNS))
    if unknown:
        raise ValueError(f"unknown column(s): {', '.join(unknown)}")
    where = " AND ".join(f"{column} = ?" for column in filters)
    sql = "SELECT * FROM entries" + (f" WHERE {where}" if where else "")
    with closing(_connect()) as conn:
        rows = conn.execute(sql + " ORDER BY posted_at, entry_id", tuple(filters.values()))
        return [dict(row) for row in rows]


def _main(argv: list[str]) -> int:
    if argv == ["reset"]:
        DB_PATH.unlink(missing_ok=True)
        print("Ledger reset to its opening entries.")
        return 0
    if argv:
        print(__doc__)
        return 2
    rows = entries()
    for row in rows:
        print(
            f"{row['entry_id']:<12} {row['posted_at'][:19]}  {row['vendor_id']:<6}"
            f" {row['invoice_number']:<16} {row['currency']} {row['amount']:>12}"
            f"  {row['reference'] or ''}"
        )
    print(f"{len(rows)} entr{'y' if len(rows) == 1 else 'ies'}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
