"""Process-local serialization for ledger.DB_PATH access.

``ledger.py`` is unchanged and always reads the global ``DB_PATH``. This module
provides a re-entrant lock so CLI snapshots/resets and worker eligibility /
posting calls cannot interleave path swaps with ledger I/O.

Does not protect independent OS processes that open the same SQLite file.
Never hold this lock across provider SDK / network calls.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import ledger

_LOCK = threading.RLock()


def ledger_lock() -> threading.RLock:
    return _LOCK


@contextmanager
def using_ledger(path: Path | str | None = None) -> Iterator[None]:
    """Hold the ledger lock; optionally use ``path`` for the duration, then restore."""
    with _LOCK:
        previous = ledger.DB_PATH
        if path is not None:
            ledger.DB_PATH = Path(path)
        try:
            yield
        finally:
            if path is not None:
                ledger.DB_PATH = previous


def set_ledger_path(path: Path | str) -> None:
    """Permanently set ``ledger.DB_PATH`` (caller must ensure no live workers)."""
    with _LOCK:
        ledger.DB_PATH = Path(path)
