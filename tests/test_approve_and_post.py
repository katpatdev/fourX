"""Assignment semantics: explicit human approve → safe post_approved."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ledger
from fourx.finance.approval import approve, get_approval, request_approval
from fourx.cli import FourXCLI
from fourx.agent.conversation import Message
from fourx.finance.posting import PostingUncertain, approve_and_post, post_approved
from fourx.persistence.storage import ConversationStore, StoredApproval


class ApproveAndPostTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmpdir.cleanup)
        root = Path(self._tmpdir.name)
        self.conv_db = root / "conversations.db"
        self.ledger_db = root / "ledger.db"
        self._db_patch = patch.object(ledger, "DB_PATH", self.ledger_db)
        self._sleep_patch = patch.object(ledger.time, "sleep", return_value=None)
        self._db_patch.start()
        self._sleep_patch.start()
        self.addCleanup(self._db_patch.stop)
        self.addCleanup(self._sleep_patch.stop)
        ledger.entries()
        self.store = ConversationStore(self.conv_db)
        self.store.create("demo", "gemini")
        self.out = io.StringIO()
        self.cli = FourXCLI(
            self.store,
            "demo",
            model_factory=lambda _p: (lambda *_a, **_k: Message(role="assistant", content="x")),
            stdout=self.out,
            ledger_db=self.ledger_db,
        )

    def _pending(self, invoice_id: str = "invoice-01"):
        with patch("ledger.entries", return_value=[]):
            return request_approval(self.store, "demo", invoice_id)

    def test_slash_approve_posts_once(self) -> None:
        pending = self._pending()
        before = len(ledger.entries())
        self.cli.handle_line("/approve")
        row = get_approval(self.store, pending.approval_id)
        self.assertEqual(row.status, "posted")
        self.assertIsNotNone(row.journal_entry_id)
        self.assertEqual(len(ledger.entries()), before + 1)
        self.assertIn("Approved and posted", self.out.getvalue())

    def test_nl_approve_posts_once(self) -> None:
        pending = self._pending("invoice-02")
        before = len(ledger.entries())
        self.cli.handle_line("I approve this")
        row = get_approval(self.store, pending.approval_id)
        self.assertEqual(row.status, "posted")
        self.assertEqual(len(ledger.entries()), before + 1)

    def test_repeat_approve_does_not_duplicate(self) -> None:
        pending = self._pending()
        self.cli.handle_line("/approve")
        je = get_approval(self.store, pending.approval_id).journal_entry_id
        count = len(ledger.entries())
        self.cli.handle_line("/approve")
        self.cli.handle_line("Approve this invoice")
        self.assertEqual(len(ledger.entries()), count)
        self.assertEqual(
            get_approval(self.store, pending.approval_id).journal_entry_id, je
        )

    def test_post_remains_for_approved_leftover(self) -> None:
        pending = self._pending()
        approve(self.store, "demo", pending.approval_id)
        self.assertEqual(
            get_approval(self.store, pending.approval_id).status, "approved"
        )
        before = len(ledger.entries())
        self.cli.handle_line("/post")
        row = get_approval(self.store, pending.approval_id)
        self.assertEqual(row.status, "posted")
        self.assertEqual(len(ledger.entries()), before + 1)

    def test_uncertain_after_approve_requires_reconcile(self) -> None:
        pending = self._pending()
        uncertain = StoredApproval(
            approval_id=pending.approval_id,
            conversation_id="demo",
            invoice_id=pending.invoice_id,
            details=dict(pending.details),
            status="posting_uncertain",
        )
        with patch(
            "fourx.cli.approve_and_post",
            side_effect=PostingUncertain(uncertain, "cannot verify"),
        ):
            self.cli.handle_line("/approve")
        self.assertIn("uncertain", self.out.getvalue().lower())

    def test_approve_and_post_idempotent_when_already_posted(self) -> None:
        pending = self._pending()
        first = approve_and_post(self.store, "demo", pending.approval_id)
        second = approve_and_post(self.store, "demo", first.approval_id)
        self.assertEqual(first.journal_entry_id, second.journal_entry_id)
        self.assertEqual(
            len(ledger.entries(invoice_number=pending.details["invoice_number"])),
            1,
        )

    def test_restart_then_approve_posts(self) -> None:
        pending = self._pending("invoice-03")
        reopened = ConversationStore(self.conv_db)
        out2 = io.StringIO()
        cli2 = FourXCLI(
            reopened,
            "demo",
            model_factory=lambda _p: (lambda *_a, **_k: Message(role="assistant", content="x")),
            stdout=out2,
            ledger_db=self.ledger_db,
        )
        before = len(ledger.entries())
        cli2.handle_line("/approve")
        row = get_approval(reopened, pending.approval_id)
        self.assertEqual(row.status, "posted")
        self.assertEqual(len(ledger.entries()), before + 1)

    def test_posting_failure_reports_durable_approved_state(self) -> None:
        pending = self._pending()
        with patch(
            "fourx.finance.posting.post_approved",
            side_effect=RuntimeError("ledger unavailable"),
        ):
            # approve() succeeds inside approve_and_post before post_approved.
            with self.assertRaises(RuntimeError):
                approve_and_post(self.store, "demo", pending.approval_id)
        row = get_approval(self.store, pending.approval_id)
        self.assertEqual(row.status, "approved")
        self.assertIsNone(row.journal_entry_id)
        # Leftover /post path can still be used after the failure.
        recovered = post_approved(self.store, "demo", pending.approval_id)
        self.assertEqual(recovered.status, "posted")


if __name__ == "__main__":
    unittest.main()
