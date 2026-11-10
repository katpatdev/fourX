"""Offline tests for safe ledger posting and crash reconciliation."""

from __future__ import annotations

import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import ledger
from approval import approve, decline, list_approvals, request_approval
from cancellation import CancellationToken, RunCancelled
from conversation import Conversation, Message, ToolCall
from harness import ApprovalRequired, run_agent_turn
from invoice_tools import BlockedVendor, DuplicateInvoice
from posting import (
    ApprovalNotPostable,
    InvoiceReservedByOther,
    LedgerAmbiguity,
    LedgerInvoiceConflict,
    PostingDetailsMismatch,
    PostingError,
    PostingUncertain,
    post_approved,
    reconcile_posting,
)
from storage import (
    ApprovalPostingClaimConflict,
    ConversationStore,
    InvoiceAlreadyReserved,
)


class FakeModel:
    def __init__(self, responses: list[Message]) -> None:
        self.responses = list(responses)

    def __call__(self, conversation: Conversation, tools: list[dict]) -> Message:
        if not self.responses:
            raise AssertionError("FakeModel exhausted")
        return self.responses.pop(0)


class PostingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        root = Path(self._tmpdir.name)
        self.conv_db = root / "conversations.db"
        self.ledger_db = root / "ledger.db"
        self.store = ConversationStore(self.conv_db)
        self.store.create("demo", "gemini")

        self._db_patch = patch.object(ledger, "DB_PATH", self.ledger_db)
        self._sleep_patch = patch.object(ledger.time, "sleep", return_value=None)
        self._db_patch.start()
        self._sleep_patch.start()
        self.addCleanup(self._db_patch.stop)
        self.addCleanup(self._sleep_patch.stop)
        # Materialize opening ledger (JE-000001 / BL-88190) in the temp DB.
        ledger.entries()

    def _approve_invoice(self, invoice_id: str, conversation_id: str = "demo"):
        with patch("ledger.entries", wraps=ledger.entries):
            # prepare_post uses ledger.entries; ensure it sees the temp DB.
            req = request_approval(self.store, conversation_id, invoice_id)
        return approve(self.store, conversation_id, req.approval_id)

    def test_01_approved_invoice_01_posts_once(self) -> None:
        approved = self._approve_invoice("invoice-01")
        with patch("ledger.post", wraps=ledger.post) as mock_post:
            result = post_approved(self.store, "demo", approved.approval_id)
            self.assertEqual(mock_post.call_count, 1)
        self.assertEqual(result.status, "posted")
        matches = ledger.entries(
            vendor_id="V-001", invoice_number="SGP/26-27/0412"
        )
        self.assertEqual(len(matches), 1)

    def test_02_exact_journal_id_recorded(self) -> None:
        approved = self._approve_invoice("invoice-01")
        result = post_approved(self.store, "demo", approved.approval_id)
        matches = ledger.entries(
            vendor_id="V-001", invoice_number="SGP/26-27/0412"
        )
        self.assertEqual(result.journal_entry_id, matches[0]["entry_id"])
        self.assertTrue(result.journal_entry_id.startswith("JE-"))

    def test_03_posted_status_survives_restart(self) -> None:
        approved = self._approve_invoice("invoice-01")
        posted = post_approved(self.store, "demo", approved.approval_id)
        reopened = ConversationStore(self.conv_db)
        with patch("ledger.post") as mock_post:
            again = post_approved(reopened, "demo", posted.approval_id)
            mock_post.assert_not_called()
        self.assertEqual(again.status, "posted")
        self.assertEqual(again.journal_entry_id, posted.journal_entry_id)

    def test_04_pending_cannot_post(self) -> None:
        with patch("ledger.entries", return_value=[]):
            pending = request_approval(self.store, "demo", "invoice-01")
        with patch("ledger.post") as mock_post:
            with self.assertRaises(ApprovalNotPostable):
                post_approved(self.store, "demo", pending.approval_id)
            mock_post.assert_not_called()

    def test_05_declined_cannot_post(self) -> None:
        with patch("ledger.entries", return_value=[]):
            pending = request_approval(self.store, "demo", "invoice-01")
        declined = decline(self.store, "demo", "no", pending.approval_id)
        with patch("ledger.post") as mock_post:
            with self.assertRaises(ApprovalNotPostable):
                post_approved(self.store, "demo", declined.approval_id)
            mock_post.assert_not_called()

    def test_06_unknown_approval_cannot_post(self) -> None:
        with patch("ledger.post") as mock_post:
            with self.assertRaises(ApprovalNotPostable):
                post_approved(self.store, "demo", "apr-missing")
            mock_post.assert_not_called()

    def test_07_wrong_conversation_cannot_post(self) -> None:
        approved = self._approve_invoice("invoice-01")
        self.store.create("other", "gemini")
        with patch("ledger.post") as mock_post:
            with self.assertRaises(ApprovalNotPostable):
                post_approved(self.store, "other", approved.approval_id)
            mock_post.assert_not_called()

    def test_08_blocked_vendor_cannot_post(self) -> None:
        # Force an approved row for invoice-04 (blocked) without prepare_post.
        details = {
            "invoice_id": "invoice-04",
            "vendor_id": "V-004",
            "vendor_name": "Orbit Facility Services",
            "invoice_number": "OFS/OCT/26/031",
            "invoice_date": "2026-10-01",
            "currency": "INR",
            "amount": "100300.00",
        }
        row = self.store.create_pending_approval("demo", details)
        self.store.set_approval_decision(row.approval_id, status="approved")
        with patch("ledger.post") as mock_post:
            with self.assertRaises(BlockedVendor):
                post_approved(self.store, "demo", row.approval_id)
            mock_post.assert_not_called()
        self.assertEqual(
            self.store.get_approval_by_id(row.approval_id).status, "approved"
        )

    def test_09_duplicate_invoice_05_cannot_post(self) -> None:
        # Opening ledger already has BL-88190 / V-002.
        details = {
            "invoice_id": "invoice-05",
            "vendor_id": "V-002",
            "vendor_name": "Bluefin Logistics LLP",
            "invoice_number": "BL-88190",
            "invoice_date": "2026-09-28",
            "currency": "INR",
            "amount": "25960.00",
        }
        row = self.store.create_pending_approval("demo", details)
        self.store.set_approval_decision(row.approval_id, status="approved")
        with patch("ledger.post") as mock_post:
            with self.assertRaises(DuplicateInvoice):
                post_approved(self.store, "demo", row.approval_id)
            mock_post.assert_not_called()
        self.assertEqual(len(ledger.entries(invoice_number="BL-88190")), 1)

    def test_10_changed_amount_after_approval_rejected(self) -> None:
        approved = self._approve_invoice("invoice-01")
        mutated = dict(approved.details)
        mutated["amount"] = "1.00"
        with self.store._connect() as conn, conn:
            conn.execute(
                "UPDATE approvals SET details_json = ? WHERE approval_id = ?",
                (__import__("json").dumps(mutated), approved.approval_id),
            )
        with patch("ledger.post") as mock_post:
            with self.assertRaises(PostingDetailsMismatch):
                post_approved(self.store, "demo", approved.approval_id)
            mock_post.assert_not_called()

    def test_11_changed_vendor_or_invoice_number_rejected(self) -> None:
        approved = self._approve_invoice("invoice-01")
        mutated = dict(approved.details)
        mutated["invoice_number"] = "OTHER-1"
        with self.store._connect() as conn, conn:
            conn.execute(
                "UPDATE approvals SET details_json = ? WHERE approval_id = ?",
                (__import__("json").dumps(mutated), approved.approval_id),
            )
        with patch("ledger.post") as mock_post:
            with self.assertRaises(PostingDetailsMismatch):
                post_approved(self.store, "demo", approved.approval_id)
            mock_post.assert_not_called()

    def test_12_duplicate_posting_request_rejected(self) -> None:
        approved = self._approve_invoice("invoice-01")
        first = post_approved(self.store, "demo", approved.approval_id)
        with patch("ledger.post") as mock_post:
            second = post_approved(self.store, "demo", approved.approval_id)
            mock_post.assert_not_called()
        self.assertEqual(second.journal_entry_id, first.journal_entry_id)
        self.assertEqual(
            len(ledger.entries(vendor_id="V-001", invoice_number="SGP/26-27/0412")),
            1,
        )

    def test_13_concurrent_attempts_only_one_posts(self) -> None:
        approved = self._approve_invoice("invoice-01")
        barrier = threading.Barrier(2)
        results: list[object] = []
        errors: list[BaseException] = []
        post_calls = {"n": 0}
        lock = threading.Lock()

        real_post = ledger.post

        def counting_post(**kwargs):
            with lock:
                post_calls["n"] += 1
            return real_post(**kwargs)

        def worker() -> None:
            barrier.wait()
            try:
                with patch("ledger.post", side_effect=counting_post):
                    results.append(
                        post_approved(self.store, "demo", approved.approval_id)
                    )
            except BaseException as exc:  # noqa: BLE001 — collect for assertion
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertTrue(all(not t.is_alive() for t in threads))
        posted = [r for r in results if getattr(r, "status", None) == "posted"]
        self.assertGreaterEqual(len(posted), 1)
        self.assertEqual(post_calls["n"], 1)
        self.assertEqual(
            len(ledger.entries(vendor_id="V-001", invoice_number="SGP/26-27/0412")),
            1,
        )

    def test_14_failed_transition_prevents_ledger_post(self) -> None:
        approved = self._approve_invoice("invoice-01")
        with patch.object(
            self.store,
            "claim_for_posting",
            side_effect=ApprovalPostingClaimConflict("claimed"),
        ):
            with patch("ledger.post") as mock_post:
                with self.assertRaises(ApprovalNotPostable):
                    post_approved(self.store, "demo", approved.approval_id)
                mock_post.assert_not_called()

    def test_15_persistence_failure_before_posting_prevents_ledger_post(self) -> None:
        approved = self._approve_invoice("invoice-01")
        with patch.object(
            self.store,
            "claim_for_posting",
            side_effect=sqlite3.OperationalError("disk full"),
        ):
            with patch("ledger.post") as mock_post:
                with self.assertRaises(sqlite3.OperationalError):
                    post_approved(self.store, "demo", approved.approval_id)
                mock_post.assert_not_called()
        self.assertEqual(
            self.store.get_approval_by_id(approved.approval_id).status, "approved"
        )

    def test_16_normal_success_reconciled(self) -> None:
        approved = self._approve_invoice("invoice-01")
        result = post_approved(self.store, "demo", approved.approval_id)
        self.assertEqual(result.status, "posted")
        self.assertIsNotNone(result.journal_entry_id)
        row = ledger.entries(entry_id=result.journal_entry_id)[0]
        self.assertEqual(row["amount"], "16862.20")

    def test_17_commit_then_exception_before_return_reconciled(self) -> None:
        approved = self._approve_invoice("invoice-01")
        real_post = ledger.post

        def commit_then_fail(**kwargs):
            entry_id = real_post(**kwargs)
            raise RuntimeError(f"lost return for {entry_id}")

        with patch("ledger.post", side_effect=commit_then_fail) as mock_post:
            # post_approved catches and reconciles — should become posted.
            result = post_approved(self.store, "demo", approved.approval_id)
            self.assertEqual(mock_post.call_count, 1)
        self.assertEqual(result.status, "posted")
        self.assertEqual(
            len(ledger.entries(vendor_id="V-001", invoice_number="SGP/26-27/0412")),
            1,
        )
        with patch("ledger.post") as mock_post2:
            again = post_approved(self.store, "demo", approved.approval_id)
            mock_post2.assert_not_called()
        self.assertEqual(again.journal_entry_id, result.journal_entry_id)

    def test_18_restart_posting_with_matching_entry_no_repost(self) -> None:
        approved = self._approve_invoice("invoice-01")
        self.store.claim_for_posting(approved.approval_id, "demo")
        entry_id = ledger.post(
            vendor_id="V-001",
            invoice_number="SGP/26-27/0412",
            invoice_date="2026-10-03",
            currency="INR",
            amount="16862.20",
            reference=approved.approval_id,
        )
        reopened = ConversationStore(self.conv_db)
        with patch("ledger.post") as mock_post:
            result = post_approved(reopened, "demo", approved.approval_id)
            mock_post.assert_not_called()
        self.assertEqual(result.status, "posted")
        self.assertEqual(result.journal_entry_id, entry_id)

    def test_19_restart_posting_without_match_does_not_repost(self) -> None:
        approved = self._approve_invoice("invoice-01")
        self.store.claim_for_posting(approved.approval_id, "demo")
        with patch("ledger.post") as mock_post:
            with self.assertRaises(PostingUncertain) as ctx:
                post_approved(self.store, "demo", approved.approval_id)
            mock_post.assert_not_called()
        self.assertEqual(ctx.exception.approval.status, "posting_uncertain")
        self.assertEqual(
            self.store.get_approval_by_id(approved.approval_id).status,
            "posting_uncertain",
        )

    def test_20_ambiguous_matches_no_automatic_repost(self) -> None:
        approved = self._approve_invoice("invoice-01")
        self.store.claim_for_posting(approved.approval_id, "demo")
        fake_rows = [
            {
                "entry_id": "JE-AAAA",
                "vendor_id": "V-001",
                "invoice_number": "SGP/26-27/0412",
                "invoice_date": "2026-10-03",
                "currency": "INR",
                "amount": "16862.20",
            },
            {
                "entry_id": "JE-BBBB",
                "vendor_id": "V-001",
                "invoice_number": "SGP/26-27/0412",
                "invoice_date": "2026-10-03",
                "currency": "INR",
                "amount": "16862.20",
            },
        ]
        with patch("ledger.entries", return_value=fake_rows):
            with patch("ledger.post") as mock_post:
                with self.assertRaises(LedgerAmbiguity):
                    reconcile_posting(self.store, "demo", approved.approval_id)
                mock_post.assert_not_called()
        self.assertEqual(
            self.store.get_approval_by_id(approved.approval_id).status,
            "posting_uncertain",
        )

    def test_21_ledger_lookup_failure_safe_non_posted(self) -> None:
        approved = self._approve_invoice("invoice-01")
        self.store.claim_for_posting(approved.approval_id, "demo")
        with patch("ledger.entries", side_effect=RuntimeError("db locked")):
            with patch("ledger.post") as mock_post:
                with self.assertRaises(PostingUncertain) as ctx:
                    reconcile_posting(self.store, "demo", approved.approval_id)
                mock_post.assert_not_called()
        self.assertEqual(ctx.exception.approval.status, "posting_uncertain")
        self.assertNotEqual(
            self.store.get_approval_by_id(approved.approval_id).status, "posted"
        )

    def test_22_cancellation_before_posting_prevents_ledger_post(self) -> None:
        approved = self._approve_invoice("invoice-01")
        token = CancellationToken()
        token.cancel()
        with patch("ledger.post") as mock_post:
            with self.assertRaises(RunCancelled):
                post_approved(
                    self.store, "demo", approved.approval_id, cancel_token=token
                )
            mock_post.assert_not_called()
        self.assertEqual(
            self.store.get_approval_by_id(approved.approval_id).status, "approved"
        )

    def test_23_cancellation_after_attempt_preserves_uncertainty(self) -> None:
        approved = self._approve_invoice("invoice-01")
        token = CancellationToken()
        real_claim = self.store.claim_for_posting

        def claim_and_cancel(approval_id, conversation_id):
            row = real_claim(approval_id, conversation_id)
            token.cancel()
            return row

        with patch.object(self.store, "claim_for_posting", side_effect=claim_and_cancel):
            with patch("ledger.post") as mock_post:
                with self.assertRaises(PostingUncertain) as ctx:
                    post_approved(
                        self.store, "demo", approved.approval_id, cancel_token=token
                    )
                mock_post.assert_not_called()
        self.assertEqual(ctx.exception.approval.status, "posting_uncertain")
        self.assertEqual(
            self.store.get_approval_by_id(approved.approval_id).status,
            "posting_uncertain",
        )

    def test_24_new_approval_after_confirmed_post(self) -> None:
        first = self._approve_invoice("invoice-01")
        post_approved(self.store, "demo", first.approval_id)
        second = request_approval(self.store, "demo", "invoice-02")
        self.assertEqual(second.status, "pending")
        self.assertEqual(second.invoice_id, "invoice-02")

    def test_25_previous_approvals_remain_in_history(self) -> None:
        first = self._approve_invoice("invoice-01")
        post_approved(self.store, "demo", first.approval_id)
        second_pending = request_approval(self.store, "demo", "invoice-02")
        approve(self.store, "demo", second_pending.approval_id)
        post_approved(self.store, "demo", second_pending.approval_id)
        history = list_approvals(self.store, "demo")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].status, "posted")
        self.assertEqual(history[1].status, "posted")
        self.assertNotEqual(history[0].approval_id, history[1].approval_id)
        self.assertNotEqual(history[0].journal_entry_id, history[1].journal_entry_id)

    def test_26_human_approve_alone_does_not_post(self) -> None:
        with patch("ledger.entries", return_value=[]):
            pending = request_approval(self.store, "demo", "invoice-01")
        with patch("ledger.post") as mock_post:
            approved = approve(self.store, "demo", pending.approval_id)
            mock_post.assert_not_called()
        self.assertEqual(approved.status, "approved")
        self.assertIsNone(approved.journal_entry_id)
        self.assertEqual(ledger.entries(invoice_number="SGP/26-27/0412"), [])

    def test_27_model_post_invoice_alone_does_not_post(self) -> None:
        loaded = self.store.load("demo")
        loaded.conversation.add(Message(role="user", content="post invoice-01"))
        self.store.save("demo", loaded.conversation)
        model = FakeModel(
            [
                Message(
                    role="assistant",
                    tool_calls=(
                        ToolCall(
                            id="p1",
                            name="post_invoice",
                            arguments={"invoice_id": "invoice-01"},
                        ),
                    ),
                )
            ]
        )
        with patch("invoice_tools.ledger.post") as mock_post:
            with patch("ledger.entries", return_value=[]):
                with self.assertRaises(ApprovalRequired):
                    run_agent_turn(store=self.store, conversation_id="demo", model=model)
            mock_post.assert_not_called()
        self.assertEqual(ledger.entries(invoice_number="SGP/26-27/0412"), [])

    def _seed_ledger_row(self, **overrides) -> str:
        kwargs = {
            "vendor_id": "V-001",
            "invoice_number": "SGP/26-27/0412",
            "invoice_date": "2026-10-03",
            "currency": "INR",
            "amount": "16862.20",
            "reference": None,
        }
        kwargs.update(overrides)
        return ledger.post(**kwargs)

    def _force_approved_invoice_01(self):
        """Create approved row without prepare_post (ledger may already conflict)."""
        details = {
            "invoice_id": "invoice-01",
            "vendor_id": "V-001",
            "vendor_name": "Shree Ganesh Packaging Pvt Ltd",
            "invoice_number": "SGP/26-27/0412",
            "invoice_date": "2026-10-03",
            "currency": "INR",
            "amount": "16862.20",
        }
        pending = self.store.create_pending_approval("demo", details)
        return self.store.set_approval_decision(pending.approval_id, status="approved")

    def test_28_exact_duplicate_blocks_fresh_post(self) -> None:
        self._seed_ledger_row()
        approved = self._force_approved_invoice_01()
        with patch("ledger.post") as mock_post:
            with self.assertRaises(DuplicateInvoice):
                post_approved(self.store, "demo", approved.approval_id)
            mock_post.assert_not_called()

    def test_29_conflicting_amount_blocks_fresh_post(self) -> None:
        self._seed_ledger_row(amount="1.00")
        approved = self._force_approved_invoice_01()
        with patch("ledger.post") as mock_post:
            with self.assertRaises(LedgerInvoiceConflict):
                post_approved(self.store, "demo", approved.approval_id)
            mock_post.assert_not_called()
        self.assertEqual(
            self.store.get_approval_by_id(approved.approval_id).status, "approved"
        )

    def test_30_conflicting_date_blocks_fresh_post(self) -> None:
        self._seed_ledger_row(invoice_date="2000-01-01")
        approved = self._force_approved_invoice_01()
        with patch("ledger.post") as mock_post:
            with self.assertRaises(LedgerInvoiceConflict):
                post_approved(self.store, "demo", approved.approval_id)
            mock_post.assert_not_called()

    def test_31_conflicting_currency_blocks_fresh_post(self) -> None:
        self._seed_ledger_row(currency="USD")
        approved = self._force_approved_invoice_01()
        with patch("ledger.post") as mock_post:
            with self.assertRaises(LedgerInvoiceConflict):
                post_approved(self.store, "demo", approved.approval_id)
            mock_post.assert_not_called()

    def test_32_multiple_identifier_rows_block_fresh_post(self) -> None:
        self._seed_ledger_row(amount="1.00")
        self._seed_ledger_row(amount="2.00")
        approved = self._force_approved_invoice_01()
        with patch("ledger.post") as mock_post:
            with self.assertRaises(LedgerInvoiceConflict):
                post_approved(self.store, "demo", approved.approval_id)
            mock_post.assert_not_called()

    def test_33_ledger_lookup_failure_blocks_fresh_post(self) -> None:
        approved = self._approve_invoice("invoice-01")
        with patch(
            "posting.list_identifier_ledger_entries",
            side_effect=RuntimeError("db locked"),
        ):
            with patch("ledger.post") as mock_post:
                with self.assertRaises(PostingError):
                    post_approved(self.store, "demo", approved.approval_id)
                mock_post.assert_not_called()

    def test_34_cross_conversation_concurrent_posts_once(self) -> None:
        self.store.create("other", "gemini")
        a = self._approve_invoice("invoice-01", "demo")
        # Second conversation: force an approved row for the same invoice.
        details = dict(a.details)
        pending = self.store.create_pending_approval("other", details)
        b = self.store.set_approval_decision(pending.approval_id, status="approved")

        barrier = threading.Barrier(2)
        post_calls = {"n": 0}
        lock = threading.Lock()
        outcomes: list[object] = []
        real_post = ledger.post

        def counting_post(**kwargs):
            with lock:
                post_calls["n"] += 1
            return real_post(**kwargs)

        def worker(conversation_id: str, approval_id: str) -> None:
            barrier.wait()
            try:
                with patch("ledger.post", side_effect=counting_post):
                    outcomes.append(
                        post_approved(self.store, conversation_id, approval_id)
                    )
            except BaseException as exc:  # noqa: BLE001
                outcomes.append(exc)

        threads = [
            threading.Thread(target=worker, args=("demo", a.approval_id)),
            threading.Thread(target=worker, args=("other", b.approval_id)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertTrue(all(not t.is_alive() for t in threads))
        self.assertEqual(post_calls["n"], 1)
        posted = [o for o in outcomes if getattr(o, "status", None) == "posted"]
        conflicts = [
            o
            for o in outcomes
            if isinstance(o, (InvoiceReservedByOther, InvoiceAlreadyReserved))
        ]
        self.assertEqual(len(posted), 1)
        self.assertGreaterEqual(len(conflicts), 1)
        reservation = self.store.get_posting_reservation(
            vendor_id="V-001", invoice_number="SGP/26-27/0412"
        )
        self.assertIsNotNone(reservation)
        self.assertEqual(reservation.approval_id, posted[0].approval_id)
        self.assertEqual(
            len(ledger.entries(vendor_id="V-001", invoice_number="SGP/26-27/0412")),
            1,
        )

    def test_35_losing_cross_conversation_gets_clear_conflict(self) -> None:
        self.store.create("other", "gemini")
        a = self._approve_invoice("invoice-01", "demo")
        post_approved(self.store, "demo", a.approval_id)
        details = dict(a.details)
        pending = self.store.create_pending_approval("other", details)
        b = self.store.set_approval_decision(pending.approval_id, status="approved")
        with patch("ledger.post") as mock_post:
            with self.assertRaises((DuplicateInvoice, InvoiceReservedByOther)):
                post_approved(self.store, "other", b.approval_id)
            mock_post.assert_not_called()

    def test_36_different_invoices_post_independently(self) -> None:
        first = self._approve_invoice("invoice-01")
        post_approved(self.store, "demo", first.approval_id)
        second_pending = request_approval(self.store, "demo", "invoice-02")
        second = approve(self.store, "demo", second_pending.approval_id)
        result = post_approved(self.store, "demo", second.approval_id)
        self.assertEqual(result.status, "posted")
        self.assertEqual(
            len(ledger.entries(invoice_number="SGP/26-27/0412")), 1
        )
        self.assertEqual(len(ledger.entries(invoice_number="BL-88213")), 1)
        r1 = self.store.get_posting_reservation(
            vendor_id="V-001", invoice_number="SGP/26-27/0412"
        )
        r2 = self.store.get_posting_reservation(
            vendor_id="V-002", invoice_number="BL-88213"
        )
        self.assertEqual(r1.approval_id, first.approval_id)
        self.assertEqual(r2.approval_id, second.approval_id)

    def test_37_reservation_and_claim_are_atomic(self) -> None:
        approved = self._approve_invoice("invoice-01")
        # Pre-insert a competing reservation; claim must not leave status=posting.
        with self.store._connect() as conn, conn:
            conn.execute(
                """
                INSERT INTO posting_reservations (
                    vendor_id, invoice_number, approval_id, conversation_id
                ) VALUES (?, ?, ?, ?)
                """,
                ("V-001", "SGP/26-27/0412", "apr-competitor", "foreign"),
            )
        with patch("ledger.post") as mock_post:
            with self.assertRaises(InvoiceReservedByOther):
                post_approved(self.store, "demo", approved.approval_id)
            mock_post.assert_not_called()
        row = self.store.get_approval_by_id(approved.approval_id)
        self.assertEqual(row.status, "approved")
        self.assertIsNone(self.store.get_reservation_for_approval(approved.approval_id))

    def test_38_forced_integrity_failure_rolls_back_claim(self) -> None:
        """Simulated UNIQUE failure after UPDATE must roll back posting status."""
        approved = self._approve_invoice("invoice-01")
        real_connect = self.store._connect

        class FlakyConn:
            def __init__(self, conn):
                self._conn = conn
                self._updates = 0

            def execute(self, sql, params=()):
                if (
                    isinstance(sql, str)
                    and "INSERT INTO posting_reservations" in sql
                ):
                    raise sqlite3.IntegrityError("forced unique")
                return self._conn.execute(sql, params)

            def __getattr__(self, name):
                return getattr(self._conn, name)

            def __enter__(self):
                self._conn.__enter__()
                return self

            def __exit__(self, *args):
                return self._conn.__exit__(*args)

        def flaky_connect():
            return FlakyConn(real_connect())

        with patch.object(self.store, "_connect", side_effect=flaky_connect):
            with patch("ledger.post") as mock_post:
                with self.assertRaises(InvoiceReservedByOther):
                    post_approved(self.store, "demo", approved.approval_id)
                mock_post.assert_not_called()
        # Re-open a clean connection to inspect durable state.
        row = ConversationStore(self.conv_db).get_approval_by_id(approved.approval_id)
        self.assertEqual(row.status, "approved")
        self.assertIsNone(
            ConversationStore(self.conv_db).get_reservation_for_approval(
                approved.approval_id
            )
        )

    def test_39_reservation_survives_restart(self) -> None:
        approved = self._approve_invoice("invoice-01")
        post_approved(self.store, "demo", approved.approval_id)
        reopened = ConversationStore(self.conv_db)
        reservation = reopened.get_posting_reservation(
            vendor_id="V-001", invoice_number="SGP/26-27/0412"
        )
        self.assertIsNotNone(reservation)
        self.assertEqual(reservation.approval_id, approved.approval_id)

    def test_40_uncertain_retains_reservation_and_owner_reconciles(self) -> None:
        approved = self._approve_invoice("invoice-01")
        self.store.claim_for_posting(approved.approval_id, "demo")
        with patch("ledger.post") as mock_post:
            with self.assertRaises(PostingUncertain):
                post_approved(self.store, "demo", approved.approval_id)
            mock_post.assert_not_called()
        self.assertEqual(
            self.store.get_approval_by_id(approved.approval_id).status,
            "posting_uncertain",
        )
        reservation = self.store.get_reservation_for_approval(approved.approval_id)
        self.assertIsNotNone(reservation)
        entry_id = ledger.post(
            vendor_id="V-001",
            invoice_number="SGP/26-27/0412",
            invoice_date="2026-10-03",
            currency="INR",
            amount="16862.20",
            reference=approved.approval_id,
        )
        with patch("ledger.post") as mock_post:
            posted = reconcile_posting(self.store, "demo", approved.approval_id)
            mock_post.assert_not_called()
        self.assertEqual(posted.status, "posted")
        self.assertEqual(posted.journal_entry_id, entry_id)
        self.assertEqual(
            self.store.get_reservation_for_approval(approved.approval_id).approval_id,
            approved.approval_id,
        )

    def test_41_posted_keeps_journal_and_protective_reservation(self) -> None:
        approved = self._approve_invoice("invoice-01")
        posted = post_approved(self.store, "demo", approved.approval_id)
        reservation = self.store.get_posting_reservation(
            vendor_id="V-001", invoice_number="SGP/26-27/0412"
        )
        self.assertEqual(reservation.approval_id, posted.approval_id)
        self.assertEqual(posted.journal_entry_id, reservation and posted.journal_entry_id)
        self.assertTrue(posted.journal_entry_id.startswith("JE-"))

    def test_42_pending_and_declined_cannot_reserve(self) -> None:
        with patch("ledger.entries", return_value=[]):
            pending = request_approval(self.store, "demo", "invoice-01")
        with self.assertRaises(ApprovalPostingClaimConflict):
            self.store.claim_for_posting(pending.approval_id, "demo")
        declined = decline(self.store, "demo", "no", pending.approval_id)
        with self.assertRaises(ApprovalPostingClaimConflict):
            self.store.claim_for_posting(declined.approval_id, "demo")
        self.assertIsNone(
            self.store.get_posting_reservation(
                vendor_id="V-001", invoice_number="SGP/26-27/0412"
            )
        )


if __name__ == "__main__":
    unittest.main()
