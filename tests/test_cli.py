"""Offline CLI command and concurrency tests (fake models, temp DBs)."""

from __future__ import annotations

import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import ledger
from fourx.cli import (
    FourXCLI,
    append_user_message,
    format_approval,
    load_dotenv,
    reconcile_trailing_cancelled_tools,
)
from fourx.agent.conversation import Conversation, Message, ToolCall
from fourx.agent.harness import RecoveryRequired, unmatched_tool_call_ids
from fourx.finance.posting import approve_and_post, post_approved
from fourx.persistence.storage import ConversationStore


class FakeModel:
    def __init__(self, responses: list[Message]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.block: threading.Event | None = None
        self.entered: threading.Event | None = None
        self.finished: threading.Event = threading.Event()

    def __call__(self, conversation: Conversation, tools: list[dict]) -> Message:
        self.calls += 1
        if self.entered is not None:
            self.entered.set()
        try:
            if self.block is not None:
                self.block.wait(timeout=30)
            if not self.responses:
                raise AssertionError("FakeModel exhausted")
            return self.responses.pop(0)
        finally:
            self.finished.set()


class CLITests(unittest.TestCase):
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
        self.model = FakeModel(
            [Message(role="assistant", content="hello from fake model")]
        )
        self.cli = FourXCLI(
            self.store,
            "demo",
            model_factory=lambda _p: self.model,
            stdout=self.out,
            ledger_db=self.ledger_db,
        )
        # Drain before TemporaryDirectory cleanup (addCleanup is LIFO).
        self.addCleanup(self._drain_workers)

    def _drain_workers(self) -> None:
        """Ensure daemon turn threads release SQLite before the temp DB is removed."""
        with self.cli._worker_lock:
            worker = self.cli._active
        if worker is None:
            return
        worker.token.cancel()
        worker.released = True
        worker.done.wait(timeout=10)
        worker.thread.join(timeout=2)

    def _send(self, line: str) -> None:
        self.cli.handle_line(line)
        self.cli._wait_turn(timeout=30)

    def test_01_starts_without_live_keys(self) -> None:
        self.cli.startup_banner()
        self.assertIn("Provider: gemini", self.out.getvalue())
        self.assertIn(str(self.ledger_db), self.out.getvalue())

    def test_02_user_messages_persisted(self) -> None:
        self._send("What invoices are available?")
        messages = self.store.load("demo").conversation.messages
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[0].content, "What invoices are available?")
        self.assertEqual(messages[-1].role, "assistant")

    def test_03_provider_switching_persists(self) -> None:
        self.cli.handle_line("/provider groq")
        self.assertEqual(self.store.load("demo").provider, "groq")
        self.cli.handle_line("/provider gemini")
        self.assertEqual(self.store.load("demo").provider, "gemini")

    def test_04_pending_approval_display_exact_details(self) -> None:
        self.model.responses = [
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
        with patch("ledger.entries", return_value=[]):
            self._send("Post invoice-01.")
        text = self.out.getvalue()
        self.assertIn("Approval required", text)
        self.assertIn("16862.20", text)
        self.assertIn("V-001", text)
        self.assertIn("SGP/26-27/0412", text)

    def test_05_approve_posts_via_safe_workflow(self) -> None:
        self.model.responses = [
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
        with patch("ledger.entries", return_value=[]):
            self._send("post invoice-01")
        before = len(ledger.entries())
        self.cli.handle_line("/approve")
        self.assertIn("Approved and posted", self.out.getvalue())
        rows = ledger.entries(invoice_number="SGP/26-27/0412")
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(ledger.entries()), before + 1)

    def test_06_decline_persists_reason(self) -> None:
        self.model.responses = [
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="p1",
                        name="post_invoice",
                        arguments={"invoice_id": "invoice-03"},
                    ),
                ),
            )
        ]
        with patch("ledger.entries", return_value=[]):
            self._send("post invoice-03")
        self.cli.handle_line("/decline PO not received yet")
        history = self.store.list_approvals("demo")
        self.assertEqual(history[0].status, "declined")
        self.assertEqual(history[0].decline_reason, "PO not received yet")

    def test_07_post_calls_only_post_approved(self) -> None:
        self.model.responses = [
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
        with patch("ledger.entries", return_value=[]):
            self._send("post invoice-01")
        with patch("fourx.cli.approve_and_post", wraps=approve_and_post) as mock_aap:
            with patch("ledger.post", wraps=ledger.post) as mock_post:
                self.cli.handle_line("/approve")
                mock_aap.assert_called_once()
                self.assertEqual(mock_post.call_count, 1)
        self.assertIn("Approved and posted", self.out.getvalue())
        # Leftover /post path stays available and idempotent.
        with patch("fourx.cli.post_approved", wraps=post_approved) as mock_pa:
            self.cli.handle_line("/post")
            mock_pa.assert_called_once()

    def test_08_restart_resumes_pending(self) -> None:
        self.model.responses = [
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="p1",
                        name="post_invoice",
                        arguments={"invoice_id": "invoice-02"},
                    ),
                ),
            )
        ]
        with patch("ledger.entries", return_value=[]):
            self._send("post invoice-02")
        reopened = ConversationStore(self.conv_db)
        out2 = io.StringIO()
        cli2 = FourXCLI(
            reopened,
            "demo",
            model_factory=lambda _p: FakeModel([]),
            stdout=out2,
            ledger_db=self.ledger_db,
        )
        cli2.startup_banner()
        self.assertIn("Pending approval", out2.getvalue())
        self.assertIn("BL-88213", out2.getvalue())
        self.assertIn("70800.00", out2.getvalue())

    def test_09_reconcile_never_posts(self) -> None:
        details = {
            "invoice_id": "invoice-01",
            "vendor_id": "V-001",
            "vendor_name": "Shree Ganesh Packaging Pvt Ltd",
            "invoice_number": "SGP/26-27/0412",
            "invoice_date": "2026-10-03",
            "currency": "INR",
            "amount": "16862.20",
        }
        row = self.store.create_pending_approval("demo", details)
        self.store.set_approval_decision(row.approval_id, status="approved")
        self.store.claim_for_posting(row.approval_id, "demo")
        with patch("ledger.post") as mock_post:
            self.cli.handle_line(f"/reconcile {row.approval_id}")
            mock_post.assert_not_called()
        self.assertIn("unresolved", self.out.getvalue().lower())

    def test_10_blocked_and_duplicate_cannot_reach_posting(self) -> None:
        # Blocked invoice-04 via model tool path
        self.model.responses = [
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="b1",
                        name="post_invoice",
                        arguments={"invoice_id": "invoice-04"},
                    ),
                ),
            ),
            Message(role="assistant", content="Vendor blocked."),
        ]
        with patch("fourx.tools.invoice_tools.ledger.post") as mock_post:
            self._send("post invoice-04")
            mock_post.assert_not_called()
        self.assertEqual(self.store.list_approvals("demo"), [])

        # Duplicate invoice-05
        self.model.responses = [
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="d1",
                        name="post_invoice",
                        arguments={"invoice_id": "invoice-05"},
                    ),
                ),
            ),
            Message(role="assistant", content="Duplicate."),
        ]
        with patch("fourx.tools.invoice_tools.ledger.post") as mock_post:
            self._send("post invoice-05")
            mock_post.assert_not_called()

    def test_11_stop_responsive_while_model_blocks(self) -> None:
        entered = threading.Event()
        block = threading.Event()
        self.model = FakeModel(
            [Message(role="assistant", content="summary done")]
        )
        self.model.entered = entered
        self.model.block = block
        self.cli.model_factory = lambda _p: self.model

        self.cli.handle_line("summarize all invoices")
        self.assertTrue(entered.wait(timeout=5))
        self.assertTrue(self.cli._blocks_new_turn())
        self.cli.handle_line("/stop")
        self.assertIn("Stop requested", self.out.getvalue())
        self.assertIn("another question", self.out.getvalue().lower())
        self.assertFalse(self.cli._blocks_new_turn())
        # Old model still blocked — input must already be free.
        self.assertTrue(block.is_set() is False)
        block.set()
        self.cli.wait_for_current_turn(timeout=10)

    def test_12_subsequent_message_works_while_old_still_blocked(self) -> None:
        entered = threading.Event()
        block = threading.Event()
        model_a = FakeModel([Message(role="assistant", content="stale-summary")])
        model_a.entered = entered
        model_a.block = block
        self.cli.model_factory = lambda _p: model_a
        self.cli.handle_line("summarize all invoices")
        self.assertTrue(entered.wait(timeout=5))
        self.cli.handle_line("/stop")

        model_b = FakeModel([Message(role="assistant", content="next answer")])
        self.cli.model_factory = lambda _p: model_b
        # B starts while A remains blocked.
        self.cli.handle_line("what has been processed?")
        self.cli.wait_for_current_turn(timeout=10)
        self.assertIn("next answer", self.out.getvalue())
        block.set()
        model_a.block.wait(timeout=1)  # ensure A can finish
        # Give A a moment to unwind after release.
        model_a.entered.wait(timeout=0.1)

    def test_13_cancelled_old_run_cannot_overwrite_new_history(self) -> None:
        entered_a = threading.Event()
        release_a = threading.Event()
        model_a = FakeModel([Message(role="assistant", content="stale-A")])
        model_a.entered = entered_a
        model_a.block = release_a

        self.cli.model_factory = lambda _p: model_a
        self.cli.handle_line("long summary")
        self.assertTrue(entered_a.wait(timeout=5))
        self.cli.handle_line("/stop")

        model_b = FakeModel([Message(role="assistant", content="fresh-B")])
        self.cli.model_factory = lambda _p: model_b
        self.cli.handle_line("follow up")
        self.cli.wait_for_current_turn(timeout=10)
        user_texts = [
            m.content
            for m in self.store.load("demo").conversation.messages
            if m.role == "user"
        ]
        self.assertEqual(user_texts[-1], "follow up")
        texts = [
            m.content
            for m in self.store.load("demo").conversation.messages
            if m.role == "assistant"
        ]
        self.assertIn("fresh-B", texts)
        self.assertEqual(texts[-1], "fresh-B")
        release_a.set()
        # After A returns, history must still end with B.
        for _ in range(50):
            if not model_a.block.is_set():
                break
        texts_after = [
            m.content
            for m in self.store.load("demo").conversation.messages
            if m.role == "assistant"
        ]
        self.assertEqual(texts_after[-1], "fresh-B")
        self.assertNotEqual(texts_after[-1], "stale-A")

    def test_14_trailing_tools_reconciled_before_next_turn(self) -> None:
        loaded = self.store.load("demo")
        loaded.conversation.add(
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(id="x1", name="list_invoices", arguments={}),
                ),
            )
        )
        self.store.save("demo", loaded.conversation)
        added = reconcile_trailing_cancelled_tools(self.store, "demo")
        self.assertEqual(added, 1)
        conv = self.store.load("demo").conversation
        self.assertEqual(unmatched_tool_call_ids(conv), [])
        self.assertEqual(conv.messages[-1].role, "tool")
        self.assertIn("SkippedForCancellation", conv.messages[-1].content)

        self.model = FakeModel([Message(role="assistant", content="reconciled ok")])
        self.cli.model_factory = lambda _p: self.model
        self._send("continue after skip")
        self.assertIn("reconciled ok", self.out.getvalue())
        self.assertEqual(self.model.calls, 1)

    def test_14b_ambiguous_incomplete_does_not_reach_provider(self) -> None:
        loaded = self.store.load("demo")
        loaded.conversation.add(
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(id="x1", name="list_invoices", arguments={}),
                ),
            )
        )
        loaded.conversation.add(Message(role="user", content="orphan after incomplete"))
        self.store.save("demo", loaded.conversation)
        self.model = FakeModel([Message(role="assistant", content="should not run")])
        self.cli.model_factory = lambda _p: self.model
        self.cli.handle_line("try again")
        self.assertIn("Recovery required", self.out.getvalue())
        self.assertEqual(self.model.calls, 0)
        with self.assertRaises(RecoveryRequired):
            append_user_message(self.store, "demo", "hello")

    def test_15_posting_uncertainty_displayed_without_retry(self) -> None:
        details = {
            "invoice_id": "invoice-01",
            "vendor_id": "V-001",
            "vendor_name": "Shree Ganesh Packaging Pvt Ltd",
            "invoice_number": "SGP/26-27/0412",
            "invoice_date": "2026-10-03",
            "currency": "INR",
            "amount": "16862.20",
        }
        row = self.store.create_pending_approval("demo", details)
        self.store.set_approval_decision(row.approval_id, status="approved")
        self.store.claim_for_posting(row.approval_id, "demo")
        with patch("ledger.post") as mock_post:
            self.cli.handle_line("/post")
            mock_post.assert_not_called()
        self.assertIn("uncertain", self.out.getvalue().lower())

    def test_16_old_worker_finally_cannot_clear_new_active(self) -> None:
        entered_a = threading.Event()
        release_a = threading.Event()
        model_a = FakeModel([Message(role="assistant", content="stale-A")])
        model_a.entered = entered_a
        model_a.block = release_a
        self.cli.model_factory = lambda _p: model_a
        self.cli.handle_line("first")
        self.assertTrue(entered_a.wait(timeout=5))
        self.cli.handle_line("/stop")

        entered_b = threading.Event()
        release_b = threading.Event()
        model_b = FakeModel([Message(role="assistant", content="fresh-B")])
        model_b.entered = entered_b
        model_b.block = release_b
        self.cli.model_factory = lambda _p: model_b
        self.cli.handle_line("second")
        self.assertTrue(entered_b.wait(timeout=5))
        with self.cli._worker_lock:
            worker_b = self.cli._active
        self.assertIsNotNone(worker_b)

        release_a.set()
        self.assertTrue(model_a.finished.wait(timeout=5))
        with self.cli._worker_lock:
            self.assertIs(self.cli._active, worker_b)
            self.assertTrue(self.cli._active.thread.is_alive())

        release_b.set()
        self.cli.wait_for_current_turn(timeout=10)
        self.assertIn("fresh-B", self.out.getvalue())
        self.assertNotIn("stale-A", self.out.getvalue())

    def test_17_repeated_stop_is_safe(self) -> None:
        entered = threading.Event()
        block = threading.Event()
        model = FakeModel([Message(role="assistant", content="x")])
        model.entered = entered
        model.block = block
        self.cli.model_factory = lambda _p: model
        self.cli.handle_line("busy work")
        self.assertTrue(entered.wait(timeout=5))
        self.cli.handle_line("/stop")
        self.cli.handle_line("/stop")
        self.cli.handle_line("/stop")
        self.assertFalse(self.cli._blocks_new_turn())
        block.set()
        self.assertTrue(model.finished.wait(timeout=5))
        self.cli.wait_for_current_turn(timeout=10)

    def test_18_two_consecutive_stop_and_new_question_cycles(self) -> None:
        # A blocked -> stop -> B blocked -> stop -> C answers
        entered_a = threading.Event()
        release_a = threading.Event()
        model_a = FakeModel([Message(role="assistant", content="A")])
        model_a.entered = entered_a
        model_a.block = release_a
        self.cli.model_factory = lambda _p: model_a
        self.cli.handle_line("question A")
        self.assertTrue(entered_a.wait(timeout=5))
        self.cli.handle_line("/stop")

        entered_b = threading.Event()
        release_b = threading.Event()
        model_b = FakeModel([Message(role="assistant", content="B")])
        model_b.entered = entered_b
        model_b.block = release_b
        self.cli.model_factory = lambda _p: model_b
        self.cli.handle_line("question B")
        self.assertTrue(entered_b.wait(timeout=5))
        self.cli.handle_line("/stop")

        model_c = FakeModel([Message(role="assistant", content="answer-C")])
        self.cli.model_factory = lambda _p: model_c
        self.cli.handle_line("question C")
        self.cli.wait_for_current_turn(timeout=10)
        self.assertIn("answer-C", self.out.getvalue())

        release_a.set()
        release_b.set()
        self.assertTrue(model_a.finished.wait(timeout=5))
        self.assertTrue(model_b.finished.wait(timeout=5))
        texts = [
            m.content
            for m in self.store.load("demo").conversation.messages
            if m.role == "assistant"
        ]
        self.assertEqual(texts[-1], "answer-C")
        users = [
            m.content
            for m in self.store.load("demo").conversation.messages
            if m.role == "user"
        ]
        self.assertEqual(users, ["question A", "question B", "question C"])

    def test_19_pending_approval_survives_stop_and_restart(self) -> None:
        self.model.responses = [
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
        with patch("ledger.entries", return_value=[]):
            self._send("post invoice-01")
        pending = self.store.list_approvals("demo")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].status, "pending")

        entered = threading.Event()
        block = threading.Event()
        slow = FakeModel([Message(role="assistant", content="should not matter")])
        slow.entered = entered
        slow.block = block
        # Pending blocks NL; clear pending path: stop is N/A for busy.
        # Simulate a second conversation turn after declining? Keep pending and
        # ensure /stop with no active turn + reopen preserves it.
        self.cli.handle_line("/stop")
        block.set()

        reopened = ConversationStore(self.conv_db)
        out2 = io.StringIO()
        cli2 = FourXCLI(
            reopened,
            "demo",
            model_factory=lambda _p: FakeModel([]),
            stdout=out2,
            ledger_db=self.ledger_db,
        )
        cli2.startup_banner()
        self.assertIn("Pending approval", out2.getvalue())
        rows = reopened.list_approvals("demo")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, "pending")
        with patch("ledger.post") as mock_post:
            cli2.handle_line("/stop")
            mock_post.assert_not_called()

    def test_20_stop_never_posts_or_approves(self) -> None:
        entered = threading.Event()
        block = threading.Event()
        model = FakeModel([Message(role="assistant", content="x")])
        model.entered = entered
        model.block = block
        self.cli.model_factory = lambda _p: model
        self.cli.handle_line("anything")
        self.assertTrue(entered.wait(timeout=5))
        with patch("ledger.post") as mock_post:
            with patch("fourx.cli.approve_and_post") as mock_aap:
                with patch("fourx.cli.decline") as mock_decline:
                    with patch("fourx.cli.post_approved") as mock_pa:
                        self.cli.handle_line("/stop")
                        mock_post.assert_not_called()
                        mock_aap.assert_not_called()
                        mock_decline.assert_not_called()
                        mock_pa.assert_not_called()
        block.set()
        self.assertTrue(model.finished.wait(timeout=5))

    def test_21_quit_exits_promptly_with_blocked_worker(self) -> None:
        import time

        entered = threading.Event()
        block = threading.Event()
        model = FakeModel([Message(role="assistant", content="late")])
        model.entered = entered
        model.block = block
        self.cli.model_factory = lambda _p: model
        self.cli.handle_line("slow turn")
        self.assertTrue(entered.wait(timeout=5))
        t0 = time.monotonic()
        cont = self.cli.handle_line("/quit")
        elapsed = time.monotonic() - t0
        self.assertFalse(cont)
        self.assertLess(elapsed, 1.0)
        self.assertIn("Goodbye", self.out.getvalue())
        # Worker still blocked; quit must not have joined it.
        self.assertFalse(model.finished.is_set())
        block.set()
        self.assertTrue(model.finished.wait(timeout=5))

    def test_22_provider_switching_intact_after_stop_cycle(self) -> None:
        entered = threading.Event()
        block = threading.Event()
        model = FakeModel([Message(role="assistant", content="x")])
        model.entered = entered
        model.block = block
        self.cli.model_factory = lambda _p: model
        self.cli.handle_line("work")
        self.assertTrue(entered.wait(timeout=5))
        self.cli.handle_line("/stop")
        block.set()
        self.assertTrue(model.finished.wait(timeout=5))
        self.cli.handle_line("/provider groq")
        self.assertEqual(self.store.load("demo").provider, "groq")
        self.model = FakeModel([Message(role="assistant", content="from groq path")])
        self.cli.model_factory = lambda _p: self.model
        self._send("hello after switch")
        self.assertIn("from groq path", self.out.getvalue())
        self.assertEqual(self.store.load("demo").provider, "groq")

    def test_load_dotenv_does_not_override(self) -> None:
        env_path = Path(self._tmpdir.name) / ".env"
        env_path.write_text("FOURX_TEST_KEY=fromfile\n", encoding="utf-8")
        with patch.dict("os.environ", {"FOURX_TEST_KEY": "existing"}, clear=False):
            load_dotenv(env_path)
            import os

            self.assertEqual(os.environ["FOURX_TEST_KEY"], "existing")

    def test_format_approval_includes_amount(self) -> None:
        from fourx.finance.approval import ApprovalRequest

        req = ApprovalRequest(
            approval_id="apr-x",
            conversation_id="demo",
            invoice_id="invoice-01",
            details={
                "invoice_id": "invoice-01",
                "vendor_id": "V-001",
                "vendor_name": "Shree Ganesh Packaging Pvt Ltd",
                "invoice_number": "SGP/26-27/0412",
                "invoice_date": "2026-10-03",
                "currency": "INR",
                "amount": "16862.20",
            },
            status="pending",
        )
        text = format_approval(req)
        self.assertIn("16862.20", text)


if __name__ == "__main__":
    unittest.main()
