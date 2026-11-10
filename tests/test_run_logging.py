"""Offline tests for per-run diagnostic logging and safe demo reset."""

from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import ledger
from approval import get_approval, request_approval
from cli import FourXCLI
from conversation import Conversation, Message, ToolCall
from harness import run_agent_turn
from posting import PostingUncertain
from run_logging import RunLogger, sanitize_text
from storage import ConversationStore, StoredApproval


class FakeModel:
    def __init__(self, responses: list[Message]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.block: threading.Event | None = None
        self.entered: threading.Event | None = None
        self.finished = threading.Event()

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


class RunLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmpdir.cleanup)
        root = Path(self._tmpdir.name)
        self.log_dir = root / "logs"
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
        self.model = FakeModel([Message(role="assistant", content="ok")])
        self.logger = RunLogger(log_dir=self.log_dir, conversation_id="demo")
        self.cli = FourXCLI(
            self.store,
            "demo",
            model_factory=lambda _p: self.model,
            stdout=self.out,
            ledger_db=self.ledger_db,
            run_logger=self.logger,
        )
        self.addCleanup(self._drain)

    def _drain(self) -> None:
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

    def _events(self) -> list[dict]:
        return self.cli.run_logger.read_recent(500)

    def _event_names(self) -> list[str]:
        return [e["event"] for e in self._events()]

    def _make_pending(self):
        with patch("ledger.entries", return_value=[]):
            return request_approval(self.store, "demo", "invoice-01")

    def test_01_unique_run_ids(self) -> None:
        other = RunLogger(log_dir=self.log_dir, conversation_id="demo")
        self.assertNotEqual(self.logger.run_id, other.run_id)
        self.assertNotEqual(self.logger.path, other.path)

    def test_02_session_and_user_input_ordering(self) -> None:
        self.cli.startup_banner()
        self._send("hello clerk")
        names = self._event_names()
        self.assertIn("session_started", names)
        self.assertIn("state_snapshot", names)
        self.assertIn("user_input", names)
        self.assertLess(names.index("session_started"), names.index("user_input"))

    def test_03_tool_call_logging(self) -> None:
        self.model.responses = [
            Message(
                role="assistant",
                tool_calls=(ToolCall(id="t1", name="list_invoices", arguments={}),),
            ),
            Message(role="assistant", content="listed"),
        ]
        self._send("list invoices")
        names = self._event_names()
        self.assertIn("model_call_started", names)
        self.assertIn("model_call_completed", names)
        self.assertIn("tool_requested", names)
        self.assertIn("tool_completed", names)

    def test_04_approval_transition_logging(self) -> None:
        pending = self._make_pending()
        self.cli.handle_line("/approve")
        events = [e for e in self._events() if e["event"] == "approval_approved"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["approval_id"], pending.approval_id)
        self.assertEqual(get_approval(self.store, pending.approval_id).status, "approved")

    def test_05_confirmed_posting_logs_journal(self) -> None:
        pending = self._make_pending()
        self.cli.handle_line("/approve")
        before = {e["entry_id"] for e in ledger.entries()}
        self.cli.handle_line("/post")
        row = get_approval(self.store, pending.approval_id)
        self.assertEqual(row.status, "posted")
        confirmed = [e for e in self._events() if e["event"] == "posting_confirmed"]
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0]["journal_entry_id"], row.journal_entry_id)
        after = {e["entry_id"] for e in ledger.entries()}
        self.assertTrue(after - before)

    def test_06_uncertain_posting_logged(self) -> None:
        pending = self._make_pending()
        self.cli.handle_line("/approve")
        uncertain = StoredApproval(
            approval_id=pending.approval_id,
            conversation_id="demo",
            invoice_id="invoice-01",
            details=dict(pending.details),
            status="posting_uncertain",
        )
        with patch(
            "cli.post_approved",
            side_effect=PostingUncertain(uncertain, "cannot verify"),
        ):
            self.cli.handle_line("/post")
        found = [e for e in self._events() if e["event"] == "posting_uncertain"]
        self.assertTrue(found)
        self.assertEqual(
            get_approval(self.store, pending.approval_id).status, "approved"
        )

    def test_07_cancellation_logged(self) -> None:
        entered = threading.Event()
        block = threading.Event()
        slow = FakeModel([Message(role="assistant", content="late")])
        slow.entered = entered
        slow.block = block
        self.cli.model_factory = lambda _p: slow
        self.cli.handle_line("long running")
        self.assertTrue(entered.wait(5))
        self.cli.handle_line("/stop")
        names = self._event_names()
        self.assertIn("cancellation_requested", names)
        block.set()
        slow.finished.wait(5)
        # Worker may emit cancellation_completed from harness.
        self.assertTrue(
            "cancellation_requested" in names or "cancellation_completed" in names
        )

    def test_08_provider_switch_logged(self) -> None:
        self.cli.handle_line("/provider groq")
        events = [e for e in self._events() if e["event"] == "provider_changed"]
        self.assertEqual(events[-1]["provider"], "groq")
        self.assertEqual(self.store.load("demo").provider, "groq")

    def test_09_state_snapshot_shape(self) -> None:
        snap = self.logger.collect_snapshot(self.store, ledger_db=self.ledger_db)
        self.assertIn("conversation", snap)
        self.assertEqual(snap["conversation"]["conversation_id"], "demo")
        self.assertIn("approvals", snap)
        self.assertIn("ledger", snap)
        self.assertIn("entry_count", snap["ledger"])
        # No invoice line payloads.
        blob = json.dumps(snap)
        self.assertNotIn("subtotal", blob)
        self.assertNotIn("taxes", blob)

    def test_10_safe_demo_reset(self) -> None:
        old_db = Path(self.store.db_path)
        old_ledger = Path(self.ledger_db)
        old_log = Path(self.logger.path)
        old_run = self.logger.run_id
        self.cli.handle_line("remember me")
        self.cli._wait_turn(30)
        # Write a marker into old log
        self.logger.emit("user_input", user_text="marker-before-reset")
        self.assertTrue(old_db.is_file())
        self.assertTrue(old_ledger.is_file())
        self.assertTrue(old_log.is_file())

        import os

        cwd = Path.cwd()
        try:
            os.chdir(self._tmpdir.name)
            self.cli.handle_line("/reset-demo")
        finally:
            os.chdir(cwd)

        self.assertTrue(old_db.is_file(), "old conversation DB must remain")
        self.assertTrue(old_ledger.is_file(), "old ledger DB must remain")
        self.assertTrue(old_log.is_file(), "old log must remain")
        old_text = old_log.read_text(encoding="utf-8")
        self.assertIn("user_text_redacted", old_text)
        self.assertNotIn("marker-before-reset", old_text)

        self.assertNotEqual(self.cli.conversation_id, "demo")
        self.assertNotEqual(self.cli.run_logger.run_id, old_run)
        self.assertEqual(self.cli.run_logger.previous_run_id, old_run)
        self.assertNotEqual(Path(self.cli.store.db_path), old_db)
        self.assertTrue(Path(self.cli.store.db_path).is_file())
        names = self.cli.run_logger.read_recent(50)
        ev = [e["event"] for e in names]
        self.assertIn("reset_completed", ev)
        self.assertIn("session_started", ev)

    def test_11_no_api_key_leakage(self) -> None:
        dirty = "GEMINI_API_KEY=AIzaSyFakeSecretValue123456 and gsk_abcdef1234567890"
        cleaned = sanitize_text(dirty)
        self.assertNotIn("AIzaSyFakeSecretValue123456", cleaned)
        self.assertNotIn("gsk_abcdef1234567890", cleaned)
        self.logger.emit("user_input", user_text=dirty)
        self.logger.emit(
            "model_call_failed",
            error=dirty,
            nested_secret={"api_key": "gsk_abcdefghijklmnop1234"},
        )
        # Unexpected / nested fields must not leak secrets.
        self.logger.emit(
            "tool_failed",
            tool_name="x",
            tool_call_id="1",
            error="ok",
            dump={"taxes": {"CGST": "1"}, "authorization": "Bearer secret"},
        )
        raw = self.logger.path.read_text(encoding="utf-8")
        self.assertNotIn("AIzaSyFakeSecretValue123456", raw)
        self.assertNotIn("gsk_abcdef1234567890", raw)
        self.assertNotIn("gsk_abcdefghijklmnop1234", raw)
        self.assertNotIn("Bearer secret", raw)
        self.assertNotIn("CGST", raw)
        self.assertNotIn(dirty, raw)

    def test_12_no_raw_invoice_payload_in_tool_logs(self) -> None:
        self.model.responses = [
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="r1",
                        name="read_invoice",
                        arguments={"invoice_id": "invoice-01"},
                    ),
                ),
            ),
            Message(role="assistant", content="done"),
        ]
        self._send("read invoice-01")
        raw = self.logger.path.read_text(encoding="utf-8")
        self.assertNotIn("CGST", raw)
        self.assertNotIn("line_items", raw)
        self.assertIn("tool_requested", raw)
        self.assertIn("invoice-01", raw)

    def test_13_log_persists_across_logger_reopen(self) -> None:
        self.logger.emit("session_started", provider="gemini")
        path = self.logger.path
        run_id = self.logger.run_id
        reopened = RunLogger(
            log_dir=self.log_dir, conversation_id="demo", run_id=run_id
        )
        reopened.emit("session_ended", reason="test")
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        self.assertGreaterEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["event"], "session_started")
        self.assertEqual(json.loads(lines[-1])["event"], "session_ended")

    def test_14_log_isolation_across_runs(self) -> None:
        self.logger.emit("user_input", user_text="run-a-unique-marker")
        other = RunLogger(log_dir=self.log_dir, conversation_id="demo")
        other.emit("user_input", user_text="run-b-unique-marker")
        a_events = self.logger.read_recent(20)
        b_events = other.read_recent(20)
        self.assertEqual(a_events[-1]["run_id"], self.logger.run_id)
        self.assertEqual(b_events[-1]["run_id"], other.run_id)
        self.assertNotEqual(self.logger.run_id, other.run_id)
        # Verbatim user text must not appear in either log.
        self.assertNotIn("run-a-unique-marker", self.logger.path.read_text())
        self.assertNotIn("run-b-unique-marker", other.path.read_text())

    def test_15_reset_refused_while_busy(self) -> None:
        entered = threading.Event()
        block = threading.Event()
        slow = FakeModel([Message(role="assistant", content="late")])
        slow.entered = entered
        slow.block = block
        self.cli.model_factory = lambda _p: slow
        self.cli.handle_line("busy work")
        self.assertTrue(entered.wait(5))
        before = self.cli.conversation_id
        self.cli.handle_line("/reset-demo")
        self.assertEqual(self.cli.conversation_id, before)
        out = self.out.getvalue()
        self.assertTrue(
            "Cannot reset" in out or "Agent is busy" in out,
            out,
        )
        block.set()
        slow.finished.wait(5)

    def test_16_harness_logs_approval_created(self) -> None:
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
        stored = self.store.load("demo")
        stored.conversation.add(Message(role="user", content="post invoice-01"))
        self.store.save("demo", stored.conversation)
        with patch("ledger.entries", return_value=[]):
            from harness import ApprovalRequired

            with self.assertRaises(ApprovalRequired):
                run_agent_turn(
                    store=self.store,
                    conversation_id="demo",
                    model=self.model,
                    run_logger=self.logger,
                )
        names = self._event_names()
        self.assertIn("approval_created", names)
        self.assertIn("tool_requested", names)

    def test_17_reset_refuses_while_superseded_worker_alive(self) -> None:
        """Turn A blocks → /stop → Turn B completes → /reset-demo must refuse."""
        entered_a = threading.Event()
        block_a = threading.Event()
        model_a = FakeModel([Message(role="assistant", content="late-a")])
        model_a.entered = entered_a
        model_a.block = block_a

        self.cli.model_factory = lambda _p: model_a
        self.cli.handle_line("turn A")
        self.assertTrue(entered_a.wait(5))
        self.cli.handle_line("/stop")
        self.assertFalse(self.cli._blocks_new_turn())

        model_b = FakeModel([Message(role="assistant", content="done-b")])
        self.cli.model_factory = lambda _p: model_b
        self._send("turn B")
        self.assertFalse(self.cli._blocks_new_turn())
        self.assertTrue(model_a.block is not None and not model_a.finished.is_set())

        before = self.cli.conversation_id
        before_ledger = Path(self.cli.ledger_db)
        import os

        cwd = Path.cwd()
        try:
            os.chdir(self._tmpdir.name)
            self.cli.handle_line("/reset-demo")
        finally:
            os.chdir(cwd)

        self.assertEqual(self.cli.conversation_id, before)
        self.assertEqual(Path(self.cli.ledger_db), before_ledger)
        self.assertIn("still alive", self.out.getvalue())

        block_a.set()
        model_a.finished.wait(5)
        model_a.block = None
        # After A exits, reset should succeed.
        try:
            os.chdir(self._tmpdir.name)
            self.cli.handle_line("/reset-demo")
        finally:
            os.chdir(cwd)
        self.assertNotEqual(self.cli.conversation_id, before)

    def test_18_snapshot_and_eligibility_do_not_cross_ledgers(self) -> None:
        """Concurrent snapshot path swap must not poison eligibility ledger reads."""
        import ledger_guard
        from invoice_tools import prepare_post

        other_ledger = Path(self._tmpdir.name) / "other-ledger.db"
        with ledger_guard.using_ledger(other_ledger):
            ledger.entries()

        seen_paths: list[str] = []
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []
        eligibility_tid = {"id": 0}
        real_entries = ledger.entries

        def tracking_entries(**filters: str):
            if threading.get_ident() == eligibility_tid["id"]:
                seen_paths.append(str(Path(ledger.DB_PATH).resolve()))
            return real_entries(**filters)

        def snapshot_worker() -> None:
            try:
                barrier.wait(timeout=5)
                for _ in range(30):
                    self.logger.collect_snapshot(
                        self.store, ledger_db=other_ledger, note="race"
                    )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def eligibility_worker() -> None:
            eligibility_tid["id"] = threading.get_ident()
            try:
                barrier.wait(timeout=5)
                with patch.object(ledger, "entries", side_effect=tracking_entries):
                    for _ in range(30):
                        try:
                            prepare_post("invoice-01")
                        except Exception as exc:  # noqa: BLE001
                            if type(exc).__name__ not in {
                                "DuplicateInvoice",
                                "BlockedVendor",
                            }:
                                raise
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        ledger.DB_PATH = self.ledger_db
        t1 = threading.Thread(target=snapshot_worker)
        t2 = threading.Thread(target=eligibility_worker)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)
        self.assertEqual(errors, [])
        self.assertTrue(seen_paths)
        session = str(Path(self.ledger_db).resolve())
        for path in seen_paths:
            self.assertEqual(path, session)
        self.assertEqual(str(Path(ledger.DB_PATH).resolve()), session)

    def test_19_reset_rollback_on_prepare_failure(self) -> None:
        before_conv = self.cli.conversation_id
        before_store = self.cli.store
        before_ledger = Path(self.cli.ledger_db)
        before_logger = self.cli.run_logger
        before_global = Path(ledger.DB_PATH)

        import os

        cwd = Path.cwd()
        try:
            os.chdir(self._tmpdir.name)
            with patch(
                "storage.ConversationStore.create",
                side_effect=RuntimeError("boom-create"),
            ):
                self.cli.handle_line("/reset-demo")
        finally:
            os.chdir(cwd)

        self.assertEqual(self.cli.conversation_id, before_conv)
        self.assertIs(self.cli.store, before_store)
        self.assertEqual(Path(self.cli.ledger_db), before_ledger)
        self.assertIs(self.cli.run_logger, before_logger)
        self.assertEqual(Path(ledger.DB_PATH).resolve(), before_global.resolve())
        self.assertIn("session unchanged", self.out.getvalue().lower())

    def test_20_recursive_sanitization_nested_secrets(self) -> None:
        from run_logging import sanitize_value

        payload = {
            "ok": True,
            "nested": {
                "authorization": "Bearer gsk_abcdefghijklmnop1234",
                "list": ["GEMINI_API_KEY=abc123456789", {"api_key": "x"}],
            },
            "taxes": {"CGST": "10"},
        }
        cleaned = sanitize_value(payload)
        blob = json.dumps(cleaned)
        self.assertNotIn("gsk_abcdefghijklmnop1234", blob)
        self.assertNotIn("abc123456789", blob)
        self.assertIn("REDACTED", blob)


if __name__ == "__main__":
    unittest.main()

