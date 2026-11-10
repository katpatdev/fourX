"""Offline regressions for NL UX, state awareness, ledger summary, stale replies."""

from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import ledger
from approval import get_approval, get_pending_approval, request_approval
from cli import FourXCLI
from conversation import Conversation, Message, ToolCall
from harness import run_agent_turn, tool_declarations, unmatched_tool_call_ids
from invoice_tools import ledger_summary
from nl_routing import (
    match_deterministic_action,
    user_message_authorizes_approve,
    user_message_authorizes_post,
    user_message_authorizes_provider_switch,
)
from providers.gemini import DEFAULT_GEMINI_MODEL, GeminiModel
from storage import ConversationStore


class FakeModel:
    def __init__(self, responses: list[Message]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.block: threading.Event | None = None
        self.entered: threading.Event | None = None
        self.seen_conversations: list[Conversation] = []
        self.finished = threading.Event()
        self.printed_late = threading.Event()

    def __call__(self, conversation: Conversation, tools: list[dict]) -> Message:
        self.calls += 1
        self.seen_conversations.append(conversation)
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


class NlUxRegressionTests(unittest.TestCase):
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
        self.model = FakeModel([Message(role="assistant", content="ok")])
        self.cli = FourXCLI(
            self.store,
            "demo",
            model_factory=lambda _p: self.model,
            stdout=self.out,
            ledger_db=self.ledger_db,
        )
        self.addCleanup(self._drain_workers)

    def _drain_workers(self) -> None:
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

    def _make_pending(self, invoice_id: str = "invoice-01"):
        with patch("ledger.entries", return_value=[]):
            return request_approval(self.store, "demo", invoice_id)

    # --- 1–4 provider switching ---

    def test_01_nl_gemini_switch(self) -> None:
        self.store.set_provider("demo", "groq")
        self.cli.handle_line("Switch to Gemini")
        self.assertEqual(self.store.load("demo").provider, "gemini")
        self.assertEqual(self.model.calls, 0)

    def test_02_polite_question_form_switch(self) -> None:
        self.assertIsNotNone(
            match_deterministic_action("Can we switch to the Gemini model?")
        )
        self.cli.handle_line("Can we switch to the Gemini model?")
        self.assertEqual(self.store.load("demo").provider, "gemini")
        self.cli.handle_line("I want to use Groq")
        self.assertEqual(self.store.load("demo").provider, "groq")
        self.cli.handle_line("Change the model to Gemini")
        self.assertEqual(self.store.load("demo").provider, "gemini")
        self.cli.handle_line("Let's continue with Groq")
        self.assertEqual(self.store.load("demo").provider, "groq")
        self.assertEqual(self.model.calls, 0)

    def test_03_asking_about_provider_does_not_switch(self) -> None:
        self.store.set_provider("demo", "groq")
        self.assertIsNone(match_deterministic_action("What is Gemini?"))
        self.assertIsNone(match_deterministic_action("Are we already using Groq?"))
        self.assertFalse(
            user_message_authorizes_provider_switch("What is Gemini?", "gemini")
        )
        self.assertFalse(
            user_message_authorizes_provider_switch(
                "Are we already using Groq?", "groq"
            )
        )
        self.model.responses = [
            Message(role="assistant", content="Gemini is a Google model."),
        ]
        before = self.store.load("demo").provider
        self._send("What is Gemini?")
        self.assertEqual(self.store.load("demo").provider, before)

    def test_04_switch_to_other_provider(self) -> None:
        self.assertEqual(self.store.load("demo").provider, "gemini")
        routed = match_deterministic_action(
            "Switch to the other provider", current_provider="gemini"
        )
        self.assertIsNotNone(routed)
        assert routed is not None
        self.assertEqual(routed.provider, "groq")
        self.cli.handle_line("Switch to the other provider")
        self.assertEqual(self.store.load("demo").provider, "groq")
        self.cli.handle_line("Switch to the other provider")
        self.assertEqual(self.store.load("demo").provider, "gemini")

    # --- 5–9 approve / post ---

    def test_05_i_want_to_approve_this(self) -> None:
        pending = self._make_pending("invoice-02")
        self.assertTrue(user_message_authorizes_approve("I want to approve this"))
        self.assertIsNotNone(match_deterministic_action("I want to approve this"))
        before = len(ledger.entries())
        self.cli.handle_line("I want to approve this")
        row = get_approval(self.store, pending.approval_id)
        self.assertEqual(row.status, "posted")
        self.assertIsNotNone(row.journal_entry_id)
        self.assertEqual(len(ledger.entries()), before + 1)

    def test_06_ambiguous_approval_blocked(self) -> None:
        self._make_pending("invoice-01")
        # Second pending is impossible (one pending max); use vague phrase instead.
        self.assertFalse(user_message_authorizes_approve("Looks good"))
        self.assertIsNone(match_deterministic_action("Looks good"))
        self.model.responses = [
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(id="a1", name="approve_pending_approval", arguments={}),
                ),
            ),
            Message(role="assistant", content="Need clearer approval."),
        ]
        pending = get_pending_approval(self.store, "demo")
        assert pending is not None
        self._send("Looks good")
        self.assertEqual(
            get_approval(self.store, pending.approval_id).status, "pending"
        )

    def test_07_post_this_invoice_succeeds(self) -> None:
        from approval import approve as record_approve

        pending = self._make_pending("invoice-02")
        record_approve(self.store, "demo", pending.approval_id)
        self.assertTrue(user_message_authorizes_post("Post this invoice"))
        self.assertIsNotNone(match_deterministic_action("Post this invoice"))
        before = len(ledger.entries())
        self.cli.handle_line("Post this invoice")
        row = get_approval(self.store, pending.approval_id)
        self.assertEqual(row.status, "posted")
        self.assertIsNotNone(row.journal_entry_id)
        self.assertEqual(len(ledger.entries()), before + 1)

    def test_08_posting_without_approval_blocked(self) -> None:
        pending = self._make_pending("invoice-03")
        before = len(ledger.entries())
        self.cli.handle_line("Post this invoice")
        self.assertEqual(
            get_approval(self.store, pending.approval_id).status, "pending"
        )
        self.assertEqual(len(ledger.entries()), before)

    def test_09_duplicate_posting_blocked(self) -> None:
        # invoice-05 matches opening JE-000001 (BL-88190).
        self.model.responses = [
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="p5",
                        name="post_invoice",
                        arguments={"invoice_id": "invoice-05"},
                    ),
                ),
            ),
        ]
        before = len(ledger.entries())
        self._send("post invoice-05")
        self.assertIsNone(get_pending_approval(self.store, "demo"))
        self.assertEqual(len(ledger.entries()), before)

    # --- 10–12 state / history ---

    def test_10_approval_and_posting_survive_restart(self) -> None:
        pending = self._make_pending("invoice-01")
        self.cli.handle_line("I want to approve this")
        journal = get_approval(self.store, pending.approval_id).journal_entry_id
        self.assertIsNotNone(journal)

        reopened = ConversationStore(self.conv_db)
        row = get_approval(reopened, pending.approval_id)
        self.assertEqual(row.status, "posted")
        self.assertEqual(row.journal_entry_id, journal)
        out2 = io.StringIO()
        cli2 = FourXCLI(
            reopened,
            "demo",
            model_factory=lambda _p: FakeModel(
                [Message(role="assistant", content="resumed")]
            ),
            stdout=out2,
            ledger_db=self.ledger_db,
        )
        cli2.handle_line("/status")
        self.assertIn("posted", out2.getvalue().lower())

    def test_11_local_approval_visible_to_subsequent_model(self) -> None:
        pending = self._make_pending("invoice-02")
        self.cli.handle_line("I want to approve this")
        events = [
            m
            for m in self.store.load("demo").conversation.messages
            if (m.extras or {}).get("application_event")
        ]
        self.assertTrue(events)
        self.assertIn("posted", (events[-1].content or "").lower())

        self.model.responses = [
            Message(role="assistant", content="It was approved and posted."),
        ]
        self._send("has the invoice been approved?")
        self.assertGreaterEqual(self.model.calls, 1)
        view = self.model.seen_conversations[-1]
        blob = " ".join(m.content or "" for m in view.messages)
        self.assertIn("trusted_application_state", blob)
        self.assertIn(pending.approval_id, blob)
        self.assertIn("posted", blob)

    def test_12_no_malformed_canonical_tool_history(self) -> None:
        self.model.responses = [
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(id="l1", name="list_invoices", arguments={}),
                ),
            ),
            Message(role="assistant", content="listed"),
        ]
        self._send("list invoices")
        conv = self.store.load("demo").conversation
        self.assertEqual(unmatched_tool_call_ids(conv), [])
        # Trusted state must not be persisted into durable history.
        for message in conv.messages:
            self.assertNotIn(
                "trusted_application_state",
                message.content or "",
            )

    # --- 13–15 ledger summary ---

    def test_13_ledger_count_equals_entries(self) -> None:
        summary = ledger_summary()
        self.assertEqual(summary["entry_count"], len(ledger.entries()))

    def test_14_initial_ledger_count_is_one(self) -> None:
        summary = ledger_summary()
        self.assertEqual(summary["entry_count"], 1)
        self.assertEqual(summary["entry_ids"][0], "JE-000001")

    def test_15_ledger_count_two_after_one_post(self) -> None:
        pending = self._make_pending("invoice-02")
        self.cli.handle_line("/approve")
        self.assertEqual(
            get_approval(self.store, pending.approval_id).status, "posted"
        )
        summary = ledger_summary()
        self.assertEqual(summary["entry_count"], 2)
        self.assertEqual(len(ledger.entries()), 2)

        self.model.responses = [
            Message(
                role="assistant",
                tool_calls=(ToolCall(id="ls1", name="ledger_summary", arguments={}),),
            ),
            Message(role="assistant", content="There are 2 journal entries."),
        ]
        names = {d["name"] for d in tool_declarations()}
        self.assertIn("ledger_summary", names)
        self._send("how many ledger entries are there?")
        self.assertIn("2", self.out.getvalue())

    # --- 16 stale response ---

    def test_16_cancelled_output_cannot_answer_newer_request(self) -> None:
        entered = threading.Event()
        block = threading.Event()
        slow = FakeModel([Message(role="assistant", content="LATE_PROVIDER_ANSWER")])
        slow.entered = entered
        slow.block = block
        self.cli.model_factory = lambda _p: slow
        self.cli.handle_line("switch topic please wait")
        self.assertTrue(entered.wait(5))
        self.cli.handle_line("/stop")
        follow = FakeModel(
            [Message(role="assistant", content="FRESH_INVOICE_ANSWER")]
        )
        self.cli.model_factory = lambda _p: follow
        self._send("what is on invoice-02?")
        block.set()
        slow.finished.wait(5)
        # Give the late worker a moment to attempt printing.
        slow.finished.wait(1)
        text = self.out.getvalue()
        self.assertIn("FRESH_INVOICE_ANSWER", text)
        self.assertNotIn("LATE_PROVIDER_ANSWER", text)

    # --- 17 Gemini model config ---

    def test_17_gemini_default_model_is_flash_lite(self) -> None:
        self.assertEqual(DEFAULT_GEMINI_MODEL, "gemini-3.5-flash-lite")
        from unittest.mock import MagicMock

        client = MagicMock()
        model = GeminiModel(api_key="test", client=client)
        self.assertEqual(model.model, "gemini-3.5-flash-lite")
        custom = GeminiModel(api_key="test", model="gemini-custom", client=client)
        self.assertEqual(custom.model, "gemini-custom")
        # Configured model is used on generate_content
        from providers.gemini import gemini_tools_from_declarations

        part = MagicMock()
        part.function_call = None
        part.text = "hello"
        content = MagicMock()
        content.parts = [part]
        content.model_dump.return_value = {"role": "model", "parts": []}
        candidate = MagicMock()
        candidate.content = content
        response = MagicMock()
        response.candidates = [candidate]
        client.models.generate_content.return_value = response
        conv = Conversation()
        conv.add(Message(role="user", content="hi"))
        out = model(conv, tool_declarations())
        self.assertEqual(out.content, "hello")
        kwargs = client.models.generate_content.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-3.5-flash-lite")
        _ = gemini_tools_from_declarations  # import retained for adapter surface


class ModeBPhraseUnitTests(unittest.TestCase):
    def test_polite_approve_and_post_phrases(self) -> None:
        for text in (
            "I want to approve this",
            "Yes, I approve this invoice",
            "Please approve it",
            "I approve the second invoice",
        ):
            self.assertTrue(user_message_authorizes_approve(text), text)
            self.assertIsNotNone(match_deterministic_action(text), text)

        for text in (
            "Post this invoice",
            "Okay, post this invoice now",
            "Please post it to the ledger",
            "Go ahead and post the approved invoice",
        ):
            self.assertTrue(user_message_authorizes_post(text), text)
            self.assertIsNotNone(match_deterministic_action(text), text)


if __name__ == "__main__":
    unittest.main()
