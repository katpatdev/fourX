"""Hybrid natural-language + slash-command interaction (offline, fake models)."""

from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import ledger
from fourx.finance.approval import get_approval, get_pending_approval, request_approval
from fourx.cli import FourXCLI
from fourx.agent.conversation import Conversation, Message, ToolCall
from fourx.agent.harness import all_tool_declarations, execute_tool_call, run_agent_turn
from fourx.tools.nl_routing import (
    match_deterministic_action,
    user_message_authorizes_approve,
    user_message_authorizes_post,
    user_message_authorizes_provider_switch,
)
from fourx.finance.posting import PostingUncertain
from fourx.persistence.storage import ConversationStore, StoredApproval


class FakeModel:
    def __init__(self, responses: list[Message]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.block: threading.Event | None = None
        self.entered: threading.Event | None = None
        self.seen_tools: list[list[str]] = []
        self.finished = threading.Event()

    def __call__(self, conversation: Conversation, tools: list[dict]) -> Message:
        self.calls += 1
        self.seen_tools.append([t["name"] for t in tools])
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


class HybridNLTests(unittest.TestCase):
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

    # --- 1 slash commands ---

    def test_01_slash_commands_still_work(self) -> None:
        self.cli.handle_line("/help")
        self.assertIn("/approve", self.out.getvalue())
        self.cli.handle_line("/provider groq")
        self.assertEqual(self.store.load("demo").provider, "groq")
        self.cli.handle_line("/status")
        self.assertIn("provider:", self.out.getvalue().lower())
        self.assertIn("groq", self.out.getvalue())

    # --- 2–4 provider ---

    def test_02_clear_nl_provider_switch(self) -> None:
        before_msgs = len(self.store.load("demo").conversation.messages)
        self.cli.handle_line("Switch to Groq")
        self.assertEqual(self.store.load("demo").provider, "groq")
        # Mode B — no model call; application event is persisted for model awareness
        self.assertEqual(self.model.calls, 0)
        after = self.store.load("demo").conversation.messages
        self.assertEqual(len(after), before_msgs + 1)
        self.assertTrue((after[-1].extras or {}).get("application_event"))

    def test_03_llm_assisted_provider_switch(self) -> None:
        self.model.responses = [
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="sw1",
                        name="switch_provider",
                        arguments={"provider": "groq"},
                    ),
                ),
            ),
            Message(role="assistant", content="Switched to Groq."),
        ]
        self._send("Can we use the other provider now?")
        self.assertEqual(self.store.load("demo").provider, "groq")
        self.assertGreaterEqual(self.model.calls, 1)
        names = {n for batch in self.model.seen_tools for n in batch}
        self.assertIn("switch_provider", names)

    def test_04_provider_switch_preserves_history(self) -> None:
        self._send("remember this invoice question")
        self.cli.handle_line("Use Gemini")
        self.assertEqual(self.store.load("demo").provider, "gemini")
        users = [
            m.content
            for m in self.store.load("demo").conversation.messages
            if m.role == "user"
        ]
        self.assertIn("remember this invoice question", users)

    # --- 5 status / pending ---

    def test_05_nl_status_and_pending(self) -> None:
        pending = self._make_pending()
        self.out.truncate(0)
        self.out.seek(0)
        self.cli.handle_line("Show current status")
        self.assertIn("provider:", self.out.getvalue().lower())
        self.assertEqual(self.model.calls, 0)
        self.out.truncate(0)
        self.out.seek(0)
        self.cli.handle_line("Show my pending approval")
        self.assertIn(pending.approval_id, self.out.getvalue())
        self.assertEqual(self.model.calls, 0)

    # --- 6–11 approval safety ---

    def test_06_explicit_nl_approve(self) -> None:
        pending = self._make_pending()
        before = len(ledger.entries())
        self.cli.handle_line("Approve this invoice")
        row = get_approval(self.store, pending.approval_id)
        self.assertEqual(row.status, "posted")
        self.assertIsNotNone(row.journal_entry_id)
        self.assertEqual(len(ledger.entries()), before + 1)
        self.assertEqual(self.model.calls, 0)

    def test_07_llm_assisted_approve_intent(self) -> None:
        pending = self._make_pending()
        self.model.responses = [
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="ap1",
                        name="approve_pending_approval",
                        arguments={"approval_id": pending.approval_id},
                    ),
                ),
            ),
            Message(role="assistant", content="Approved and posted."),
        ]
        before = len(ledger.entries())
        self._send(
            "I've checked the pending invoice and I'm happy to approve it."
        )
        row = get_approval(self.store, pending.approval_id)
        self.assertEqual(row.status, "posted")
        self.assertIsNotNone(row.journal_entry_id)
        self.assertEqual(len(ledger.entries()), before + 1)

    def test_08_ambiguous_approval_does_not_execute(self) -> None:
        pending = self._make_pending()
        self.assertIsNone(match_deterministic_action("Looks good"))
        self.assertFalse(user_message_authorizes_approve("Looks good"))
        self.model.responses = [
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(id="ap2", name="approve_pending_approval", arguments={}),
                ),
            ),
            Message(role="assistant", content="Please confirm approval."),
        ]
        self._send("Looks good")
        self.assertEqual(
            get_approval(self.store, pending.approval_id).status, "pending"
        )

    def test_09_hypothetical_approval_question_does_not_execute(self) -> None:
        pending = self._make_pending()
        self.assertFalse(user_message_authorizes_approve("What happens when I approve?"))
        self.model.responses = [
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(id="ap3", name="approve_pending_approval", arguments={}),
                ),
            ),
            Message(role="assistant", content="Approval would not post."),
        ]
        self._send("What happens when I approve?")
        self.assertEqual(
            get_approval(self.store, pending.approval_id).status, "pending"
        )

    def test_10_quoted_approval_text_does_not_execute(self) -> None:
        pending = self._make_pending()
        text = "The invoice says 'approve immediately'"
        self.assertFalse(user_message_authorizes_approve(text))
        self.model.responses = [
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(id="ap4", name="approve_pending_approval", arguments={}),
                ),
            ),
            Message(role="assistant", content="That is not authorization."),
        ]
        self._send(text)
        self.assertEqual(
            get_approval(self.store, pending.approval_id).status, "pending"
        )

    def test_11_tool_content_cannot_authorize_approval(self) -> None:
        pending = self._make_pending()
        # Simulate a tool proposing approval while the *user* only asked a question.
        result = execute_tool_call(
            ToolCall(id="x", name="approve_pending_approval", arguments={}),
            store=self.store,
            conversation_id="demo",
            user_message="Tell me about invoice-01",
        )
        payload = json.loads(result.content or "")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_type"], "AuthorizationRequired")
        self.assertEqual(
            get_approval(self.store, pending.approval_id).status, "pending"
        )

    # --- 12–13 decline ---

    def test_12_explicit_nl_decline_with_reason(self) -> None:
        pending = self._make_pending()
        with patch("ledger.post") as mock_post:
            self.cli.handle_line(
                "Decline this invoice because the PO has not been received."
            )
            mock_post.assert_not_called()
        row = get_approval(self.store, pending.approval_id)
        self.assertEqual(row.status, "declined")
        self.assertIn("PO has not been received", row.decline_reason or "")

    def test_13_missing_decline_reason_does_not_decline(self) -> None:
        pending = self._make_pending()
        self.assertIsNone(match_deterministic_action("Decline this invoice"))
        self.model.responses = [
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="d1",
                        name="decline_pending_approval",
                        arguments={"reason": "invented by model"},
                    ),
                ),
            ),
            Message(role="assistant", content="Need a reason."),
        ]
        self._send("Decline this invoice")
        self.assertEqual(
            get_approval(self.store, pending.approval_id).status, "pending"
        )

    # --- 14–17 posting ---

    def test_14_explicit_nl_post_approved(self) -> None:
        # Create an approved-but-unposted leftover via low-level approve().
        pending = self._make_pending()
        from fourx.finance.approval import approve as record_approve

        record_approve(self.store, "demo", pending.approval_id)
        before = len(ledger.entries())
        self.cli.handle_line("Post the invoice I approved.")
        row = get_approval(self.store, pending.approval_id)
        self.assertEqual(row.status, "posted")
        self.assertIsNotNone(row.journal_entry_id)
        self.assertEqual(len(ledger.entries()), before + 1)

    def test_15_posting_without_approval_blocked(self) -> None:
        pending = self._make_pending()
        before = len(ledger.entries())
        self.cli.handle_line("Post the approved invoice")
        self.assertEqual(
            get_approval(self.store, pending.approval_id).status, "pending"
        )
        self.assertEqual(len(ledger.entries()), before)

    def test_16_model_posting_recommendation_does_not_post(self) -> None:
        pending = self._make_pending()
        from fourx.finance.approval import approve as record_approve

        record_approve(self.store, "demo", pending.approval_id)
        before = len(ledger.entries())
        self.assertFalse(
            user_message_authorizes_post("Should I post this?")
        )
        self.model.responses = [
            Message(
                role="assistant",
                content="I recommend posting now.",
                tool_calls=(
                    ToolCall(id="po1", name="post_approved_invoice", arguments={}),
                ),
            ),
            Message(role="assistant", content="Waiting for an explicit post request."),
        ]
        self._send("Should I post this?")
        self.assertEqual(
            get_approval(self.store, pending.approval_id).status, "approved"
        )
        self.assertEqual(len(ledger.entries()), before)

    def test_17_explicit_approve_posts_once(self) -> None:
        pending = self._make_pending()
        before = len(ledger.entries())
        self.cli.handle_line("I approve this invoice")
        row = get_approval(self.store, pending.approval_id)
        self.assertEqual(row.status, "posted")
        self.assertIsNotNone(row.journal_entry_id)
        self.assertEqual(len(ledger.entries()), before + 1)
        # Repeated approve cannot create a duplicate ledger row.
        self.cli.handle_line("I approve this invoice")
        self.assertEqual(len(ledger.entries()), before + 1)

    # --- 18–19 stop ---

    def test_18_nl_stop_while_model_blocked(self) -> None:
        entered = threading.Event()
        block = threading.Event()
        slow = FakeModel([Message(role="assistant", content="late")])
        slow.entered = entered
        slow.block = block
        self.cli.model_factory = lambda _p: slow
        self.cli.handle_line("summarize all invoices")
        self.assertTrue(entered.wait(5))
        self.assertTrue(self.cli._blocks_new_turn())
        self.cli.handle_line("Stop")
        self.assertFalse(self.cli._blocks_new_turn())
        block.set()
        slow.finished.wait(5)

    def test_19_message_works_immediately_after_nl_stop(self) -> None:
        entered = threading.Event()
        block = threading.Event()
        slow = FakeModel([Message(role="assistant", content="late")])
        slow.entered = entered
        slow.block = block
        self.cli.model_factory = lambda _p: slow
        self.cli.handle_line("long running")
        self.assertTrue(entered.wait(5))
        self.cli.handle_line("Cancel the current task")
        self.assertFalse(self.cli._blocks_new_turn())
        follow = FakeModel([Message(role="assistant", content="next answer")])
        self.cli.model_factory = lambda _p: follow
        self._send("what has been processed?")
        self.assertIn("what has been processed?", [
            m.content
            for m in self.store.load("demo").conversation.messages
            if m.role == "user"
        ])
        block.set()
        slow.finished.wait(5)

    def test_app_control_tools_in_all_declarations(self) -> None:
        names = {d["name"] for d in all_tool_declarations()}
        self.assertIn("approve_pending_approval", names)
        self.assertIn("post_invoice", names)
        # Invoice-only list stays small for adapters.
        from fourx.agent.harness import tool_declarations

        self.assertEqual(
            {d["name"] for d in tool_declarations()},
            {
                "list_invoices",
                "read_invoice",
                "lookup_vendor",
                "post_invoice",
                "ledger_summary",
            },
        )

    def test_pending_survives_provider_nl_switch(self) -> None:
        pending = self._make_pending()
        self.cli.handle_line("Switch to Groq")
        self.assertEqual(self.store.load("demo").provider, "groq")
        still = get_pending_approval(self.store, "demo")
        self.assertIsNotNone(still)
        self.assertEqual(still.approval_id, pending.approval_id)

    def test_llm_reject_arbitrary_provider(self) -> None:
        result = execute_tool_call(
            ToolCall(
                id="bad",
                name="switch_provider",
                arguments={"provider": "openai"},
            ),
            store=self.store,
            conversation_id="demo",
            user_message="switch to openai",
        )
        payload = json.loads(result.content or "")
        self.assertFalse(payload["ok"])
        self.assertEqual(self.store.load("demo").provider, "gemini")


class HybridNLSafetyCorrectionTests(unittest.TestCase):
    """Focused fixes: posting response, separate turns, provider auth."""

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

    def _sgp_entries(self) -> list[dict]:
        return [
            e
            for e in ledger.entries()
            if e.get("invoice_number") == "SGP/26-27/0412"
        ]

    def test_01_mode_c_post_returns_ok_with_journal(self) -> None:
        from fourx.finance.approval import approve as record_approve

        pending = self._make_pending()
        record_approve(self.store, "demo", pending.approval_id)
        before = len(self._sgp_entries())
        # Phrasing authorizes post but avoids Mode-B whole-utterance shortcut.
        post_line = "Please post the approved request."
        self.assertIsNone(match_deterministic_action(post_line))
        self.assertTrue(user_message_authorizes_post(post_line))
        self.model.responses = [
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(id="p1", name="post_approved_invoice", arguments={}),
                ),
            ),
            Message(role="assistant", content="Posted."),
        ]
        self._send(post_line)
        row = get_approval(self.store, pending.approval_id)
        self.assertEqual(row.status, "posted")
        self.assertIsNotNone(row.journal_entry_id)
        self.assertEqual(len(self._sgp_entries()), before + 1)
        tool_msgs = [
            m
            for m in self.store.load("demo").conversation.messages
            if m.role == "tool" and m.tool_call_id == "p1"
        ]
        self.assertEqual(len(tool_msgs), 1)
        payload = json.loads(tool_msgs[0].content or "")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"], "posted")
        self.assertEqual(payload["journal_entry_id"], row.journal_entry_id)
        self.assertEqual(payload["approval_id"], pending.approval_id)

    def test_02_repeat_post_is_idempotent(self) -> None:
        pending = self._make_pending()
        self.cli.handle_line("/approve")
        row = get_approval(self.store, pending.approval_id)
        je = row.journal_entry_id
        count = len(self._sgp_entries())
        post_line = "Please post the approved request."
        self.model.responses = [
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="p2",
                        name="post_approved_invoice",
                        arguments={"approval_id": pending.approval_id},
                    ),
                ),
            ),
            Message(role="assistant", content="Already posted."),
        ]
        self._send(post_line)
        row2 = get_approval(self.store, pending.approval_id)
        self.assertEqual(row2.status, "posted")
        self.assertEqual(row2.journal_entry_id, je)
        self.assertEqual(len(self._sgp_entries()), count)

    def test_03_posting_uncertain_not_success(self) -> None:
        from fourx.finance.approval import approve as record_approve

        pending = self._make_pending()
        record_approve(self.store, "demo", pending.approval_id)
        uncertain = StoredApproval(
            approval_id=pending.approval_id,
            conversation_id="demo",
            invoice_id="invoice-01",
            details=dict(pending.details),
            status="posting_uncertain",
            journal_entry_id=None,
        )
        before = len(ledger.entries())
        with patch(
            "fourx.tools.app_actions.post_approved",
            side_effect=PostingUncertain(uncertain, "cannot verify"),
        ):
            result = execute_tool_call(
                ToolCall(id="u1", name="post_approved_invoice", arguments={}),
                store=self.store,
                conversation_id="demo",
                user_message="Post the approved invoice",
            )
        payload = json.loads(result.content or "")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_type"], "PostingUncertain")
        self.assertEqual(payload["status"], "posting_uncertain")
        self.assertIsNone(payload.get("journal_entry_id"))
        self.assertEqual(len(ledger.entries()), before)
        # Durable row still approved (mock prevented real post).
        self.assertEqual(
            get_approval(self.store, pending.approval_id).status, "approved"
        )

    def test_04_same_turn_approve_posts_once_second_post_blocked(self) -> None:
        pending = self._make_pending()
        before = len(self._sgp_entries())
        um = "Please approve and then post the approved invoice"
        self.model.responses = [
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(id="a1", name="approve_pending_approval", arguments={}),
                    ToolCall(id="p1", name="post_approved_invoice", arguments={}),
                ),
            ),
            Message(role="assistant", content="Approved and posted."),
        ]
        self._send(um)
        row = get_approval(self.store, pending.approval_id)
        self.assertEqual(row.status, "posted")
        self.assertIsNotNone(row.journal_entry_id)
        self.assertEqual(len(self._sgp_entries()), before + 1)
        payloads = {
            m.tool_call_id: json.loads(m.content or "")
            for m in self.store.load("demo").conversation.messages
            if m.role == "tool" and m.tool_call_id in {"a1", "p1"}
        }
        self.assertTrue(payloads["a1"]["ok"])
        self.assertEqual(payloads["a1"]["status"], "posted")
        self.assertFalse(payloads["p1"]["ok"])
        self.assertEqual(payloads["p1"]["error_type"], "SeparateTurnRequired")

    def test_05_second_model_batch_cannot_double_post(self) -> None:
        pending = self._make_pending()
        before = len(self._sgp_entries())
        um = "Please approve and then post the approved invoice"
        self.model.responses = [
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(id="a1", name="approve_pending_approval", arguments={}),
                ),
            ),
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(id="p1", name="post_approved_invoice", arguments={}),
                ),
            ),
            Message(role="assistant", content="Already handled."),
        ]
        self._send(um)
        row = get_approval(self.store, pending.approval_id)
        self.assertEqual(row.status, "posted")
        self.assertEqual(len(self._sgp_entries()), before + 1)
        post_payload = next(
            json.loads(m.content or "")
            for m in self.store.load("demo").conversation.messages
            if m.role == "tool" and m.tool_call_id == "p1"
        )
        self.assertFalse(post_payload["ok"])
        self.assertEqual(post_payload["error_type"], "SeparateTurnRequired")

    def test_06_approve_posts_immediately(self) -> None:
        pending = self._make_pending()
        before = len(self._sgp_entries())
        self.cli.handle_line("Approve this invoice")
        row = get_approval(self.store, pending.approval_id)
        self.assertEqual(row.status, "posted")
        self.assertIsNotNone(row.journal_entry_id)
        self.assertEqual(len(self._sgp_entries()), before + 1)

    def test_07_slash_approve_posts_and_post_is_idempotent(self) -> None:
        pending = self._make_pending()
        before = len(self._sgp_entries())
        self.cli.handle_line("/approve")
        row = get_approval(self.store, pending.approval_id)
        self.assertEqual(row.status, "posted")
        self.assertEqual(len(self._sgp_entries()), before + 1)
        self.cli.handle_line("/post")
        self.assertEqual(len(self._sgp_entries()), before + 1)
        self.assertEqual(
            get_approval(self.store, pending.approval_id).journal_entry_id,
            row.journal_entry_id,
        )

    def test_08_unsolicited_provider_switch_rejected(self) -> None:
        result = execute_tool_call(
            ToolCall(
                id="s1",
                name="switch_provider",
                arguments={"provider": "groq"},
            ),
            store=self.store,
            conversation_id="demo",
            user_message="what invoices are available?",
        )
        payload = json.loads(result.content or "")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_type"], "AuthorizationRequired")
        self.assertEqual(self.store.load("demo").provider, "gemini")

    def test_09_explicit_provider_switch_authorized(self) -> None:
        self.assertTrue(
            user_message_authorizes_provider_switch("Switch to Groq", "groq")
        )
        self.model.responses = [
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="s1",
                        name="switch_provider",
                        arguments={"provider": "groq"},
                    ),
                ),
            ),
            Message(role="assistant", content="Using Groq."),
        ]
        # Mode B would catch "Switch to Groq"; use LLM-style phrasing.
        self._send("Can we continue with Groq instead?")
        self.assertEqual(self.store.load("demo").provider, "groq")

    def test_10_incidental_provider_mention_no_switch(self) -> None:
        self.assertFalse(
            user_message_authorizes_provider_switch(
                "gemini is sometimes slow on tools", "gemini"
            )
        )
        result = execute_tool_call(
            ToolCall(
                id="s1",
                name="switch_provider",
                arguments={"provider": "groq"},
            ),
            store=self.store,
            conversation_id="demo",
            user_message="gemini is sometimes slow on tools",
        )
        self.assertFalse(json.loads(result.content or "")["ok"])
        self.assertEqual(self.store.load("demo").provider, "gemini")

    def test_11_provider_switch_preserves_pending_and_history(self) -> None:
        self._send("remember invoice context")
        pending = self._make_pending()
        self.cli.handle_line("Switch to Groq")
        self.assertEqual(self.store.load("demo").provider, "groq")
        still = get_pending_approval(self.store, "demo")
        self.assertIsNotNone(still)
        self.assertEqual(still.approval_id, pending.approval_id)
        users = [
            m.content
            for m in self.store.load("demo").conversation.messages
            if m.role == "user"
        ]
        self.assertIn("remember invoice context", users)

    def test_12_post_error_reflects_durable_state(self) -> None:
        pending = self._make_pending()
        result = execute_tool_call(
            ToolCall(id="p1", name="post_approved_invoice", arguments={}),
            store=self.store,
            conversation_id="demo",
            user_message="Post the approved invoice",
        )
        payload = json.loads(result.content or "")
        self.assertFalse(payload["ok"])
        # Still pending — message must not claim posted.
        self.assertEqual(
            get_approval(self.store, pending.approval_id).status, "pending"
        )
        self.assertNotEqual(payload.get("status"), "posted")


if __name__ == "__main__":
    unittest.main()

