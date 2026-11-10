"""Offline tests for durable human approval (no ledger posting)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from approval import (
    ApprovalAlreadyPending,
    ApprovalBlockedByApproved,
    ApprovalNotFound,
    InvalidDeclineReason,
    approve,
    decline,
    get_approval,
    get_pending_approval,
    list_approvals,
    request_approval,
)
from conversation import Conversation, Message, ToolCall
from harness import (
    ApprovalRequired,
    PersistenceError,
    RecoveryRequired,
    run_agent_turn,
    tool_declarations,
    unmatched_tool_call_ids,
)
from providers.gemini import gemini_tools_from_declarations
from providers.groq import groq_tools_from_declarations
from storage import ConversationStore


class FakeModel:
    def __init__(self, responses: list[Message]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, conversation: Conversation, tools: list[dict]) -> Message:
        self.calls += 1
        if not self.responses:
            raise AssertionError("FakeModel exhausted")
        return self.responses.pop(0)


class ApprovalLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.db_path = Path(self._tmpdir.name) / "conversations.db"
        self.store = ConversationStore(self.db_path)
        self.store.create("demo", "gemini")

    def _add_user(self, text: str) -> None:
        loaded = self.store.load("demo")
        loaded.conversation.add(Message(role="user", content=text))
        self.store.save("demo", loaded.conversation)

    def _post_model(self, invoice_id: str, call_id: str = "post1") -> FakeModel:
        return FakeModel(
            [
                Message(
                    role="assistant",
                    tool_calls=(
                        ToolCall(
                            id=call_id,
                            name="post_invoice",
                            arguments={"invoice_id": invoice_id},
                        ),
                    ),
                )
            ]
        )

    @patch("invoice_tools.ledger.post")
    @patch("ledger.entries", return_value=[])
    def test_eligible_invoice_01_creates_pending_with_exact_details(
        self, mock_entries, mock_post
    ) -> None:
        self._add_user("post invoice-01")
        with self.assertRaises(ApprovalRequired) as ctx:
            run_agent_turn(
                store=self.store, conversation_id="demo", model=self._post_model("invoice-01")
            )
        approval = ctx.exception.approval
        self.assertEqual(approval.status, "pending")
        self.assertTrue(approval.approval_id.startswith("apr-"))
        self.assertEqual(
            approval.display(),
            {
                "invoice_id": "invoice-01",
                "vendor_id": "V-001",
                "vendor_name": "Shree Ganesh Packaging Pvt Ltd",
                "invoice_number": "SGP/26-27/0412",
                "invoice_date": "2026-10-03",
                "currency": "INR",
                "amount": "16862.20",
            },
        )
        mock_post.assert_not_called()

    @patch("ledger.entries", return_value=[])
    def test_approved_record_survives_and_blocks_overwrite(self, mock_entries) -> None:
        first = request_approval(self.store, "demo", "invoice-01")
        approve(self.store, "demo", first.approval_id)
        with self.assertRaises(ApprovalBlockedByApproved):
            request_approval(self.store, "demo", "invoice-02")
        history = list_approvals(self.store, "demo")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].approval_id, first.approval_id)
        self.assertEqual(history[0].status, "approved")
        self.assertEqual(history[0].details["amount"], "16862.20")

    @patch("ledger.entries", return_value=[])
    def test_declined_record_survives_and_allows_new_pending(self, mock_entries) -> None:
        first = request_approval(self.store, "demo", "invoice-03")
        decline(self.store, "demo", "PO not received yet", first.approval_id)
        second = request_approval(self.store, "demo", "invoice-01")
        history = list_approvals(self.store, "demo")
        self.assertEqual(len(history), 2)
        self.assertNotEqual(history[0].approval_id, history[1].approval_id)
        self.assertEqual(history[0].status, "declined")
        self.assertEqual(history[0].decline_reason, "PO not received yet")
        self.assertEqual(history[1].status, "pending")
        self.assertEqual(second.invoice_id, "invoice-01")

    @patch("ledger.entries", return_value=[])
    def test_two_records_have_distinct_ids(self, mock_entries) -> None:
        a = request_approval(self.store, "demo", "invoice-03")
        decline(self.store, "demo", "later", a.approval_id)
        b = request_approval(self.store, "demo", "invoice-01")
        self.assertNotEqual(a.approval_id, b.approval_id)

    @patch("ledger.entries", return_value=[])
    def test_existing_pending_not_overwritten(self, mock_entries) -> None:
        first = request_approval(self.store, "demo", "invoice-01")
        with self.assertRaises(ApprovalAlreadyPending):
            request_approval(self.store, "demo", "invoice-02")
        pending = get_pending_approval(self.store, "demo")
        self.assertEqual(pending.approval_id, first.approval_id)
        self.assertEqual(pending.details["invoice_number"], "SGP/26-27/0412")

    @patch("ledger.entries", return_value=[])
    def test_repeated_approve_rejected(self, mock_entries) -> None:
        first = request_approval(self.store, "demo", "invoice-01")
        approve(self.store, "demo", first.approval_id)
        with self.assertRaises(ApprovalNotFound):
            approve(self.store, "demo")
        from storage import ApprovalDecisionConflict

        with self.assertRaises(ApprovalDecisionConflict):
            self.store.set_approval_decision(first.approval_id, status="approved")

    @patch("ledger.entries", return_value=[])
    def test_repeated_decline_rejected(self, mock_entries) -> None:
        first = request_approval(self.store, "demo", "invoice-01")
        decline(self.store, "demo", "not now", first.approval_id)
        with self.assertRaises(ApprovalNotFound):
            decline(self.store, "demo", "again")
        from storage import ApprovalDecisionConflict

        with self.assertRaises(ApprovalDecisionConflict):
            self.store.set_approval_decision(
                first.approval_id, status="declined", decline_reason="again"
            )

    @patch("ledger.entries", return_value=[])
    def test_empty_decline_reason_rejected(self, mock_entries) -> None:
        request_approval(self.store, "demo", "invoice-01")
        with self.assertRaises(InvalidDeclineReason):
            decline(self.store, "demo", "   ")
        self.assertEqual(get_pending_approval(self.store, "demo").status, "pending")

    @patch("ledger.entries", return_value=[])
    def test_model_message_cannot_approve(self, mock_entries) -> None:
        request_approval(self.store, "demo", "invoice-01")
        loaded = self.store.load("demo")
        loaded.conversation.add(Message(role="assistant", content="Approved"))
        self.store.save("demo", loaded.conversation)
        pending = get_pending_approval(self.store, "demo")
        self.assertIsNotNone(pending)
        self.assertEqual(pending.status, "pending")

    @patch("invoice_tools.ledger.post")
    def test_invoice_04_blocked_no_pending(self, mock_post) -> None:
        self._add_user("post invoice-04")
        model = FakeModel(
            [
                Message(
                    role="assistant",
                    tool_calls=(
                        ToolCall(
                            id="p4",
                            name="post_invoice",
                            arguments={"invoice_id": "invoice-04"},
                        ),
                    ),
                ),
                Message(role="assistant", content="Vendor is blocked."),
            ]
        )
        final = run_agent_turn(store=self.store, conversation_id="demo", model=model)
        self.assertIn("blocked", final.content or "")
        payload = json.loads(self.store.load("demo").conversation.messages[2].content or "")
        self.assertEqual(payload["error_type"], "BlockedVendor")
        self.assertEqual(list_approvals(self.store, "demo"), [])
        mock_post.assert_not_called()

    @patch("invoice_tools.ledger.post")
    @patch(
        "ledger.entries",
        return_value=[{"entry_id": "JE-000001", "invoice_number": "BL-88190"}],
    )
    def test_invoice_05_duplicate_no_pending(self, mock_entries, mock_post) -> None:
        self._add_user("post invoice-05")
        model = FakeModel(
            [
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
                Message(role="assistant", content="Already in the ledger."),
            ]
        )
        run_agent_turn(store=self.store, conversation_id="demo", model=model)
        payload = json.loads(self.store.load("demo").conversation.messages[2].content or "")
        self.assertEqual(payload["error_type"], "DuplicateInvoice")
        self.assertEqual(list_approvals(self.store, "demo"), [])
        mock_post.assert_not_called()

    @patch("ledger.entries", return_value=[])
    def test_atomic_pending_and_tool_result_commit(self, mock_entries) -> None:
        self._add_user("post invoice-01")
        with self.assertRaises(ApprovalRequired) as ctx:
            run_agent_turn(
                store=self.store, conversation_id="demo", model=self._post_model("invoice-01")
            )
        approval = ctx.exception.approval
        msgs = self.store.load("demo").conversation.messages
        self.assertEqual([m.role for m in msgs], ["user", "assistant", "tool"])
        payload = json.loads(msgs[2].content or "")
        self.assertEqual(payload["status"], "pending")
        self.assertEqual(get_approval(self.store, approval.approval_id).status, "pending")

    @patch("ledger.entries", return_value=[])
    def test_atomic_failure_leaves_no_pending_or_tool_result(self, mock_entries) -> None:
        self._add_user("post invoice-01")
        model = self._post_model("invoice-01")
        with patch.object(
            ConversationStore,
            "commit_pending_approval_with_conversation",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaises(PersistenceError):
                run_agent_turn(store=self.store, conversation_id="demo", model=model)
        self.assertEqual(list_approvals(self.store, "demo"), [])
        msgs = self.store.load("demo").conversation.messages
        self.assertEqual([m.role for m in msgs], ["user", "assistant"])
        self.assertEqual(unmatched_tool_call_ids(self.store.load("demo").conversation), ["post1"])
        self.assertEqual(model.calls, 1)

    @patch("ledger.entries", return_value=[])
    def test_failed_transaction_does_not_call_model_again(self, mock_entries) -> None:
        self._add_user("post")
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
                ),
                Message(role="assistant", content="should not run"),
            ]
        )
        with patch.object(
            ConversationStore,
            "commit_pending_approval_with_conversation",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaises(PersistenceError):
                run_agent_turn(store=self.store, conversation_id="demo", model=model)
        self.assertEqual(model.calls, 1)

    @patch("ledger.entries", return_value=[])
    def test_pending_blocks_model_after_restart(self, mock_entries) -> None:
        self._add_user("post invoice-01")
        with self.assertRaises(ApprovalRequired):
            run_agent_turn(
                store=self.store, conversation_id="demo", model=self._post_model("invoice-01")
            )
        reopened = ConversationStore(self.db_path)
        model = FakeModel([Message(role="assistant", content="should not run")])
        with self.assertRaises(ApprovalRequired) as ctx:
            run_agent_turn(store=reopened, conversation_id="demo", model=model)
        self.assertEqual(ctx.exception.approval.status, "pending")
        self.assertEqual(model.calls, 0)

    @patch("ledger.entries", return_value=[])
    def test_incomplete_history_requires_recovery(self, mock_entries) -> None:
        self._add_user("post")
        loaded = self.store.load("demo")
        loaded.conversation.add(
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="orphan",
                        name="post_invoice",
                        arguments={"invoice_id": "invoice-01"},
                    ),
                ),
            )
        )
        self.store.save("demo", loaded.conversation)
        model = FakeModel([Message(role="assistant", content="nope")])
        with self.assertRaises(RecoveryRequired):
            run_agent_turn(store=self.store, conversation_id="demo", model=model)
        self.assertEqual(model.calls, 0)

    @patch("ledger.entries", return_value=[])
    def test_read_then_post_order(self, mock_entries) -> None:
        self._add_user("read and post")
        model = FakeModel(
            [
                Message(
                    role="assistant",
                    tool_calls=(
                        ToolCall(
                            id="r1",
                            name="read_invoice",
                            arguments={"invoice_id": "invoice-01"},
                        ),
                        ToolCall(
                            id="p1",
                            name="post_invoice",
                            arguments={"invoice_id": "invoice-01"},
                        ),
                    ),
                )
            ]
        )
        with self.assertRaises(ApprovalRequired):
            run_agent_turn(store=self.store, conversation_id="demo", model=model)
        msgs = self.store.load("demo").conversation.messages
        self.assertEqual([m.role for m in msgs], ["user", "assistant", "tool", "tool"])
        self.assertEqual(msgs[2].tool_call_id, "r1")
        self.assertTrue(json.loads(msgs[2].content or "")["ok"])
        self.assertEqual(msgs[3].tool_call_id, "p1")
        self.assertEqual(json.loads(msgs[3].content or "")["status"], "pending")
        self.assertEqual(unmatched_tool_call_ids(self.store.load("demo").conversation), [])

    @patch("harness.lookup_vendor")
    @patch("ledger.entries", return_value=[])
    def test_post_then_lookup_skips_without_executing(self, mock_entries, mock_lookup) -> None:
        self._add_user("post then lookup")
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
                        ToolCall(
                            id="l1",
                            name="lookup_vendor",
                            arguments={"vendor_id": "V-002"},
                        ),
                    ),
                )
            ]
        )
        with self.assertRaises(ApprovalRequired):
            run_agent_turn(store=self.store, conversation_id="demo", model=model)
        mock_lookup.assert_not_called()
        msgs = self.store.load("demo").conversation.messages
        self.assertEqual(msgs[2].tool_call_id, "p1")
        self.assertEqual(msgs[3].tool_call_id, "l1")
        skipped = json.loads(msgs[3].content or "")
        self.assertFalse(skipped["ok"])
        self.assertEqual(skipped["error_type"], "SkippedForApproval")
        self.assertEqual(unmatched_tool_call_ids(self.store.load("demo").conversation), [])
        self.assertEqual(model.calls, 1)

    @patch("ledger.entries", return_value=[])
    def test_two_post_invoice_no_duplicate_pending(self, mock_entries) -> None:
        self._add_user("double post")
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
                        ToolCall(
                            id="p2",
                            name="post_invoice",
                            arguments={"invoice_id": "invoice-02"},
                        ),
                    ),
                )
            ]
        )
        with self.assertRaises(ApprovalRequired) as ctx:
            run_agent_turn(store=self.store, conversation_id="demo", model=model)
        self.assertEqual(ctx.exception.approval.invoice_id, "invoice-01")
        self.assertEqual(len(list_approvals(self.store, "demo")), 1)
        msgs = self.store.load("demo").conversation.messages
        self.assertEqual(json.loads(msgs[3].content or "")["error_type"], "SkippedForApproval")
        self.assertEqual(msgs[3].tool_call_id, "p2")

    @patch("invoice_tools.ledger.post")
    @patch("ledger.entries", return_value=[])
    def test_approve_and_decline_never_post(self, mock_entries, mock_post) -> None:
        first = request_approval(self.store, "demo", "invoice-01")
        approve(self.store, "demo", first.approval_id)
        mock_post.assert_not_called()

        self.store.create("other", "groq")
        second = request_approval(self.store, "other", "invoice-02")
        decline(self.store, "other", "hold", second.approval_id)
        mock_post.assert_not_called()

    def test_provider_schemas_include_post_invoice(self) -> None:
        decls = tool_declarations()
        gemini_names = {
            fn.name for fn in gemini_tools_from_declarations(decls)[0].function_declarations
        }
        groq_names = {
            item["function"]["name"] for item in groq_tools_from_declarations(decls)
        }
        self.assertIn("post_invoice", gemini_names)
        self.assertIn("post_invoice", groq_names)

    @patch("invoice_tools.ledger.post")
    def test_providers_do_not_execute_ledger_post(self, mock_post) -> None:
        from providers.gemini import GeminiModel
        from providers.groq import GroqModel
        from types import SimpleNamespace

        fake_gemini = MagicMock()
        fake_gemini.models.generate_content.return_value = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[SimpleNamespace(text="ok", function_call=None)],
                        model_dump=lambda **kwargs: {"role": "model", "parts": [{"text": "ok"}]},
                    )
                )
            ]
        )
        GeminiModel(api_key="x", client=fake_gemini)(
            Conversation(messages=[Message(role="user", content="hi")]),
            tool_declarations(),
        )
        fake_groq = MagicMock()
        fake_groq.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))]
        )
        GroqModel(api_key="x", client=fake_groq)(
            Conversation(messages=[Message(role="user", content="hi")]),
            tool_declarations(),
        )
        mock_post.assert_not_called()

    @patch("invoice_tools.ledger.post")
    def test_readonly_tools_still_work(self, mock_post) -> None:
        self._add_user("list")
        model = FakeModel(
            [
                Message(
                    role="assistant",
                    tool_calls=(ToolCall(id="c1", name="list_invoices", arguments={}),),
                ),
                Message(role="assistant", content="Six invoices."),
            ]
        )
        final = run_agent_turn(store=self.store, conversation_id="demo", model=model)
        self.assertEqual(final.content, "Six invoices.")
        mock_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
