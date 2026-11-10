"""Offline tests for the provider-independent agent harness."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from conversation import Conversation, Message, ToolCall
from approval import approve, get_approval
from harness import (
    ApprovalRequired,
    InvalidModelResponse,
    IterationLimitReached,
    PersistenceError,
    execute_tool_call,
    run_agent_turn,
    tool_declarations,
)
from storage import ConversationStore


class FakeModel:
    """Returns predetermined assistant Messages in order."""

    def __init__(self, responses: list[Message | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[Conversation] = []

    def __call__(self, conversation: Conversation, tools: list[dict]) -> Message:
        self.calls.append(
            Conversation(messages=list(conversation.messages))
        )
        if not self.responses:
            raise AssertionError("FakeModel has no more responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class HarnessTests(unittest.TestCase):
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

    def test_final_text_only_response(self) -> None:
        self._add_user("hello")
        model = FakeModel([Message(role="assistant", content="Hello from Acme AP.")])
        final = run_agent_turn(store=self.store, conversation_id="demo", model=model)
        self.assertEqual(final.content, "Hello from Acme AP.")
        self.assertEqual(len(model.calls), 1)

    @patch("invoice_tools.ledger.post")
    def test_list_invoices_tool(self, mock_post) -> None:
        self._add_user("list invoices")
        model = FakeModel(
            [
                Message(
                    role="assistant",
                    tool_calls=(ToolCall(id="c1", name="list_invoices", arguments={}),),
                ),
                Message(role="assistant", content="There are six invoices."),
            ]
        )
        final = run_agent_turn(store=self.store, conversation_id="demo", model=model)
        self.assertEqual(final.content, "There are six invoices.")
        tool_msg = self.store.load("demo").conversation.messages[2]
        self.assertEqual(tool_msg.tool_call_id, "c1")
        payload = json.loads(tool_msg.content or "")
        self.assertTrue(payload["ok"])
        self.assertEqual(len(payload["result"]), 6)
        mock_post.assert_not_called()

    @patch("invoice_tools.ledger.post")
    def test_read_invoice_06(self, mock_post) -> None:
        self._add_user("what's on invoice-06?")
        model = FakeModel(
            [
                Message(
                    role="assistant",
                    tool_calls=(
                        ToolCall(
                            id="read6",
                            name="read_invoice",
                            arguments={"invoice_id": "invoice-06"},
                        ),
                    ),
                ),
                Message(
                    role="assistant",
                    content=(
                        "Invoice-06 is from Shree Ganesh Packaging Pvt Ltd, "
                        "total INR 21948.00."
                    ),
                ),
            ]
        )
        final = run_agent_turn(store=self.store, conversation_id="demo", model=model)
        self.assertIn("21948.00", final.content or "")
        payload = json.loads(self.store.load("demo").conversation.messages[2].content or "")
        self.assertEqual(payload["result"]["total"], "21948.00")
        mock_post.assert_not_called()

    def test_lookup_vendor_tool(self) -> None:
        self._add_user("who is Orbit?")
        model = FakeModel(
            [
                Message(
                    role="assistant",
                    tool_calls=(
                        ToolCall(
                            id="v1",
                            name="lookup_vendor",
                            arguments={"name": "Orbit Facility Services"},
                        ),
                    ),
                ),
                Message(role="assistant", content="Orbit is blocked."),
            ]
        )
        run_agent_turn(store=self.store, conversation_id="demo", model=model)
        payload = json.loads(self.store.load("demo").conversation.messages[2].content or "")
        self.assertEqual(payload["result"]["vendor_id"], "V-004")
        self.assertEqual(payload["result"]["status"], "blocked")

    def test_multiple_tool_calls_preserve_ids(self) -> None:
        self._add_user("look up two things")
        model = FakeModel(
            [
                Message(
                    role="assistant",
                    content="Checking.",
                    tool_calls=(
                        ToolCall(id="a", name="list_invoices", arguments={}),
                        ToolCall(
                            id="b",
                            name="lookup_vendor",
                            arguments={"vendor_id": "V-001"},
                        ),
                    ),
                ),
                Message(role="assistant", content="Done."),
            ]
        )
        run_agent_turn(store=self.store, conversation_id="demo", model=model)
        msgs = self.store.load("demo").conversation.messages
        self.assertEqual(msgs[2].tool_call_id, "a")
        self.assertEqual(msgs[3].tool_call_id, "b")
        self.assertTrue(json.loads(msgs[2].content or "")["ok"])
        self.assertEqual(json.loads(msgs[3].content or "")["result"]["vendor_id"], "V-001")

    def test_tool_error_returned_to_model(self) -> None:
        self._add_user("read missing")
        model = FakeModel(
            [
                Message(
                    role="assistant",
                    tool_calls=(
                        ToolCall(
                            id="bad",
                            name="read_invoice",
                            arguments={"invoice_id": "invoice-99"},
                        ),
                    ),
                ),
                Message(role="assistant", content="That invoice is not on file."),
            ]
        )
        run_agent_turn(store=self.store, conversation_id="demo", model=model)
        # Second model call must have seen the tool error in history.
        second_history = model.calls[1]
        tool_msg = second_history.messages[2]
        payload = json.loads(tool_msg.content or "")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_type"], "InvoiceNotFound")

    def test_unknown_tool_rejected_safely(self) -> None:
        result = execute_tool_call(ToolCall(id="u1", name="delete_everything", arguments={}))
        payload = json.loads(result.content or "")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_type"], "UnknownTool")

    @patch("invoice_tools.ledger.post")
    @patch("ledger.entries", return_value=[])
    def test_post_invoice_creates_pending_without_posting(self, mock_entries, mock_post) -> None:
        self._add_user("post invoice-01")
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
                Message(role="assistant", content="should not continue"),
            ]
        )
        with self.assertRaises(ApprovalRequired) as ctx:
            run_agent_turn(store=self.store, conversation_id="demo", model=model)
        self.assertEqual(ctx.exception.approval.invoice_id, "invoice-01")
        self.assertEqual(ctx.exception.approval.details["amount"], "16862.20")
        payload = json.loads(self.store.load("demo").conversation.messages[2].content or "")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"], "pending")
        pending_id = ctx.exception.approval.approval_id
        self.assertEqual(get_approval(self.store, pending_id).status, "pending")
        mock_post.assert_not_called()
        self.assertEqual(len(model.calls), 1)
        approve(self.store, "demo", pending_id)
        mock_post.assert_not_called()
        self.assertFalse(Path("ledger.db").exists())

    def test_final_answer_after_tools(self) -> None:
        self._add_user("summarise")
        model = FakeModel(
            [
                Message(
                    role="assistant",
                    tool_calls=(ToolCall(id="c1", name="list_invoices", arguments={}),),
                ),
                Message(role="assistant", content="Six invoices on file."),
            ]
        )
        final = run_agent_turn(store=self.store, conversation_id="demo", model=model)
        self.assertEqual(final.content, "Six invoices on file.")
        self.assertEqual(len(model.calls), 2)

    def test_assistant_tool_calls_persisted_before_execution(self) -> None:
        self._add_user("list")
        seen: list[str] = []

        real_execute = execute_tool_call

        def wrapped(call: ToolCall, **_kwargs) -> Message:
            roles = [m.role for m in self.store.load("demo").conversation.messages]
            seen.append(",".join(roles))
            return real_execute(call)

        model = FakeModel(
            [
                Message(
                    role="assistant",
                    tool_calls=(ToolCall(id="c1", name="list_invoices", arguments={}),),
                ),
                Message(role="assistant", content="ok"),
            ]
        )
        with patch("harness.execute_tool_call", side_effect=wrapped):
            run_agent_turn(store=self.store, conversation_id="demo", model=model)
        # Before the tool runs, history already includes the assistant tool request.
        self.assertEqual(seen[0], "user,assistant")

    def test_each_tool_result_persisted_separately(self) -> None:
        self._add_user("two tools")
        saves_after: list[int] = []
        real_save = self.store.save

        def counting_save(conversation_id: str, conversation: Conversation) -> None:
            real_save(conversation_id, conversation)
            saves_after.append(len(conversation.messages))

        model = FakeModel(
            [
                Message(
                    role="assistant",
                    tool_calls=(
                        ToolCall(id="a", name="list_invoices", arguments={}),
                        ToolCall(
                            id="b",
                            name="lookup_vendor",
                            arguments={"vendor_id": "V-002"},
                        ),
                    ),
                ),
                Message(role="assistant", content="done"),
            ]
        )
        with patch.object(self.store, "save", side_effect=counting_save):
            run_agent_turn(store=self.store, conversation_id="demo", model=model)
        # user already saved; then assistant, tool a, tool b, final assistant
        self.assertIn(2, saves_after)  # after assistant with two tool calls
        self.assertIn(3, saves_after)  # after first tool result
        self.assertIn(4, saves_after)  # after second tool result

    def test_final_messages_survive_reload(self) -> None:
        self._add_user("what's on invoice-06?")
        model = FakeModel(
            [
                Message(
                    role="assistant",
                    tool_calls=(
                        ToolCall(
                            id="read6",
                            name="read_invoice",
                            arguments={"invoice_id": "invoice-06"},
                        ),
                    ),
                ),
                Message(
                    role="assistant",
                    content=(
                        "Invoice-06 is from Shree Ganesh Packaging Pvt Ltd, "
                        "total INR 21948.00."
                    ),
                ),
            ]
        )
        run_agent_turn(store=self.store, conversation_id="demo", model=model)
        reloaded = ConversationStore(self.db_path).load("demo")
        roles = [m.role for m in reloaded.conversation.messages]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant"])
        self.assertEqual(reloaded.conversation.messages[1].tool_calls[0].id, "read6")
        self.assertEqual(reloaded.conversation.messages[2].tool_call_id, "read6")
        self.assertIn("21948.00", reloaded.conversation.messages[3].content or "")
        self.assertEqual(reloaded.provider, "gemini")

    def test_invalid_model_response_fails_predictably(self) -> None:
        self._add_user("hi")

        def bad_model(conversation: Conversation, tools: list[dict]) -> Message:
            return "not a message"  # type: ignore[return-value]

        with self.assertRaises(InvalidModelResponse):
            run_agent_turn(store=self.store, conversation_id="demo", model=bad_model)
        roles = [m.role for m in self.store.load("demo").conversation.messages]
        self.assertEqual(roles, ["user"])

    def test_model_exception_does_not_corrupt_history(self) -> None:
        self._add_user("hi")
        model = FakeModel([RuntimeError("provider down")])
        with self.assertRaises(RuntimeError):
            run_agent_turn(store=self.store, conversation_id="demo", model=model)
        self.assertEqual(
            [m.role for m in self.store.load("demo").conversation.messages],
            ["user"],
        )

    def test_iteration_limit_stops_tool_loop(self) -> None:
        self._add_user("loop")

        def endless(conversation: Conversation, tools: list[dict]) -> Message:
            return Message(
                role="assistant",
                tool_calls=(ToolCall(id="x", name="list_invoices", arguments={}),),
            )

        with self.assertRaises(IterationLimitReached):
            run_agent_turn(
                store=self.store,
                conversation_id="demo",
                model=endless,
                max_model_calls=3,
            )
        msgs = self.store.load("demo").conversation.messages
        # user + 3*(assistant+tool) = 7
        self.assertEqual(len(msgs), 7)
        self.assertEqual(msgs[-1].role, "tool")

    def test_persistence_failure_stops_further_tools(self) -> None:
        self._add_user("two")
        real_save = ConversationStore.save
        state = {"n": 0}

        def flaky(store_self: ConversationStore, conversation_id: str, conversation: Conversation) -> None:
            state["n"] += 1
            # Allow initial user save path if any, and first assistant persist.
            # Fail when persisting the first tool result (after assistant save).
            if state["n"] >= 2 and conversation.messages and conversation.messages[-1].role == "tool":
                raise OSError("disk full")
            return real_save(store_self, conversation_id, conversation)

        model = FakeModel(
            [
                Message(
                    role="assistant",
                    tool_calls=(
                        ToolCall(id="a", name="list_invoices", arguments={}),
                        ToolCall(
                            id="b",
                            name="lookup_vendor",
                            arguments={"vendor_id": "V-001"},
                        ),
                    ),
                ),
                Message(role="assistant", content="should not reach"),
            ]
        )
        with patch.object(ConversationStore, "save", flaky):
            with self.assertRaises(PersistenceError):
                run_agent_turn(store=self.store, conversation_id="demo", model=model)

        # Assistant with two tool calls was saved; first tool may or may not be
        # partially in memory — durable store must not have continued to final.
        loaded = ConversationStore(self.db_path).load("demo")
        roles = [m.role for m in loaded.conversation.messages]
        self.assertEqual(roles[0], "user")
        self.assertEqual(roles[1], "assistant")
        self.assertEqual(len(roles), 2)  # failed before durable tool persist
        self.assertEqual(len(model.calls), 1)  # never called model again

    def test_tool_declarations_include_approval_gated_post(self) -> None:
        names = {item["name"] for item in tool_declarations()}
        self.assertEqual(
            names, {"list_invoices", "read_invoice", "lookup_vendor", "post_invoice"}
        )
        post = next(item for item in tool_declarations() if item["name"] == "post_invoice")
        self.assertEqual(post["parameters"]["required"], ["invoice_id"])


if __name__ == "__main__":
    unittest.main()
