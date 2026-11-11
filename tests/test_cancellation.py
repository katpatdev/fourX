"""Offline cooperative-cancellation tests (no network, no ledger posting)."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fourx.finance.approval import get_pending_approval, list_approvals, request_approval
from fourx.agent.cancellation import (
    CancellationToken,
    RunCancelled,
    StaleRunError,
    assert_run_active,
    begin_run,
    end_run,
    run_write,
)
from fourx.agent.conversation import Conversation, Message, ToolCall
from fourx.agent.harness import (
    ApprovalRequired,
    run_agent_turn,
    unmatched_tool_call_ids,
)
import fourx.agent.harness as harness_mod
from fourx.persistence.storage import ConversationStore


class GateModel:
    """Fake model that can block until a gate Event is set."""

    def __init__(
        self,
        *,
        response: Message,
        entered: threading.Event,
        release: threading.Event,
        calls: list | None = None,
    ) -> None:
        self.response = response
        self.entered = entered
        self.release = release
        self.calls = calls if calls is not None else []
        self.extra: list[Message] = []

    def __call__(self, conversation: Conversation, tools: list[dict]) -> Message:
        self.calls.append(Conversation(messages=list(conversation.messages)))
        self.entered.set()
        self.release.wait(timeout=5)
        if self.extra:
            return self.extra.pop(0)
        return self.response


class SequenceModel:
    def __init__(self, responses: list[Message]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, conversation: Conversation, tools: list[dict]) -> Message:
        self.calls += 1
        if not self.responses:
            raise AssertionError("SequenceModel exhausted")
        return self.responses.pop(0)


class CancellationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.db_path = Path(self._tmpdir.name) / "c.db"
        self.store = ConversationStore(self.db_path)
        self.store.create("demo", "gemini")

    def _add_user(self, text: str) -> None:
        loaded = self.store.load("demo")
        loaded.conversation.add(Message(role="user", content=text))
        self.store.save("demo", loaded.conversation)

    def test_cancel_before_first_model_call(self) -> None:
        self._add_user("hi")
        token = CancellationToken()
        token.cancel()
        model = SequenceModel([Message(role="assistant", content="nope")])
        with self.assertRaises(RunCancelled):
            run_agent_turn(
                store=self.store, conversation_id="demo", model=model, cancel=token
            )
        self.assertEqual(model.calls, 0)
        roles = [m.role for m in self.store.load("demo").conversation.messages]
        self.assertEqual(roles, ["user"])

    def test_cancel_while_model_blocked(self) -> None:
        self._add_user("work")
        entered = threading.Event()
        release = threading.Event()
        token = CancellationToken()
        model = GateModel(
            response=Message(
                role="assistant",
                tool_calls=(
                    ToolCall(id="t1", name="list_invoices", arguments={}),
                    ToolCall(id="t2", name="list_invoices", arguments={}),
                ),
            ),
            entered=entered,
            release=release,
        )
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                run_agent_turn(
                    store=self.store, conversation_id="demo", model=model, cancel=token
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(entered.wait(timeout=2))
        token.cancel()
        release.set()
        thread.join(timeout=5)
        self.assertTrue(thread.is_alive() is False)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RunCancelled)
        msgs = self.store.load("demo").conversation.messages
        self.assertEqual(msgs[1].role, "assistant")
        self.assertEqual(len(msgs[1].tool_calls), 2)
        # Both tools skipped — none executed successfully.
        self.assertEqual([m.role for m in msgs[2:]], ["tool", "tool"])
        for tool_msg in msgs[2:]:
            payload = json.loads(tool_msg.content or "")
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error_type"], "SkippedForCancellation")
        self.assertEqual(unmatched_tool_call_ids(self.store.load("demo").conversation), [])

    def test_cancel_immediately_after_model_returns_text(self) -> None:
        self._add_user("hi")
        entered = threading.Event()
        release = threading.Event()
        token = CancellationToken()
        model = GateModel(
            response=Message(role="assistant", content="Hello."),
            entered=entered,
            release=release,
        )
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                run_agent_turn(
                    store=self.store, conversation_id="demo", model=model, cancel=token
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(entered.wait(timeout=2))
        token.cancel()
        release.set()
        thread.join(timeout=5)
        self.assertIsInstance(errors[0], RunCancelled)
        msgs = self.store.load("demo").conversation.messages
        self.assertEqual([m.role for m in msgs], ["user", "assistant"])
        self.assertEqual(msgs[1].content, "Hello.")

    @patch("fourx.agent.harness.list_invoices")
    def test_cancel_between_two_tools(self, mock_list) -> None:
        mock_list.return_value = [{"invoice_id": "invoice-01"}]
        self._add_user("two tools")
        # First tool runs; cancel before second via side_effect on execute path.
        token = CancellationToken()
        call_count = {"n": 0}

        real_execute = __import__(
            "fourx.agent.harness", fromlist=["execute_tool_call"]
        ).execute_tool_call

        def wrapping(call: ToolCall, **kwargs) -> Message:
            call_count["n"] += 1
            if call_count["n"] == 1:
                result = real_execute(call, **kwargs)
                token.cancel()
                return result
            raise AssertionError("second tool should not execute")

        model = SequenceModel(
            [
                Message(
                    role="assistant",
                    tool_calls=(
                        ToolCall(id="a", name="list_invoices", arguments={}),
                        ToolCall(id="b", name="list_invoices", arguments={}),
                    ),
                )
            ]
        )
        with patch("fourx.agent.harness.execute_tool_call", side_effect=wrapping):
            with self.assertRaises(RunCancelled):
                run_agent_turn(
                    store=self.store, conversation_id="demo", model=model, cancel=token
                )
        self.assertEqual(call_count["n"], 1)
        msgs = self.store.load("demo").conversation.messages
        self.assertEqual(msgs[2].tool_call_id, "a")
        self.assertTrue(json.loads(msgs[2].content or "")["ok"])
        self.assertEqual(msgs[3].tool_call_id, "b")
        self.assertEqual(
            json.loads(msgs[3].content or "")["error_type"], "SkippedForCancellation"
        )
        self.assertEqual(unmatched_tool_call_ids(self.store.load("demo").conversation), [])

    def test_no_tools_after_cancel_observed(self) -> None:
        self._add_user("x")
        token = CancellationToken()
        executed: list[str] = []

        def tracking(call: ToolCall, **_kwargs) -> Message:
            executed.append(call.id)
            raise AssertionError("should not run tools after cancel")

        model = SequenceModel(
            [
                Message(
                    role="assistant",
                    tool_calls=(ToolCall(id="z", name="list_invoices", arguments={}),),
                )
            ]
        )
        # Cancel after model by wrapping model
        original = model

        def cancelling_model(conversation: Conversation, tools: list[dict]) -> Message:
            msg = original(conversation, tools)
            token.cancel()
            return msg

        with patch("fourx.agent.harness.execute_tool_call", side_effect=tracking):
            with self.assertRaises(RunCancelled):
                run_agent_turn(
                    store=self.store,
                    conversation_id="demo",
                    model=cancelling_model,
                    cancel=token,
                )
        self.assertEqual(executed, [])
        payload = json.loads(self.store.load("demo").conversation.messages[2].content or "")
        self.assertEqual(payload["error_type"], "SkippedForCancellation")

    def test_previous_messages_remain(self) -> None:
        self._add_user("first")
        loaded = self.store.load("demo")
        loaded.conversation.add(Message(role="assistant", content="prior answer"))
        self.store.save("demo", loaded.conversation)
        self._add_user("second")
        token = CancellationToken()
        token.cancel()
        with self.assertRaises(RunCancelled):
            run_agent_turn(
                store=self.store,
                conversation_id="demo",
                model=SequenceModel([Message(role="assistant", content="x")]),
                cancel=token,
            )
        roles = [m.role for m in self.store.load("demo").conversation.messages]
        self.assertEqual(roles, ["user", "assistant", "user"])
        self.assertEqual(
            self.store.load("demo").conversation.messages[1].content, "prior answer"
        )

    def test_next_request_works_after_cancel(self) -> None:
        self._add_user("stop me")
        entered = threading.Event()
        release = threading.Event()
        token = CancellationToken()
        model = GateModel(
            response=Message(
                role="assistant",
                tool_calls=(ToolCall(id="t1", name="list_invoices", arguments={}),),
            ),
            entered=entered,
            release=release,
        )
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                run_agent_turn(
                    store=self.store, conversation_id="demo", model=model, cancel=token
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(entered.wait(timeout=2))
        token.cancel()
        release.set()
        thread.join(timeout=5)
        self.assertIsInstance(errors[0], RunCancelled)

        # Fresh run with a new token / no cancel.
        self._add_user("what is left?")
        final = run_agent_turn(
            store=self.store,
            conversation_id="demo",
            model=SequenceModel([Message(role="assistant", content="Ready again.")]),
        )
        self.assertEqual(final.content, "Ready again.")
        self.assertEqual(unmatched_tool_call_ids(self.store.load("demo").conversation), [])

    def test_cancel_signal_does_not_leak_to_next_run(self) -> None:
        self._add_user("a")
        token1 = CancellationToken()
        token1.cancel()
        with self.assertRaises(RunCancelled):
            run_agent_turn(
                store=self.store,
                conversation_id="demo",
                model=SequenceModel([Message(role="assistant", content="x")]),
                cancel=token1,
            )
        self._add_user("b")
        # New token, not cancelled.
        final = run_agent_turn(
            store=self.store,
            conversation_id="demo",
            model=SequenceModel([Message(role="assistant", content="ok")]),
            cancel=CancellationToken(),
        )
        self.assertEqual(final.content, "ok")

    def test_stale_cancelled_run_cannot_append_after_new_run(self) -> None:
        self._add_user("old")
        entered = threading.Event()
        release = threading.Event()
        ownership_taken = threading.Event()
        old_token = CancellationToken()
        model = GateModel(
            response=Message(role="assistant", content="stale"),
            entered=entered,
            release=release,
        )
        errors: list[BaseException] = []

        def old_worker() -> None:
            try:
                run_agent_turn(
                    store=self.store,
                    conversation_id="demo",
                    model=model,
                    cancel=old_token,
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=old_worker)
        thread.start()
        self.assertTrue(entered.wait(timeout=2))
        old_token.cancel()

        # New run takes ownership while the old model call is still blocked.
        self._add_user("new")
        new_model = SequenceModel([Message(role="assistant", content="fresh")])

        def start_new() -> None:
            run_agent_turn(store=self.store, conversation_id="demo", model=new_model)
            ownership_taken.set()

        new_thread = threading.Thread(target=start_new)
        new_thread.start()
        new_thread.join(timeout=5)
        self.assertTrue(ownership_taken.is_set())
        release.set()
        thread.join(timeout=5)
        self.assertTrue(any(isinstance(e, (StaleRunError, RunCancelled)) for e in errors))
        texts = [
            m.content
            for m in self.store.load("demo").conversation.messages
            if m.role == "assistant"
        ]
        self.assertIn("fresh", texts)
        self.assertNotIn("stale", texts)

    @patch("ledger.entries", return_value=[])
    @patch("fourx.tools.invoice_tools.ledger.post")
    def test_pending_approval_survives_cancellation(self, mock_post, mock_entries) -> None:
        pending = request_approval(self.store, "demo", "invoice-01")
        token = CancellationToken()
        token.cancel()
        # Pending approval is surfaced even when a stop was requested.
        with self.assertRaises(ApprovalRequired) as ctx:
            run_agent_turn(
                store=self.store,
                conversation_id="demo",
                model=SequenceModel([Message(role="assistant", content="x")]),
                cancel=token,
            )
        self.assertEqual(ctx.exception.approval.approval_id, pending.approval_id)
        reopened = ConversationStore(self.db_path)
        loaded = get_pending_approval(reopened, "demo")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.approval_id, pending.approval_id)
        self.assertEqual(loaded.status, "pending")
        mock_post.assert_not_called()

    @patch("ledger.entries", return_value=[])
    @patch("fourx.tools.invoice_tools.ledger.post")
    def test_cancel_does_not_approve_or_decline(self, mock_post, mock_entries) -> None:
        request_approval(self.store, "demo", "invoice-01")
        token = CancellationToken()
        token.cancel()
        with self.assertRaises(ApprovalRequired):
            run_agent_turn(
                store=self.store,
                conversation_id="demo",
                model=SequenceModel([]),
                cancel=token,
            )
        rows = list_approvals(self.store, "demo")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, "pending")
        mock_post.assert_not_called()

    @patch("fourx.tools.invoice_tools.ledger.post")
    def test_cancel_never_calls_ledger_post(self, mock_post) -> None:
        self._add_user("x")
        token = CancellationToken()
        token.cancel()
        with self.assertRaises(RunCancelled):
            run_agent_turn(
                store=self.store,
                conversation_id="demo",
                model=SequenceModel([Message(role="assistant", content="x")]),
                cancel=token,
            )
        mock_post.assert_not_called()

    def test_unexecuted_tools_never_marked_success(self) -> None:
        self._add_user("x")
        token = CancellationToken()

        def model(conversation: Conversation, tools: list[dict]) -> Message:
            token.cancel()
            return Message(
                role="assistant",
                tool_calls=(
                    ToolCall(id="r1", name="read_invoice", arguments={"invoice_id": "invoice-01"}),
                ),
            )

        with self.assertRaises(RunCancelled):
            run_agent_turn(
                store=self.store, conversation_id="demo", model=model, cancel=token
            )
        payload = json.loads(self.store.load("demo").conversation.messages[2].content or "")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_type"], "SkippedForCancellation")

    def test_a_superseded_before_write_section_rejected(self) -> None:
        """Run A paused before run_write; B supersedes; A write fails; B kept."""
        self._add_user("old")
        reached = threading.Event()
        proceed = threading.Event()
        errors: list[BaseException] = []
        real_checkpoint = harness_mod._checkpoint

        def gated_checkpoint(store, conversation_id, conversation, token):
            reached.set()
            self.assertTrue(proceed.wait(timeout=5))
            return real_checkpoint(store, conversation_id, conversation, token)

        def worker() -> None:
            try:
                run_agent_turn(
                    store=self.store,
                    conversation_id="demo",
                    model=SequenceModel(
                        [Message(role="assistant", content="stale-from-A")]
                    ),
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        with patch.object(harness_mod, "_checkpoint", side_effect=gated_checkpoint):
            thread = threading.Thread(target=worker)
            thread.start()
            self.assertTrue(reached.wait(timeout=2))
            token_b = CancellationToken()
            begin_run("demo", token_b)
            loaded = self.store.load("demo")
            loaded.conversation.add(Message(role="assistant", content="fresh-from-B"))
            with run_write("demo", token_b):
                self.store.save("demo", loaded.conversation)
            proceed.set()
            thread.join(timeout=5)

        self.assertTrue(any(isinstance(e, StaleRunError) for e in errors))
        texts = [
            m.content
            for m in self.store.load("demo").conversation.messages
            if m.role == "assistant"
        ]
        self.assertIn("fresh-from-B", texts)
        self.assertNotIn("stale-from-A", texts)
        end_run("demo", token_b)

    def test_b_begin_waits_until_protected_write_finishes(self) -> None:
        """Run B cannot supersede while A holds run_write during save."""
        self._add_user("a-writing")
        in_save = threading.Event()
        release_save = threading.Event()
        b_began = threading.Event()
        b_finished_begin = threading.Event()
        real_save = ConversationStore.save

        def slow_save(self_store, conversation_id, conversation):
            in_save.set()
            self.assertTrue(release_save.wait(timeout=5))
            return real_save(self_store, conversation_id, conversation)

        errors: list[BaseException] = []

        def run_a() -> None:
            try:
                with patch.object(ConversationStore, "save", slow_save):
                    run_agent_turn(
                        store=self.store,
                        conversation_id="demo",
                        model=SequenceModel(
                            [Message(role="assistant", content="committed-by-A")]
                        ),
                    )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread_a = threading.Thread(target=run_a)
        thread_a.start()
        self.assertTrue(in_save.wait(timeout=2))

        token_b = CancellationToken()

        def run_b_begin() -> None:
            b_began.set()
            begin_run("demo", token_b)
            b_finished_begin.set()

        thread_b = threading.Thread(target=run_b_begin)
        thread_b.start()
        self.assertTrue(b_began.wait(timeout=2))
        # B must still be blocked on the ownership lock while A saves.
        self.assertFalse(b_finished_begin.wait(timeout=0.2))

        release_save.set()
        thread_a.join(timeout=5)
        thread_b.join(timeout=5)
        self.assertEqual(errors, [])
        self.assertTrue(b_finished_begin.is_set())
        assert_run_active("demo", token_b)
        loaded = self.store.load("demo")
        self.assertIn(
            "committed-by-A",
            [m.content for m in loaded.conversation.messages if m.role == "assistant"],
        )
        end_run("demo", token_b)

    @patch("ledger.entries", return_value=[])
    @patch("fourx.tools.invoice_tools.ledger.post")
    def test_c_superseded_before_approval_commit(self, mock_post, mock_entries) -> None:
        """Run A cannot create approval after B supersedes before protected commit."""
        self._add_user("post please")
        reached = threading.Event()
        proceed = threading.Event()
        errors: list[BaseException] = []
        real_commit = harness_mod._commit_pending_approval

        def gated_commit(store, conversation_id, details, conversation, token):
            reached.set()
            self.assertTrue(proceed.wait(timeout=5))
            return real_commit(store, conversation_id, details, conversation, token)

        def worker() -> None:
            try:
                run_agent_turn(
                    store=self.store,
                    conversation_id="demo",
                    model=SequenceModel(
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
                    ),
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        with patch.object(harness_mod, "_commit_pending_approval", side_effect=gated_commit):
            thread = threading.Thread(target=worker)
            thread.start()
            self.assertTrue(reached.wait(timeout=2))
            token_b = CancellationToken()
            begin_run("demo", token_b)
            loaded = self.store.load("demo")
            # Assistant tool request may already be on disk from A's earlier checkpoint.
            loaded.conversation.add(Message(role="assistant", content="B-kept"))
            with run_write("demo", token_b):
                self.store.save("demo", loaded.conversation)
            proceed.set()
            thread.join(timeout=5)

        self.assertTrue(any(isinstance(e, StaleRunError) for e in errors))
        self.assertEqual(list_approvals(self.store, "demo"), [])
        texts = [
            m.content
            for m in self.store.load("demo").conversation.messages
            if m.role == "assistant" and m.content
        ]
        self.assertIn("B-kept", texts)
        mock_post.assert_not_called()
        end_run("demo", token_b)

    def test_d_old_end_run_does_not_clear_newer(self) -> None:
        token_a = CancellationToken()
        token_b = CancellationToken()
        begin_run("demo", token_a)
        begin_run("demo", token_b)
        end_run("demo", token_a)
        assert_run_active("demo", token_b)
        with run_write("demo", token_b):
            loaded = self.store.load("demo")
            loaded.conversation.add(Message(role="assistant", content="B-ok"))
            self.store.save("demo", loaded.conversation)
        end_run("demo", token_b)
        self.assertIn(
            "B-ok",
            [m.content for m in self.store.load("demo").conversation.messages if m.role == "assistant"],
        )

    def test_e_cancelled_run_does_not_append_after_next_turn(self) -> None:
        """Stop-then-next-request: cancelled A cannot overwrite B afterward."""
        self._add_user("stop me")
        entered = threading.Event()
        release = threading.Event()
        token_a = CancellationToken()
        model = GateModel(
            response=Message(role="assistant", content="stale-A"),
            entered=entered,
            release=release,
        )
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                run_agent_turn(
                    store=self.store,
                    conversation_id="demo",
                    model=model,
                    cancel=token_a,
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(entered.wait(timeout=2))
        token_a.cancel()

        self._add_user("next")
        final = run_agent_turn(
            store=self.store,
            conversation_id="demo",
            model=SequenceModel([Message(role="assistant", content="fresh-B")]),
        )
        self.assertEqual(final.content, "fresh-B")
        release.set()
        thread.join(timeout=5)
        self.assertTrue(any(isinstance(e, (StaleRunError, RunCancelled)) for e in errors))
        texts = [
            m.content
            for m in self.store.load("demo").conversation.messages
            if m.role == "assistant"
        ]
        self.assertIn("fresh-B", texts)
        self.assertNotIn("stale-A", texts)


if __name__ == "__main__":
    unittest.main()
