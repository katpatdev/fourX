"""Offline tests for SQLite conversation persistence."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fourx.agent.conversation import Conversation, Message, ToolCall
from fourx.persistence.storage import (
    ConversationExists,
    ConversationNotFound,
    ConversationStore,
    CorruptConversationData,
    InvalidProvider,
)


class ConversationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.db_path = Path(self._tmpdir.name) / "conversations.db"
        self.store = ConversationStore(self.db_path)

    def test_create_and_load_empty_conversation(self) -> None:
        created = self.store.create("demo", "gemini")
        self.assertEqual(created.conversation_id, "demo")
        self.assertEqual(created.provider, "gemini")
        self.assertEqual(created.conversation.messages, [])

        loaded = self.store.load("demo")
        self.assertEqual(loaded.provider, "gemini")
        self.assertEqual(loaded.conversation.messages, [])

    def test_save_and_load_user_and_assistant_messages(self) -> None:
        self.store.create("demo", "gemini")
        conversation = Conversation()
        conversation.add(Message(role="user", content="list invoices"))
        conversation.add(Message(role="assistant", content="There are six invoices."))
        self.store.save("demo", conversation)

        loaded = ConversationStore(self.db_path).load("demo")
        self.assertEqual([m.role for m in loaded.conversation.messages], ["user", "assistant"])
        self.assertEqual(loaded.conversation.messages[0].content, "list invoices")
        self.assertEqual(loaded.conversation.messages[1].content, "There are six invoices.")

    def test_save_and_load_tool_call_history(self) -> None:
        self.store.create("demo", "gemini")
        conversation = Conversation()
        conversation.add(Message(role="user", content="read invoice-06"))
        conversation.add(
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="call_1",
                        name="read_invoice",
                        arguments={"invoice_id": "invoice-06"},
                    ),
                ),
            )
        )
        self.store.save("demo", conversation)

        loaded = ConversationStore(self.db_path).load("demo")
        call = loaded.conversation.messages[1].tool_calls[0]
        self.assertEqual(call.id, "call_1")
        self.assertEqual(call.name, "read_invoice")
        self.assertEqual(call.arguments, {"invoice_id": "invoice-06"})

    def test_preserves_multiple_tool_calls_in_order(self) -> None:
        self.store.create("demo", "gemini")
        conversation = Conversation()
        conversation.add(
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(id="call_a", name="read_invoice", arguments={"invoice_id": "invoice-01"}),
                    ToolCall(id="call_b", name="lookup_vendor", arguments={"name": "Bluefin"}),
                ),
            )
        )
        self.store.save("demo", conversation)

        loaded = self.store.load("demo")
        ids = [call.id for call in loaded.conversation.messages[0].tool_calls]
        self.assertEqual(ids, ["call_a", "call_b"])

    def test_preserves_tool_results_with_call_ids(self) -> None:
        self.store.create("demo", "gemini")
        conversation = Conversation()
        conversation.add(
            Message(
                role="assistant",
                tool_calls=(ToolCall(id="call_1", name="read_invoice", arguments={"invoice_id": "invoice-06"}),),
            )
        )
        conversation.add(
            Message(
                role="tool",
                tool_call_id="call_1",
                content='{"total":"21948.00"}',
            )
        )
        self.store.save("demo", conversation)

        loaded = ConversationStore(self.db_path).load("demo")
        self.assertEqual(loaded.conversation.messages[1].tool_call_id, "call_1")
        self.assertIn("21948.00", loaded.conversation.messages[1].content or "")

    def test_provider_survives_reopen(self) -> None:
        self.store.create("demo", "gemini")
        reopened = ConversationStore(self.db_path)
        self.assertEqual(reopened.load("demo").provider, "gemini")

    def test_change_provider_gemini_to_groq_and_back(self) -> None:
        self.store.create("demo", "gemini")
        self.store.set_provider("demo", "groq")
        self.assertEqual(ConversationStore(self.db_path).load("demo").provider, "groq")
        self.store.set_provider("demo", "gemini")
        self.assertEqual(ConversationStore(self.db_path).load("demo").provider, "gemini")

    def test_two_conversation_ids_are_independent(self) -> None:
        self.store.create("alpha", "gemini")
        self.store.create("beta", "groq")

        alpha = Conversation()
        alpha.add(Message(role="user", content="from alpha"))
        self.store.save("alpha", alpha)

        beta = Conversation()
        beta.add(Message(role="user", content="from beta"))
        self.store.save("beta", beta)

        loaded_alpha = self.store.load("alpha")
        loaded_beta = self.store.load("beta")
        self.assertEqual(loaded_alpha.provider, "gemini")
        self.assertEqual(loaded_beta.provider, "groq")
        self.assertEqual(loaded_alpha.conversation.messages[0].content, "from alpha")
        self.assertEqual(loaded_beta.conversation.messages[0].content, "from beta")

    def test_reject_duplicate_creation(self) -> None:
        self.store.create("demo", "gemini")
        with self.assertRaises(ConversationExists):
            self.store.create("demo", "groq")

    def test_missing_conversation_is_predictable(self) -> None:
        with self.assertRaises(ConversationNotFound):
            self.store.load("missing")
        with self.assertRaises(ConversationNotFound):
            self.store.save("missing", Conversation())
        with self.assertRaises(ConversationNotFound):
            self.store.set_provider("missing", "groq")

    def test_corrupt_stored_json_is_predictable(self) -> None:
        self.store.create("demo", "gemini")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE conversations SET messages_json = ? WHERE conversation_id = ?",
                ("{not-json", "demo"),
            )
            conn.commit()
        with self.assertRaises(CorruptConversationData):
            ConversationStore(self.db_path).load("demo")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE conversations SET messages_json = ? WHERE conversation_id = ?",
                (json.dumps([{"role": "system", "content": "bad"}]), "demo"),
            )
            conn.commit()
        with self.assertRaises(CorruptConversationData):
            ConversationStore(self.db_path).load("demo")

    def test_reject_invalid_provider(self) -> None:
        with self.assertRaises(InvalidProvider):
            self.store.create("demo", "openai")
        self.store.create("demo", "gemini")
        with self.assertRaises(InvalidProvider):
            self.store.set_provider("demo", "claude")


if __name__ == "__main__":
    unittest.main()
