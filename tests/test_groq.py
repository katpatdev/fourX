"""Offline unit tests for the Groq provider adapter (mocked SDK)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from conversation import Conversation, Message, ToolCall
from harness import tool_declarations
from providers.groq import (
    DEFAULT_GROQ_MODEL,
    GroqModel,
    MissingAPIKeyError,
    ProviderResponseError,
    conversation_to_groq_messages,
    groq_response_to_message,
    groq_tools_from_declarations,
)
from storage import ConversationStore


def _groq_response(
    *,
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls or [])
            )
        ]
    )


def _tool_call(call_id: str, name: str, arguments: dict | str) -> SimpleNamespace:
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class GroqAdapterTests(unittest.TestCase):
    def test_user_message_conversion(self) -> None:
        conversation = Conversation()
        conversation.add(Message(role="user", content="list invoices"))
        messages = conversation_to_groq_messages(conversation)
        self.assertEqual(messages, [{"role": "user", "content": "list invoices"}])

    def test_assistant_text_conversion(self) -> None:
        conversation = Conversation()
        conversation.add(Message(role="user", content="hi"))
        conversation.add(Message(role="assistant", content="Hello from Groq."))
        messages = conversation_to_groq_messages(conversation)
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], "Hello from Groq.")
        self.assertNotIn("tool_calls", messages[1])

    def test_tool_call_response_conversion(self) -> None:
        response = _groq_response(
            tool_calls=[
                _tool_call("call_read6", "read_invoice", {"invoice_id": "invoice-06"})
            ]
        )
        message = groq_response_to_message(response)
        self.assertEqual(message.tool_calls[0].id, "call_read6")
        self.assertEqual(message.tool_calls[0].arguments["invoice_id"], "invoice-06")

    def test_tool_result_history_reconstruction(self) -> None:
        conversation = Conversation()
        conversation.add(Message(role="user", content="read"))
        conversation.add(
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="call_read6",
                        name="read_invoice",
                        arguments={"invoice_id": "invoice-06"},
                    ),
                ),
            )
        )
        conversation.add(
            Message(
                role="tool",
                tool_call_id="call_read6",
                content='{"ok":true,"result":{"total":"21948.00"}}',
            )
        )
        messages = conversation_to_groq_messages(conversation)
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call_read6")
        self.assertEqual(messages[2]["role"], "tool")
        self.assertEqual(messages[2]["tool_call_id"], "call_read6")

    def test_multiple_tool_calls(self) -> None:
        response = _groq_response(
            content="Checking both.",
            tool_calls=[
                _tool_call("a", "list_invoices", {}),
                _tool_call("b", "lookup_vendor", {"name": "Bluefin Logistics LLP"}),
            ],
        )
        message = groq_response_to_message(response)
        self.assertEqual(message.content, "Checking both.")
        self.assertEqual([c.id for c in message.tool_calls], ["a", "b"])

    def test_preserves_tool_call_ids(self) -> None:
        response = _groq_response(tool_calls=[_tool_call("keep-groq", "list_invoices", {})])
        message = groq_response_to_message(response)
        self.assertEqual(message.tool_calls[0].id, "keep-groq")

    def test_text_and_tool_calls_together(self) -> None:
        response = _groq_response(
            content="One sec.",
            tool_calls=[_tool_call("x", "read_invoice", {"invoice_id": "invoice-06"})],
        )
        message = groq_response_to_message(response)
        self.assertEqual(message.content, "One sec.")
        self.assertEqual(len(message.tool_calls), 1)

    def test_invalid_json_tool_arguments_fail(self) -> None:
        response = _groq_response(tool_calls=[_tool_call("bad", "read_invoice", "{nope")])
        with self.assertRaises(ProviderResponseError):
            groq_response_to_message(response)

    def test_missing_api_key_fails_clearly(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(MissingAPIKeyError):
                GroqModel(api_key=None, client=None)

    def test_tools_include_approval_gated_post_invoice(self) -> None:
        tools = groq_tools_from_declarations(tool_declarations())
        names = {item["function"]["name"] for item in tools}
        self.assertEqual(
            names, {"list_invoices", "read_invoice", "lookup_vendor", "post_invoice"}
        )
        post = next(item for item in tools if item["function"]["name"] == "post_invoice")
        self.assertIn("invoice_id", post["function"]["parameters"]["properties"])

    def test_adapter_does_not_execute_tools_or_post(self) -> None:
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _groq_response(content="ok")
        model = GroqModel(api_key="test-key", client=fake_client)
        conversation = Conversation()
        conversation.add(Message(role="user", content="hi"))
        with patch("invoice_tools.ledger.post") as mock_post:
            with patch("invoice_tools.read_invoice") as mock_read:
                result = model(conversation, tool_declarations())
        self.assertEqual(result.content, "ok")
        mock_post.assert_not_called()
        mock_read.assert_not_called()
        self.assertEqual(model.model, DEFAULT_GROQ_MODEL)

    def test_provider_change_persists_and_history_converts_both_ways(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ConversationStore(Path(tmp) / "c.db")
            store.create("demo", "groq")
            conversation = Conversation()
            conversation.add(Message(role="user", content="what's on invoice-06?"))
            conversation.add(
                Message(
                    role="assistant",
                    tool_calls=(
                        ToolCall(
                            id="read6",
                            name="read_invoice",
                            arguments={"invoice_id": "invoice-06"},
                        ),
                    ),
                )
            )
            conversation.add(
                Message(
                    role="tool",
                    tool_call_id="read6",
                    content=json.dumps(
                        {
                            "ok": True,
                            "result": {
                                "vendor_name": "Shree Ganesh Packaging Pvt Ltd",
                                "total": "21948.00",
                                "lines": [1],
                            },
                        }
                    ),
                )
            )
            store.save("demo", conversation)
            store.set_provider("demo", "gemini")
            loaded = store.load("demo")
            self.assertEqual(loaded.provider, "gemini")
            from providers.gemini import conversation_to_gemini_contents

            gemini_contents = conversation_to_gemini_contents(loaded.conversation)
            self.assertEqual(gemini_contents[1].role, "model")
            self.assertEqual(gemini_contents[2].role, "user")
            store.set_provider("demo", "groq")
            again = store.load("demo")
            self.assertEqual(again.provider, "groq")
            groq_messages = conversation_to_groq_messages(again.conversation)
            self.assertEqual(groq_messages[1]["tool_calls"][0]["id"], "read6")


class Invoice06SwitchDemoTests(unittest.TestCase):
    """Manual acceptance scenario using mocked provider responses."""

    def test_mocked_gemini_then_groq_then_gemini_invoice_06(self) -> None:
        from providers.gemini import gemini_response_to_message
        from providers.groq import groq_response_to_message
        from tests.test_gemini import _gemini_response, _part_function

        conversation = Conversation()
        conversation.add(Message(role="user", content="what's on invoice-06?"))

        gemini_assistant = gemini_response_to_message(
            _gemini_response(
                [
                    _part_function(
                        "read_invoice",
                        {"invoice_id": "invoice-06"},
                        "read6",
                    )
                ]
            )
        )
        conversation.add(gemini_assistant)
        self.assertEqual(gemini_assistant.tool_calls[0].id, "read6")

        # Harness would execute read_invoice; we append the real tool result shape.
        from invoice_tools import read_invoice

        invoice = read_invoice("invoice-06")
        conversation.add(
            Message(
                role="tool",
                tool_call_id="read6",
                content=json.dumps({"ok": True, "result": invoice}),
            )
        )

        # Switch to Groq — same canonical history.
        groq_messages = conversation_to_groq_messages(conversation)
        self.assertEqual(groq_messages[1]["tool_calls"][0]["id"], "read6")
        groq_answer = groq_response_to_message(
            _groq_response(
                content=(
                    "Vendor is Shree Ganesh Packaging Pvt Ltd, "
                    "total with tax INR 21948.00."
                )
            )
        )
        conversation.add(groq_answer)
        self.assertIn("21948.00", groq_answer.content or "")
        self.assertIn("Shree Ganesh Packaging Pvt Ltd", groq_answer.content or "")

        # Switch back to Gemini — history still has tool id + line items.
        from providers.gemini import conversation_to_gemini_contents

        gemini_contents = conversation_to_gemini_contents(conversation)
        self.assertEqual(len(gemini_contents) >= 3, True)
        gemini_final = gemini_response_to_message(
            _gemini_response(
                [SimpleNamespace(text="It has one line item.", function_call=None)]
            )
        )
        conversation.add(gemini_final)
        self.assertIn("one line", gemini_final.content or "")
        self.assertEqual(len(invoice["lines"]), 1)


if __name__ == "__main__":
    unittest.main()
