"""Offline unit tests for the canonical conversation representation."""

from __future__ import annotations

import json
import unittest

from conversation import Conversation, Message, ToolCall


class ConversationRepresentationTests(unittest.TestCase):
    def test_user_message_can_be_represented(self) -> None:
        message = Message(role="user", content="list the invoices")
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "list the invoices")
        self.assertEqual(message.tool_calls, ())
        self.assertIsNone(message.tool_call_id)

    def test_assistant_text_response(self) -> None:
        message = Message(role="assistant", content="There are six invoices on file.")
        self.assertEqual(message.content, "There are six invoices on file.")
        self.assertEqual(message.tool_calls, ())

    def test_tool_call_keeps_name_arguments_and_id(self) -> None:
        call = ToolCall(
            id="call_read_1",
            name="read_invoice",
            arguments={"invoice_id": "invoice-06"},
        )
        message = Message(role="assistant", tool_calls=(call,))
        self.assertEqual(len(message.tool_calls), 1)
        self.assertEqual(message.tool_calls[0].id, "call_read_1")
        self.assertEqual(message.tool_calls[0].name, "read_invoice")
        self.assertEqual(message.tool_calls[0].arguments, {"invoice_id": "invoice-06"})

    def test_tool_result_keeps_matching_call_id(self) -> None:
        message = Message(
            role="tool",
            tool_call_id="call_read_1",
            content='{"invoice_number":"SGP/26-27/0431"}',
        )
        self.assertEqual(message.tool_call_id, "call_read_1")
        self.assertIn("SGP/26-27/0431", message.content or "")

    def test_multiple_tool_calls_in_one_response_remain_distinct(self) -> None:
        message = Message(
            role="assistant",
            content="Looking those up.",
            tool_calls=(
                ToolCall(id="call_a", name="read_invoice", arguments={"invoice_id": "invoice-01"}),
                ToolCall(id="call_b", name="lookup_vendor", arguments={"name": "Bluefin Logistics LLP"}),
            ),
        )
        self.assertEqual([call.id for call in message.tool_calls], ["call_a", "call_b"])
        self.assertEqual(message.tool_calls[0].name, "read_invoice")
        self.assertEqual(message.tool_calls[1].name, "lookup_vendor")

    def test_round_trip_preserves_order_and_fields(self) -> None:
        conversation = Conversation()
        conversation.add(Message(role="user", content="what's on invoice-06?"))
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
        conversation.add(
            Message(
                role="tool",
                tool_call_id="call_1",
                content='{"total":"21948.00","vendor_name":"Shree Ganesh Packaging Pvt Ltd"}',
            )
        )
        conversation.add(
            Message(
                role="assistant",
                content="Invoice-06 totals INR 21948.00 from Shree Ganesh Packaging Pvt Ltd.",
            )
        )

        encoded = conversation.to_jsonable()
        # Prove the payload is JSON-serializable (as later persistence will need).
        as_text = json.dumps(encoded)
        decoded = Conversation.from_jsonable(json.loads(as_text))

        self.assertEqual(len(decoded.messages), 4)
        self.assertEqual([m.role for m in decoded.messages], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(decoded.messages[0].content, "what's on invoice-06?")
        self.assertEqual(decoded.messages[1].tool_calls[0].id, "call_1")
        self.assertEqual(decoded.messages[1].tool_calls[0].arguments["invoice_id"], "invoice-06")
        self.assertEqual(decoded.messages[2].tool_call_id, "call_1")
        self.assertEqual(
            decoded.messages[3].content,
            "Invoice-06 totals INR 21948.00 from Shree Ganesh Packaging Pvt Ltd.",
        )

    def test_invalid_structures_raise_predictably(self) -> None:
        with self.assertRaises(ValueError):
            Message(role="user", content="")
        with self.assertRaises(ValueError):
            Message(role="user", content="hi", tool_calls=(ToolCall(id="x", name="n", arguments={}),))
        with self.assertRaises(ValueError):
            Message(role="assistant")
        with self.assertRaises(ValueError):
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(id="dup", name="a", arguments={}),
                    ToolCall(id="dup", name="b", arguments={}),
                ),
            )
        with self.assertRaises(ValueError):
            Message(role="tool", content="ok")
        with self.assertRaises(ValueError):
            Message.from_jsonable({"role": "system", "content": "nope"})
        with self.assertRaises(ValueError):
            Message.from_jsonable({"role": "user", "content": "hi", "extra": 1})
        with self.assertRaises(ValueError):
            Conversation.from_jsonable({"not": "a list"})

    def test_direct_tool_call_rejects_invalid_fields(self) -> None:
        with self.assertRaises(ValueError):
            ToolCall(id="", name="read_invoice", arguments={})
        with self.assertRaises(ValueError):
            ToolCall(id="call_1", name="", arguments={})
        with self.assertRaises(ValueError):
            ToolCall(id="call_1", name="read_invoice", arguments=["not", "a", "dict"])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ToolCall(id="call_1", name="read_invoice", arguments={"bad": {1, 2, 3}})


if __name__ == "__main__":
    unittest.main()
