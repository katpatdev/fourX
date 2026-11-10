"""Offline unit tests for the Gemini provider adapter (mocked SDK)."""

from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from conversation import Conversation, Message, ToolCall
from harness import tool_declarations
from providers.gemini import (
    DEFAULT_GEMINI_MODEL,
    GeminiModel,
    MissingAPIKeyError,
    ProviderResponseError,
    conversation_to_gemini_contents,
    gemini_response_to_message,
    gemini_tools_from_declarations,
)
from storage import ConversationStore


def _part_text(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text, function_call=None)


def _part_function(name: str, args: dict, call_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        text=None,
        function_call=SimpleNamespace(name=name, args=args, id=call_id),
    )


def _gemini_response(parts: list[SimpleNamespace]) -> SimpleNamespace:
    content = SimpleNamespace(
        role="model",
        parts=parts,
        model_dump=lambda mode="json", exclude_none=True: {
            "role": "model",
            "parts": [
                (
                    {"text": p.text}
                    if p.function_call is None
                    else {
                        "function_call": {
                            "name": p.function_call.name,
                            "args": p.function_call.args,
                            "id": p.function_call.id,
                        }
                    }
                )
                for p in parts
            ],
        },
    )
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])


class GeminiAdapterTests(unittest.TestCase):
    def test_user_message_conversion(self) -> None:
        conversation = Conversation()
        conversation.add(Message(role="user", content="what's on invoice-06?"))
        contents = conversation_to_gemini_contents(conversation)
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0].role, "user")
        self.assertEqual(contents[0].parts[0].text, "what's on invoice-06?")

    def test_assistant_text_conversion(self) -> None:
        conversation = Conversation()
        conversation.add(Message(role="user", content="hi"))
        conversation.add(Message(role="assistant", content="Hello."))
        contents = conversation_to_gemini_contents(conversation)
        self.assertEqual(contents[1].role, "model")
        self.assertEqual(contents[1].parts[0].text, "Hello.")

    def test_tool_call_response_conversion(self) -> None:
        response = _gemini_response(
            [
                _part_function(
                    "read_invoice",
                    {"invoice_id": "invoice-06"},
                    call_id="call_read6",
                )
            ]
        )
        message = gemini_response_to_message(response)
        self.assertEqual(message.role, "assistant")
        self.assertEqual(len(message.tool_calls), 1)
        self.assertEqual(message.tool_calls[0].id, "call_read6")
        self.assertEqual(message.tool_calls[0].name, "read_invoice")
        self.assertEqual(message.tool_calls[0].arguments, {"invoice_id": "invoice-06"})
        self.assertIn("gemini_content", message.extras or {})

    def test_tool_result_history_reconstruction(self) -> None:
        conversation = Conversation()
        conversation.add(Message(role="user", content="read it"))
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
                content=json.dumps({"ok": True, "result": {"total": "21948.00"}}),
            )
        )
        contents = conversation_to_gemini_contents(conversation)
        self.assertEqual(contents[1].role, "model")
        self.assertEqual(contents[2].role, "user")
        self.assertEqual(contents[2].parts[0].function_response.name, "read_invoice")
        self.assertEqual(
            contents[2].parts[0].function_response.response["result"]["total"],
            "21948.00",
        )

    def test_multiple_function_calls(self) -> None:
        response = _gemini_response(
            [
                _part_text("Looking up."),
                _part_function("list_invoices", {}, "c1"),
                _part_function("lookup_vendor", {"vendor_id": "V-001"}, "c2"),
            ]
        )
        message = gemini_response_to_message(response)
        self.assertEqual(message.content, "Looking up.")
        self.assertEqual([c.id for c in message.tool_calls], ["c1", "c2"])

    def test_preserves_tool_call_ids(self) -> None:
        response = _gemini_response(
            [_part_function("list_invoices", {}, "keep-me")]
        )
        message = gemini_response_to_message(response)
        self.assertEqual(message.tool_calls[0].id, "keep-me")

    def test_text_and_tool_calls_together(self) -> None:
        response = _gemini_response(
            [
                _part_text("One moment."),
                _part_function("read_invoice", {"invoice_id": "invoice-01"}, "x"),
            ]
        )
        message = gemini_response_to_message(response)
        self.assertEqual(message.content, "One moment.")
        self.assertEqual(message.tool_calls[0].name, "read_invoice")

    def test_invalid_json_tool_arguments_fail(self) -> None:
        response = _gemini_response(
            [
                SimpleNamespace(
                    text=None,
                    function_call=SimpleNamespace(
                        name="read_invoice",
                        args="{not-json",
                        id="bad",
                    ),
                )
            ]
        )
        with self.assertRaises(ProviderResponseError):
            gemini_response_to_message(response)

    def test_missing_api_key_fails_clearly(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(MissingAPIKeyError):
                GeminiModel(api_key=None, client=None)

    def test_tools_include_approval_gated_post_invoice(self) -> None:
        tools = gemini_tools_from_declarations(tool_declarations())
        names = [fn.name for fn in tools[0].function_declarations]
        self.assertEqual(
            set(names),
            {
                "list_invoices",
                "read_invoice",
                "lookup_vendor",
                "post_invoice",
                "ledger_summary",
            },
        )
        post = next(fn for fn in tools[0].function_declarations if fn.name == "post_invoice")
        self.assertIn("invoice_id", (post.parameters_json_schema or {}).get("properties", {}))

    def test_adapter_does_not_execute_tools_or_post(self) -> None:
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value = _gemini_response(
            [_part_text("done")]
        )
        model = GeminiModel(api_key="test-key", client=fake_client)
        conversation = Conversation()
        conversation.add(Message(role="user", content="hi"))
        with patch("invoice_tools.ledger.post") as mock_post:
            with patch("invoice_tools.list_invoices") as mock_list:
                result = model(conversation, tool_declarations())
        self.assertEqual(result.content, "done")
        mock_post.assert_not_called()
        mock_list.assert_not_called()
        self.assertEqual(model.model, DEFAULT_GEMINI_MODEL)

    def test_native_gemini_content_round_trip_in_history(self) -> None:
        response = _gemini_response(
            [_part_function("list_invoices", {}, "native-1")]
        )
        assistant = gemini_response_to_message(response)
        conversation = Conversation()
        conversation.add(Message(role="user", content="list"))
        conversation.add(assistant)
        contents = conversation_to_gemini_contents(conversation)
        # Uses stored extras rather than rebuilding from ToolCall alone.
        self.assertEqual(contents[1].role, "model")
        self.assertTrue(assistant.extras)


class ProviderSwitchWithGeminiHistoryTests(unittest.TestCase):
    def test_provider_switch_retains_canonical_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ConversationStore(Path(tmp) / "c.db")
            store.create("demo", "gemini")
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
                    extras={"gemini_content": {"role": "model", "parts": []}},
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
                                "lines": [{"description": "Stretch wrap rolls, 500 mm"}],
                            },
                        }
                    ),
                )
            )
            store.save("demo", conversation)
            store.set_provider("demo", "groq")
            loaded = store.load("demo")
            self.assertEqual(loaded.provider, "groq")
            self.assertEqual(loaded.conversation.messages[1].tool_calls[0].id, "read6")
            self.assertEqual(loaded.conversation.messages[2].tool_call_id, "read6")
            # Groq can consume the same canonical history.
            from providers.groq import conversation_to_groq_messages

            groq_messages = conversation_to_groq_messages(loaded.conversation)
            self.assertEqual(groq_messages[1]["tool_calls"][0]["id"], "read6")
            store.set_provider("demo", "gemini")
            again = store.load("demo")
            self.assertEqual(again.provider, "gemini")
            self.assertEqual(len(again.conversation.messages), 3)


def _sdk_function_call_response(
    *,
    name: str,
    args: dict,
    call_id: str | None = None,
    thought_signature: bytes | None = None,
):
    """Build a generate_content-shaped response using real google.genai.types."""
    from google.genai import types

    function_call = types.FunctionCall(name=name, args=args, id=call_id)
    part = types.Part(function_call=function_call, thought_signature=thought_signature)
    content = types.Content(role="model", parts=[part])
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])


class GeminiSdkNativeRoundTripTests(unittest.TestCase):
    """Corrections A–C: real SDK Content / Part serialization and IDs."""

    def test_native_function_call_with_signature_survives_store_reload(self) -> None:
        from google.genai import types

        signature = base64.b64encode(b"fourx-sim-thought-sig")
        # types.Part accepts raw bytes; model_dump(mode="json") base64-encodes them.
        response = _sdk_function_call_response(
            name="read_invoice",
            args={"invoice_id": "invoice-06"},
            call_id="call_read6",
            thought_signature=signature,
        )
        assistant = gemini_response_to_message(response)
        self.assertEqual(assistant.tool_calls[0].id, "call_read6")
        self.assertIn("gemini_content", assistant.extras or {})
        native = assistant.extras["gemini_content"]
        self.assertEqual(native["parts"][0]["function_call"]["id"], "call_read6")
        self.assertIn("thought_signature", native["parts"][0])

        conversation = Conversation()
        conversation.add(Message(role="user", content="what's on invoice-06?"))
        conversation.add(assistant)
        conversation.add(
            Message(
                role="tool",
                tool_call_id="call_read6",
                content=json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "vendor_name": "Shree Ganesh Packaging Pvt Ltd",
                            "total": "21948.00",
                            "lines": [{"description": "Stretch wrap rolls, 500 mm"}],
                        },
                    }
                ),
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            store = ConversationStore(Path(tmp) / "c.db")
            store.create("sig-demo", "gemini")
            store.save("sig-demo", conversation)
            loaded = store.load("sig-demo").conversation

        restored_assistant = loaded.messages[1]
        self.assertEqual(restored_assistant.tool_calls[0].id, "call_read6")
        self.assertEqual(
            restored_assistant.extras["gemini_content"]["parts"][0]["thought_signature"],
            native["parts"][0]["thought_signature"],
        )

        contents = conversation_to_gemini_contents(loaded)
        self.assertIsInstance(contents[1], types.Content)
        self.assertEqual(contents[1].parts[0].function_call.name, "read_invoice")
        self.assertEqual(
            contents[1].parts[0].function_call.args["invoice_id"], "invoice-06"
        )
        self.assertEqual(contents[1].parts[0].function_call.id, "call_read6")
        self.assertTrue(contents[1].parts[0].thought_signature)

        from providers.groq import conversation_to_groq_messages

        groq_messages = conversation_to_groq_messages(loaded)
        self.assertEqual(groq_messages[1]["tool_calls"][0]["id"], "call_read6")
        self.assertNotIn("extras", groq_messages[1])
        self.assertNotIn("gemini_content", groq_messages[1])
        self.assertNotIn("thought_signature", json.dumps(groq_messages))

    def test_id_less_calls_get_unique_stable_ids(self) -> None:
        first = gemini_response_to_message(
            _sdk_function_call_response(name="list_invoices", args={})
        )
        second = gemini_response_to_message(
            _sdk_function_call_response(name="list_invoices", args={})
        )
        id_a = first.tool_calls[0].id
        id_b = second.tool_calls[0].id
        self.assertTrue(id_a)
        self.assertTrue(id_b)
        self.assertNotEqual(id_a, id_b)
        self.assertEqual(
            first.extras["gemini_content"]["parts"][0]["function_call"]["id"], id_a
        )
        self.assertEqual(
            second.extras["gemini_content"]["parts"][0]["function_call"]["id"], id_b
        )

        conversation = Conversation()
        conversation.add(Message(role="user", content="list"))
        conversation.add(first)
        with tempfile.TemporaryDirectory() as tmp:
            store = ConversationStore(Path(tmp) / "c.db")
            store.create("id-demo", "gemini")
            store.save("id-demo", conversation)
            reloaded = store.load("id-demo").conversation.messages[1]
        self.assertEqual(reloaded.tool_calls[0].id, id_a)
        self.assertEqual(
            reloaded.extras["gemini_content"]["parts"][0]["function_call"]["id"], id_a
        )

    def test_native_metadata_serialization_failure_raises(self) -> None:
        content = SimpleNamespace(
            parts=[
                SimpleNamespace(
                    text=None,
                    function_call=SimpleNamespace(
                        name="list_invoices", args={}, id="keep"
                    ),
                )
            ],
            model_dump=MagicMock(side_effect=TypeError("cannot dump")),
        )
        response = SimpleNamespace(candidates=[SimpleNamespace(content=content)])
        with self.assertRaises(ProviderResponseError) as ctx:
            gemini_response_to_message(response)
        self.assertIsInstance(ctx.exception.__cause__, TypeError)
        self.assertIn("serialize Gemini native content", str(ctx.exception))

    def test_function_response_uses_user_role_for_gemini_38(self) -> None:
        """Gemini rejects role='tool'; map function responses to USER."""
        from google.genai import types

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
                content=json.dumps({"ok": True, "result": {"total": "21948.00"}}),
            )
        )
        contents = conversation_to_gemini_contents(conversation)
        tool_content = contents[2]
        self.assertEqual(tool_content.role, "user")
        self.assertIsInstance(tool_content, types.Content)
        dumped = tool_content.model_dump(mode="json", exclude_none=True)
        restored = types.Content.model_validate(dumped)
        self.assertEqual(restored.role, "user")
        self.assertEqual(restored.parts[0].function_response.name, "read_invoice")


if __name__ == "__main__":
    unittest.main()
