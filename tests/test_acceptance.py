"""Offline end-to-end acceptance scenarios AT1–AT7 via the CLI (fake models)."""

from __future__ import annotations

import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import ledger
from approval import list_approvals
from cli import FourXCLI
from conversation import Conversation, Message, ToolCall
from storage import ConversationStore


class ScriptedModel:
    """Returns scripted assistant messages; tracks provider via factory binding."""

    def __init__(self, provider: str, queues: dict[str, list[Message]]) -> None:
        self.provider = provider
        self.queues = queues
        self.block: threading.Event | None = None
        self.entered: threading.Event | None = None

    def __call__(self, conversation: Conversation, tools: list[dict]) -> Message:
        if self.entered is not None:
            self.entered.set()
        if self.block is not None:
            self.block.wait(timeout=30)
        queue = self.queues[self.provider]
        if not queue:
            raise AssertionError(f"ScriptedModel exhausted for {self.provider}")
        return queue.pop(0)


class AcceptanceTests(unittest.TestCase):
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
        # Opening JE-000001 / BL-88190
        ledger.entries()

        self.store = ConversationStore(self.conv_db)
        self.store.create("accept", "gemini")
        self.out = io.StringIO()
        self.queues: dict[str, list[Message]] = {"gemini": [], "groq": []}
        self.cli = FourXCLI(
            self.store,
            "accept",
            model_factory=self._factory,
            stdout=self.out,
            ledger_db=self.ledger_db,
        )
        # Drain before TemporaryDirectory cleanup (addCleanup is LIFO).
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

    def _factory(self, provider: str) -> ScriptedModel:
        return ScriptedModel(provider, self.queues)

    def _send(self, line: str) -> None:
        self.cli.handle_line(line)
        self.cli._wait_turn(timeout=60)

    def _post_tool(self, invoice_id: str, call_id: str) -> Message:
        return Message(
            role="assistant",
            tool_calls=(
                ToolCall(
                    id=call_id,
                    name="post_invoice",
                    arguments={"invoice_id": invoice_id},
                ),
            ),
        )

    def test_all_seven_acceptance_scenarios(self) -> None:
        # --- AT1: Approve and post invoice-01 ---
        self.queues["gemini"] = [self._post_tool("invoice-01", "at1")]
        self._send("post invoice-01")
        self.assertIn("Approval required", self.out.getvalue())
        self.cli.handle_line("/approve")
        inv01 = ledger.entries(vendor_id="V-001", invoice_number="SGP/26-27/0412")
        self.assertEqual(len(inv01), 1)
        self.assertEqual(inv01[0]["amount"], "16862.20")
        self.assertIn("Approved and posted", self.out.getvalue())

        # --- AT2: Pending invoice-02 survives restart ---
        self.queues["gemini"] = [self._post_tool("invoice-02", "at2")]
        self._send("post invoice-02")
        pending_before = self.store.get_pending_approval("accept")
        self.assertIsNotNone(pending_before)
        self.assertEqual(pending_before.invoice_id, "invoice-02")

        reopened = ConversationStore(self.conv_db)
        self.out = io.StringIO()
        self.store = reopened
        self.cli = FourXCLI(
            reopened,
            "accept",
            model_factory=self._factory,
            stdout=self.out,
            ledger_db=self.ledger_db,
        )
        self.cli.startup_banner()
        self.assertIn("Pending approval", self.out.getvalue())
        self.assertIn("BL-88213", self.out.getvalue())
        self.assertIn("70800.00", self.out.getvalue())
        self.cli.handle_line("/approve")
        inv02 = ledger.entries(invoice_number="BL-88213")
        self.assertEqual(len(inv02), 1)
        self.assertEqual(inv02[0]["amount"], "70800.00")
        self.assertIn("Approved and posted", self.out.getvalue())

        # --- AT3: Decline invoice-03 ---
        self.queues["gemini"] = [self._post_tool("invoice-03", "at3")]
        self._send("post invoice-03")
        self.cli.handle_line("/decline PO not received yet")
        declined = [
            a for a in list_approvals(self.store, "accept") if a.invoice_id == "invoice-03"
        ]
        self.assertEqual(len(declined), 1)
        self.assertEqual(declined[0].status, "declined")
        self.assertEqual(declined[0].decline_reason, "PO not received yet")
        self.assertEqual(ledger.entries(invoice_number="KOS-2026-1187"), [])

        # --- AT4: Switch providers around invoice-06 ---
        # Gemini reads invoice-06
        self.queues["gemini"] = [
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="r6",
                        name="read_invoice",
                        arguments={"invoice_id": "invoice-06"},
                    ),
                ),
            ),
            Message(
                role="assistant",
                content="Invoice-06 loaded for Shree Ganesh Packaging Pvt Ltd.",
            ),
        ]
        self._send("what's on invoice-06?")
        self.assertIn("Shree Ganesh", self.out.getvalue())

        self.cli.handle_line("/provider groq")
        self.assertEqual(self.store.load("accept").provider, "groq")
        # Groq answers vendor and total from history/tools (scripted text using fixture facts)
        self.queues["groq"] = [
            Message(
                role="assistant",
                content="Vendor Shree Ganesh Packaging Pvt Ltd, total INR 21948.00.",
            )
        ]
        self._send("which vendor was that from, and what's the total with tax?")
        self.assertIn("Shree Ganesh Packaging Pvt Ltd", self.out.getvalue())
        self.assertIn("21948.00", self.out.getvalue())

        self.cli.handle_line("/provider gemini")
        self.assertEqual(self.store.load("accept").provider, "gemini")
        self.queues["gemini"] = [
            Message(role="assistant", content="Invoice-06 has 1 line item.")
        ]
        self._send("and how many lines did it have?")
        self.assertIn("1 line", self.out.getvalue())

        # History still contains the earlier tool call/result across switches.
        roles = [m.role for m in self.store.load("accept").conversation.messages]
        self.assertIn("tool", roles)

        # --- AT5: Stop an active summary, then continue while old model still blocked ---
        entered = threading.Event()
        block = threading.Event()
        summary_model = ScriptedModel(
            "gemini",
            {
                "gemini": [Message(role="assistant", content="Summary of all invoices.")],
                "groq": [],
            },
        )
        summary_model.entered = entered
        summary_model.block = block
        self.cli.model_factory = lambda _p: summary_model
        before_approvals = len(list_approvals(self.store, "accept"))
        self.cli.handle_line("summarize all invoices")
        self.assertTrue(entered.wait(timeout=5))
        self.cli.handle_line("/stop")
        self.assertIn("Stop requested", self.out.getvalue())
        self.assertFalse(self.cli._blocks_new_turn())
        self.assertEqual(len(list_approvals(self.store, "accept")), before_approvals)

        # Next question must start without waiting for the blocked summary model.
        self.cli.model_factory = self._factory
        self.queues["gemini"] = [
            Message(role="assistant", content="Processed so far: invoice-01 and invoice-02 posted.")
        ]
        self._send("what has been processed?")
        self.assertIn("Processed so far", self.out.getvalue())
        block.set()

        # --- AT6: Blocked invoice-04 ---
        self.queues["gemini"] = [
            self._post_tool("invoice-04", "at6"),
            Message(
                role="assistant",
                content="Cannot post: vendor V-004 bank details under verification.",
            ),
        ]
        before = len(list_approvals(self.store, "accept"))
        with patch("invoice_tools.ledger.post") as mock_post:
            self._send("post invoice-04")
            mock_post.assert_not_called()
        self.assertEqual(len(list_approvals(self.store, "accept")), before)
        lower = self.out.getvalue().lower()
        self.assertTrue(
            "v-004" in lower and "verification" in lower,
            msg="expected blocked-vendor messaging for invoice-04",
        )

        # --- AT7: Duplicate invoice-05 ---
        self.queues["gemini"] = [
            self._post_tool("invoice-05", "at7"),
            Message(
                role="assistant",
                content="Duplicate: BL-88190 already posted as JE-000001.",
            ),
        ]
        before = len(list_approvals(self.store, "accept"))
        with patch("invoice_tools.ledger.post") as mock_post:
            self._send("post invoice-05")
            mock_post.assert_not_called()
        self.assertEqual(len(list_approvals(self.store, "accept")), before)

        # Final ledger: opening + invoice-01 + invoice-02
        all_entries = ledger.entries()
        self.assertEqual(len(all_entries), 3)
        numbers = {e["invoice_number"] for e in all_entries}
        self.assertEqual(numbers, {"BL-88190", "SGP/26-27/0412", "BL-88213"})
        self.assertEqual(len(ledger.entries(invoice_number="BL-88190")), 1)
        self.assertEqual(len(ledger.entries(invoice_number="SGP/26-27/0412")), 1)
        self.assertEqual(len(ledger.entries(invoice_number="BL-88213")), 1)


if __name__ == "__main__":
    unittest.main()
