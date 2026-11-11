"""Focused checks for the fourx package layout (offline only)."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class PackageStructureTests(unittest.TestCase):
    def test_package_imports(self) -> None:
        from fourx.cli import main
        from fourx.agent import harness, conversation, cancellation
        from fourx.tools import invoice_tools, app_actions, nl_routing
        from fourx.finance import approval, posting
        from fourx.persistence import storage, ledger_guard
        from fourx.diagnostics import run_logging, app_state
        from fourx.providers import make_model

        self.assertTrue(callable(main))
        self.assertTrue(hasattr(harness, "run_agent_turn"))
        self.assertTrue(hasattr(conversation, "Message"))
        self.assertTrue(hasattr(cancellation, "CancellationToken"))
        self.assertTrue(hasattr(invoice_tools, "list_invoices"))
        self.assertTrue(hasattr(app_actions, "execute_app_control_tool"))
        self.assertTrue(hasattr(nl_routing, "match_deterministic_action"))
        self.assertTrue(hasattr(approval, "approve"))
        self.assertTrue(hasattr(posting, "approve_and_post"))
        self.assertTrue(hasattr(storage, "ConversationStore"))
        self.assertTrue(hasattr(ledger_guard, "using_ledger"))
        self.assertTrue(hasattr(run_logging, "RunLogger"))
        self.assertTrue(hasattr(app_state, "build_trusted_state_summary"))
        self.assertTrue(callable(make_model))

    def test_fixture_paths_resolve_to_repo_root(self) -> None:
        from fourx.tools.invoice_tools import INVOICES_DIR, ROOT, VENDORS_PATH, list_invoices

        self.assertEqual(ROOT.resolve(), REPO_ROOT.resolve())
        self.assertTrue(INVOICES_DIR.is_dir())
        self.assertTrue(VENDORS_PATH.is_file())
        self.assertEqual(len(list_invoices()), 6)

    def test_single_ledger_module_identity(self) -> None:
        import ledger
        from fourx.persistence import ledger_guard
        from fourx.tools import invoice_tools
        from fourx.finance import posting

        self.assertIs(invoice_tools.ledger, ledger)
        self.assertIs(posting.ledger, ledger)
        self.assertIs(ledger_guard.ledger, ledger)

    def test_single_ledger_guard_identity(self) -> None:
        from fourx.persistence import ledger_guard as lg1
        from fourx.persistence.ledger_guard import ledger_lock
        import fourx.persistence.ledger_guard as lg2

        self.assertIs(lg1, lg2)
        self.assertIs(lg1.ledger_lock(), ledger_lock())

    def test_cli_module_entrypoints_importable(self) -> None:
        root_cli = importlib.import_module("cli")
        pkg_main = importlib.import_module("fourx.__main__")
        self.assertTrue(callable(root_cli.main))
        self.assertTrue(callable(pkg_main.main))
        self.assertIs(root_cli.main, pkg_main.main)

    def test_launch_from_different_cwd_lists_invoices(self) -> None:
        """Fixture resolution must not depend on the process working directory."""
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            code = (
                "from fourx.tools.invoice_tools import list_invoices, ROOT; "
                "print(len(list_invoices())); print(ROOT)"
            )
            proc = subprocess.run(
                [sys.executable, "-c", code],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            lines = proc.stdout.strip().splitlines()
            self.assertEqual(lines[0], "6")
            self.assertEqual(Path(lines[1]).resolve(), REPO_ROOT.resolve())

    def test_existing_db_resume_after_restructure(self) -> None:
        import tempfile
        from unittest.mock import patch

        import ledger
        from fourx.agent.conversation import Message
        from fourx.cli import FourXCLI
        from fourx.persistence.storage import ConversationStore

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            conv_db = root / "c.db"
            ledger_db = root / "l.db"
            with patch.object(ledger, "DB_PATH", ledger_db):
                with patch.object(ledger.time, "sleep", return_value=None):
                    ledger.entries()
                    store = ConversationStore(conv_db)
                    store.create("resume", "gemini")
                    loaded = store.load("resume")
                    loaded.conversation.add(Message(role="user", content="hello"))
                    store.save("resume", loaded.conversation)

                    reopened = ConversationStore(conv_db)
                    cli = FourXCLI(
                        reopened,
                        "resume",
                        model_factory=lambda _p: (
                            lambda *_a, **_k: Message(role="assistant", content="ok")
                        ),
                        ledger_db=ledger_db,
                    )
                    again = reopened.load("resume")
                    self.assertEqual(again.provider, "gemini")
                    self.assertEqual(len(again.conversation.messages), 1)
                    self.assertEqual(again.conversation.messages[0].content, "hello")
                    self.assertEqual(cli.conversation_id, "resume")


if __name__ == "__main__":
    unittest.main()
