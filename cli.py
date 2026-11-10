"""Interactive FourX clerk CLI — thin shell over harness / approval / posting.

Launch::

    python cli.py --conversation demo

Optional::

    python cli.py --conversation demo --db ./conversations.db --ledger-db ./ledger.db

Ordinary lines go to the selected model via ``run_agent_turn``. Slash commands
handle provider, approval, posting, stop, and status. The CLI never calls
``ledger.post`` directly.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import ledger
from approval import (
    ApprovalAlreadyDecided,
    ApprovalNotFound,
    ApprovalRequest,
    InvalidDeclineReason,
    approve,
    decline,
    get_pending_approval,
    list_approvals,
)
from cancellation import CancellationToken, RunCancelled, StaleRunError, begin_run, end_run, run_write
from conversation import Message
from harness import (
    ApprovalRequired,
    PersistenceError,
    RecoveryRequired,
    apply_skipped_for_cancellation,
    run_agent_turn,
    unmatched_tool_call_ids,
)
from invoice_tools import BlockedVendor, DuplicateInvoice
from nl_routing import RoutedAction, match_deterministic_action
from posting import (
    ApprovalNotPostable,
    InvoiceReservedByOther,
    LedgerAmbiguity,
    LedgerInvoiceConflict,
    PostingDetailsMismatch,
    PostingError,
    PostingUncertain,
    post_approved,
    reconcile_posting,
)
from providers import make_model
from storage import ConversationNotFound, ConversationStore, StoredApproval

HELP_TEXT = """\
Natural language is the primary interaction mode. Slash commands are optional shortcuts.

Commands:
  /help                         Show this help
  /provider gemini|groq         Switch provider (persisted)
  /approve [approval_id]        Record human approval (does not post)
  /decline <reason>             Decline pending approval with a reason
  /post [approval_id]           Post an approved request via posting.post_approved
  /reconcile [approval_id]      Reconcile posting/posting_uncertain (never posts)
  /pending                      Show unresolved pending approval
  /status                       Provider, approvals, reservations overview
  /stop                         Cancel the active turn; you may ask another question immediately
  /quit                         Exit promptly (state already durable on disk)

Clear phrases like "switch to groq", "approve this invoice", or "stop" are handled
locally. Other requests go to the model; Python validates every application action.
Approval and posting stay separate. Ambiguous financial commands ask for clarification.
A cancelled model worker may still await an SDK response; it cannot overwrite newer turns.
"""

ModelFactory = Callable[[str], Any]


@dataclass
class _TurnWorker:
    """One agent-turn worker. Identity is used so stale finally blocks stay isolated."""

    token: CancellationToken
    thread: threading.Thread
    # After /stop: no longer blocks NL input; may still be unwinding an SDK call.
    released: bool = False
    done: threading.Event = field(default_factory=threading.Event)


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal KEY=VALUE loader. Does not override existing environment variables."""
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def format_approval(approval: ApprovalRequest | StoredApproval) -> str:
    display = (
        approval.display()
        if isinstance(approval, ApprovalRequest)
        else {k: approval.details.get(k) for k in (
            "invoice_id",
            "vendor_id",
            "vendor_name",
            "invoice_number",
            "invoice_date",
            "currency",
            "amount",
        )}
    )
    lines = [
        f"  approval_id   {approval.approval_id}",
        f"  status        {approval.status}",
        f"  invoice_id    {display.get('invoice_id')}",
        f"  vendor        {display.get('vendor_id')}  {display.get('vendor_name')}",
        f"  invoice       {display.get('invoice_number')}  dated {display.get('invoice_date')}",
        f"  amount        {display.get('currency')} {display.get('amount')}",
    ]
    journal = getattr(approval, "journal_entry_id", None)
    if journal:
        lines.append(f"  journal       {journal}")
    reason = getattr(approval, "decline_reason", None)
    if reason:
        lines.append(f"  decline       {reason}")
    return "\n".join(lines)


def reconcile_trailing_cancelled_tools(
    store: ConversationStore, conversation_id: str
) -> int:
    """Fill a trailing incomplete tool batch with SkippedForCancellation under ownership.

    Supersedes any previous run before writing so a stale worker cannot race the
    reconciliation save. Returns the number of skip results persisted.
    """
    token = CancellationToken()
    begin_run(conversation_id, token)
    try:
        with run_write(conversation_id, token):
            stored = store.load(conversation_id)
            if not unmatched_tool_call_ids(stored.conversation):
                return 0
            added = apply_skipped_for_cancellation(stored.conversation)
            if unmatched_tool_call_ids(stored.conversation):
                raise RecoveryRequired(
                    f"conversation {conversation_id!r} still has incomplete tool "
                    f"results after cancellation reconciliation"
                )
            if added:
                store.save(conversation_id, stored.conversation)
            return added
    finally:
        end_run(conversation_id, token)


def append_user_message(
    store: ConversationStore, conversation_id: str, text: str
) -> None:
    """Persist one user message under the same ownership fence as harness writes.

    ``begin_run`` supersedes any cancelled/stale worker before load+save, so an
    old turn cannot overwrite this snapshot afterward.
    """
    token = CancellationToken()
    begin_run(conversation_id, token)
    try:
        with run_write(conversation_id, token):
            stored = store.load(conversation_id)
            missing = unmatched_tool_call_ids(stored.conversation)
            if missing:
                raise RecoveryRequired(
                    f"conversation {conversation_id!r} has incomplete tool results "
                    f"for tool_call_id(s) {missing}; recover before sending messages"
                )
            stored.conversation.add(Message(role="user", content=text))
            store.save(conversation_id, stored.conversation)
    finally:
        end_run(conversation_id, token)


class FourXCLI:
    """Small interactive shell. Injectable I/O and model factory for offline tests."""

    def __init__(
        self,
        store: ConversationStore,
        conversation_id: str,
        *,
        model_factory: ModelFactory | None = None,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
        ledger_db: Path | None = None,
    ) -> None:
        self.store = store
        self.conversation_id = conversation_id
        self.model_factory = model_factory or (lambda provider: make_model(provider))
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self.ledger_db = Path(ledger_db) if ledger_db is not None else Path(ledger.DB_PATH)

        self._worker_lock = threading.Lock()
        self._active: _TurnWorker | None = None

    def print(self, text: str = "") -> None:
        self.stdout.write(text + ("\n" if not text.endswith("\n") else ""))
        self.stdout.flush()

    def _provider(self) -> str:
        return self.store.load(self.conversation_id).provider

    def ensure_conversation(self, default_provider: str = "gemini") -> None:
        try:
            self.store.load(self.conversation_id)
        except ConversationNotFound:
            self.store.create(self.conversation_id, default_provider)

    def startup_banner(self) -> None:
        stored = self.store.load(self.conversation_id)
        n = len(stored.conversation.messages)
        kind = "resumed" if n else "new"
        self.print(
            f'Conversation "{self.conversation_id}" ({kind}, {n} messages). '
            f"Provider: {stored.provider}."
        )
        self.print(f"Conversation DB: {self.store.db_path}")
        self.print(f"Ledger DB:       {self.ledger_db}")
        self.print("Type /help for commands.")
        pending = get_pending_approval(self.store, self.conversation_id)
        if pending is not None:
            self.print("\nPending approval (no model call until resolved):")
            self.print(format_approval(pending))
            self.print("Use /approve or /decline <reason>.")
        self._announce_unresolved_posting()

    def _announce_unresolved_posting(self) -> None:
        for row in self.store.list_unprocessed_approved(self.conversation_id):
            if row.status in {"posting", "posting_uncertain"}:
                self.print(
                    f"\nUnresolved posting attempt {row.approval_id} "
                    f"(status={row.status}). Use /reconcile — never auto-retries."
                )
                self.print(format_approval(row))

    def _blocks_new_turn(self) -> bool:
        """True only while an authorized, not-yet-stopped turn is running."""
        with self._worker_lock:
            active = self._active
            return (
                active is not None
                and not active.released
                and active.thread.is_alive()
            )

    def handle_line(self, line: str) -> bool:
        """Handle one input line. Returns False when the session should exit."""
        text = line.rstrip("\n")
        stripped = text.strip()
        if not stripped:
            return True

        if stripped.startswith("/"):
            return self._handle_command(stripped)

        # Mode B / local stop while a model call is blocked — no LLM wait.
        routed = match_deterministic_action(stripped)
        if self._blocks_new_turn():
            if routed is not None and routed.action == "stop":
                return self._handle_command("/stop")
            if routed is not None and routed.action == "quit":
                return self._handle_command("/quit")
            if routed is not None and routed.action in {"status", "pending", "help"}:
                return self._dispatch_routed(routed)
            self.print(
                "Agent is busy. Type /stop or say 'stop' to cancel, then send your message."
            )
            return True

        if routed is not None:
            return self._dispatch_routed(routed)

        pending = get_pending_approval(self.store, self.conversation_id)
        # With a pending approval, still allow the model so the clerk can approve,
        # decline, or ask clarifying questions (Mode C tools validate authorization).
        allow_pending = pending is not None

        try:
            # Supersede any cancelled worker before persisting the new user line.
            reconcile_trailing_cancelled_tools(self.store, self.conversation_id)
            append_user_message(self.store, self.conversation_id, stripped)
        except RecoveryRequired as exc:
            self.print(f"Recovery required: {exc}")
            return True
        except StaleRunError as exc:
            self.print(f"Could not save user message: {exc}")
            return True

        self._start_agent_turn(
            user_message=stripped,
            allow_pending_resolution=allow_pending,
        )
        return True

    def _dispatch_routed(self, routed: RoutedAction) -> bool:
        """Execute a Mode-B deterministic natural-language shortcut."""
        if routed.action == "stop":
            return self._handle_command("/stop")
        if routed.action == "quit":
            return self._handle_command("/quit")
        if routed.action == "help":
            return self._handle_command("/help")
        if routed.action == "status":
            return self._handle_command("/status")
        if routed.action == "pending":
            return self._handle_command("/pending")
        if routed.action == "provider" and routed.provider:
            return self._handle_command(f"/provider {routed.provider}")
        if routed.action == "approve":
            return self._handle_command("/approve")
        if routed.action == "decline" and routed.reason:
            return self._handle_command(f"/decline {routed.reason}")
        if routed.action == "post":
            # Mode-B post phrases are whole-utterance and do not authorize approve.
            return self._handle_command("/post")
        self.print("Could not interpret that shortcut. Try /help.")
        return True

    def _handle_command(self, command: str) -> bool:
        parts = command.split(maxsplit=1)
        name = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if name in {"/quit", "/exit"}:
            with self._worker_lock:
                active = self._active
                if active is not None:
                    active.token.cancel()
                    active.released = True
            self.print(
                "Goodbye. "
                "Any cancelled model worker may still await an SDK response; "
                "it cannot overwrite durable conversation state."
            )
            return False

        if name == "/help":
            self.print(HELP_TEXT)
            return True

        if name == "/stop":
            with self._worker_lock:
                active = self._active
                if active is None or not active.thread.is_alive():
                    self.print("No active model turn to stop.")
                    return True
                active.token.cancel()
                active.released = True
            self.print(
                "Stop requested. You can enter another question now. "
                "In-flight provider calls may finish; superseded results are discarded."
            )
            return True

        if self._blocks_new_turn() and name not in {"/status", "/pending", "/help"}:
            self.print("Agent is busy. Type /stop first.")
            return True

        if name == "/provider":
            return self._cmd_provider(arg)
        if name in {"/approve"}:
            return self._cmd_approve(arg)
        if name == "/decline":
            return self._cmd_decline(arg)
        if name == "/post":
            return self._cmd_post(arg)
        if name == "/reconcile":
            return self._cmd_reconcile(arg)
        if name == "/pending":
            return self._cmd_pending()
        if name == "/status":
            return self._cmd_status()

        self.print(f"Unknown command {name!r}. Type /help.")
        return True

    def _cmd_provider(self, arg: str) -> bool:
        provider = arg.strip().lower()
        if provider not in {"gemini", "groq"}:
            self.print("Usage: /provider gemini|groq")
            return True
        try:
            self.store.set_provider(self.conversation_id, provider)
        except Exception as exc:  # noqa: BLE001
            self.print(f"Could not switch provider: {exc}")
            return True
        self.print(f"Switched to {provider}.")
        return True

    def _cmd_approve(self, arg: str) -> bool:
        approval_id = arg.strip() or None
        try:
            result = approve(self.store, self.conversation_id, approval_id)
        except (ApprovalNotFound, ApprovalAlreadyDecided) as exc:
            self.print(f"Approve failed: {exc}")
            return True
        self.print("Approved. Not yet posted. Use /post to post to the ledger.")
        self.print(format_approval(result))
        return True

    def _cmd_decline(self, arg: str) -> bool:
        if not arg.strip():
            self.print("Usage: /decline <reason>")
            return True
        try:
            result = decline(self.store, self.conversation_id, arg)
        except (ApprovalNotFound, ApprovalAlreadyDecided, InvalidDeclineReason) as exc:
            self.print(f"Decline failed: {exc}")
            return True
        self.print("Declined. Nothing was posted.")
        self.print(format_approval(result))
        return True

    def _resolve_approval_id(self, arg: str) -> str | None:
        if arg.strip():
            return arg.strip()
        pending = get_pending_approval(self.store, self.conversation_id)
        if pending is not None:
            self.print("Cannot post while a pending approval exists; /approve first.")
            return None
        rows = self.store.list_unprocessed_approved(self.conversation_id)
        approved = [r for r in rows if r.status == "approved"]
        postingish = [r for r in rows if r.status in {"posting", "posting_uncertain"}]
        if len(approved) == 1 and not postingish:
            return approved[0].approval_id
        if len(postingish) == 1 and not approved:
            return postingish[0].approval_id
        if approved or postingish:
            self.print("Multiple unresolved approvals; pass approval_id explicitly.")
            for row in rows:
                self.print(format_approval(row))
            return None
        # Fall back to most recent posted for idempotent re-display? Prefer clear miss.
        history = list_approvals(self.store, self.conversation_id)
        posted = [a for a in history if a.status == "posted"]
        if len(posted) == 1:
            return posted[-1].approval_id
        self.print("No approved/posting request found. Approve a pending request first.")
        return None

    def _cmd_post(self, arg: str) -> bool:
        approval_id = self._resolve_approval_id(arg)
        if approval_id is None:
            return True
        self.print(f"Posting via posting.post_approved (ledger DB: {self.ledger_db}) ...")
        try:
            result = post_approved(self.store, self.conversation_id, approval_id)
        except PostingUncertain as exc:
            self.print(f"Posting uncertain: {exc}")
            self.print(format_approval(exc.approval))
            self.print("Use /reconcile after inspecting the ledger. No automatic retry.")
            return True
        except (
            ApprovalNotPostable,
            DuplicateInvoice,
            BlockedVendor,
            LedgerInvoiceConflict,
            InvoiceReservedByOther,
            PostingDetailsMismatch,
            PostingError,
        ) as exc:
            self.print(f"Posting refused: {type(exc).__name__}: {exc}")
            return True
        except Exception as exc:  # noqa: BLE001
            self.print(f"Posting failed: {type(exc).__name__}: {exc}")
            return True
        self.print(
            f"Posting confirmed as {result.journal_entry_id} "
            f"(status={result.status})."
        )
        self.print(format_approval(result))
        return True

    def _cmd_reconcile(self, arg: str) -> bool:
        approval_id = arg.strip()
        if not approval_id:
            rows = [
                r
                for r in self.store.list_unprocessed_approved(self.conversation_id)
                if r.status in {"posting", "posting_uncertain"}
            ]
            if len(rows) != 1:
                self.print("Usage: /reconcile <approval_id>")
                for row in rows:
                    self.print(format_approval(row))
                return True
            approval_id = rows[0].approval_id
        try:
            result = reconcile_posting(self.store, self.conversation_id, approval_id)
        except PostingUncertain as exc:
            self.print(f"Still unresolved: {exc}")
            self.print(format_approval(exc.approval))
            return True
        except (ApprovalNotPostable, PostingError) as exc:
            self.print(f"Reconcile refused: {exc}")
            return True
        self.print(
            f"Reconciled: status={result.status} journal={result.journal_entry_id}"
        )
        self.print(format_approval(result))
        return True

    def _cmd_pending(self) -> bool:
        pending = get_pending_approval(self.store, self.conversation_id)
        if pending is None:
            self.print("No pending approval.")
            return True
        self.print("Pending approval:")
        self.print(format_approval(pending))
        return True

    def _cmd_status(self) -> bool:
        stored = self.store.load(self.conversation_id)
        with self._worker_lock:
            active = self._active
            busy = (
                active is not None
                and not active.released
                and active.thread.is_alive()
            )
            released_pending = (
                active is not None
                and active.released
                and active.thread.is_alive()
            )
        self.print(f"conversation: {self.conversation_id}")
        self.print(f"provider:     {stored.provider}")
        self.print(f"messages:     {len(stored.conversation.messages)}")
        self.print(f"busy:         {busy}")
        self.print(f"unwinding:    {released_pending}")
        self.print(f"conv DB:      {self.store.db_path}")
        self.print(f"ledger DB:    {self.ledger_db}")
        history = list_approvals(self.store, self.conversation_id)
        if not history:
            self.print("approvals:    (none)")
            return True
        self.print("approvals:")
        for item in history:
            self.print(format_approval(item))
            self.print("")
        return True

    def _start_agent_turn(
        self,
        *,
        user_message: str = "",
        allow_pending_resolution: bool = False,
    ) -> None:
        with self._worker_lock:
            if (
                self._active is not None
                and not self._active.released
                and self._active.thread.is_alive()
            ):
                self.print("Agent already running.")
                return
            token = CancellationToken()
            # Placeholder thread replaced below; dataclass needs a Thread instance.
            worker = _TurnWorker(
                token=token,
                thread=threading.Thread(target=lambda: None),
            )
            thread = threading.Thread(
                target=self._run_turn_worker,
                args=(worker, user_message, allow_pending_resolution),
                name=f"fourx-agent-turn-{token.run_id[:8]}",
                daemon=True,
            )
            worker.thread = thread
            self._active = worker
            thread.start()

    def _run_turn_worker(
        self,
        worker: _TurnWorker,
        user_message: str = "",
        allow_pending_resolution: bool = False,
    ) -> None:
        try:
            provider = self._provider()
            try:
                model = self.model_factory(provider)
            except Exception as exc:  # noqa: BLE001
                self.print(f"Model setup failed: {exc}")
                return
            try:
                final = run_agent_turn(
                    store=self.store,
                    conversation_id=self.conversation_id,
                    model=model,
                    cancel=worker.token,
                    user_message=user_message,
                    allow_pending_resolution=allow_pending_resolution,
                )
                # If this worker was superseded/released, do not print late answers.
                with self._worker_lock:
                    still_current = self._active is worker and not worker.released
                if not still_current:
                    return
                if final.content:
                    self.print(final.content)
                else:
                    self.print("(assistant returned no text)")
            except ApprovalRequired as exc:
                with self._worker_lock:
                    still_current = self._active is worker and not worker.released
                if still_current:
                    self.print("Approval required before posting:")
                    self.print(format_approval(exc.approval))
                    self.print("Type /approve or /decline <reason>.")
            except RunCancelled:
                if not worker.released:
                    self.print("Turn cancelled.")
            except StaleRunError:
                # Expected after /stop + newer turn; stay quiet unless still current.
                if not worker.released:
                    self.print("Turn superseded; discarded stale results.")
            except RecoveryRequired as exc:
                if not worker.released:
                    self.print(f"Recovery required: {exc}")
            except PersistenceError as exc:
                if not worker.released:
                    self.print(f"Persistence error: {exc}")
            except Exception as exc:  # noqa: BLE001
                if not worker.released:
                    self.print(f"Agent error: {type(exc).__name__}: {exc}")
        finally:
            worker.done.set()
            with self._worker_lock:
                # Never clear a newer worker's slot.
                if self._active is worker:
                    self._active = None

    def wait_for_current_turn(self, timeout: float | None = None) -> None:
        """Wait for the current authorized worker to finish (tests / helpers)."""
        with self._worker_lock:
            worker = self._active
        if worker is None:
            return
        worker.done.wait(timeout=timeout)

    def _wait_turn(self, timeout: float | None = None) -> None:
        """Backward-compatible alias used by older tests."""
        self.wait_for_current_turn(timeout=timeout)

    def run_interactive(self) -> int:
        """Main stdin loop. After /stop, accepts the next question immediately."""
        self.ensure_conversation()
        self.startup_banner()
        while True:
            try:
                self.stdout.write("you> ")
                self.stdout.flush()
                line = self.stdin.readline()
            except KeyboardInterrupt:
                self.print("\n(Use /stop to cancel a turn, or /quit to exit.)")
                with self._worker_lock:
                    if self._active is not None:
                        self._active.token.cancel()
                        self._active.released = True
                continue
            if line == "":
                with self._worker_lock:
                    if self._active is not None:
                        self._active.token.cancel()
                        self._active.released = True
                self.print(
                    "Goodbye. Cancelled workers may still await an SDK response."
                )
                return 0
            cont = self.handle_line(line)
            # While an authorized (not stopped) turn runs, keep reading for /stop.
            while self._blocks_new_turn():
                if self.stdin is sys.stdin and sys.stdin.isatty():
                    self.stdout.write("you> ")
                    self.stdout.flush()
                    try:
                        cmd = self.stdin.readline()
                    except KeyboardInterrupt:
                        with self._worker_lock:
                            if self._active is not None:
                                self._active.token.cancel()
                                self._active.released = True
                        self.print("\nStop requested. You can enter another question now.")
                        break
                    if cmd == "":
                        with self._worker_lock:
                            if self._active is not None:
                                self._active.token.cancel()
                                self._active.released = True
                        self.print("Goodbye.")
                        return 0
                    if not self.handle_line(cmd):
                        return 0
                else:
                    self.wait_for_current_turn(timeout=120.0)
            if not cont:
                return 0
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FourX interactive AP clerk agent")
    parser.add_argument(
        "--conversation",
        "-c",
        default="demo",
        help="Conversation id to create or resume (default: demo)",
    )
    parser.add_argument(
        "--db",
        default="conversations.db",
        help="Path to the conversation/approval SQLite database",
    )
    parser.add_argument(
        "--ledger-db",
        default=None,
        help="Override ledger.DB_PATH for this process (default: ledger.db beside ledger.py)",
    )
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "groq"],
        help="Initial provider when creating a new conversation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_arg_parser().parse_args(argv)
    if args.ledger_db:
        ledger.DB_PATH = Path(args.ledger_db)
    store = ConversationStore(args.db)
    app = FourXCLI(
        store,
        args.conversation,
        ledger_db=ledger.DB_PATH,
    )
    app.ensure_conversation(default_provider=args.provider)
    # If conversation existed with another provider, --provider only applies to create.
    return app.run_interactive()


if __name__ == "__main__":
    raise SystemExit(main())
