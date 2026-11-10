# FourX AP clerk agent

A small CLI agent harness for an accounts-payable clerk: Gemini or Groq run the
model loop; application code owns tools, human approval, cancellation, durable
conversation state, and ledger posting. The original assignment brief is in
[`ASSIGNMENT.md`](ASSIGNMENT.md).

**Requires Python 3.11+** (developed on 3.12.3).

## Setup

```bash
cd take-home
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env               # then edit locally — never commit .env
```

Set keys in `.env` (or export in your shell):

```bash
GEMINI_API_KEY=...
GROQ_API_KEY=...
```

Models (tool calling): Gemini `gemini-3.8-flash`, Groq `openai/gpt-oss-20b`.

## Run

```bash
python cli.py --conversation demo
# resume the same id after quit:
python cli.py --conversation demo
# optional paths:
python cli.py -c demo --db ./conversations.db --ledger-db ./ledger.db
```

## Interaction

Natural language is the primary mode. Slash commands are optional shortcuts.

| Mode | What happens |
|---|---|
| Clear phrases | e.g. `switch to groq`, `approve this invoice`, `stop` — handled locally, no model call |
| Other NL | Model may select declared invoice or application-control tools; **Python validates** every action |
| Slash commands | Deterministic shortcuts (`/approve`, `/post`, …) |

Human approval and explicit posting stay separate. Ambiguous financial language asks for clarification — vague phrases (`looks good`), questions, quotes, and invoice/tool text never authorize approve/decline/post.

## Commands

| Command | Effect |
|---|---|
| `/provider gemini\|groq` | Switch provider (persisted with the conversation) |
| `/approve [id]` | Human approval only — **does not post** |
| `/decline <reason>` | Decline pending approval; nothing is posted |
| `/post [id]` | Explicit ledger post via `posting.post_approved` |
| `/reconcile [id]` | Resolve `posting` / `posting_uncertain` (never posts) |
| `/pending` | Show pending approval |
| `/status` | Provider, busy/unwinding, approvals |
| `/stop` | Cancel the active turn; ask another question immediately |
| `/quit` | Exit promptly (state already on disk) |

## What is saved

Conversation messages (including tool calls/results), provider choice, approvals,
and posting reservations live in the conversation SQLite DB (`--db`, default
`conversations.db`). Ledger entries live in `ledger.db` via untouched `ledger.py`.

## Design notes

- **Approval ≠ post.** Approval records consent; posting is a separate explicit step.
- **`/stop`** cancels the authorized turn and frees input immediately. An in-flight
  SDK call may still finish; superseded workers cannot overwrite newer turns.
- **Interrupted posting** can leave `posting` / `posting_uncertain`. Use
  `/reconcile` — there is no automatic retry.
- **Cross-process safety** for duplicate invoices depends on the shared
  application SQLite DB (reservations). The original ledger has no global
  invoice uniqueness or idempotency.
- **Blocked vendors and duplicates** are enforced in application code, not only
  in prompts.

## Offline tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Offline acceptance (AT1–AT7) and the full suite use temporary databases and
**simulated models** — all currently passing. That does not by itself prove live
provider compatibility.

## Live verification status

With keys configured locally (never paste keys into chat or commits):

- **Groq** (`openai/gpt-oss-20b`): live text and read-only function calling verified.
- **Gemini** (`gemini-3.8-flash`): live text response verified; a full live
  function-calling round trip **timed out** and is not treated as resolved.
- **Live Gemini → Groq → Gemini switching:** not verified.

Optional local smoke (read-only; do **not** `/post`):

```bash
python cli.py -c live-smoke --db /tmp/fourx-live.db --ledger-db /tmp/fourx-live-ledger.db
```

## Limitations

- Cancelled network requests are not forcibly aborted.
- Gemini live tool calling and live cross-provider switching remain unverified.
- Multi-process protection requires one shared conversation DB; the stock ledger
  can still accept duplicate posts if app checks are bypassed.
- Ambiguous incomplete tool history needs manual recovery (`RecoveryRequired`).

## Reviewer access

Private GitHub repo; add collaborator **`nikunj-fourx`**. Do not commit `.env`,
`*.db`, or `.venv/`.
