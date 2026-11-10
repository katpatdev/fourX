# FourX — end-to-end system guide

This document explains **what the program does**, **how the pieces fit together**,
**what you type**, and **what happens in the databases**. It is a learning /
interview aid. For a short runbook see [`README.md`](README.md). For the original
brief see [`ASSIGNMENT.md`](ASSIGNMENT.md).

---

## 1. What this program is

You are building a **CLI accounts-payable clerk assistant**.

- The clerk types normal English (or slash commands).
- A **model** (Gemini or Groq) may answer in text or ask to run **tools**.
- **Your Python code** runs those tools, saves history, asks a human before any
  ledger post, and never lets blocked/duplicate invoices through on trust of the
  model alone.
- The fake company ledger is untouched `ledger.py` (SQLite `ledger.db`).

Think of it as:

```text
You (clerk)  →  cli.py  →  harness (agent loop)  →  Gemini/Groq
                              ↓
                    tools + approval + posting
                              ↓
              conversations.db          ledger.db
              (your app state)          (accounting)
```

---

## 2. How to start it

### Configuration

| Item | Where | Purpose |
|------|--------|---------|
| Python | ≥ 3.11 (developed on 3.12) | Runtime |
| Dependencies | `pyproject.toml` | `google-genai`, `groq` only |
| API keys | `.env` (from `.env.example`) | `GEMINI_API_KEY`, `GROQ_API_KEY` |
| Gemini model | `providers/gemini.py` | Default `gemini-3.8-flash` |
| Groq model | `providers/groq.py` | Default `openai/gpt-oss-20b` |
| Conversation DB | `--db` (default `conversations.db`) | Chat, approvals, reservations |
| Ledger DB | `--ledger-db` / `ledger.DB_PATH` | Journal entries |

Never commit `.env`. Keys are loaded by `cli.load_dotenv()` without overriding
variables already set in the shell.

### Launch

```bash
source .venv/bin/activate
python cli.py --conversation demo
# same id resumes the same chat:
python cli.py -c demo --db ./conversations.db --ledger-db ./ledger.db
```

On start you see provider, DB paths, and any **pending approval** or unresolved
posting status.

---

## 3. What you type

### Ordinary text (natural language)

Anything that does **not** start with `/` is a user message.

Examples:

```text
you> what invoices are available?
you> what's on invoice-06?
you> post invoice-01
you> which invoices are from Bluefin?
```

Flow for ordinary text:

1. If an **authorized** agent turn is still running → “busy; use `/stop`”.
2. If a **pending approval** exists → must `/approve` or `/decline` first.
3. Otherwise:
   - Reconcile any trailing incomplete tool calls (if safe).
   - Save the user message under run-ownership.
   - Start a **background worker** that calls `run_agent_turn`.

### Slash commands

| You type | What happens |
|----------|----------------|
| `/help` | Show command help |
| `/provider gemini` or `/provider groq` | Switch model provider; persisted on the conversation |
| `/approve` or `/approve <approval_id>` | Approve pending request **and post** via `approve_and_post` |
| `/decline <reason>` | Decline pending approval; reason stored; nothing posted |
| `/post` or `/post <approval_id>` | Actually call `ledger.post` via `posting.post_approved` |
| `/reconcile` or `/reconcile <id>` | Resolve `posting` / `posting_uncertain` by looking at the ledger (never posts again blindly) |
| `/pending` | Show the pending approval |
| `/status` | Provider, message count, busy/unwinding, approvals |
| `/stop` | Cancel the current turn; you may ask another question **immediately** |
| `/quit` or `/exit` | Exit promptly (state already on disk) |

**Important:** the model cannot approve or post. Only you can, with `/approve`
(which posts safely) or `/post` for approved/uncertain leftovers.

---

## 4. Module map (what each file is for)

| File | Role |
|------|------|
| `cli.py` | Interactive shell, workers, `/stop` / `/quit`, wires everything |
| `harness.py` | Agent loop: call model → execute tools → save → repeat |
| `conversation.py` | In-memory `Message` / `ToolCall` / `Conversation` |
| `storage.py` | SQLite for conversations, approvals, posting reservations |
| `invoice_tools.py` | Read fixtures + `prepare_post` eligibility checks |
| `approval.py` | Create / approve / decline human approval records |
| `posting.py` | Safe `post_approved` + `reconcile_posting` |
| `cancellation.py` | `CancellationToken`, run ownership, `run_write` |
| `providers/gemini.py` | Official Gemini adapter (`generate_content`, manual tools) |
| `providers/groq.py` | Official Groq adapter (chat completions, manual tools) |
| `providers/__init__.py` | `make_model("gemini"\|"groq")` |
| `ledger.py` | **Do not modify** — fake accounting API |
| `invoices/*.json`, `vendors.json` | Fixtures |

---

## 5. Tools the model can request

Declared in `harness.tool_declarations()` and executed only by **your** code
(never by the SDK automatically):

| Tool | Input | What it does |
|------|--------|----------------|
| `list_invoices` | (none) | Summaries of `invoice-01` … `invoice-06` |
| `read_invoice` | `invoice_id` | Full invoice JSON |
| `lookup_vendor` | `name` and/or `vendor_id` | Vendor row from `vendors.json` |
| `post_invoice` | `invoice_id` | Runs eligibility (`prepare_post`); if OK, creates a **pending approval** and stops the turn. **Does not** call `ledger.post` |

Blocked vendor or duplicate invoice → error tool result, **no** approval, **no** post.

---

## 6. Core flows

### 6.1 Normal Q&A / read-only tools

```text
you> what's on invoice-06?
        │
        ▼
  save user message → start worker → run_agent_turn
        │
        ▼
  model may call read_invoice
        │
        ▼
  harness executes tool, saves tool result, calls model again
        │
        ▼
  model returns final text → printed to you
```

History (user / assistant / tool) is stored in `conversations.messages_json`.

### 6.2 Posting an invoice (happy path)

#### Assignment wording (implemented)

`ASSIGNMENT.md`:

- “a person approves every post”
- “**Ask before posting.** … **Approve**: post it, and the agent carries on.”
- Demo: `you> /approve` → `agent> Posted as JE-…`

Flow:

```text
you> post invoice-01
        │
        ▼
  model calls post_invoice(invoice_id="invoice-01")
        │
        ▼
  prepare_post: load invoice, match vendor, check not blocked,
                check not already in ledger, validate amount
        │
        ▼
  pending approval saved (details snapshot shown)
  turn stops with "Approval required…"
        │
you> /approve  (or "I approve this" / "Approve this invoice")
        │
        ▼
  approve_and_post:
    record approved → post_approved (re-check, reserve, claim,
    ledger.post once, verify journal id)
        │
        ▼
  ledger.db gains a JE-… row
```

`/post` remains for approved-but-unposted or `posting_uncertain` leftovers.
Ambiguous phrases never authorize. The model cannot approve or post on its own.

If you **decline**:

```text
you> /decline PO not received yet
→ status=declined, reason stored, ledger untouched
```

### 6.3 Provider switch

```text
you> /provider groq
→ conversations.provider updated
→ same messages_json (including prior tool calls/results)
→ next NL turn uses GroqModel
you> /provider gemini
→ switch back; history still there
```

### 6.4 Stop / cancel

```text
you> summarize all invoices     ← worker starts, may block in SDK
you> /stop                      ← cancel token + mark worker “released”
→ input free immediately
you> what has been processed?   ← new turn, new token, supersedes old run
```

Notes:

- An old SDK HTTP call may still finish; it must not overwrite the newer turn
  (run ownership + `StaleRunError`).
- `/stop` never approves, declines, or posts.

### 6.5 Uncertain posting / reconcile

If the process dies after `ledger.post` may have committed but before status is
confirmed, approval can be `posting` or `posting_uncertain`.

```text
you> /reconcile
→ inspect ledger.entries; never auto-retry ledger.post
→ may become posted (with journal id) or stay uncertain
```

---

## 7. Agent loop (harness) in more detail

`run_agent_turn(store, conversation_id, model, cancel)` roughly:

1. `begin_run` — this token owns the conversation for writes.
2. Refuse to start if pending approval or incomplete tool history (unless cancel path).
3. Loop (bounded model calls):
   - `cancel.check()`
   - Call `model(conversation, tool_declarations)`
   - Persist assistant message
   - If **no** tool calls → return final answer
   - If tools: run each in order
     - Read-only tools execute immediately
     - `post_invoice` → eligibility; on success, commit pending approval + any
       skipped-later tools, then raise `ApprovalRequired`
   - Persist tool results; continue loop
4. On cancel: write `SkippedForCancellation` for unanswered tools when safe.
5. `end_run` in `finally`.

Providers only **translate** history ↔ SDK and return a canonical `Message`.
They do **not** execute tools.

---

## 8. Databases and schema

### Two databases

| DB | Default file | Owner | Purpose |
|----|--------------|--------|---------|
| Application | `conversations.db` | `storage.ConversationStore` | Chat + approvals + reservations |
| Ledger | `ledger.db` | `ledger.py` (untouched) | Posted journal entries |

### Application DB tables

**`conversations`**

| Column | Meaning |
|--------|---------|
| `conversation_id` | PK (e.g. `demo`) |
| `provider` | `gemini` or `groq` |
| `messages_json` | Full transcript JSON |

**`approvals`**

| Column | Meaning |
|--------|---------|
| `approval_id` | PK |
| `conversation_id` | FK to conversation |
| `invoice_id` | e.g. `invoice-01` |
| `details_json` | Frozen post payload (vendor, number, date, amount, …) |
| `status` | `pending` / `approved` / `declined` / `posting` / `posted` / `posting_uncertain` |
| `decline_reason` | Optional |
| `journal_entry_id` | Ledger `JE-…` when posted |

Unique partial index: at most one **pending** approval per conversation.

**`posting_reservations`**

| Column | Meaning |
|--------|---------|
| `vendor_id` + `invoice_number` | Composite PK |
| `approval_id` | Owner |
| `conversation_id` | Which chat |

Prevents two chats that share this app DB from both posting the same invoice.
The stock ledger itself has **no** uniqueness constraint.

### Ledger DB table

**`entries`**

| Column | Meaning |
|--------|---------|
| `entry_id` | `JE-…` |
| `vendor_id`, `invoice_number`, `invoice_date`, `currency`, `amount` | Posted invoice |
| `reference` | Optional |
| `posted_at` | Timestamp |

Opens with `JE-000001` / `BL-88190` (invoice-05 duplicate case).

---

## 9. Key functions (cheat sheet)

### CLI (`cli.py`)

- `FourXCLI.handle_line` — one input line
- `_start_agent_turn` / `_run_turn_worker` — daemon worker + identity-safe cleanup
- `append_user_message` / `reconcile_trailing_cancelled_tools` — durable writes under ownership

### Harness (`harness.py`)

- `run_agent_turn` — main loop
- `tool_declarations` / `execute_tool_call`
- `apply_skipped_for_cancellation`

### Tools (`invoice_tools.py`)

- `list_invoices`, `read_invoice`, `lookup_vendor`, `prepare_post`

### Approval (`approval.py`)

- `request_approval`, `approve`, `decline`, `get_pending_approval`, `list_approvals`

### Posting (`posting.py`)

- `post_approved`, `reconcile_posting`

### Cancellation (`cancellation.py`)

- `CancellationToken`, `begin_run` / `end_run`, `run_write`, `assert_run_active`

### Storage (`storage.py`)

- `ConversationStore.create/load/save/set_provider`
- Approval CRUD + claim/post/uncertain helpers
- Reservation insert / lookup

### Providers

- `make_model(provider)` → `GeminiModel` or `GroqModel`
- Both: `__call__(conversation, declarations) -> Message`

---

## 10. Example session (end-to-end)

```text
$ python cli.py -c demo
Conversation "demo" (new, 0 messages). Provider: gemini.
...

you> list the invoices briefly
agent> (may call list_invoices, then summarize)

you> post invoice-01
agent> Approval required before posting:
         vendor V-001 … invoice SGP/26-27/0412 … INR 16862.20
       Type /approve or /decline <reason>.

you> /approve
Approved and posted as JE-… (status=posted).

you> /provider groq
Switched to groq.

you> what was the total we just posted?
agent> INR 16862.20 …

you> /quit
```

Blocked / duplicate examples:

```text
you> post invoice-04     → blocked vendor; no approval
you> post invoice-05     → already in ledger (BL-88190); no approval
```

Stop example:

```text
you> read every invoice and summarise each
you> /stop
Stop requested. You can enter another question now.
you> which invoices did you get through?
```

---

## 11. Tests and live status (context)

```bash
.venv/bin/python -m unittest discover -s tests -v
```

- Offline suite (280 tests) uses **fake models** and temp DBs — includes AT1–AT7.
- Live: Groq text + read-only tools verified; Gemini `gemini-3.5-flash-lite`
  text + read-only tool round-trip verified. Live cross-provider switch under
  load is still lightly covered. See README.

---

## 12. Mental model to remember for an interview

1. **Model suggests; code decides.** Especially for money.
2. **Approve → post.** Explicit human `/approve` (or clear NL) runs `approve_and_post`.
3. **Two databases.** App state vs ledger; only the posting workflow hits `ledger.post`.
4. **History is durable JSON** plus structured approval rows + trusted state.
5. **Stop is cooperative.** Cancel signal + supersession; SDK may still be in flight.
6. **Safety is in Python** (blocked vendor, duplicate, reservations), not in the prompt.

That is the whole system from keyboard input to journal entry.
