# FourX take-home: build a small agent harness

## How this works

1. You get this folder.
2. You build the agent described below.
3. You send us a link to your GitHub repo.
4. We have a **one-hour call**: you run it on your machine, walk us through your code, and
   we talk about how it behaves in situations beyond the tests.

Questions at any point: email nikunj@fourx.ai, the way you'd ask a customer.

## Why this exercise

A model on its own only answers one request. The code around it, the **harness**, turns
it into an agent: it runs the loop of model calls and tool calls, saves the
conversation, pauses when a person must approve something, lets someone stop it, and
can move the conversation from one model provider to another.

At FourX, agents do finance work for companies, and the harness is where most of the
hard problems are. This take-home is a small version of that harness for one job. We
care about the plumbing, not about prompts, polish or UI: is the loop right, is
everything saved, can a person stay in control, and do you understand every part of it?

## The scenario

You're building an assistant for an accounts payable clerk at **Acme Foods** (made up,
like everything here). Vendors send invoices. The clerk checks each one against the
vendor list and **posts** it to the company's **ledger** (its accounting system). Once an
invoice is posted it gets paid, so posting is the step that matters: a person approves
every post, and some invoices must never be posted at all.

## What's in this folder

| File | What it is |
|---|---|
| `invoices/invoice-01.json` … `invoice-06.json` | Six vendor invoices |
| `vendors.json` | The vendor list |
| `ledger.py` | A fake ledger (the accounting system). Use it as it is: don't change it |
| `pyproject.toml` | Add your dependencies here |
| `.gitignore` | Already ignores `ledger.db`, `.venv/` and `.env` |

**An invoice** has the vendor's `invoice_number`, `invoice_date`, `due_date`,
`vendor_name`, `currency`, `lines`, `subtotal`, `taxes` (Indian GST) and `total`. Money
is written as text (`"16862.20"`) so it's exact.

**A vendor** has a `vendor_id` (`V-001` …), a `name` matching the invoice's
`vendor_name`, and a `status`: `active`, or `blocked` with a `blocked_reason`. A blocked
vendor's invoices must never be posted.

**The ledger** is `ledger.py`: see the next section.

## The ledger: how your agent posts

`ledger.py` stands in for the company's accounting system. It's a plain Python file, not
a server: there's nothing to start or keep running. It keeps its data in `ledger.db`, a
SQLite file it creates next to itself the first time it's used, with one invoice
already posted.

You use it in two ways.

### 1. Your code imports it

Put your code in this folder, next to `ledger.py`, and import it like any Python module.
These are the only two functions there are:

```python
import ledger

entry_id = ledger.post(
    vendor_id="V-001",
    invoice_number="SGP/26-27/0412",
    invoice_date="2026-10-03",
    currency="INR",
    amount="16862.20",               # the invoice total
    reference="anything-you-like",   # optional free text
)
# entry_id is the new entry's id, e.g. "JE-4F1A92C0"

ledger.entries()                           # every posted entry, oldest first (a list of dicts)
ledger.entries(vendor_id="V-002")          # filter on any column
ledger.entries(invoice_number="BL-88190")
```

It behaves like a real accounting system:
- `post` takes about 2 seconds.
- Nothing stops the same invoice being posted twice: that's your code's job.
- The entry is saved partway through the call. If your program dies before `post`
  returns, the entry is in the ledger but you never got its id.
- An amount that isn't a positive number raises `ValueError`.

### 2. You run it in the terminal

To see what's been posted, or to start over:

```bash
python ledger.py          # list every posted entry
python ledger.py reset    # delete everything except the one opening entry
```

On a fresh folder, `python ledger.py` prints:

```text
JE-000001    2026-10-01T10:15:00  V-002  BL-88190         INR     25960.00
1 entry
```

### How a post flows through your agent

The model never touches the ledger. It can only ask your program to run a tool, and your
program decides whether and how to run it:

1. The clerk types "post invoice-01".
2. Your program sends the conversation and the list of tools to the model.
3. The model replies with a tool call, for example `post_invoice(invoice_id="invoice-01")`.
4. Your program shows the clerk what would be posted and waits for approval.
5. On approve, your program runs its `post_invoice` function, which calls
   `ledger.post(...)`. The entry is now in `ledger.db`.
6. Your program sends the result (the entry id) back to the model.
7. The model answers: "Posted as JE-4F1A92C0."

The tool behind step 5 can be as small as this. The checks and how approval works are
yours to design:

```python
import json
from pathlib import Path

import ledger


def post_invoice(invoice_id: str) -> dict:
    """Your program calls this only after the clerk approves."""
    invoice = json.loads(Path(f"invoices/{invoice_id}.json").read_text())
    # Your checks go here too: a blocked vendor, an invoice already in the ledger.
    entry_id = ledger.post(
        vendor_id=...,  # look it up in vendors.json
        invoice_number=invoice["invoice_number"],
        invoice_date=invoice["invoice_date"],
        currency=invoice["currency"],
        amount=invoice["total"],
    )
    return {"entry_id": entry_id}
```

## What to build

A command-line agent the clerk chats with in plain English ("post invoice-01", "which
invoices are from Bluefin?").

**Its tools.** At least: list the invoices, read an invoice, look up a vendor, and post an
invoice to the ledger. You choose the tools' names and inputs, and which checks happen in
your code rather than being left to the model.

**It must:**

1. **Run the agent loop.** Call the model, run the tools it asks for, send back the
   results, and repeat until it answers.
2. **Ask before posting.** When the model wants to post, show the person exactly what
   will be posted (vendor, invoice number, date, amount) and wait. **Approve**: post it,
   and the agent carries on. **Decline** (with an optional reason): don't post; tell the
   agent, and it carries on without posting.
3. **Switch providers.** A command switches to a model from a different provider. The
   conversation carries on with the same history, including earlier tool calls and
   their results, and you can switch back.
4. **Save everything and resume.** The conversation is saved to disk (SQLite or files,
   your choice) as it goes. Quit at any point, start the program again, and pick up the
   same conversation, including an approval that was still waiting.
5. **Stop.** While the agent is working (waiting on the model or running a tool), the
   person can stop it. It stops within a few seconds, and the next message works as
   normal.
6. **Never post what it shouldn't.** Nothing from a blocked vendor, and nothing already
   in the ledger (same vendor and invoice number). Enforce both in your code, not only
   in the prompt.

**Out of scope:** a web UI, PDFs, logins, multiple users, deployment, streaming, and
building a general framework. Keep it to what this job needs.

## What a session could look like

This is only an example: design your own interface. An interactive prompt with commands
like these is the simplest.

```text
$ python agent.py --conversation demo
Conversation "demo" (new). Model: gemini. Commands: /model gemini|groq, /approve,
/decline <reason>, /quit. Press Ctrl-C while the agent is working to stop it.

you> post invoice-01
agent> Invoice SGP/26-27/0412 from Shree Ganesh Packaging Pvt Ltd (V-001, active),
       total INR 16862.20. It isn't in the ledger yet.

  Approval needed: post to the ledger
    vendor    V-001  Shree Ganesh Packaging Pvt Ltd
    invoice   SGP/26-27/0412  dated 2026-10-03
    amount    INR 16862.20
  Type /approve or /decline <reason>

you> /approve
agent> Posted as JE-4F1A92C0.

you> /model groq
Switched to groq.

you> what was the total of the invoice you just posted?
agent> INR 16862.20, for invoice SGP/26-27/0412.

you> /quit

$ python agent.py --conversation demo
Conversation "demo" (resumed, 6 messages). Model: groq.
you> ...
```

## Rules

- **Python 3.11 or later.**
- **Two different providers, each through its own official SDK.** Write the agent loop
  yourself. Don't use agent frameworks or libraries that make every provider look the
  same (LangChain, LangGraph, LlamaIndex, LiteLLM, CrewAI, the providers' agent SDKs and
  the like): we want to see your loop and your conversion between providers. Small
  helpers are fine (`python-dotenv`, `rich`, `pytest`).
- **Use `ledger.py` as it is.**
- **Coding agents are welcome** (Claude Code, Codex, Cursor): we use them every day. Be
  ready to explain every part of your code.
- **Keep it small.** A working core and an honest README beat extra features. If
  something's missing, say so in your README.

## Setup

```bash
python3 --version                  # needs 3.11 or later
cd take-home
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install google-genai groq      # the SDKs of the two providers you pick
python ledger.py                   # shows the one opening entry: BL-88190
```

If `python3 --version` shows something older than 3.11, install a newer Python from
[python.org](https://www.python.org/downloads/) (or with Homebrew, pyenv or uv) and use
it to create the `.venv`.

Write your code in this folder, next to `ledger.py` (for example `agent.py`), so
`import ledger` works.

Put your keys in a `.env` file (already git-ignored) or export them in your shell:

```bash
GEMINI_API_KEY=...
GROQ_API_KEY=...
```

Pick a model from each provider that supports **tool calling** (also called function
calling): each provider's models page says which do.

## Getting API keys

Two providers have **free tiers that need no card**:

**Google Gemini** (SDK: `google-genai`)
1. Go to [aistudio.google.com](https://aistudio.google.com) and sign in with a Google
   account.
2. Click **Get API key**, then **Create API key**.
3. Save it as `GEMINI_API_KEY`.

**Groq** (SDK: `groq`; it hosts open-weight models, such as OpenAI's gpt-oss)
1. Go to [console.groq.com](https://console.groq.com) and sign up.
2. Open **API Keys** and click **Create API Key**.
3. Save it as `GROQ_API_KEY`.

Menus may be named slightly differently by the time you read this.

Free tiers allow only a few requests a minute and a limited number a day. That's enough
here, but you may hit a rate limit (HTTP 429): handling it with a short wait and retry is
a plus. Free tiers may also use what you send for training, which is why all the data
here is made up.

Other options:
- **Claude** ([console.anthropic.com](https://console.anthropic.com), SDK `anthropic`) or
  **OpenAI** ([platform.openai.com](https://platform.openai.com), SDK `openai`): paid,
  but this exercise costs well under a dollar on their smaller models.
- **Ollama** ([ollama.com](https://ollama.com), SDK `ollama`): runs a model on your own
  laptop with no key. Needs about 8 GB of free memory and is slower; pick a model that
  supports tools.

Any two different providers are fine.

## Test cases

Start with `python ledger.py reset` and a new conversation, then run these in order in
the same conversation. "Approval" means however your program asks the person.

| # | Do this | Expect this |
|---|---|---|
| 1 | Say "post invoice-01". When asked, **approve** | The approval shows V-001, SGP/26-27/0412, 2026-10-03, INR 16862.20. After approving, `python ledger.py` shows SGP/26-27/0412 **once** |
| 2 | Say "post invoice-02". When the approval appears, **quit the program**. Start it again on the same conversation and **approve** | After restarting, the waiting approval is still there. After approving, BL-88213 (INR 70800.00) is in the ledger once |
| 3 | Say "post invoice-03". **Decline** with the reason "PO not received yet" | Nothing posted. The agent says it didn't post invoice-03 and why |
| 4 | Ask "what's on invoice-06?" Then **switch provider** and ask "which vendor was that from, and what's the total with tax?" Then **switch back** and ask "and how many lines did it have?" | Correct answers on both providers, from the conversation so far: Shree Ganesh Packaging Pvt Ltd, INR 21948.00, one line. No errors after either switch |
| 5 | Say "read every invoice and summarise each one in a line". **Stop it** while it's working. Then ask "which invoices did you get through?" | It stops within a few seconds and nothing is posted. The next answer comes back normally, with no error |
| 6 | Say "post invoice-04" | Not posted: vendor V-004 (Orbit Facility Services) is blocked, bank details under verification. No approval is asked for |
| 7 | Say "post invoice-05" | Not posted: BL-88190 is already in the ledger (JE-000001). No approval is asked for |

**At the end,** `python ledger.py` shows exactly **3 entries**: BL-88190 (the opening
entry), SGP/26-27/0412 and BL-88213.

Test 5 can be over in a few seconds on a fast model. If it finishes before you can stop
it, slow one of your tools down while you test (for example `time.sleep(3)` in your
read-invoice tool).

Writing a few of these as automated tests is welcome but not required.

## What to send us

1. **A GitHub repo** with your code. Private is fine: add **`nikunj-fourx`** as a
   collaborator. Commit as you go rather than all at the end. Don't commit API keys or
   `ledger.db`.
2. **Your own README** (one page at most):
   - how to run it, and which providers and models you used
   - what you save, where, and when
   - how you store the conversation, and how switching providers works with it
   - how approval and stop work
   - what you didn't handle, and what you'd do next

Reply to our email with the link.

## What we look at

- The test cases above work.
- The code is small and clear, and you can explain all of it.
- Your README is honest about what works and what doesn't.

In the call we'll go beyond the test cases, for example what happens if the program is
killed at an awkward moment. Thinking about those cases matters more than handling every
one of them in code.
