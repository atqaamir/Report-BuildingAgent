# DocDacity Document Assistant

DocDacity is a command-line assistant that answers questions about a small set
of sample financial and healthcare documents. It uses LangChain tools and a
LangGraph workflow to route each request to a Q&A, summarization, or
calculation agent.

> **Project status:** This is a sample application, not a production document
> platform. The included retriever searches five documents defined in code; it
> does not ingest arbitrary files or use a vector database. See
> [Known limitations](#known-limitations) before relying on session persistence
> or running the full workflow.

## Requirements

- Python 3.10 or later
- An API key accepted by the OpenAI-compatible endpoint configured in
  `src/assistant.py` (`https://openai.vocareum.com/v1`)

## Setup

Run these commands from the `starter` directory.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and replace `your_openai_api_key_here` with your own API key.
Keep `.env` private; do not commit it or paste its contents into issues or
messages. The repository's `.gitignore` excludes it.

The current command-line entry point reads `OPENAI_API_KEY`. Although
`.env.example` also lists `MODEL_NAME`, `TEMPERATURE`, and
`SESSION_STORAGE_PATH`, `main.py` currently supplies the model and temperature
directly, and the assistant uses a fixed API base URL.

## Run

With the virtual environment active, run:

```bash
python main.py
```

Enter a user ID when prompted, or press Enter to use `demo_user`. Then ask a
question about the sample documents. The interactive commands are:

| Command | Action |
| --- | --- |
| `/help` | Show command and example-query help |
| `/docs` | List the documents currently available to the retriever |
| `/quit` | Exit the assistant |

Example questions:

- “What is the total due on invoice INV-002?”
- “Summarize contract CON-001.”
- “Add up the invoice totals.”
- “Find documents over $50,000.”

## Included sample documents

The in-memory `SimulatedRetriever` loads these records when the application
starts:

| ID | Type | Example content |
| --- | --- | --- |
| `INV-001` | Invoice | Acme Corporation consulting invoice |
| `INV-002` | Invoice | TechStart Inc. enterprise license invoice |
| `INV-003` | Invoice | Global Corp annual subscription invoice |
| `CON-001` | Contract | Healthcare Partners service agreement |
| `CLM-001` | Claim | Medical expense reimbursement claim |

This data is defined in `src/retrieval.py`; restarting the program resets the
retriever to these examples.

## How it works

1. `src/agent.py` classifies the request and routes it to an agent.
2. The selected agent uses the prompt and structured response schema for its
   task.
3. Available tools search documents, read a document, report collection
   statistics, and calculate basic arithmetic expressions.
4. The memory node summarizes the messages and records document IDs.
5. LangGraph uses an in-memory checkpointer keyed by the session ID while the
   process is running.

![LangGraph agent architecture](docs/langgraph_agent_architecture.png)

## Project layout

```text
starter/
├── main.py                 # Interactive command-line entry point
├── requirements.txt        # Python dependencies
├── .env.example            # Example local configuration
├── .gitignore              # Excludes local credentials and runtime files
├── docs/
│   └── langgraph_agent_architecture.png
└── src/
    ├── agent.py             # Agent state, nodes, and LangGraph workflow
    ├── assistant.py         # LLM, tools, and session orchestration
    ├── prompts.py           # Intent, agent, and memory prompts
    ├── retrieval.py         # In-memory sample documents and search
    ├── schemas.py           # Pydantic response and state schemas
    └── tools.py             # Calculator, search, reader, and statistics tools
```

## Tests

There is no automated test suite in the repository yet. The project has not
been validated end to end against the configured language-model endpoint.

## Known limitations

- **Intent classification needs a correction:** the classifier enables
  `with_structured_output(UserIntent)` and then passes its result to a
  `PydanticOutputParser`, which expects text rather than an already-parsed
  model. This can stop requests before they reach an agent.
- **Session save and resume need validation/fixes:** the assistant stores
  LangChain message objects in its JSON session history, but its JSON encoder
  only handles datetimes. Saving a completed turn can therefore fail, and the
  in-memory LangGraph checkpointer itself does not survive a process restart.
- **Some tool and display paths are incomplete:** the document search tool's
  `"all"` option can fall through to query-based search, and the CLI checks for
  `active_documents` even though the assistant returns the list as `sources`.
- **Some schema defaults need cleanup:** several list/dictionary fields use
  default factories that return the `list` or `dict` type instead of an empty
  value.
- **The retriever is a demonstration:** document search is keyword/metadata
  based over hard-coded examples, not semantic search over user-provided files.
- **The CLI currently uses fixed model settings and a fixed API base URL.**
  The corresponding optional values in `.env.example` are not wired to the
  entry point.
- **No automated tests are included.** Add unit tests for retrieval, tools,
  graph routing, and session persistence before extending or deploying the
  application.
