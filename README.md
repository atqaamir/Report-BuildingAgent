# DocDacity Document Assistant

DocDacity is a command-line assistant that answers questions about a small set
of sample financial and healthcare documents. It uses LangChain tools and a
LangGraph workflow to route each request to a Q&A, summarization, or
calculation agent.


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

  entry point.
- **No automated tests are included.** Add unit tests for retrieval, tools,
  graph routing, and session persistence before extending or deploying the
  application.
