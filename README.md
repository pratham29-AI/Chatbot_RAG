# RAG Chatbot API

A production-ready conversational AI chatbot built with **FastAPI**, **LangChain**, **OpenAI GPT-4o**, and **FAISS**.  
The assistant can answer questions from uploaded documents, search the web in real time, and perform safe mathematical calculations — all through a single streaming chat API.

---

## Table of Contents

1. [Features](#features)  
2. [Architecture](#architecture)  
3. [Tech Stack & Rationale](#tech-stack--rationale)  
4. [Project Structure](#project-structure)  
5. [Installation](#installation)  
6. [Environment Variables](#environment-variables)  
7. [Running the Application](#running-the-application)  
8. [API Reference](#api-reference)  
9. [End-to-End Usage Examples](#end-to-end-usage-examples)  
10. [Design Decisions & Trade-offs](#design-decisions--trade-offs)  
11. [Future Improvements](#future-improvements)  

---

## Features

| Feature | Details |
|---|---|
| **Streaming chat** | Token-by-token SSE responses via `text/event-stream` |
| **Session history** | Full multi-turn conversation memory per session |
| **Document Q&A (RAG)** | Upload PDF / TXT → chunk → embed → FAISS → retrieve → answer |
| **Deny out-of-scope queries** | Explicit refusal when the answer is not in the documents and no appropriate tool applies |
| **Tool 1 — Web Search** | Tavily API: current events, real-time facts, news |
| **Tool 2 — Calculator** | Safe AST-based math evaluator (no `eval`), supports trig / log / sqrt |
| **Autonomous tool selection** | GPT-4o decides which tool(s) to call based on the query |
| **FAISS index persistence** | Index survives restarts; new documents are appended incrementally |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        Client                            │
│  (curl / browser / any HTTP client)                      │
└──────────────┬───────────────────────────────────────────┘
               │  HTTP / SSE
               ▼
┌──────────────────────────────────────────────────────────┐
│                     FastAPI App                          │
│                                                          │
│  POST /documents/upload ──► RAG Service                  │
│                               │ PyPDF / TextLoader       │
│                               │ RecursiveCharacterSplitter│
│                               │ OpenAI Embeddings         │
│                               ▼                          │
│                            FAISS Index (disk-persisted)  │
│                                                          │
│  POST /chat/sessions/{id}/message                        │
│    │                                                     │
│    ▼                                                     │
│  Session Service  ──► chat_history (HumanMessage /       │
│  (in-memory dict)      AIMessage list)                   │
│    │                                                     │
│    ▼                                                     │
│  LangChain AgentExecutor  (shared singleton)             │
│    │                                                     │
│    ├─► search_documents ──► FAISS similarity search      │
│    ├─► web_search       ──► Tavily API                   │
│    └─► calculator       ──► safe AST evaluator           │
│    │                                                     │
│    ▼                                                     │
│  OpenAI GPT-4o  (streaming)                              │
│    │                                                     │
│    ▼                                                     │
│  SSE token stream ──► Client                             │
└──────────────────────────────────────────────────────────┘
```

### Request lifecycle (chat message)

1. Client sends `POST /chat/sessions/{session_id}/message` with a JSON body.
2. The router fetches (or creates) the session and its message history.
3. The `AgentExecutor` receives `{input: message, chat_history: [...]}`.
4. GPT-4o decides whether to call tools or answer directly.
5. Tool results are fed back to GPT-4o, which generates the final answer.
6. Tokens stream to the client via SSE as they are produced.
7. After the stream completes, the exchange is appended to session history.

---

## Tech Stack & Rationale

| Component | Choice | Why |
|---|---|---|
| **LLM** | OpenAI GPT-4o | Best-in-class tool-calling accuracy; native function schema support |
| **Embeddings** | `text-embedding-3-small` | Best price/performance ratio; 1536 dimensions |
| **Vector DB** | FAISS (CPU) | Zero infrastructure, disk-persisted, fast cosine search, no external service needed |
| **Agent framework** | LangChain `create_openai_tools_agent` | Mature, well-documented, native OpenAI tool-calling support |
| **Web search** | Tavily | Purpose-built for LLMs; returns clean snippets, not raw HTML; 1 000 free calls/month |
| **Calculator** | Custom AST evaluator | No `eval()` security risk; supports full arithmetic + math functions |
| **Framework** | FastAPI | Async-native, automatic OpenAPI docs, `StreamingResponse` for SSE |
| **Streaming** | `astream_events(version="v2")` | Event-level granularity; allows filtering intermediate tool-call tokens |

---

## Project Structure

```
chatbot_rag/
├── app/
│   ├── main.py                  # FastAPI app, lifespan, middleware, routers
│   ├── config.py                # Pydantic Settings (reads .env)
│   ├── prompts/
│   │   └── system_prompt.py     # Aria's persona + tool-usage rules
│   ├── tools/
│   │   ├── web_search_tool.py   # Tool 1: Tavily web search
│   │   └── calculator_tool.py   # Tool 2: safe AST math evaluator
│   ├── services/
│   │   ├── rag_service.py       # FAISS indexing + retrieval (singleton)
│   │   ├── session_service.py   # In-memory session + history store
│   │   └── agent_service.py     # AgentExecutor factory + streaming helper
│   └── routers/
│       ├── chat.py              # /chat/* endpoints
│       └── documents.py         # /documents/* endpoints
├── data/
│   └── faiss_index/             # Auto-created; gitignored
│       ├── index.faiss          # FAISS binary index
│       ├── index.pkl            # Docstore pickle
│       └── documents.json       # Lightweight document metadata
├── .env.example                 # Template — copy to .env and fill in keys
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.11 or 3.12
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [Tavily API key](https://app.tavily.com) (free tier)

### Steps

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd suffescom_rag

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env .env
# Open .env and fill in OPENAI_API_KEY and TAVILY_API_KEY
```

---

## Environment Variables

Copy `.env` to `.env` and set the values below.

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | OpenAI API key |
| `TAVILY_API_KEY` | ✅ | — | Tavily Search API key |

---

## Running the Application

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now available at **http://localhost:8000**.  
Interactive docs: **http://localhost:8000/docs**

---

## API Reference

### Health

| Method | Path | Description |
|---|---|---|
| GET | `/` | Basic status check |
| GET | `/health` | Status + indexed document count |

---

### Documents

#### Upload a document

```
POST /documents/upload
Content-Type: multipart/form-data
```

| Field | Type | Description |
|---|---|---|
| `file` | file | PDF or TXT file (max 50 MB) |

**Response**
```json
{
  "message": "'report.pdf' indexed successfully.",
  "filename": "report.pdf",
  "chunk_count": 42,
  "page_count": 8
}
```

**curl example**
```bash
curl -X POST http://localhost:8000/documents/upload \
     -F "file=@/path/to/report.pdf"
```

---

#### List indexed documents

```
GET /documents
```

```json
{
  "document_count": 1,
  "documents": [
    {"filename": "report.pdf", "chunk_count": 42, "page_count": 8}
  ]
}
```

---

#### Clear all documents

```
DELETE /documents
```

```json
{"message": "All documents have been removed from the index."}
```

---

### Chat

#### Create a session

```
POST /chat/sessions
```

```json
{"session_id": "3f7a8b2c-...", "message": "Session created."}
```

---

#### Send a message (streaming)

```
POST /chat/sessions/{session_id}/message
Content-Type: application/json
```

**Request body**
```json
{"message": "Summarise the key findings in the uploaded report."}
```

**Response** — `text/event-stream`

Each token:
```
data: {"type": "token", "content": "The"}

data: {"type": "token", "content": " key"}

data: {"type": "done"}
```

On error:
```
data: {"type": "error", "content": "..."}
```

**curl example (streaming)**
```bash
curl -N -X POST http://localhost:8000/chat/sessions/YOUR_SESSION_ID/message \
     -H "Content-Type: application/json" \
     -d '{"message": "What does the document say about revenue?"}'
```

---

#### Get session history

```
GET /chat/sessions/{session_id}
```

```json
{
  "session_id": "3f7a8b2c-...",
  "message_count": 4,
  "created_at": "2025-01-01T10:00:00Z",
  "last_active": "2025-01-01T10:05:00Z",
  "history": [
    {"role": "user", "content": "What is the report about?"},
    {"role": "assistant", "content": "The report covers..."}
  ]
}
```

---

#### Delete a session

```
DELETE /chat/sessions/{session_id}
```

---

## End-to-End Usage Examples

### Example 1: Document Q&A

```bash
# Step 1 — upload a PDF
curl -X POST http://localhost:8000/documents/upload \
     -F "file=@annual_report.pdf"

# Step 2 — create a session
SESSION=$(curl -s -X POST http://localhost:8000/chat/sessions | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

# Step 3 — ask a question about the document
curl -N -X POST http://localhost:8000/chat/sessions/$SESSION/message \
     -H "Content-Type: application/json" \
     -d '{"message": "What was the total revenue in Q3?"}'

# Step 4 — follow-up question (history is maintained)
curl -N -X POST http://localhost:8000/chat/sessions/$SESSION/message \
     -H "Content-Type: application/json" \
     -d '{"message": "How does that compare to Q2?"}'
```

---

### Example 2: Web Search (tool auto-selected)

```bash
curl -N -X POST http://localhost:8000/chat/sessions/$SESSION/message \
     -H "Content-Type: application/json" \
     -d '{"message": "What is the latest news about OpenAI?"}'
```

The agent will call `web_search("latest news about OpenAI")` automatically and synthesise the results into a response.

---

### Example 3: Calculator (tool auto-selected)

```bash
curl -N -X POST http://localhost:8000/chat/sessions/$SESSION/message \
     -H "Content-Type: application/json" \
     -d '{"message": "What is sqrt(144) + 2 to the power of 8?"}'
```

The agent calls `calculator("sqrt(144) + 2 ** 8")` and returns `148`.

---

### Example 4: Out-of-scope denial

```bash
curl -N -X POST http://localhost:8000/chat/sessions/$SESSION/message \
     -H "Content-Type: application/json" \
     -d '{"message": "Write me a poem about dragons"}'
```

The agent responds politely that it is designed for document Q&A, web lookups, and calculations, and declines.

---

### Example 5: Python client (collect full response)

```python
import httpx, json

BASE = "http://localhost:8000"

# Create session
session_id = httpx.post(f"{BASE}/chat/sessions").json()["session_id"]

def chat(message: str) -> str:
    full = []
    with httpx.stream(
        "POST",
        f"{BASE}/chat/sessions/{session_id}/message",
        json={"message": message},
        timeout=60,
    ) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if event["type"] == "token":
                    full.append(event["content"])
                    print(event["content"], end="", flush=True)
                elif event["type"] == "done":
                    break
    print()
    return "".join(full)

chat("Upload a document first, then ask: What is the main topic?")
```

---

## Design Decisions & Trade-offs

### 1. In-memory session store
**Decision:** Sessions are stored in a Python `dict` on the application process.  
**Why:** Simplest correct solution for a single-process deployment. No external dependency.  
**Trade-off:** Sessions are lost on restart. In production, replace with Redis or a database — only the `SessionService._store` needs to change; all callers are unaffected by the interface.

### 2. FAISS over ChromaDB / Pinecone
**Decision:** `faiss-cpu` with disk persistence.  
**Why:** Zero external infrastructure. Runs fully offline after setup. Fast for document collections up to millions of vectors. The `save_local` / `load_local` API handles persistence transparently.  
**Trade-off:** No native metadata filtering, no real-time multi-process sync. For a hosted multi-tenant system, Pinecone or Qdrant would be better choices.

### 3. Single global FAISS index
**Decision:** All uploaded documents share one FAISS index, regardless of session.  
**Why:** Keeps the implementation simple and makes all documents available to all sessions (a common use case for a knowledge-base assistant).  
**Trade-off:** No per-session or per-user document isolation. If isolation is needed, maintain one index per user/session and load by ID.

### 4. `create_openai_tools_agent` over LangGraph
**Decision:** LangChain's `AgentExecutor` with OpenAI tools agent.  
**Why:** Sufficient for this task; less boilerplate than LangGraph for a linear tool-use loop. GPT-4o's native tool-calling handles multi-step tool chaining correctly.  
**Trade-off:** Less control over agent state transitions compared to LangGraph. For complex multi-agent workflows, LangGraph is the better choice.

### 5. Manual history management over `RunnableWithMessageHistory`
**Decision:** History is fetched from `SessionService`, passed as `chat_history`, and written back after streaming completes.  
**Why:** More transparent and debuggable. Avoids the implicit session-key contract that `RunnableWithMessageHistory` requires.  
**Trade-off:** Slightly more code in the router, but the behavior is explicit and easy to trace.

### 6. SSE over WebSocket
**Decision:** Server-Sent Events (`text/event-stream`) for streaming.  
**Why:** SSE is one-directional (server → client), which matches a chat response stream exactly. It works over plain HTTP/1.1, requires no protocol upgrade, and is natively supported by browsers via `EventSource`.  
**Trade-off:** WebSocket would allow bidirectional streaming (e.g., the client cancelling mid-stream). For a simple Q&A chatbot, SSE is sufficient and simpler to implement and test with curl.

### 7. Relevance score threshold for RAG
**Decision:** Chunks below `RETRIEVAL_SCORE_THRESHOLD` (default 0.30) are discarded even if they are the "top" results.  
**Why:** Prevents the agent from hallucinating answers when no genuinely relevant chunk exists. The system prompt reinforces this by instructing the agent to deny rather than guess.  
**Trade-off:** A threshold that is too high will cause false negatives (relevant content not retrieved). 0.30 is a conservative default; tune based on your document corpus.

---

## Future Improvements

- **Redis-backed sessions** — persist conversations across restarts
- **Per-user document isolation** — multiple FAISS indices keyed by user ID
- **Document deletion** — remove a specific document and rebuild the index
- **Hybrid search** — combine BM25 (keyword) with FAISS (semantic) for better recall
- **Re-ranking** — apply a cross-encoder to re-rank retrieved chunks before passing to the LLM
- **Streaming tool output** — surface tool-call progress events to the client (e.g., "Searching the web…")
- **Rate limiting** — protect the API with per-IP or per-session limits
- **Authentication** — API key or JWT middleware
- **Docker Compose** — one-command local setup
