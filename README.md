# Advanced RAG System — Production Grade with LangGraph + GCP

> **A fully enterprise-ready Retrieval-Augmented Generation system built for learning and production.**
> Includes LangGraph orchestration, Hybrid Retrieval, Semantic Re-ranking (True Data vs Noisy Data),
> LLM Gateway, Guardrails, RAGAS Evaluations, a beautiful chat UI, Docker, and GCP Terraform.

---

## What This Project Teaches You

| Topic | Where to Find It |
|---|---|
| LangGraph state machines | `src/graph/pipeline.py`, `src/graph/nodes.py` |
| Hybrid retrieval (Vector + BM25) | `src/retrieval/hybrid_retriever.py` |
| Cross-encoder re-ranking | `src/retrieval/hybrid_retriever.py` (SemanticReRanker) |
| True Data vs Noisy Data | `src/retrieval/hybrid_retriever.py` (reranker_threshold) |
| LLM Gateway (LiteLLM) | `src/gateway/llm_gateway.py` |
| Input/Output Guardrails | `src/guardrails/` |
| PII detection (Presidio) | `src/guardrails/input_guardrails.py` |
| RAGAS Evaluations | `src/evals/evaluator.py` |
| FastAPI REST API | `src/api/` |
| Structured logging | `src/observability/logger.py` |
| LangFuse tracing | `src/observability/tracer.py` |
| Docker multi-stage builds | `docker/Dockerfile` |
| GCP Terraform | `infra/terraform/` |

---

## System Architecture

```
User Browser (HTML/JS UI)
        |
        v
FastAPI REST API  (src/api/)
        |
        v
LangGraph Pipeline  (src/graph/)
        |
   +----|----+--------+--------+--------+--------+--------+
   |         |        |        |        |        |        |
Input     Query    History  Hybrid   Context  LLM Gen  Output
Guards   Planner  Rewriter Retrieval Assembly  (LiteLLM) Guards
   |                          |                          |
   |               Vector + BM25 + RRF             RAGAS Eval
   |               + CrossEncoder Rerank
   |               + True/Noisy Classification
   |
   v
ChromaDB (local)  OR  Vertex AI Vector Search (GCP)
```

---

## Cost: 100% FREE for Learning

| Component | Local (Free) | Production (GCP) |
|---|---|---|
| LLM | Gemini API free tier | Vertex AI Gemini |
| Embeddings | sentence-transformers (local) | Vertex AI text-embedding |
| Vector DB | ChromaDB (local) | Vertex AI Vector Search |
| Tracing | LangFuse free tier | LangFuse / Cloud Logging |
| Deploy | Docker on laptop | Cloud Run |

**You need ZERO paid services to run and learn from this project locally.**

---

## Quick Start (5 Minutes, Free)

### Prerequisites
- Python 3.11+
- pip
- (Optional) Docker Desktop

### Step 1: Clone and Setup

```bash
# Navigate to the project
cd advanced-rag-gcp

# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy the example config
copy .env.example .env

# The defaults work for local development (no GCP needed)
# Only set OPENAI_API_KEY or GCP credentials if you have them
```

The default `.env` uses:
- **ChromaDB** (free, runs locally)
- **sentence-transformers** (free, runs locally)
- **Gemini free tier** for LLM (no key needed with gcloud ADC)

### Step 3: Start the Server

If `.env` sets `OLLAMA_MODEL=llama3.2`, install and start the local model
service first. Leave `GEMINI_API_KEY` and `OPENAI_API_KEY` empty in `.env`:
any non-empty value makes the app use that provider instead of Ollama.

**Install Ollama once** (the installer is not stored in this repository):

| OS | Install |
|---|---|
| macOS | `brew install ollama`, or the app from https://ollama.com/download |
| Windows | Run `OllamaSetup.exe` from https://ollama.com/download, or `winget install Ollama.Ollama` |
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` |

On Windows, the installed app runs the Ollama service in the background, so
skip `ollama serve` below. On Linux, the install script registers a systemd
service; if `ollama list` fails, start it with `sudo systemctl start ollama`
or run `ollama serve` in a terminal.

Download the model once and check it is available:

```bash
ollama pull llama3.2
ollama list        # should show llama3.2
```

Download the embedding and re-ranking models once (the server runs offline):

```bash
# macOS/Linux (use your virtual environment's python, e.g. .venv/bin/python on Linux)
.venv-macos/bin/python download_model.py
# Windows
.venv\Scripts\python.exe download_model.py
```

Then start the services (macOS shown):

```bash
# Terminal 1: keep the model service running
ollama serve
```

```bash
# Terminal 2: start the app
.venv-macos/bin/python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

On Linux, use your virtual environment's Python the same way (for example
`.venv/bin/python -m uvicorn ...`). On Windows (PowerShell):

```powershell
.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

`.venv-macos` is the macOS environment created for this checkout. If using
your own environment, use its Python executable instead. FastAPI serves both
the frontend and backend; no separate frontend server is needed. An Ollama
connection error means the model service is unavailable on port 11434.

```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

### Step 4: Open the UI

Open your browser at: **http://localhost:8000**

You should see the dark-themed chat interface.

### Step 5: Ingest a Document and Ask Questions

Phase 2 document management is available under **Knowledge Sources → Documents**.
Uploads from either that tab or the chat sidebar register a source, retain the
original in `data/documents/`, and index chunks with the original filename and
source ID. Each document has a status, chunk count, **Reindex**, and **Delete**.
Reindex replaces that source's chunks; delete removes its original, registry
entry, vectors, and keyword-search entries. Uploads accept PDF, DOCX, UTF-8 TXT,
and Markdown up to 50 MB each. Legacy `.doc` files should be saved as `.docx`.
Files uploaded before document management appear under **Older uploads** with a
text preview; they stay searchable and can be removed, but not reindexed.

### Network policy (company networks)

Set `ALLOW_EXTERNAL_NETWORK=false` in `.env` to keep every call inside the
company network. Only hosts under `INTERNAL_DOMAINS` (default `nxp.com`, e.g.
`bitbucket.sw.nxp.com`), `localhost` (Ollama) and private IP addresses can be
contacted. Cloud LLMs, Hugging Face downloads, LiteLLM's price-list fetch,
Chroma / LiteLLM telemetry, Langfuse cloud, spaCy model downloads, Google Cloud
Storage, bitbucket.org and Jira Cloud are blocked with a clear error, and only
the local Ollama model is used even if API keys are set. Run
`download_model.py` while external access is still allowed, then switch it off.

### Indexing speed

Each chunk stores a hash of its text; re-indexing and Sync reuse the stored
vector of every unchanged chunk, so only changed files are embedded again (the
first index of a repository still embeds everything). Embedding uses all CPU
cores but one (`EMBEDDING_THREADS`). Several sources can index at the same time.

### Bitbucket and Jira

Add a repository (**Knowledge Sources → Bitbucket**) or project (**→ Jira**),
then click **Index**. Indexing runs in the background and the card shows its
progress; previously indexed content stays searchable until it succeeds.

- **Bitbucket Cloud and Bitbucket Server / Data Center** are both supported. In
  **Add Repository**, choose the type. For a company-hosted server, paste the
  repository page link (e.g. `https://bitbucket.company.com/projects/KEY/repos/repo/browse`)
  to fill in the project key and repository, and use an **HTTP access token**
  (avatar > Manage account > HTTP access tokens, *Repository read*); the username
  is optional. Cloud uses the workspace, account email and an API token.
  On networks that inspect HTTPS, `pip-system-certs` (in requirements) makes
  Python trust the company certificate.
- **Bitbucket** downloads the branch's latest commit as one archive and indexes
  readable source and text files (max 400 KB each), plus a file-tree overview.
  It skips build/dependency folders, lock files, binaries, and likely secrets
  (`.env`, keys). If the configured branch doesn't exist, the repository's
  default branch is used. **Sync** re-indexes only when there are new commits.
  The token needs repository read access.
- **Add many at once:** in **Add Repository**, leave Repository empty and click
  **Browse repositories** to list a project's repositories (or, on a company
  server with no project key, every repository you can access); filter, tick or
  **Select all**, then **Add N selected**. Each becomes its own source; indexing
  is queued and runs `INDEX_PARALLEL_JOBS` (default 2) at a time.
- **Confluence** (Cloud or Server / Data Center) indexes **every current page of
  a space**. Paste a space or page link (the space key is filled in), or click
  **Browse spaces** to add several spaces at once. Company servers use a
  Personal Access Token (avatar > Profile > Personal Access Tokens); Cloud uses
  email + API token. Pages keep their title, page-tree path and link; Sync
  re-fetches all pages and reuses vectors of unchanged ones.
- **Jira Server / Data Center** (e.g. `jira.company.com`) uses a Personal Access
  Token with the email left empty; Jira Cloud uses email + API token.
- **Jira** indexes each issue's key fields, description, and latest 30 comments
  (up to 5,000 issues) using an Atlassian API token. **Sync** re-fetches all issues.

Under **Search In** in the chat sidebar, untick sources (or whole types) to limit
which sources an answer may use; with everything ticked, all sources are searched.

Document management currently uses local ChromaDB and a single API worker.
Keep `data/documents/`, `data/sources.db`, and the vector database together when
backing up. Documents uploaded before Phase 2 remain searchable but have no
retained originals or source registry entries; they are not automatically
converted into managed sources. Uploading them again creates managed copies and
does not remove the older chunks. Bitbucket indexing and chat source filtering
are still later phases.

Document source endpoints:

```text
POST   /api/v1/sources/documents                    multipart file upload
GET    /api/v1/sources/documents                    list documents
GET    /api/v1/sources/documents/{source_id}         inspect document
POST   /api/v1/sources/documents/{source_id}/reindex replace indexed chunks
DELETE /api/v1/sources/documents/{source_id}         remove document
```

1. Click **"Choose Files"** in the left panel
2. Upload any `.txt`, `.pdf`, or `.docx` file
3. Wait for "X chunks added" confirmation
4. Type a question about the document in the chat box
5. Watch the **RAG Internals** panel on the right — it shows:
   - Which pipeline nodes ran
   - Which chunks are **True Data** vs **Noisy Data**
   - **RAGAS evaluation scores** per response

---

## Running with Docker (Alternative)

```bash
# Copy and fill in .env
copy .env.example .env

# Start all services (API + ChromaDB)
docker-compose up

# Open http://localhost:8000
```

---

## Project Structure

```
advanced-rag-gcp/
|
|-- src/
|   |-- config.py                    # All settings (pydantic-settings)
|   |
|   |-- graph/
|   |   |-- state.py                 # LangGraph state definition
|   |   |-- nodes.py                 # One function per pipeline node
|   |   +-- pipeline.py              # Graph wiring + RAGPipeline runner
|   |
|   |-- ingestion/
|   |   |-- document_loader.py       # Load/chunk PDF, DOCX, TXT, MD
|   |   |-- embedder.py              # Embedding model wrapper
|   |   +-- ingestion_pipeline.py    # End-to-end ingest orchestration
|   |
|   |-- retrieval/
|   |   |-- vector_store.py          # ChromaDB / Vertex AI abstraction
|   |   +-- hybrid_retriever.py      # BM25 + Vector + RRF + Re-ranking
|   |
|   |-- gateway/
|   |   +-- llm_gateway.py           # LiteLLM with retry/fallback/budget
|   |
|   |-- guardrails/
|   |   |-- input_guardrails.py      # PII, injection, length checks
|   |   +-- output_guardrails.py     # Hallucination, toxicity, refusal
|   |
|   |-- evals/
|   |   +-- evaluator.py             # RAGAS faithfulness/relevancy/precision
|   |
|   |-- observability/
|   |   |-- logger.py                # Structured JSON logging (structlog)
|   |   +-- tracer.py                # LangFuse distributed tracing
|   |
|   +-- api/
|       |-- main.py                  # FastAPI app + startup/shutdown
|       +-- routes/
|           |-- query.py             # POST /query, /query/stream
|           |-- ingest.py            # POST /ingest/file, /ingest/gcs
|           +-- health.py            # GET /health/live, /ready, /stats
|
|-- ui/
|   |-- index.html                   # 3-panel chat UI
|   |-- style.css                    # Dark theme, responsive
|   +-- app.js                       # Upload, chat, RAG internals display
|
|-- tests/
|   |-- test_guardrails.py           # Unit tests for guardrail logic
|   +-- test_api.py                  # Integration tests for API endpoints
|
|-- infra/
|   +-- terraform/                   # GCP infrastructure as code
|
|-- docker/
|   +-- Dockerfile                   # Multi-stage production container
|
|-- docker-compose.yml               # Local dev stack
|-- requirements.txt                 # Python dependencies
+-- .env.example                     # Environment variable template
```

---

## API Reference

### POST /api/v1/query

Ask a question about your documents.

**Request:**
```json
{
  "question": "What are the key benefits of transformer models?",
  "session_id": "sess-abc123",
  "conversation_history": [
    {"role": "user", "content": "Tell me about BERT"},
    {"role": "assistant", "content": "BERT is a transformer model..."}
  ]
}
```

**Response:**
```json
{
  "answer": "According to Chunk 1, transformer models provide...",
  "sources": ["research_paper.pdf"],
  "true_data_chunks": [...],
  "noisy_data_chunks": [...],
  "eval_metrics": {
    "faithfulness": 0.92,
    "answer_relevancy": 0.88,
    "context_precision": 0.85,
    "overall_score": 0.88,
    "evaluated": true
  },
  "pipeline_steps": ["input_guardrails", "query_planner", "history_rewriter",
                     "retrieval", "context_assembly", "generation",
                     "output_guardrails", "evaluation"],
  "hallucination_score": 0.91,
  "model_used": "gemini/gemini-1.5-flash",
  "latency_ms": 2340,
  "input_safe": true,
  "output_safe": true
}
```

### POST /api/v1/ingest/file

Upload and ingest a document (multipart form data).

### GET /api/v1/health/stats

System statistics including document count, model info, usage.

Full API docs available at: **http://localhost:8000/docs**

---

## Understanding True Data vs Noisy Data

This is the key innovation in this RAG system.

**The Problem:** Standard RAG retrieves the top-K chunks by vector similarity.
But similarity doesn't equal relevance. Many retrieved chunks are "noisy" —
they appear similar but don't actually help answer the question.

**Our Solution:**
1. Retrieve top-10 candidates using hybrid search (vector + BM25)
2. Pass all 10 to a **Cross-Encoder re-ranker**
3. The cross-encoder scores each (query, chunk) pair together
4. Chunks scoring **above** `RERANKER_THRESHOLD` (0.3) → **True Data**
5. Chunks scoring **below** the threshold → **Noisy Data** (filtered out)
6. Only **True Data** chunks go into the LLM prompt

The UI shows you which chunks were classified as True Data vs Noisy Data
in the right panel after every query.

---

## LangGraph Pipeline Flow

```
START
  |
  v
[input_guardrails] -- blocked --> [error_response] --> END
  |
  v (safe)
[query_planner]     -- classifies intent: factual/analytical/conversational
  |
  v
[history_rewriter]  -- rewrites query using conversation history
  |
  v
[retrieval]         -- hybrid search + re-ranking + True/Noisy split
  |
  v
[context_assembly]  -- format True Data chunks into LLM prompt
  |
  v
[generation]        -- call LLM via LiteLLM gateway
  |
  v
[output_guardrails] -- check hallucination, toxicity, PII
  |
  v
[evaluation]        -- RAGAS faithfulness/relevancy scores
  |
  v
END
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_guardrails.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Deploying to GCP (Production)

```bash
# 1. Authenticate with GCP
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Apply Terraform infrastructure
cd infra/terraform
terraform init
terraform plan
terraform apply

# 3. Build and push Docker image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/advanced-rag:latest

# 4. Deploy to Cloud Run
gcloud run deploy advanced-rag \
  --image gcr.io/YOUR_PROJECT_ID/advanced-rag:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## Key Learning Resources

- **LangGraph docs:** https://langchain-ai.github.io/langgraph/
- **RAGAS docs:** https://docs.ragas.io/
- **LiteLLM docs:** https://docs.litellm.ai/
- **LangFuse docs:** https://langfuse.com/docs
- **ChromaDB docs:** https://docs.trychroma.com/
- **FastAPI docs:** https://fastapi.tiangolo.com/

---

## License

MIT License - free to use for learning and commercial projects.
