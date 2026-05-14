# Intelligent Document Q&A System

A production-quality RAG (Retrieval-Augmented Generation) system that ingests
PDF documents and answers natural language questions grounded in the document
content — with source citations, confidence indicators, and highlighted
relevant text.

Built as a portfolio project demonstrating spec-driven AI engineering applied
to a real-world use case: querying government policy papers, financial
reports, and regulatory manuals from the Government of Vanuatu.

**Runs entirely locally — no cloud accounts, no API keys, no ongoing costs.**

---

## Features

- **PDF ingestion** — text-native and scanned (OCR) documents supported
- **Hybrid search** — combines dense vector similarity with BM25 keyword search
- **Cross-encoder reranking** — improves answer quality by rescoring retrieved chunks
- **Multi-turn conversation** — maintains context across follow-up questions
- **Source citations** — every answer links back to the source document and page
- **Confidence indicator** — High / Medium / Low based on retrieval quality
- **Highlight extraction** — the most relevant sentence per source, identified by the LLM
- **Async ingestion** — large documents process in the background with job polling
- **Document management** — upload, list, and delete documents via the UI or API
- **Evaluation harness** — automated scoring on retrieval recall, answer similarity, and not-found accuracy

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │            INGESTION PIPELINE            │
                    │                                          │
  PDF Upload ──────▶│ PyMuPDF ──▶ Chunker ──▶ Ollama Embed   │──▶ Qdrant
                    │  (parse)    (split)     (nomic-embed)   │   (store)
                    └─────────────────────────────────────────┘

                    ┌─────────────────────────────────────────┐
                    │              QUERY PIPELINE              │
                    │                                          │
  Question ────────▶│ Ollama Embed ──▶ Hybrid Search (Qdrant) │
                    │  (nomic-embed)   (vector + BM25)        │
                    │                      │                  │
                    │              Cross-Encoder Rerank       │
                    │              (ms-marco-MiniLM-L-6-v2)   │
                    │                      │                  │
                    │              LangChain RAG Chain        │──▶ Answer
                    │              (qwen2.5:7b via Ollama)    │   + Sources
                    └─────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| API framework | FastAPI |
| PDF parsing | PyMuPDF + Tesseract OCR |
| Text chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | Ollama — `nomic-embed-text` (768 dimensions) |
| Vector store | Qdrant (hybrid dense + sparse BM25) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers |
| LLM | Ollama — `qwen2.5:7b` |
| LLM orchestration | LangChain |
| Frontend | Streamlit |
| Containers | Docker + Docker Compose |
| CI | GitHub Actions |
| Testing | Pytest — 124 tests |

---

## Prerequisites

The following must be installed before running the system:

| Tool | Purpose | Install |
|---|---|---|
| Docker Desktop | Runs the API, Qdrant, and frontend containers | [docker.com](https://www.docker.com/products/docker-desktop/) |
| Ollama | Runs LLM and embedding models locally | [ollama.com](https://ollama.com) |
| Git | Clone the repository | [git-scm.com](https://git-scm.com) |

**Minimum hardware:**

| Resource | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB or more |
| Disk space | 15 GB free | 25 GB free |
| CPU | Apple Silicon or modern x86-64 | Apple M1/M2/M3 or AMD Ryzen / Intel Core i7+ |

> **Apple Silicon (M1/M2/M3):** All models run natively on ARM. No GPU required.
>
> **Windows / Linux x86-64:** All models run on CPU. No GPU required,
> though a GPU will significantly speed up inference if available.

---

## Installation

### macOS (Apple Silicon or Intel)

#### Step 1 — Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Step 2 — Install prerequisites

```bash
brew install git
brew install --cask docker
```

Install Ollama:
```bash
brew install ollama
```

Or download the macOS installer from [ollama.com](https://ollama.com).

#### Step 3 — Pull Ollama models

```bash
# Start Ollama
ollama serve

# In a new terminal tab — pull both models (downloads ~5 GB total)
ollama pull nomic-embed-text
ollama pull qwen2.5:7b

# Verify
ollama list
```

Both `nomic-embed-text` and `qwen2.5:7b` should appear in the list.

#### Step 4 — Start Docker Desktop

Open Docker Desktop from Applications. Wait for the whale icon in the
menu bar to stop animating.

**Recommended Docker Desktop settings (Settings → Resources):**
- Memory: 8 GB or more
- Virtual disk limit: 60 GB or more

#### Step 5 — Clone the repository

```bash
git clone https://github.com/jlivu/doc-qa-system.git
cd doc-qa-system
```

#### Step 6 — Configure environment

```bash
cp .env.example .env
```

The default values work without any changes on macOS:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_LLM_MODEL=qwen2.5:7b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
QDRANT_URL=http://qdrant:6333
CORS_ORIGINS=["http://localhost:8501"]
```

> `host.docker.internal` is the special hostname Docker containers use
> to reach services running on your Mac. Ollama runs on your Mac
> (not in Docker), so the API container connects to it via this address.

#### Step 7 — Build and start

```bash
docker compose up --build -d
```

The first build takes 10–20 minutes — it downloads PyTorch and other
large dependencies. Subsequent starts use cached layers and are fast.

---

### Windows 11 / Windows 10

#### Step 1 — Install WSL2 (Windows Subsystem for Linux)

Open PowerShell as Administrator and run:

```powershell
wsl --install
```

Restart your computer when prompted. WSL2 installs Ubuntu by default.

#### Step 2 — Install Docker Desktop

Download and install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/).

During installation, ensure **"Use WSL2 instead of Hyper-V"** is selected.

After installation, open Docker Desktop:
- Go to Settings → General → check **"Use the WSL2 based engine"**
- Go to Settings → Resources → WSL Integration → enable your Ubuntu distro
- Go to Settings → Resources → Memory: set to 8 GB or more
- Go to Settings → Resources → Disk image size: set to 60 GB or more

#### Step 3 — Install Ollama for Windows

Download the Windows installer from [ollama.com](https://ollama.com) and
run it. Ollama installs as a Windows service and starts automatically.

Open a Command Prompt or PowerShell and pull the models:

```powershell
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
ollama list
```

#### Step 4 — Install Git for Windows

Download from [git-scm.com](https://git-scm.com/download/win) and install
with default settings.

#### Step 5 — Clone and configure (use Git Bash or PowerShell)

```bash
git clone https://github.com/jlivu/doc-qa-system.git
cd doc-qa-system
copy .env.example .env
```

Open `.env` in Notepad and verify these values:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_LLM_MODEL=qwen2.5:7b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
QDRANT_URL=http://qdrant:6333
CORS_ORIGINS=["http://localhost:8501"]
```

> On Windows, `host.docker.internal` resolves to the Windows host machine,
> where Ollama is running. This is the same as on macOS.

#### Step 6 — Build and start

Open PowerShell or Command Prompt in the project folder:

```powershell
docker compose up --build -d
```

The first build takes 10–20 minutes. Progress is shown in the terminal.

---

### Linux (Ubuntu / Debian)

#### Step 1 — Install Docker Engine

```bash
# Remove old versions if present
sudo apt remove docker docker-engine docker.io containerd runc

# Install prerequisites
sudo apt update
sudo apt install -y ca-certificates curl gnupg

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Allow running Docker without sudo
sudo usermod -aG docker $USER
newgrp docker
```

#### Step 2 — Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Ollama starts automatically as a systemd service. Pull the models:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
ollama list
```

#### Step 3 — Clone and configure

```bash
git clone https://github.com/jlivu/doc-qa-system.git
cd doc-qa-system
cp .env.example .env
```

Open `.env` and make **one change** — on Linux, Docker containers reach
the host machine via `172.17.0.1` (the default Docker bridge gateway),
not `host.docker.internal`:

```env
OLLAMA_BASE_URL=http://172.17.0.1:11434
OLLAMA_LLM_MODEL=qwen2.5:7b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
QDRANT_URL=http://qdrant:6333
CORS_ORIGINS=["http://localhost:8501"]
```

> **Alternative:** You can also use `host.docker.internal` on Linux by
> adding `--add-host=host.docker.internal:host-gateway` to the api service
> in `docker-compose.yml` under the `extra_hosts` key.

Also ensure Ollama listens on all interfaces (not just localhost):

```bash
# Edit the Ollama systemd service
sudo systemctl edit ollama

# Add these lines between the comments:
[Service]
Environment="OLLAMA_HOST=0.0.0.0"

# Restart Ollama
sudo systemctl restart ollama
```

#### Step 4 — Build and start

```bash
docker compose up --build -d
```

---

## Verifying the Installation

After starting the system on any platform, run these checks:

```bash
# 1. Confirm all three containers are running
docker compose ps

# 2. Check the API health endpoint
curl -s http://localhost:8001/health
# Expected: {"status":"ok","qdrant":"ok"}

# 3. Confirm the reranker model loaded at startup
docker logs doc-qa-api | grep "Loading weights"
# Expected: Loading weights: 100%|██████████| 105/105

# 4. Open the frontend
# Navigate to http://localhost:8501 in your browser
```

---

## Usage

### Using the Frontend (Recommended)

Open **http://localhost:8501** in your browser.

**Upload and ingest a document:**
1. Click **Upload** in the Document Library sidebar
2. Select a PDF file (up to 50 MB — text-native or scanned)
3. Click **Ingest** — the sidebar shows a progress indicator
4. Once complete, the document appears in the library with its page and chunk counts

**Ask a question:**
1. Type your question in the Question field
2. Optionally scope the search to a specific document using the Scope dropdown
3. Click **Ask**
4. The answer appears with:
   - A **confidence badge** (🟢 High / 🟡 Medium / 🔴 Low)
   - **Expandable source citations** with filename, page number, and relevance score
   - **Highlighted text** — the most relevant sentence per source

**Multi-turn conversation:**
- Ask follow-up questions — the system maintains context
- The **Conversation** sidebar shows your full history
- Click **Clear conversation** to start fresh (a confirmation dialog appears)

**Delete a document:**
- Click the trash icon next to any document
- Confirm in the dialog — this removes the document and all its stored chunks

### Using the API Directly

Interactive API documentation: **http://localhost:8001/docs**

**Ingest a document (async):**
```bash
curl -s -X POST http://localhost:8001/ingest \
  -F "file=@/path/to/document.pdf" \
  | python -m json.tool
```

Returns immediately with a `job_id`. Poll for completion:

```bash
curl -s http://localhost:8001/jobs/{job_id} | python -m json.tool
```

Status progresses: `pending` → `running` → `completed` (or `failed`)

**Ask a question:**
```bash
curl -s -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the procurement methods?"}' \
  | python -m json.tool
```

**Ask with a document filter:**
```bash
curl -s -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the procurement methods?",
    "filters": {"filename": "GOV_Contracts.pdf"}
  }' | python -m json.tool
```

**Ask a follow-up question:**
```bash
curl -s -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the threshold for direct contracting?",
    "conversation_history": [
      {"role": "user", "content": "What are the procurement methods?"},
      {"role": "assistant", "content": "The procurement methods include open tenders..."}
    ]
  }' | python -m json.tool
```

**List all ingested documents:**
```bash
curl -s http://localhost:8001/documents | python -m json.tool
```

**Delete a document:**
```bash
curl -s -X DELETE http://localhost:8001/documents/{document_id}
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check — verifies API and Qdrant |
| `/ingest` | POST | Upload a PDF — returns `job_id` immediately (async) |
| `/jobs/{job_id}` | GET | Poll ingestion job status |
| `/jobs` | GET | List all ingestion jobs |
| `/query` | POST | Ask a question with optional filters and history |
| `/documents` | GET | List all ingested documents with metadata |
| `/documents/{id}` | DELETE | Delete a document and all its chunks |

Full interactive documentation with request/response schemas:
**http://localhost:8001/docs**

---

## Development

### Run tests

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt

# Run the full test suite (no Docker or Ollama needed)
python -m pytest tests/ -v
```

All 124 tests pass with all external services mocked.

### Run the evaluation harness

Scores the live system against a golden dataset of 15 real questions.

```bash
# Ensure the system is running and documents are ingested
source .venv/bin/activate
python scripts/evaluate.py --api-url http://localhost:8001
```

Current benchmark results (all targets met):

| Metric | Score | Target |
|---|---|---|
| Retrieval recall@5 | 0.92 | ≥ 0.80 ✓ |
| Answer similarity | 0.73 | ≥ 0.70 ✓ |
| Not-found accuracy | 1.00 | 1.00 ✓ |

Results are saved to `eval/results/` as JSON and a Markdown summary.

### Run without Docker (development mode)

```bash
# Start only Qdrant in Docker
docker compose up qdrant -d

# Activate virtual environment
source .venv/bin/activate

# Configure .env for local development
cp .env.example .env
# Set: QDRANT_URL=http://localhost:6333
# Set: OLLAMA_BASE_URL=http://localhost:11434

# Start the API with hot reload
uvicorn app.main:app --reload --port 8000

# In a separate terminal — start the frontend
streamlit run frontend/app.py
```

---

## Project Structure

```
doc-qa-system/
│
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── ingest.py        # POST /ingest — async PDF ingestion
│   │   │   ├── query.py         # POST /query — full RAG pipeline
│   │   │   ├── documents.py     # GET/DELETE /documents
│   │   │   └── jobs.py          # GET /jobs — ingestion job polling
│   │   └── schemas.py           # Pydantic request/response models
│   │
│   ├── core/
│   │   ├── config.py            # Settings loaded from .env
│   │   └── dependencies.py      # FastAPI dependency injection
│   │
│   ├── ingestion/
│   │   ├── parser.py            # PyMuPDF + Tesseract OCR
│   │   ├── chunker.py           # Text splitting
│   │   ├── embedder.py          # Ollama nomic-embed-text
│   │   ├── hasher.py            # SHA-256 deduplication
│   │   ├── validator.py         # File type and size validation
│   │   ├── job_store.py         # In-memory async job tracking
│   │   └── exceptions.py        # Custom exception hierarchy
│   │
│   ├── retrieval/
│   │   ├── vector_store.py      # Qdrant — hybrid search, upsert, delete
│   │   ├── retriever.py         # Query embedding + retrieval
│   │   └── reranker.py          # Cross-encoder reranking
│   │
│   ├── qa/
│   │   ├── chain.py             # LangChain RAG chain + highlight extraction
│   │   ├── context.py           # Context builder + confidence scoring
│   │   └── prompts.py           # OCR-aware system prompt
│   │
│   └── query/
│       └── validator.py         # Query request validation
│
├── frontend/
│   └── app.py                   # Streamlit UI
│
├── tests/                       # 124 pytest tests — one file per module
│
├── eval/
│   ├── golden_dataset.json      # 15 Q&A pairs for evaluation
│   └── results/                 # Generated evaluation reports
│
├── scripts/
│   └── evaluate.py              # Evaluation runner
│
├── docs/
│   ├── specs/                   # Phase specifications
│   └── plans/                   # Implementation plans
│
├── sample_docs/
│   └── README.md                # Where to find test documents
│
├── docker-compose.yml
├── Dockerfile                   # API image
├── Dockerfile.frontend          # Frontend image
├── requirements.txt
└── .env.example
```

---

## Troubleshooting

**`docker compose up` fails — "address already in use"**

Something else is using port 8001, 8501, or 6333.

```bash
# macOS / Linux — find and stop the conflicting process
lsof -i :8001
kill -9 <PID>

# Windows PowerShell
netstat -ano | findstr :8001
taskkill /PID <PID> /F
```

**Ollama not reachable from Docker containers**

Confirm Ollama is running:
```bash
ollama list          # should show the models
curl http://localhost:11434   # should return a response
```

On Linux, confirm Ollama is listening on all interfaces (see Linux
installation step 3).

**API health check shows `"qdrant": "error"`**

Qdrant may still be starting. Wait 10 seconds and retry. If it
persists:
```bash
docker compose restart qdrant
sleep 5
curl -s http://localhost:8001/health
```

**`CORS_ORIGINS` parsing error — API crashes on startup**

The value in `.env` must be valid JSON with double quotes:
```env
# Correct
CORS_ORIGINS=["http://localhost:8501"]

# Wrong — will crash
CORS_ORIGINS=http://localhost:8501
CORS_ORIGINS=['http://localhost:8501']
```

**Docker build fails — "no space left on device"**

The build ran out of disk space. Free up Docker's cache:
```bash
docker builder prune -f
docker system prune -f
```

Then increase Docker Desktop's virtual disk limit under
Settings → Resources → Virtual disk limit and retry the build.

**Reranker model fails to load**

The cross-encoder downloads from HuggingFace on first startup (~80 MB).
If the download fails, restart the container:
```bash
docker compose restart api
docker logs doc-qa-api | grep -i "weights\|rerank\|error"
```

**Ingestion job stays "pending" forever**

Check the API logs for errors:
```bash
docker logs doc-qa-api --tail 50
```

If Ollama is unreachable, embedding will fail. Check that `OLLAMA_BASE_URL`
in `.env` is correct for your platform.

**Windows — containers can't reach Ollama**

Confirm Ollama is running (check the system tray). Then verify the
connection from inside Docker:
```powershell
docker run --rm curlimages/curl curl http://host.docker.internal:11434
```

If this fails, check that Windows Firewall is not blocking port 11434.

---

## Roadmap

- [x] Phase 1 — Ingestion pipeline (parser, chunker, embedder, vector store, deduplication)
- [x] Phase 2 — Query pipeline (hybrid search, RAG chain, conversation history)
- [x] Phase 3 — Frontend UI (document library, confidence badges, source highlights)
- [x] Phase 4 — Cross-encoder reranking, async ingestion, evaluation harness
- [ ] Phase 5 — GCP deployment

---

## Author

**Joe Livu** — AI Engineer | Data Scientist | Application Developer

- GitHub: [github.com/jlivu](https://github.com/jlivu)
- Location: Port Vila, Vanuatu

Built following a spec → plan → test → implement workflow across four
phases, demonstrating production AI/ML engineering applied to real
government document use cases in Vanuatu.
