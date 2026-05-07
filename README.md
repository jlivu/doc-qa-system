# Intelligent Document Q&A System

A RAG (Retrieval-Augmented Generation) system that ingests PDF documents and
answers natural language questions grounded in the document content, with
source citations.

Built to demonstrate production-grade AI engineering using a real-world use
case: querying government policy papers, financial reports, and regulatory
manuals from the Government of Vanuatu.

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐
│  PyMuPDF    │───▶│ Chunker  │───▶│ Embedder │───▶│ Qdrant │
│  (parse)    │    │ (split)  │    │ (OpenAI) │    │(store) │
└─────────────┘    └──────────┘    └──────────┘    └────────┘

User Question
    │
    ▼
┌──────────┐    ┌────────┐    ┌──────────────┐    ┌────────┐
│ Embedder │───▶│ Qdrant │───▶│ LangChain    │───▶│ Answer │
│ (query)  │    │(search)│    │ RAG Chain    │    │+ Sources│
└──────────┘    └────────┘    └──────────────┘    └────────┘
```

## Tech stack

| Layer | Technology |
|---|---|
| API framework | FastAPI |
| PDF parsing | PyMuPDF |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | Qdrant |
| LLM | OpenAI `gpt-4o-mini` |
| Frontend | Streamlit |
| Containers | Docker + Docker Compose |
| CI | GitHub Actions |

---

## Quick start

### Prerequisites

- Docker and Docker Compose
- An OpenAI API key

### 1. Clone and configure

```bash
git clone https://github.com/jlivu/doc-qa-system.git
cd doc-qa-system
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 2. Start the system

```bash
docker compose up --build
```

This starts three services:
- **API** at http://localhost:8000
- **Qdrant** at http://localhost:6333 (dashboard at http://localhost:6333/dashboard)
- **Frontend** at http://localhost:8501

### 3. Ingest a document

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@sample_docs/your_document.pdf"
```

### 4. Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the total revenue in 2024?"}'
```

Or open the Streamlit UI at http://localhost:8501.

---

## API reference

Interactive docs available at http://localhost:8000/docs once running.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/ingest` | POST | Upload and ingest a PDF |
| `/query` | POST | Ask a question |

---

## Development

### Run without Docker

```bash
# Start Qdrant only
docker compose up qdrant -d

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Set QDRANT_URL=http://localhost:6333 in .env

# Run the API
uvicorn app.main:app --reload

# Run the frontend (separate terminal)
streamlit run frontend/app.py
```

### Run tests

```bash
pytest tests/ -v
```

---

## Project structure

```
doc-qa-system/
├── app/
│   ├── api/            # FastAPI routes and Pydantic schemas
│   ├── core/           # Config and shared dependencies
│   ├── ingestion/      # PDF parsing, chunking, embedding
│   ├── retrieval/      # Qdrant wrapper and retriever
│   └── qa/             # LangChain RAG chain and prompts
├── frontend/           # Streamlit UI
├── tests/              # Pytest test suite
├── sample_docs/        # Sample PDFs for testing
├── docker-compose.yml
├── Dockerfile
└── .env.example
```

---

## Roadmap

- [x] Phase 1 — Ingestion pipeline (parser, chunker, embedder, vector store)
- [ ] Phase 2 — Query pipeline (retriever, RAG chain, query endpoint)
- [ ] Phase 3 — Frontend UI and hybrid search
- [ ] Phase 4 — Metadata filtering, reranking, evaluation

---

## Author

Joe Livu — [github.com/jlivu](https://github.com/jlivu)

Built as part of a portfolio demonstrating AI/ML engineering applied to
real government document use cases in Vanuatu.
