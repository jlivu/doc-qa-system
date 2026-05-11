# Phase 4 — Reranking, Evaluation & Async Ingestion Specification

**Status:** Approved  
**Version:** 1.0  
**Date:** 2026-05-12  
**Author:** Joe Livu  

---

## 1. Overview

Phase 4 delivers three backend improvements that raise the quality and
robustness of the system without changing the user-facing interface:

1. **Reranking** — a cross-encoder model rescores retrieved chunks after
   hybrid search, replacing RRF scores with more accurate relevance scores
   before the context is passed to the LLM.

2. **Evaluation** — a golden dataset of real questions with reference
   answers and expected source pages, evaluated automatically on three
   metrics: retrieval recall, answer similarity, and not-found accuracy.

3. **Async ingestion** — large PDFs are ingested in a background thread
   rather than blocking the HTTP response. A job status endpoint lets
   the client poll for completion.

All work is local — no cloud infrastructure is introduced in this phase.

---

## 2. Scope

### In scope

- Cross-encoder reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`
  via `sentence-transformers`
- Reranker loaded once at startup and cached for the lifetime of the
  process
- `GET /jobs/{job_id}` — poll ingestion job status
- `POST /ingest` returns a `job_id` immediately and processes the PDF
  in the background
- In-memory job store (lost on server restart — acceptable for Phase 4)
- Frontend updated to poll job status after upload rather than waiting
  for synchronous completion
- Golden dataset — 15 question/answer pairs drawn from ingested Vanuatu
  government documents
- Evaluation script `scripts/evaluate.py` — runs all 15 questions
  through the live pipeline and reports retrieval recall, answer
  similarity, and not-found accuracy
- Evaluation results written to `eval/results/` as JSON and a Markdown
  summary report

### Out of scope

- Celery, Redis, or any external task queue
- Persistent job storage across server restarts
- GCP deployment
- LangGraph agent workflows
- Query decomposition or rewriting
- User authentication

---

## 3. Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Reranker model | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Runs on CPU, loads in <1s, reranks 5 chunks in <100ms on M1 |
| Reranker placement | After hybrid search, before context building | Reranking requires all candidates to be retrieved first |
| Reranker loading | Singleton cached at FastAPI startup via lifespan | Avoid reloading 80MB model on every request |
| Async backend | FastAPI `BackgroundTasks` | No new services; adequate for single-server hobby project |
| Job store | In-memory Python dict | Simple, zero dependencies; loss on restart acceptable |
| Job ID | UUID4 | Consistent with document_id pattern already in use |
| Evaluation metrics | Retrieval recall + answer similarity + not-found accuracy | Automated, fast, no LLM call needed |
| Similarity scoring | Cosine similarity using the existing embedding model | Reuses infrastructure already in the project |
| Golden dataset size | 15 questions | Enough for meaningful signal; manageable to write |
| Evaluation trigger | Manual script — `python scripts/evaluate.py` | Not blocking CI; run before releases |

---

## 4. Feature 1 — Reranking

### 4.1 What reranking does

Hybrid search returns the top-k chunks ordered by RRF score. RRF is a
good retrieval heuristic but does not directly measure how relevant a
chunk is to the specific question asked. A cross-encoder takes each
(question, chunk) pair and produces a relevance score that directly
models their relationship. This consistently improves answer quality,
especially for questions that require precise text matching.

### 4.2 Pipeline change

```
Before Phase 4:
  hybrid_search() → top-5 chunks (RRF scores) → build_context() → LLM

After Phase 4:
  hybrid_search() → top-10 chunks (RRF scores)
      → rerank() → top-5 chunks (cross-encoder scores)
          → build_context() → LLM
```

The retrieval step fetches more candidates (10 instead of 5) to give
the reranker enough material to work with. The reranker then selects
the best 5 for the LLM.

### 4.3 New module — `app/retrieval/reranker.py`

**Interface:**

```python
def load_reranker(model_name: str) -> CrossEncoder:
    """Load and return the cross-encoder model."""

def rerank(
    question: str,
    chunks: list[SourceChunk],
    top_k: int,
    reranker: CrossEncoder,
) -> list[SourceChunk]:
    """Rerank chunks by cross-encoder score and return top_k."""
```

**Behaviour of `rerank`:**

- Constructs `(question, chunk.text)` pairs for each chunk
- Calls `reranker.predict(pairs)` — returns a float score per pair
- Replaces each chunk's `score` field with the cross-encoder score
- Returns the top-k chunks ordered by cross-encoder score descending
- If `chunks` is empty, returns empty list without calling the model
- Cross-encoder scores are raw logits — not bounded to 0–1. The
  confidence thresholds in `context.py` must be recalibrated for
  cross-encoder scores (see Section 4.5)

**Behaviour of `load_reranker`:**

- Calls `CrossEncoder(model_name)` from `sentence-transformers`
- Model is downloaded on first call (~80MB) and cached by
  `sentence-transformers` in `~/.cache/huggingface/`
- Returns the loaded `CrossEncoder` instance

### 4.4 Changes to `app/core/config.py`

Add two new settings:

```python
reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker_top_k: int = 5      # number of chunks to pass to the LLM after reranking
retrieval_top_k: int = 10    # increase from 5 to 10 — more candidates for reranker
```

`retrieval_top_k` is increased from 5 to 10 so the reranker has more
candidates to choose from. The value passed to the LLM is controlled
by `reranker_top_k`.

### 4.5 Changes to `app/main.py`

Load the reranker at startup using the lifespan handler and store it
in `app.state`:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.retrieval.reranker import load_reranker
    app.state.reranker = load_reranker(settings.reranker_model)
    # ... existing Qdrant startup ...
    yield
```

Add a new dependency in `app/core/dependencies.py`:

```python
def get_reranker(request: Request) -> CrossEncoder:
    return request.app.state.reranker
```

### 4.6 Changes to `app/api/routes/query.py`

Add `reranker` as a dependency. After `hybrid_search`, call `rerank`:

```python
chunks = hybrid_search(query_vector, question, qdrant, settings, 
                       settings.retrieval_top_k, ...)
chunks = rerank(question, chunks, settings.reranker_top_k, reranker)
```

### 4.7 Confidence threshold recalibration

Cross-encoder scores are raw logits, typically ranging from about
-10 to +10 for this model. A score above 0 generally indicates
relevance. Update `compute_confidence` thresholds in
`app/qa/context.py`:

| Cross-encoder score | Confidence |
|---|---|
| ≥ 3.0 | `"high"` |
| ≥ 0.0 | `"medium"` |
| < 0.0 | `"low"` |

Update `is_not_found` threshold from 0.0 to -5.0 — only flag as
not-found when cross-encoder scores are strongly negative, indicating
the retrieved chunks are genuinely irrelevant.

### 4.8 Changes to `app/api/schemas.py`

No schema changes — `score` on `SourceChunk` remains a float. The
cross-encoder score replaces the RRF score in the same field.

### 4.9 `.env.example` additions

```
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_TOP_K=5
RETRIEVAL_TOP_K=10
```

---

## 5. Feature 2 — Async ingestion

### 5.1 Current behaviour vs new behaviour

**Current:** `POST /ingest` blocks until the entire pipeline completes
(parse → chunk → embed → store). For a 300-page document like PSSRM.pdf
this can take 30–60 seconds, during which the HTTP connection is held
open and the frontend spinner blocks.

**New:** `POST /ingest` validates the file, assigns a `job_id`, launches
the pipeline in a background thread, and immediately returns HTTP 202
with the `job_id`. The client polls `GET /jobs/{job_id}` until status
is `completed` or `failed`.

### 5.2 New job status store — `app/ingestion/job_store.py`

```python
from enum import Enum

class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"

class JobRecord(TypedDict):
    job_id: str
    status: JobStatus
    filename: str
    created_at: str        # ISO 8601
    updated_at: str        # ISO 8601
    document_id: str | None
    pages: int | None
    chunks: int | None
    replaced: bool | None
    error: str | None

# Module-level in-memory store
_jobs: dict[str, JobRecord] = {}

def create_job(job_id: str, filename: str) -> JobRecord: ...
def update_job(job_id: str, **kwargs) -> None: ...
def get_job(job_id: str) -> JobRecord | None: ...
def list_jobs() -> list[JobRecord]: ...
```

### 5.3 Updated `POST /ingest` endpoint

**New response — HTTP 202 Accepted:**

```json
{
  "job_id": "a1b2c3d4-...",
  "filename": "PSSRM.pdf",
  "status": "pending",
  "message": "Ingestion started. Poll GET /jobs/{job_id} for status."
}
```

**New schema: `IngestAcceptedResponse`:**

```python
class IngestAcceptedResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    message: str
```

**Route logic:**

```
1. validate_upload() — same as before; return 415/413 immediately
2. pdf_bytes = await file.read()
3. job_id = str(uuid4())
4. create_job(job_id, filename)
5. background_tasks.add_task(_run_ingestion, job_id, pdf_bytes,
                              filename, qdrant, settings)
6. return IngestAcceptedResponse(job_id=job_id, ...)  HTTP 202
```

**Background task `_run_ingestion`:**

```
update_job(job_id, status=RUNNING)
try:
    sha256 = compute_sha256(pdf_bytes)
    existing_id = find_existing_document(sha256, qdrant, settings)
    if existing_id:
        delete_by_document_id(existing_id, qdrant, settings)
        document_id = existing_id
        replaced = True
    else:
        document_id = str(uuid4())
        replaced = False
    pages = extract_text(pdf_bytes)
    chunks = chunk_pages(pages, document_id, filename, sha256, settings)
    if not chunks:
        raise EmptyPDFError("No text could be extracted")
    embedded = embed_chunks(chunks, settings)
    chunk_count = upsert_chunks(embedded, qdrant, settings)
    update_job(job_id, status=COMPLETED, document_id=document_id,
               pages=len(pages), chunks=chunk_count, replaced=replaced)
except Exception as exc:
    update_job(job_id, status=FAILED, error=str(exc))
```

### 5.4 New `GET /jobs/{job_id}` endpoint

**Location:** `app/api/routes/jobs.py` (new file)

**Success response — HTTP 200 (job found):**

```json
{
  "job_id": "a1b2c3d4-...",
  "status": "completed",
  "filename": "PSSRM.pdf",
  "created_at": "2026-05-12T10:00:00Z",
  "updated_at": "2026-05-12T10:00:45Z",
  "document_id": "9ceca0eb-...",
  "pages": 323,
  "chunks": 951,
  "replaced": false,
  "error": null
}
```

**Error response — HTTP 404:**

```json
{
  "error": "JOB_NOT_FOUND",
  "detail": "No job found with id a1b2c3d4-..."
}
```

**Additional endpoint — `GET /jobs`** — returns all jobs in the store,
ordered by `created_at` descending. Useful for debugging.

### 5.5 New exception

Add to `app/ingestion/exceptions.py`:

```python
class JobNotFoundError(IngestionError): pass
```

### 5.6 Frontend update — `frontend/app.py`

Replace the synchronous ingest call with an async polling loop:

```
1. POST /ingest → receive job_id and status=pending
2. Show spinner: "Ingesting {filename}…"
3. Every 2 seconds, GET /jobs/{job_id}
4. If status == "running": update spinner message to 
   "Processing… (this may take a moment for large documents)"
5. If status == "completed": 
   - Show success banner: "{pages} pages, {chunks} chunks"
   - Refresh document list
   - Stop polling
6. If status == "failed":
   - Show error banner with job.error message
   - Stop polling
7. Timeout after 300 seconds — show timeout error
```

Use `st.empty()` for the spinner and status messages so they update
in place without rerunning the full page.

---

## 6. Feature 3 — Evaluation

### 6.1 Structure

```
eval/
├── golden_dataset.json     ← 15 hand-written Q&A pairs
├── README.md               ← instructions for running evaluation
└── results/                ← generated by the evaluation script
    └── .gitkeep

scripts/
└── evaluate.py             ← evaluation runner
```

### 6.2 Golden dataset format — `eval/golden_dataset.json`

```json
[
  {
    "id": "q01",
    "question": "What are the procurement methods for government contracts?",
    "reference_answer": "Procurement methods include open tenders, limited tenders, and request for quotation.",
    "expected_source_pages": [5],
    "expected_found": true,
    "document_filename": "GOV_Contracts_and_Tenders_Regulation_Consolidation_Edition_2021_Order_No.160_of_2021.pdf"
  },
  ...
]
```

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique identifier for this question |
| `question` | string | The natural language question |
| `reference_answer` | string | The expected correct answer |
| `expected_source_pages` | list[int] | Page numbers that should appear in sources |
| `expected_found` | boolean | Whether the system should return `found: true` |
| `document_filename` | string | Which document to scope the query to |

### 6.3 Golden dataset composition — 15 questions

| Count | Type | Source document |
|---|---|---|
| 6 | Answerable questions | GOV Contracts and Tenders Regulation |
| 5 | Answerable questions | 2023 Financial Statements |
| 2 | Answerable questions | PSSRM |
| 2 | Unanswerable questions | Any (expected `found: false`) |

### 6.4 Evaluation metrics

**Metric 1 — Retrieval recall @ k**

For each answerable question, check whether at least one of the
`expected_source_pages` appears in the top-k returned sources.

```
recall@5 = (questions where expected page in top-5 sources) / total answerable questions
```

Target: ≥ 0.80 (80% of questions have the right page in top 5)

**Metric 2 — Answer similarity**

Embed both the generated answer and the reference answer using the
existing embedding model. Compute cosine similarity. Average across
all answerable questions.

```
avg_similarity = mean(cosine_similarity(embed(answer), embed(reference)))
```

Target: ≥ 0.70 (generated answers are semantically close to references)

**Metric 3 — Not-found accuracy**

For the 2 unanswerable questions, check whether the system correctly
returns `found: false`.

```
not_found_accuracy = correct not-found responses / total unanswerable questions
```

Target: 1.0 (both unanswerable questions correctly return not-found)

### 6.5 Evaluation script — `scripts/evaluate.py`

**Usage:**

```bash
# Run against the local API (must be running)
python scripts/evaluate.py

# Run with a custom API URL
python scripts/evaluate.py --api-url http://localhost:8001

# Save results to a specific directory
python scripts/evaluate.py --output-dir eval/results/
```

**Script logic:**

```
1. Load eval/golden_dataset.json
2. For each question:
   a. POST /query with question and document_filename filter
   b. Record answer, sources, found, confidence, latency
3. Compute the three metrics
4. Write results to eval/results/{timestamp}.json
5. Write eval/results/latest.md — Markdown summary report
6. Print summary to stdout
7. Exit code 0 if all targets met, exit code 1 if any target missed
```

**Stdout output format:**

```
Evaluation results — 2026-05-12 10:00:00
==========================================
Questions evaluated:  15
Answerable:           13
Unanswerable:          2

Retrieval recall@5:   0.85  ✓  (target: ≥ 0.80)
Answer similarity:    0.73  ✓  (target: ≥ 0.70)
Not-found accuracy:   1.00  ✓  (target: 1.00)

All targets met. Results saved to eval/results/2026-05-12T10:00:00.json
```

**Markdown summary — `eval/results/latest.md`:**

Includes the metrics table, per-question results with generated vs
reference answer, and source page recall for each question.

---

## 7. New and modified files

| File | Action |
|---|---|
| `app/retrieval/reranker.py` | Create |
| `app/ingestion/job_store.py` | Create |
| `app/api/routes/jobs.py` | Create |
| `app/api/schemas.py` | Add `IngestAcceptedResponse`, `JobStatusResponse`, `JobListResponse` |
| `app/ingestion/exceptions.py` | Add `JobNotFoundError` |
| `app/core/config.py` | Add `reranker_model`, `reranker_top_k`, update `retrieval_top_k` default to 10 |
| `app/core/dependencies.py` | Add `get_reranker` dependency |
| `app/main.py` | Load reranker in lifespan, register jobs router |
| `app/api/routes/ingest.py` | Switch to async — return 202, launch BackgroundTask |
| `app/api/routes/query.py` | Add reranker dependency, call `rerank()` after hybrid search |
| `app/qa/context.py` | Recalibrate confidence thresholds for cross-encoder scores |
| `frontend/app.py` | Replace synchronous ingest with polling loop |
| `requirements.txt` | Add `sentence-transformers` |
| `.env.example` | Add reranker env vars |
| `eval/golden_dataset.json` | Create |
| `eval/README.md` | Create |
| `eval/results/.gitkeep` | Create |
| `scripts/evaluate.py` | Create |

---

## 8. Acceptance criteria

### Reranking (AC-RERANK)

- **AC-RERANK-01** — The reranker model loads at startup without error
- **AC-RERANK-02** — `rerank()` returns exactly `top_k` chunks when
  input has more than `top_k` chunks
- **AC-RERANK-03** — `rerank()` returns all chunks when input has fewer
  than `top_k` chunks
- **AC-RERANK-04** — Returned chunks are ordered by cross-encoder score
  descending
- **AC-RERANK-05** — `rerank()` returns empty list for empty input
- **AC-RERANK-06** — `SourceChunk.score` reflects the cross-encoder
  score after reranking, not the RRF score
- **AC-RERANK-07** — The query route calls `rerank()` between
  `hybrid_search()` and `build_context()`

### Async ingestion (AC-ASYNC)

- **AC-ASYNC-01** — `POST /ingest` returns HTTP 202 immediately with
  a `job_id`
- **AC-ASYNC-02** — `GET /jobs/{job_id}` returns `status: "pending"`
  or `"running"` while the job is in progress
- **AC-ASYNC-03** — `GET /jobs/{job_id}` returns `status: "completed"`
  with `document_id`, `pages`, and `chunks` after successful ingestion
- **AC-ASYNC-04** — `GET /jobs/{job_id}` returns `status: "failed"`
  with an `error` message for an invalid PDF
- **AC-ASYNC-05** — `GET /jobs/{unknown_id}` returns HTTP 404
- **AC-ASYNC-06** — `GET /jobs` returns a list of all jobs
- **AC-ASYNC-07** — A document ingested via async route is queryable
  after the job reaches `completed`

### Evaluation (AC-EVAL)

- **AC-EVAL-01** — `eval/golden_dataset.json` contains exactly 15
  questions in the correct format
- **AC-EVAL-02** — `scripts/evaluate.py` runs without error when the
  API is available
- **AC-EVAL-03** — The script exits with code 0 when all three metric
  targets are met
- **AC-EVAL-04** — The script exits with code 1 when any target is
  missed
- **AC-EVAL-05** — A results JSON file is written to `eval/results/`
- **AC-EVAL-06** — `eval/results/latest.md` is written with a
  Markdown summary

---

## 9. Notes for the plan

- The reranker singleton on `app.state` means tests that use
  `TestClient` need the lifespan to run or must mock `get_reranker`.
  The plan must address this explicitly.
- `BackgroundTasks` runs in a thread pool — `_run_ingestion` must not
  use `async`/`await` internally since it calls synchronous functions
  (PyMuPDF, sentence-transformers, OpenAI-compat embeddings). This is
  already the case — all existing ingestion functions are synchronous.
- The frontend polling loop uses `time.sleep(2)` inside a `st.empty()`
  context. Streamlit reruns the script on every interaction — the
  polling state must be stored in `st.session_state` so it survives
  reruns.
- `sentence-transformers` is a heavy dependency (~500MB with model).
  It must be added to `requirements.txt` and the `Dockerfile` must
  rebuild to include it.
- The golden dataset questions must be written against documents that
  are actually ingested. The evaluation script must check that the
  required documents exist before running and print a helpful error
  if they are missing.
- `retrieval_top_k` changing from 5 to 10 affects the Phase 2 tests
  that assert on the `limit` parameter in Qdrant search calls. The
  plan must flag these tests for updating.
- Cross-encoder scores are not bounded — the confidence thresholds
  (3.0 / 0.0) are initial calibration values. The evaluation script
  will reveal whether they need tuning after Phase 4 is deployed.
- `eval/results/` must be in `.gitignore` except for `.gitkeep` —
  generated evaluation output should not be committed.
