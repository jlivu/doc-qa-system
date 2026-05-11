# Phase 4 — Reranking, Evaluation & Async Ingestion Plan

**Spec:** `docs/specs/phase-4-reranking-eval-async-spec.md` v1.0

---

## 1. Build Order

```
Step  1  Requirements + Dockerfile    requirements.txt, Dockerfile
Step  2  Config                       app/core/config.py
Step  3  Exceptions                   app/ingestion/exceptions.py
Step  4  Schemas                      app/api/schemas.py
Step  5  Reranker module              app/retrieval/reranker.py           (new)
Step  6  Dependencies + lifespan      app/core/dependencies.py, app/main.py
Step  7  Confidence recalibration     app/qa/context.py
Step  8  Query route                  app/api/routes/query.py
Step  9  Job store                    app/ingestion/job_store.py          (new)
Step 10  Async ingest route           app/api/routes/ingest.py
Step 11  Jobs route                   app/api/routes/jobs.py              (new)
Step 12  Register jobs router         app/main.py
Step 13  Frontend polling             frontend/app.py
Step 14  Golden dataset               eval/golden_dataset.json, eval/README.md, eval/results/.gitkeep
Step 15  Evaluation script            scripts/evaluate.py
Step 16  .env.example + .gitignore    .env.example, .gitignore
Step 17  Tests                        all test files
```

**Why this order:**

Step 1 first — `sentence-transformers` must be installable before the
reranker module (step 5) can import `CrossEncoder`. The Dockerfile must
be updated to handle the larger image size.

Step 2 (config) before steps 5–12 — every module reads `reranker_model`,
`reranker_top_k`, and the updated `retrieval_top_k` default (5 → 10).

Step 3 (exceptions) before steps 10–11 — `JobNotFoundError` is used by
the jobs route.

Step 4 (schemas) before steps 8–11 — `IngestAcceptedResponse`,
`JobStatusResponse`, and `JobListResponse` are used by the ingest and
jobs routes.

Step 5 (reranker module) is a leaf — depends only on
`sentence-transformers` and `SourceChunk`. Must exist before the query
route (step 8) can call `rerank()`.

Step 6 (dependencies + lifespan) loads the reranker singleton at startup
and exposes `get_reranker` for dependency injection. Must come after the
reranker module (step 5) and before the query route (step 8).

Step 7 (confidence recalibration) updates thresholds for cross-encoder
scores. Must come before step 8 uses `compute_confidence`.

Step 8 (query route) adds the reranker dependency and calls `rerank()`
between `hybrid_search()` and `build_context()`.

Steps 9–12 (async ingestion) form a dependency chain: job store → ingest
route → jobs route → router registration.

Step 13 (frontend) replaces the synchronous ingest call with a polling
loop against `GET /jobs/{job_id}`.

Steps 14–15 (evaluation) are independent of the API changes. They can
be built last.

Step 16 updates `.env.example` and `.gitignore`.

Step 17 (tests) last — references final public interfaces.

---

## 2. Scaffold Changes

### 2.1 `requirements.txt`

Add one line:

```
sentence-transformers>=2.7.0
```

This pulls in `transformers`, `torch` (CPU), and `huggingface-hub`.
Total added size: ~500MB in the Docker image.

### 2.2 `Dockerfile`

No structural changes needed. `sentence-transformers` installs via pip
in the existing builder stage. The runtime stage already copies
`/install` from the builder. The image will be larger (~1.5GB total)
but the build process is unchanged.

### 2.3 `app/core/config.py`

Add three fields to `Settings`:

```python
# Reranker
reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker_top_k: int = 5

# Retrieval — changed from 5 to 10
retrieval_top_k: int = 10
```

### 2.4 `app/ingestion/exceptions.py`

Add one new class:

```python
class JobNotFoundError(IngestionError):
    """No job found with the given ID."""
```

### 2.5 `app/api/schemas.py`

Add three new models:

```python
class IngestAcceptedResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    created_at: str
    updated_at: str
    document_id: str | None = None
    pages: int | None = None
    chunks: int | None = None
    replaced: bool | None = None
    error: str | None = None

class JobListResponse(BaseModel):
    jobs: list[JobStatusResponse]
    total: int
```

Keep `IngestResponse` — it's still used by tests for the old response
shape. The route changes from 201 to 202 with a different response model.

### 2.6 New files

| File | Purpose |
|------|---------|
| `app/retrieval/reranker.py` | `load_reranker()`, `rerank()` |
| `app/ingestion/job_store.py` | In-memory job store: `create_job`, `update_job`, `get_job`, `list_jobs` |
| `app/api/routes/jobs.py` | `GET /jobs/{job_id}`, `GET /jobs` |
| `eval/golden_dataset.json` | 15 question/answer pairs |
| `eval/README.md` | Instructions for running evaluation |
| `eval/results/.gitkeep` | Placeholder for generated results |
| `scripts/evaluate.py` | Evaluation runner |

### 2.7 New directories

```
eval/
eval/results/
scripts/
```

---

## 3. Implementation Strategy — Module by Module

### 3.1 Reranker — `app/retrieval/reranker.py`

**`load_reranker(model_name: str) -> CrossEncoder`**

```python
from sentence_transformers import CrossEncoder

def load_reranker(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)
```

The model is downloaded on first call (~80MB) and cached by
`huggingface_hub` in `~/.cache/huggingface/`. Subsequent calls use the
cached model. In Docker, the cache directory is inside the container —
the model is re-downloaded on each container rebuild. For production,
mount a volume to `~/.cache/huggingface/` to persist the model.

**`rerank(question, chunks, top_k, reranker) -> list[SourceChunk]`**

```python
def rerank(
    question: str,
    chunks: list[SourceChunk],
    top_k: int,
    reranker: CrossEncoder,
) -> list[SourceChunk]:
    if not chunks:
        return []
    pairs = [(question, c.text) for c in chunks]
    scores = reranker.predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)
    ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
    return ranked[:top_k]
```

**Pitfalls:**

- `reranker.predict()` returns numpy floats. Must cast to `float()` for
  Pydantic serialisation.
- Cross-encoder scores are raw logits (typically -10 to +10), not
  probabilities. A score above 0 generally indicates relevance.
- If `len(chunks) <= top_k`, return all chunks (sorted by score).
- The reranker is CPU-only. On an M1 Mac, reranking 10 chunks takes
  <100ms. No GPU required.

### 3.2 Job Store — `app/ingestion/job_store.py`

Module-level `_jobs: dict[str, JobRecord]` dict. Four functions:

```python
import threading
from datetime import datetime, timezone

_jobs: dict[str, dict] = {}
_lock = threading.Lock()

def create_job(job_id: str, filename: str) -> dict:
    with _lock:
        job = {
            "job_id": job_id, "status": "pending", "filename": filename,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "document_id": None, "pages": None, "chunks": None,
            "replaced": None, "error": None,
        }
        _jobs[job_id] = job
        return dict(job)

def update_job(job_id: str, **kwargs) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)
            _jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

def get_job(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None

def list_jobs() -> list[dict]:
    with _lock:
        return sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)
```

The `threading.Lock` is necessary because `_run_ingestion` runs in a
background thread (via `BackgroundTasks`) while the main thread serves
`GET /jobs/{job_id}` requests. Without the lock, concurrent reads and
writes to `_jobs` are a race condition.

### 3.3 Async Ingest Route — `app/api/routes/ingest.py`

Replace the synchronous pipeline with:

1. Validate the upload (same as before — still synchronous, still
   returns 415/413 immediately).
2. Read `pdf_bytes = await file.read()`.
3. Create a job: `job_id = str(uuid4())`, `create_job(job_id, filename)`.
4. Add the background task:
   `background_tasks.add_task(_run_ingestion, job_id, pdf_bytes, filename, qdrant, settings)`
5. Return HTTP 202 with `IngestAcceptedResponse`.

The `_run_ingestion` function is a plain (non-async) function that runs
the full pipeline: hash → dedup → parse → chunk → embed → store. On
success it calls `update_job(job_id, status="completed", ...)`. On
failure it calls `update_job(job_id, status="failed", error=str(exc))`.

The route function gains a `background_tasks: BackgroundTasks` parameter
from FastAPI.

**Pitfalls:**

- `_run_ingestion` must NOT use `async def` — it runs in a thread pool
  and all functions it calls (PyMuPDF, embedder, Qdrant client) are
  synchronous.
- The `qdrant` and `settings` objects must be passed to the background
  task, not looked up inside it. FastAPI's dependency injection scope
  is per-request — the DI objects may be garbage-collected after the
  request returns if not captured by the task closure.

### 3.4 Jobs Route — `app/api/routes/jobs.py`

Router prefix: `/jobs`.

**`GET /jobs/{job_id}`** — calls `get_job()`, returns 404 if `None`.

**`GET /jobs`** — calls `list_jobs()`, returns `JobListResponse`.

### 3.5 Query Route Changes — `app/api/routes/query.py`

Two changes:

1. Add `reranker` as a FastAPI dependency:
   ```python
   from app.core.dependencies import RerankerDep
   ```
   The route signature gains `reranker: RerankerDep`.

2. After `hybrid_search()`, before converting to `SourceChunk`, call:
   ```python
   from app.retrieval.reranker import rerank
   chunks_as_source = [...]  # convert SearchResult → SourceChunk
   chunks_as_source = rerank(payload.question, chunks_as_source,
                              settings.reranker_top_k, reranker)
   ```

The `hybrid_search` now fetches `settings.retrieval_top_k` (10)
candidates. The reranker selects the best `settings.reranker_top_k` (5).

### 3.6 Frontend Polling — `frontend/app.py`

Replace the synchronous ingest block with a polling loop. On
`POST /ingest` receiving 202, store the `job_id` in session state.
Then poll `GET /jobs/{job_id}` every 2 seconds using `st.empty()` for
in-place status updates. Stop on `completed`, `failed`, or 300s timeout.

**Session state keys added:**
- `polling_job_id` — the job ID currently being polled, or `None`
- `polling_filename` — the filename for display during polling

---

## 4. Reranker Singleton

### 4.1 Lifespan handler changes in `app/main.py`

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load reranker at startup
    try:
        from app.retrieval.reranker import load_reranker
        app.state.reranker = load_reranker(settings.reranker_model)
    except Exception:
        app.state.reranker = None  # Tests will override via DI

    # Existing Qdrant startup (unchanged)
    try:
        client = get_qdrant_client()
        ...
    except Exception:
        pass
    yield
```

The reranker load is wrapped in try/except so tests that don't have
`sentence-transformers` installed can still start the app. The
dependency override in tests ensures the mock is used.

### 4.2 Dependency in `app/core/dependencies.py`

```python
from fastapi import Depends, Request

def get_reranker(request: Request):
    return request.app.state.reranker

RerankerDep = Annotated[object, Depends(get_reranker)]
```

The return type is `object` (not `CrossEncoder`) to avoid importing
`sentence_transformers` in the dependencies module.

### 4.3 How tests mock `get_reranker`

Tests add to the existing fixture:

```python
from app.core.dependencies import get_reranker

mock_reranker = MagicMock()
mock_reranker.predict.return_value = [0.9, 0.5, 0.1]
app.dependency_overrides[get_reranker] = lambda: mock_reranker
```

The lifespan's try/except ensures `app.state.reranker` is set to `None`
if the model can't load (e.g., in CI without the model cached). The
dependency override replaces it with the mock for all test requests.

---

## 5. BackgroundTasks Constraints

### 5.1 Why `_run_ingestion` must be synchronous

FastAPI's `BackgroundTasks` runs tasks in a thread pool (via Starlette's
`run_in_threadpool`). If the task function is `async def`, Starlette
runs it on the event loop — blocking the loop for the duration of
the CPU-heavy ingestion pipeline. By using a plain `def`, the task
runs in a separate thread. All existing ingestion functions are
synchronous, so this works naturally.

### 5.2 Thread safety of job state

The `_jobs` dict is accessed from the main thread (`create_job`,
`get_job`) and the background thread (`update_job`). A
`threading.Lock` wrapping every access prevents race conditions. The
lock is fine-grained (held only during dict operations) so it does not
create contention.

### 5.3 Object lifetime

The `qdrant` and `settings` objects are passed explicitly to
`_run_ingestion` via `add_task()`. This captures them in the task
closure, preventing garbage collection after the request completes.

---

## 6. Frontend Polling Loop

### 6.1 Session state management

Two new keys: `polling_job_id` (str | None) and `polling_filename`
(str). Set when POST /ingest returns 202, cleared on completion,
failure, or timeout.

### 6.2 `st.empty()` pattern

`st.empty()` creates a placeholder updated in-place. The loop calls
`status_box.info(...)`, `.success(...)`, or `.error(...)` on each
iteration, replacing the previous message.

### 6.3 Timeout handling

300 seconds (5 minutes). On timeout, show error and clear polling state.
The background job continues — the document will appear in the list on
next page load.

### 6.4 Streamlit rerun behaviour

The polling loop blocks the script thread via `time.sleep(2)`. The UI
is unresponsive during polling. Acceptable for Phase 4.

---

## 7. Golden Dataset

15 questions drawn from the actual documents in `sample_docs/`:

**GOV Contracts and Tenders Regulation (6 questions):**
1. What are the procurement methods available under this regulation?
2. What is the threshold for open tendering?
3. Who is responsible for approving tender evaluation reports?
4. What are the grounds for disqualifying a tenderer?
5. How many days must a tender notice be published before the deadline?
6. What is the process for handling complaints about tender awards?

**2023 Financial Statements (5 questions):**
7. What was the total government expenditure in 2023?
8. What was the revenue from taxes in the 2023 financial year?
9. What is the largest category of government spending?
10. What was the budget deficit or surplus in 2023?
11. How much was spent on education in 2023?

**PSSRM (2 questions):**
12. What is the annual leave entitlement for public servants?
13. What are the disciplinary procedures for misconduct?

**Unanswerable (2 questions):**
14. What is the population of Vanuatu in 2025?
15. What is the exchange rate between Vatu and USD?

Questions 14–15 have `expected_found: false` and empty
`expected_source_pages`. The exact `reference_answer` and
`expected_source_pages` values must be filled in by reading the actual
documents before writing the file.

---

## 8. Evaluation Script Architecture

### 8.1 Argument parsing

```python
parser = argparse.ArgumentParser()
parser.add_argument("--api-url", default="http://localhost:8001")
parser.add_argument("--output-dir", default="eval/results")
parser.add_argument("--dataset", default="eval/golden_dataset.json")
```

### 8.2 Main loop

For each question: POST /query with filename filter, record answer,
sources, found, confidence, latency.

### 8.3 Metric computation

- **Retrieval recall@5:** proportion of answerable questions where at
  least one expected page appears in top-5 sources.
- **Answer similarity:** cosine similarity of embedded answer vs
  reference answer, averaged over answerable questions. Uses
  `sentence-transformers` (already installed) for local embedding.
- **Not-found accuracy:** proportion of unanswerable questions where
  `found == false`.

### 8.4 Output files

- `eval/results/{timestamp}.json` — full results
- `eval/results/latest.md` — Markdown summary

### 8.5 Exit code

0 if all targets met (recall >= 0.80, similarity >= 0.70, not-found
accuracy == 1.0). 1 otherwise.

---

## 9. Tests That Break

**`retrieval_top_k` change (5 → 10):** No existing tests break. All
hybrid search tests pass explicit `top_k` values. API query tests mock
`hybrid_search` entirely.

**Ingest route change (201 → 202):** These 4 tests WILL break:

- `test_ingest_valid_pdf_returns_201` — expects 201
- `test_ingest_response_includes_all_fields` — expects 201 + full body
- `test_ingest_duplicate_returns_replaced_true` — expects 201
- `test_ingest_duplicate_same_chunk_count` — expects 201

They must be updated to expect 202 and `IngestAcceptedResponse`. The
dedup and chunk-count tests must be restructured to poll
`GET /jobs/{job_id}` until completion.

**Confidence threshold change:** `test_confidence.py` tests pass
explicit scores. They must be updated from RRF range (0.025/0.015)
to cross-encoder range (3.0/0.0).

---

## 10. Mocking Strategy

| Test file | Module | What is mocked | How |
|-----------|--------|----------------|-----|
| `test_reranker.py` (new) | `rerank()` | `CrossEncoder` | Pass MagicMock with `.predict()` returning float list |
| `test_reranker.py` | `load_reranker()` | `CrossEncoder` constructor | `@patch("app.retrieval.reranker.CrossEncoder")` |
| `test_job_store.py` (new) | job store | Nothing | Pure in-memory dict |
| `test_api.py` | `POST /ingest` | `_run_ingestion` | `@patch("app.api.routes.ingest._run_ingestion")` — no-op |
| `test_api.py` | `GET /jobs/{id}` | `get_job` | `@patch("app.api.routes.jobs.get_job")` |
| `test_api.py` | query + reranker | `get_reranker` | `dependency_overrides[get_reranker]` — MagicMock |
| `test_confidence.py` | `compute_confidence` | Nothing | Update scores to cross-encoder range |

---

## 11. Acceptance Criteria Traceability

### Reranking (AC-RERANK)

| AC | Test function | File |
|----|---------------|------|
| AC-RERANK-01 | `test_load_reranker_returns_model` | `tests/test_reranker.py` |
| AC-RERANK-02 | `test_rerank_returns_top_k` | `tests/test_reranker.py` |
| AC-RERANK-03 | `test_rerank_returns_all_when_fewer_than_top_k` | `tests/test_reranker.py` |
| AC-RERANK-04 | `test_rerank_ordered_by_score_descending` | `tests/test_reranker.py` |
| AC-RERANK-05 | `test_rerank_empty_input` | `tests/test_reranker.py` |
| AC-RERANK-06 | `test_rerank_replaces_rrf_score` | `tests/test_reranker.py` |
| AC-RERANK-07 | `test_query_route_calls_rerank` | `tests/test_api.py` |

### Async ingestion (AC-ASYNC)

| AC | Test function | File |
|----|---------------|------|
| AC-ASYNC-01 | `test_ingest_returns_202_with_job_id` | `tests/test_api.py` |
| AC-ASYNC-02 | `test_job_status_pending_or_running` | `tests/test_api.py` |
| AC-ASYNC-03 | `test_job_status_completed` | `tests/test_api.py` |
| AC-ASYNC-04 | `test_job_status_failed_for_invalid_pdf` | `tests/test_api.py` |
| AC-ASYNC-05 | `test_job_not_found_returns_404` | `tests/test_api.py` |
| AC-ASYNC-06 | `test_list_jobs_returns_all` | `tests/test_api.py` |
| AC-ASYNC-07 | `test_async_ingested_document_is_queryable` | `tests/test_api.py` |

### Evaluation (AC-EVAL)

| AC | Test function | File |
|----|---------------|------|
| AC-EVAL-01 | `test_golden_dataset_has_15_questions` | `tests/test_evaluation.py` |
| AC-EVAL-02 | Manual — `python scripts/evaluate.py` | — |
| AC-EVAL-03 | Manual — check exit code 0 | — |
| AC-EVAL-04 | Manual — tamper dataset, check exit code 1 | — |
| AC-EVAL-05 | Manual — check `eval/results/*.json` exists | — |
| AC-EVAL-06 | Manual — check `eval/results/latest.md` exists | — |

---

## 12. Definition of Done Checklist

- [ ] `sentence-transformers` installs and `CrossEncoder` loads
- [ ] Reranker singleton loads at startup in lifespan
- [ ] `rerank()` produces correctly ordered, truncated results
- [ ] Query route calls rerank between hybrid_search and context
- [ ] Confidence thresholds recalibrated for cross-encoder scores
- [ ] `POST /ingest` returns 202 with job_id
- [ ] `GET /jobs/{job_id}` returns pending/running/completed/failed
- [ ] Background ingestion completes and updates job store
- [ ] Frontend polls job status and shows progress
- [ ] All 103 existing tests updated and passing
- [ ] All new Phase 4 tests passing
- [ ] Golden dataset contains 15 well-formed questions
- [ ] `python scripts/evaluate.py` runs end-to-end
- [ ] Evaluation results written to `eval/results/`
- [ ] `.env.example` updated with reranker vars
- [ ] `eval/results/` in `.gitignore`
- [ ] `ruff check app/ tests/ scripts/` clean
- [ ] Docker build succeeds
