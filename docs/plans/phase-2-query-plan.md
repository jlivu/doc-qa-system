# Phase 2 — Query Pipeline Implementation Plan

**Spec:** `docs/specs/phase-2-query-spec.md` v1.0
**Date:** 2026-05-08

---

## 1. Build Order

Steps are ordered so that every dependency is satisfied before the code
that imports it. Each step is independently testable before moving on.

```
Step  1  Scaffold               new directories, __init__.py
Step  2  Schemas                app/api/schemas.py
Step  3  Exceptions             app/ingestion/exceptions.py
Step  4  Query validator        app/query/validator.py           (new)
Step  5  Retriever rewrite      app/retrieval/retriever.py
Step  6  Context builder        app/qa/context.py                (new)
Step  7  Vector store + ingest  app/retrieval/vector_store.py
                                app/ingestion/embedder.py
Step  8  RAG chain              app/qa/chain.py
Step  9  Query route            app/api/routes/query.py
Step 10  Tests                  all test files
```

**Why this order:**

Step 1 comes first because `app/query/` must exist as a Python package
before `validator.py` can be created inside it.

Step 2 (schemas) comes before any module code because every downstream
module — validator, retriever, context builder, chain, and route —
imports from `app/api/schemas`. The updated `QueryRequest` adds a
`conversation_history` field, and `QueryResponse` replaces the `question`
field with `found` and `conversation_history`. These must exist before any
module that returns or accepts them.

Step 3 (exceptions) comes before steps 4–9 because every module imports
its exception class from `app/ingestion/exceptions.py`. Five new exception
classes are added: `InvalidQuestionError`, `InvalidFiltersError`,
`InvalidHistoryError`, `RetrievalError`, and `GenerationError`. The
existing `EmbeddingError` is reused for query embedding failures.

Step 4 (query validator) is a leaf module with no internal dependencies
beyond schemas and exceptions. It can be built immediately after those
are in place.

Step 5 (retriever) depends on schemas and config. The existing
`retriever.py` references `settings.openai_api_key` and
`settings.embedding_model`, both removed during the Ollama migration. It
must be rewritten to use Ollama's OpenAI-compatible endpoint before the
route can call it.

Step 6 (context builder) depends only on the `SourceChunk` schema. It
contains three pure functions (`build_context`, `is_not_found`,
`build_not_found_answer`) used by the route to format prompts and detect
the not-found condition.

Step 7 (vector store + embedder) adds the sparse vector infrastructure.
The vector store gains `text_to_sparse_vector`, `hybrid_search`, and
`_reciprocal_rank_fusion`. The `_ensure_collection` function is updated
to create both dense and sparse vector configs. The `upsert_chunks`
function switches from unnamed vectors to named `{"dense": ...,
"sparse": ...}` vectors. The embedder itself does not change — sparse
vectors are computed from chunk text at upsert time, not during
embedding. These changes are grouped because they share the sparse vector
concept and must be consistent. They must come before step 9 wires up
hybrid search.

Step 8 (RAG chain) depends on the context builder (step 6) and schemas
(step 2). The `answer()` function gains a `history` parameter and wraps
LLM failures in `GenerationError`. It must be ready before the route
calls it.

Step 9 (query route) orchestrates all modules — validator, retriever,
vector store, context builder, and chain. Every import must already exist.

Step 10 (tests) comes last because tests reference the final public
interfaces of every module.

---

## 2. Scaffold Changes

These changes must be made before any module implementation begins.

### 2.1 New directory: `app/query/`

Create `app/query/__init__.py` as an empty file. The query validator
lives here per spec section 5.1. This is a new package separate from
`app/ingestion/` (which handles PDF processing) and `app/qa/` (which
handles LLM interaction).

### 2.2 Schema updates — `app/api/schemas.py`

Three changes to the existing file. All other schemas (`QueryFilters`,
`SourceChunk`, `IngestResponse`, `ErrorResponse`, `HealthResponse`)
remain unchanged.

**Add `ConversationTurn`** — a new model placed in the Query section,
before `QueryRequest`:

```python
from typing import Literal

class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)
```

The `Literal` type constrains role to exactly two values. Pydantic
validates this at parse time, rejecting any other string with a 422.
`min_length=1` prevents empty messages.

**Replace `QueryRequest`** — add `conversation_history`:

```python
class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    filters: QueryFilters | None = Field(default=None)
    top_k: int | None = Field(default=None, ge=1, le=20)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
```

The `conversation_history` defaults to an empty list so first-turn
queries work without the field. The `description` keyword on
`QueryRequest.question` is removed (it was informational only).

**Replace `QueryResponse`** — add `found` and `conversation_history`;
drop `question`:

```python
class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    found: bool
    conversation_history: list[ConversationTurn]
```

The `question` field is removed because the spec does not include it in
the response (the client already knows the question it sent). The new
`found` boolean distinguishes a grounded answer (`true`) from a
not-found suggestion (`false`). The `conversation_history` returns the
full updated history including the current turn.

**Test impact:** The existing `test_query_rejects_empty_question` sends
`{"question": "ab"}` and expects 422. Pydantic's `min_length=3` on
`QueryRequest.question` still triggers this, so the test continues to
pass. The existing `test_query_returns_501_before_phase2` will break
once the route is wired (it expects 501 but Phase 2 returns 200). That
test must be replaced in step 10.

### 2.3 New exceptions — `app/ingestion/exceptions.py`

Add five new exception classes below the existing six. All inherit from
`IngestionError` (the shared base class):

| Class | Raised by | HTTP status | Error code |
|-------|-----------|-------------|------------|
| `InvalidQuestionError` | Query validator | 422 | `INVALID_QUESTION` |
| `InvalidFiltersError` | Query validator | 422 | `INVALID_FILTERS` |
| `InvalidHistoryError` | Query validator | 422 | `INVALID_HISTORY` |
| `RetrievalError` | Vector store | 500 | `RETRIEVAL_ERROR` |
| `GenerationError` | RAG chain | 500 | `GENERATION_ERROR` |

`EmbeddingError` already exists from Phase 1 and is reused for query
embedding failures.

---

## 3. Implementation Strategy — Module by Module

### 3.1 Query Validator — `app/query/validator.py` (new file)

**Purpose:** Validate the incoming query request before any expensive
operations begin. Fast, cheap checks that fail early.

**Interface:**

```python
def validate_query_request(
    question: str,
    filters: QueryFilters | None,
    history: list[ConversationTurn] | None,
) -> None:
```

**Internal logic:**

1. If `len(question.strip()) < 3`, raise `InvalidQuestionError` with a
   message including the stripped length. This catches whitespace-only
   questions like `"   "` that Pydantic's raw-length check lets through.

2. If `len(question) > 1000`, raise `InvalidQuestionError`. This is a
   belt-and-suspenders check — Pydantic's `max_length=1000` on
   `QueryRequest.question` catches this first, but the validator protects
   against direct function calls outside the HTTP layer.

3. If `filters` is not None and both `filters.document_id` and
   `filters.filename` are non-None, raise `InvalidFiltersError`. This is
   the primary unique validation this function provides — the
   `QueryFilters` Pydantic model allows both fields to be set because
   they are independently optional. Only the validator enforces the
   mutual-exclusion constraint.

4. If `history` is not None, iterate over each entry. If
   `entry.role not in ("user", "assistant")`, raise
   `InvalidHistoryError`. If `not entry.content.strip()`, raise
   `InvalidHistoryError`. The role check is supplementary —
   `ConversationTurn.role` is typed as `Literal["user", "assistant"]` so
   Pydantic catches invalid roles in HTTP requests. But calling
   `validate_query_request` directly with raw dicts bypasses Pydantic,
   so the validator provides a safety net.

5. Return `None` on success.

**Pitfalls:**

The stripped-length check for questions must use `question.strip()`, not
`question`. A question of `"   "` (three spaces) passes Pydantic's
`min_length=3` but is semantically empty.

---

### 3.2 Retriever Rewrite — `app/retrieval/retriever.py`

**Current state:** The existing `retriever.py` imports
`from openai import OpenAI` and references `settings.openai_api_key` and
`settings.embedding_model`. Both settings were removed during the Ollama
migration. The file is completely broken and must be rewritten.

**Changes:**

1. Rename `_embed_query` to `embed_query` (make it public — the route
   calls it directly per spec section 5.2).

2. Use `settings.ollama_base_url + "/v1"` and
   `settings.ollama_embedding_model`, matching the pattern established in
   `app/ingestion/embedder.py`. Use `api_key="ollama"` (Ollama does not
   validate API keys, but the OpenAI client requires a non-empty value).

3. Wrap the entire function body in try/except and raise
   `EmbeddingError` on any failure. No retry logic — the spec says
   "do not retry (Ollama is local)".

4. Update `retrieve()` to call `embed_query` (the renamed public
   function) and pass results to `hybrid_search` instead of `search`.
   The `retrieve()` function remains the high-level orchestrator. It
   maps `SearchResult` objects to `SourceChunk` objects with
   `score=round(r.score, 4)`.

**Library usage:** The `openai` Python package is used with a custom
`base_url` pointing at Ollama's OpenAI-compatible endpoint. This is the
same approach used by `embedder.py` and avoids introducing a separate
embedding client for queries.

**Pitfalls:**

The `EmbeddingError` catch must use a bare `except Exception` rather
than catching a specific OpenAI exception type. The OpenAI client can
raise `httpx.ConnectError` (Ollama not running), `openai.APIError`
(model not found), or other exceptions depending on the failure mode.
Catching `Exception` and re-raising as `EmbeddingError` normalises all
failures to a single type.

---

### 3.3 Context Builder — `app/qa/context.py` (new file)

**Purpose:** Three pure functions that format retrieved chunks for the
LLM prompt and handle the not-found condition.

**`build_context(chunks: list[SourceChunk]) -> str`:**

Iterates over chunks and formats each as:

```
[Source 1] filename.pdf, page 7 (score: 0.8923)
The actual chunk text here...
```

Chunks are separated by `\n\n---\n\n`. This is an evolution of the
existing `_build_context` in `chain.py` — the Phase 2 version adds
`(score: {chunk.score})` to the source label for transparency.

**`is_not_found(chunks: list[SourceChunk], threshold: float = 0.3) -> bool`:**

Returns `True` if `chunks` is empty. Returns `True` if the highest score
across all chunks is below `threshold`. Returns `False` otherwise.

The default threshold of 0.3 reflects RRF scores, which are
significantly lower than raw cosine similarity scores. A document
appearing at rank 1 in both dense and sparse results gets a maximum RRF
score of `2 / (60 + 1) ≈ 0.033`. After many documents overlap, scores
accumulate but rarely exceed 0.1. The 0.3 threshold is intentionally
generous — it catches only clearly irrelevant results. Tuning this
value requires evaluation against real queries on the target document
corpus.

**`build_not_found_answer(chunks: list[SourceChunk]) -> str`:**

Called only when `is_not_found` returns `True`. Two code paths:

If `chunks` is empty (nothing found at all), return a generic message:
`"I could not find any relevant information in the available documents. Try rephrasing your question or uploading additional documents."`

If chunks exist but all scores are below threshold, extract the set of
unique filenames from the chunks and return:
`"I could not find a direct answer to your question in the available documents. The documents do contain information about: {filenames}. Try rephrasing your question around one of these topics."`

This avoids hallucination (no LLM is called for the not-found case)
while still being helpful.

---

### 3.4 RAG Chain — `app/qa/chain.py` (modify existing)

**Current state:** The `answer()` function accepts `(question, chunks,
settings)` and builds a two-message list: `[SystemMessage, HumanMessage]`.
It has a private `_build_context` helper and an `if not chunks` early
return.

**Changes:**

1. **Remove `_build_context`** — it moves to `app/qa/context.py` as the
   public `build_context`. Import it from there.

2. **Add `history` parameter** — the signature becomes
   `answer(question, chunks, history, settings)` where `history` is
   `list[ConversationTurn]`.

3. **Build the message list with history.** The order is:
   - `SystemMessage` with context (from `build_context`)
   - All prior turns from `history`, mapped to `HumanMessage` (for
     `role == "user"`) or `AIMessage` (for `role == "assistant"`)
   - Current question as a final `HumanMessage`

   This requires importing `AIMessage` from `langchain_core.messages`
   (not currently imported).

4. **Wrap the LLM call** in try/except. Catch any exception from
   `llm.invoke(messages)` and re-raise as `GenerationError`. This gives
   the route a single exception type to map to HTTP 500.

5. **Remove the `if not chunks` early return.** The route now handles
   the not-found condition before calling `answer()` — it checks
   `is_not_found(chunks)` and returns a `QueryResponse` with
   `found=False` directly. The chain is only called when chunks are
   present.

**Library usage:** `langchain_core.messages.AIMessage` is added to the
existing `HumanMessage` and `SystemMessage` imports.
`langchain_ollama.ChatOllama` is already imported. No new packages.

**Pitfalls:**

The `_get_llm` function is `@lru_cache`d. It takes `(model, base_url)`
as arguments. These are hashable strings, so caching works. Do not pass
mutable objects (like `settings`) as arguments to `_get_llm` — they are
not hashable and will break the cache.

---

## 4. Sparse Vector Index — Detailed Implementation

This is the most complex new feature. It touches `vector_store.py`
(three changes) and has a downstream impact on re-ingestion.

### 4.1 `text_to_sparse_vector()` — `app/retrieval/vector_store.py`

**Purpose:** Convert arbitrary text into a Qdrant `SparseVector` using
term-frequency weighting. This function is used at both ingestion time
(to compute sparse vectors for stored chunks) and query time (to
compute sparse vectors for the search query).

**Algorithm:**

1. Lowercase the text and tokenise with `re.findall(r'\w+', text.lower())`.
   This produces a list of alphanumeric tokens, stripping punctuation.

2. Count term frequencies with `collections.Counter(tokens)`.

3. For each unique token, compute:
   - **Index:** `hash(token) % (2**31)`. This maps each token to a
     stable integer in the range `[0, 2^31)`. The Python `hash()`
     function is deterministic within a process, but its seed varies
     across Python invocations due to hash randomisation
     (`PYTHONHASHSEED`). To make indices stable across restarts, set
     `PYTHONHASHSEED=0` in the environment, or use a deterministic hash
     like `hashlib.md5(token.encode()).digest()[:4]` interpreted as an
     int. For Phase 2, `hash()` with `PYTHONHASHSEED=0` is acceptable.
   - **Value:** `math.log(1 + count)`. Sublinear term frequency — a
     token appearing 10 times is not 10x more important than one
     appearing once.

4. Return `SparseVector(indices=indices, values=values)`.

**Key decisions:**

- **No IDF.** True BM25 uses inverse document frequency across the
  corpus. Computing IDF requires scanning all documents at query time or
  maintaining a global term-frequency table. This is deferred to Phase 4.
  The TF-only approach still provides keyword matching — it just does not
  down-weight common terms.

- **No external libraries.** Only `re`, `math`, `collections.Counter`
  from the stdlib. The spec explicitly avoids `fastembed` for Phase 2.

- **Hash collisions.** With 2^31 buckets (~2 billion), the probability
  of two tokens colliding is negligible for typical document vocabularies
  (10k–100k unique terms).

### 4.2 Collection schema update — `_ensure_collection()`

**Current state:** Creates a collection with a single unnamed dense
vector:

```python
VectorParams(size=vector_size, distance=Distance.COSINE)
```

**New state:** Creates a collection with named dense and sparse vectors:

```python
vectors_config={
    "dense": VectorParams(size=vector_size, distance=Distance.COSINE),
},
sparse_vectors_config={
    "sparse": SparseVectorParams(
        index=SparseIndexParams(on_disk=False),
    ),
},
```

The `SparseVectorParams` and `SparseIndexParams` are imported from
`qdrant_client.models`. `on_disk=False` keeps the sparse index in memory
for fast access.

**Critical impact on existing data:** The current collection uses an
unnamed vector config. Switching to named vectors is a schema-level
change. Qdrant does not support migrating from unnamed to named vectors
on an existing collection. Points stored with unnamed vectors cannot be
queried with named-vector syntax (`("dense", query_vector)`).

**Re-ingestion requirement:** After Phase 2 code is deployed, the
existing Qdrant collection must be deleted and all documents re-ingested.
The smoke test procedure is:

1. Stop the API.
2. Delete the collection (or drop the Qdrant volume):
   `curl -X DELETE http://localhost:6333/collections/documents`
3. Start the API.
4. Re-ingest all documents via `POST /ingest`.

The spec acknowledges this in section 8: "Existing ingested documents do
not have sparse vectors. After the Phase 2 code is deployed, documents
must be re-ingested to populate the sparse index."

### 4.3 Changes to `upsert_chunks()` — `vector_store.py`

Currently, each `PointStruct` stores an unnamed vector:

```python
PointStruct(
    id=chunk["chunk_id"],
    vector=chunk["vector"],
    payload={...},
)
```

After the change, each point stores named dense and sparse vectors:

```python
PointStruct(
    id=chunk["chunk_id"],
    vector={
        "dense": chunk["vector"],
        "sparse": text_to_sparse_vector(chunk["text"]),
    },
    payload={...},
)
```

The sparse vector is computed from `chunk["text"]` at upsert time. This
means `text_to_sparse_vector` is called once per chunk during ingestion.
For a 42-page document with ~187 chunks, this adds negligible overhead
(pure Python string processing).

No changes to `embedder.py` are needed. The `EmbeddedChunk` type does
not gain a sparse vector field. Sparse vectors are derived from text,
not from the embedding model, so they belong in the storage layer.

### 4.4 Changes to `search()` — the existing dense-only function

The existing `search()` uses unnamed vectors:

```python
client.search(query_vector=query_vector, ...)
```

With named vectors this becomes:

```python
client.search(query_vector=("dense", query_vector), ...)
```

This is a backward-incompatible change at the Qdrant API level.
However, existing Phase 1 tests that call `search()` use a `MagicMock`
for the Qdrant client. The mock does not validate the `query_vector`
argument format, so these tests continue to pass without modification.
After Phase 2, `retrieve()` calls `hybrid_search()` instead of
`search()`, so `search()` becomes internal.

---

## 5. RRF Merge Logic — Step by Step

Reciprocal Rank Fusion (RRF) combines ranked result lists from different
retrieval methods without requiring score normalisation or weight tuning.

### 5.1 The `hybrid_search()` function

**Signature:**

```python
def hybrid_search(
    query_vector: list[float],
    query_text: str,
    client: QdrantClient,
    settings: Settings,
    top_k: int,
    document_id: str | None = None,
    filename: str | None = None,
) -> list[SearchResult]:
```

**Step-by-step logic:**

1. **Build the metadata filter.** Reuse the same filter-building logic
   as the existing `search()` function: if `document_id` is provided,
   add a `FieldCondition` on `document_id`. If `filename` is provided,
   add a `FieldCondition` on `filename`. If neither, no filter.

2. **Run the dense search.** Call `client.search()` with
   `query_vector=("dense", query_vector)`, the metadata filter, and
   `limit=top_k * 2`. The over-fetch factor of 2 ensures enough
   candidates for RRF merging — some results may appear in only one of
   the two lists.

3. **Run the sparse search.** Call `client.search()` with
   `query_vector=NamedSparseVector(name="sparse", vector=text_to_sparse_vector(query_text))`,
   the same metadata filter, and `limit=top_k * 2`. Wrap this call in
   try/except. If it raises (e.g., the collection has no sparse index
   because documents were ingested before Phase 2), fall back to
   dense-only results: return the dense results directly, truncated to
   `top_k`, with scores rounded to 4 decimal places.

4. **Merge with RRF.** Call `_reciprocal_rank_fusion(dense_results,
   sparse_results, top_k)`.

5. **Return** the merged `list[SearchResult]`.

### 5.2 The `_reciprocal_rank_fusion()` helper

**Signature:**

```python
RRF_K = 60

def _reciprocal_rank_fusion(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
```

**Step-by-step logic:**

1. **Initialise two dictionaries:**
   - `scores: dict[str, float]` — maps `chunk_id` to accumulated RRF
     score.
   - `results_by_id: dict[str, SearchResult]` — maps `chunk_id` to
     the `SearchResult` object (for building the final list).

2. **Score the dense results.** For each result at 1-based rank `r`,
   add `1 / (RRF_K + r)` to `scores[result.chunk_id]`. Store the
   result in `results_by_id`.

   With `RRF_K = 60`, rank 1 contributes `1/61 ≈ 0.0164`, rank 2
   contributes `1/62 ≈ 0.0161`, etc. The constant 60 dampens the
   difference between adjacent ranks.

3. **Score the sparse results.** Same formula. If a chunk appears in
   both lists, its scores are summed. A chunk at rank 1 in both lists
   gets `1/61 + 1/61 ≈ 0.0328`.

4. **Sort by accumulated RRF score descending.** Take the top `top_k`
   chunk IDs.

5. **Build the output list.** For each selected chunk ID, retrieve the
   `SearchResult` from `results_by_id`. Set `result.score` to the
   rounded RRF score (`round(scores[cid], 4)`). Append to the output
   list.

6. **Return** the merged list.

**Why RRF_K = 60:** This is the standard constant from the original RRF
paper (Cormack, Clarke, Buettcher 2009). It has been widely adopted
without modification. It balances the contribution of top-ranked vs
lower-ranked results.

**Fallback behaviour:** If the sparse search fails (no sparse index),
`hybrid_search` returns the dense results directly. Each dense result
retains its original cosine similarity score, not an RRF score. This
provides graceful degradation: queries work before re-ingestion, they
just use dense-only search.

---

## 6. Mocking Strategy for Tests

### 6.1 Principles

Same as Phase 1: mock at the boundary (Ollama, Qdrant), run real code
for pure functions (validator, context builder, `text_to_sparse_vector`).

### 6.2 Per-module mocking table

| Test file | Module under test | What is mocked | How |
|-----------|-------------------|----------------|-----|
| `test_query_validator.py` | query validator | Nothing | Pure function — no external calls |
| `test_context.py` | context builder | Nothing | Pure functions operating on `SourceChunk` objects |
| `test_query_chain.py` | chain `answer()` | `ChatOllama` via `_get_llm` | `@patch("app.qa.chain._get_llm")` — return a `MagicMock` whose `.invoke()` returns a mock response with a `.content` attribute set to a known answer string |
| `test_retrieval.py` | `embed_query` | `OpenAI` client | `@patch("app.retrieval.retriever.OpenAI")` — configure `mock_cls.return_value` so that `client.embeddings.create()` returns a mock response with `.data[0].embedding` set to a list of floats |
| `test_retrieval.py` | `hybrid_search` | `QdrantClient` | Pass a `MagicMock()` as `client`. Use `client.search.side_effect` to return different results for the two `client.search()` calls (dense first, sparse second). For the fallback test (AC-HSEARCH-06), make the second call raise an exception |
| `test_retrieval.py` | `text_to_sparse_vector` | Nothing | Pure function — only uses stdlib (`re`, `math`, `Counter`). Assert that the return value is a `SparseVector` with non-empty `indices` and `values` |
| `test_api.py` | query route (E2E) | `embed_query`, `hybrid_search`, `answer` | Patch at the route module level: `@patch("app.api.routes.query.answer")` returns `{"answer": "test", "sources": chunks}`. `@patch("app.api.routes.query.hybrid_search")` returns a list of mock `SearchResult` objects. `@patch("app.api.routes.query.embed_query")` returns `[0.1] * 768`. This avoids any real Ollama or Qdrant calls |

### 6.3 Existing test compatibility

`test_query_returns_501_before_phase2` must be **removed and replaced**
since the route no longer returns 501. It is replaced by the AC-ROUTE
query tests.

`test_query_rejects_empty_question` **remains valid**. The updated
`QueryRequest` still has `min_length=3` on `question`, so `"ab"` still
triggers Pydantic's 422.

All existing Phase 1 tests (`test_ingestion.py`, ingest tests in
`test_api.py`, upsert/delete/search tests in `test_retrieval.py`,
`test_hasher.py`, `test_validator.py`) must continue to pass. The schema
changes add new fields to `QueryResponse` but do not alter
`IngestResponse`, `SourceChunk`, or `QueryFilters`. The `search()`
change from unnamed to named vector syntax is invisible to mocked tests
because `MagicMock` does not validate argument values.

### 6.4 `hybrid_search` mock detail for `test_api.py`

The route calls `hybrid_search()` which returns `list[SearchResult]`.
The mock must return objects that the route can convert to `SourceChunk`
instances. Use `MagicMock` objects with attributes matching
`SearchResult`: `chunk_id`, `score`, `document_id`, `filename`,
`page_number`, `text`. The existing `_make_mock_hit` helper in
`test_retrieval.py` shows this pattern.

For the not-found test (AC-ROUTE-04), make the mock return results with
scores below 0.3 (the `is_not_found` threshold). The route will detect
this and return `found=False`.

---

## 7. Acceptance Criteria Traceability

Every AC from spec section 10 is mapped to a specific test function and
file.

### Validation (AC-QVAL)

| AC | Test function | File |
|----|---------------|------|
| AC-QVAL-01 | `test_validate_rejects_short_question` | `tests/test_query_validator.py` |
| AC-QVAL-02 | `test_validate_rejects_long_question` | `tests/test_query_validator.py` |
| AC-QVAL-03 | `test_validate_rejects_dual_filters` | `tests/test_query_validator.py` |
| AC-QVAL-04 | `test_validate_rejects_invalid_history_role` | `tests/test_query_validator.py` |
| AC-QVAL-05 | `test_validate_accepts_valid_request` | `tests/test_query_validator.py` |

### Embedding (AC-QEMB)

| AC | Test function | File |
|----|---------------|------|
| AC-QEMB-01 | `test_embed_query_returns_float_list` | `tests/test_retrieval.py` |
| AC-QEMB-02 | `test_embed_query_raises_embedding_error` | `tests/test_retrieval.py` |

### Hybrid search (AC-HSEARCH)

| AC | Test function | File |
|----|---------------|------|
| AC-HSEARCH-01 | `test_hybrid_search_ordered_by_rrf_score` | `tests/test_retrieval.py` |
| AC-HSEARCH-02 | `test_hybrid_search_document_id_filter` | `tests/test_retrieval.py` |
| AC-HSEARCH-03 | `test_hybrid_search_filename_filter` | `tests/test_retrieval.py` |
| AC-HSEARCH-04 | `test_hybrid_search_no_filter_returns_all` | `tests/test_retrieval.py` |
| AC-HSEARCH-05 | `test_hybrid_search_empty_collection` | `tests/test_retrieval.py` |
| AC-HSEARCH-06 | `test_hybrid_search_falls_back_to_dense_only` | `tests/test_retrieval.py` |

### Context builder (AC-CTX)

| AC | Test function | File |
|----|---------------|------|
| AC-CTX-01 | `test_build_context_format` | `tests/test_context.py` |
| AC-CTX-02 | `test_is_not_found_empty_chunks` | `tests/test_context.py` |
| AC-CTX-03 | `test_is_not_found_low_scores` | `tests/test_context.py` |
| AC-CTX-04 | `test_is_not_found_above_threshold` | `tests/test_context.py` |
| AC-CTX-05 | `test_not_found_answer_with_low_score_chunks` | `tests/test_context.py` |
| AC-CTX-06 | `test_not_found_answer_empty_chunks` | `tests/test_context.py` |

### RAG chain (AC-CHAIN)

| AC | Test function | File |
|----|---------------|------|
| AC-CHAIN-01 | `test_answer_returns_non_empty_string` | `tests/test_query_chain.py` |
| AC-CHAIN-02 | `test_answer_returns_input_chunks_as_sources` | `tests/test_query_chain.py` |
| AC-CHAIN-03 | `test_answer_includes_history_in_messages` | `tests/test_query_chain.py` |
| AC-CHAIN-04 | `test_answer_raises_generation_error` | `tests/test_query_chain.py` |

### End-to-end route (AC-ROUTE)

| AC | Test function | File |
|----|---------------|------|
| AC-ROUTE-01 | `test_query_valid_question_returns_200` | `tests/test_api.py` |
| AC-ROUTE-02 | `test_query_response_includes_all_fields` | `tests/test_api.py` |
| AC-ROUTE-03 | `test_query_found_true_when_relevant` | `tests/test_api.py` |
| AC-ROUTE-04 | `test_query_found_false_when_not_relevant` | `tests/test_api.py` |
| AC-ROUTE-05 | `test_query_sources_include_all_metadata` | `tests/test_api.py` |
| AC-ROUTE-06 | `test_query_history_appended_in_response` | `tests/test_api.py` |
| AC-ROUTE-07 | `test_query_followup_with_history` | `tests/test_api.py` |
| AC-ROUTE-08 | `test_query_document_id_filter` | `tests/test_api.py` |
| AC-ROUTE-09 | `test_query_invalid_question_returns_422` | `tests/test_api.py` |

---

## 8. Definition of Done Checklist

- [ ] **All new files created** — `app/query/__init__.py`,
  `app/query/validator.py`, `app/qa/context.py`
- [ ] **All existing Phase 1 tests pass** — `pytest tests/ -v` shows
  the 47 pre-existing Phase 1 tests still passing (the one test
  `test_query_returns_501_before_phase2` is replaced, not broken)
- [ ] **All 30 Phase 2 acceptance criteria pass** — 30 new tests
  covering every AC from spec section 10 (see traceability table above)
- [ ] **Re-ingestion works** — after code changes, the old Qdrant
  collection is deleted and the sample document is re-ingested to
  populate both dense and sparse vectors
- [ ] **CI green** — the GitHub Actions workflow passes on a PR branch
  with all Phase 2 changes
- [ ] **Lint clean** — `ruff check app/ tests/` reports zero violations
- [ ] **Manual smoke test passes:**
  ```bash
  # Delete old collection and re-ingest
  curl -X DELETE http://localhost:6333/collections/documents
  curl -X POST http://localhost:8001/ingest -F "file=@sample.pdf"

  # Happy path query
  curl -s -X POST http://localhost:8001/query \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the total expenditure?"}' | python -m json.tool
  # → 200, found=true, sources non-empty, conversation_history has 2 entries

  # Follow-up with history
  curl -s -X POST http://localhost:8001/query \
    -H "Content-Type: application/json" \
    -d '{"question": "How does that compare to last year?",
         "conversation_history": [
           {"role": "user", "content": "What is the total expenditure?"},
           {"role": "assistant", "content": "The total expenditure is..."}
         ]}' | python -m json.tool
  # → 200, history has 4 entries

  # Not-found
  curl -s -X POST http://localhost:8001/query \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the speed of light?"}' | python -m json.tool
  # → 200, found=false, sources=[]

  # Validation error
  curl -s -X POST http://localhost:8001/query \
    -H "Content-Type: application/json" \
    -d '{"question": "ab"}' | python -m json.tool
  # → 422
  ```
- [ ] **Error response format** — every error returns
  `{"error": "CODE", "detail": "..."}`, not FastAPI's default format
- [ ] **No Phase 1 regressions** — the ingest pipeline is unaffected;
  all ingest tests pass; `POST /ingest` with a PDF still returns 201
