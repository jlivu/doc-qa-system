# Phase 3 — Frontend & Quality Improvements Implementation Plan

**Spec:** `docs/specs/phase-3-frontend-spec.md` v1.0
**Date:** 2026-05-11

---

## 1. Build Order

```
Step  1  Scaffold             new files, __init__.py stubs
Step  2  Schemas              app/api/schemas.py — 3 new models, 2 field additions
Step  3  Exceptions           app/ingestion/exceptions.py — 1 new class
Step  4  Vector store         app/retrieval/vector_store.py — list_documents, find_document_by_id
Step  5  Context              app/qa/context.py — compute_confidence
Step  6  Prompts              app/qa/prompts.py — OCR-aware prompt with HIGHLIGHT markers
Step  7  Chain                app/qa/chain.py — highlight extraction, strip markers
Step  8  Query route update   app/api/routes/query.py — add confidence to response
Step  9  Documents route      app/api/routes/documents.py — GET + DELETE
Step 10  Main router          app/main.py — register documents router
Step 11  Frontend             frontend/app.py — full rewrite
Step 12  Tests                all test files
```

**Why this order:**

Step 1 creates the new file `app/api/routes/documents.py` so later
steps can import from it. No new packages needed.

Step 2 (schemas) comes before any module code because the documents
route, query route, and frontend all reference the new models
(`DocumentMetadataResponse`, `DocumentListResponse`,
`DeleteDocumentResponse`) and the updated `SourceChunk` (+ `highlight`)
and `QueryResponse` (+ `confidence`).

Step 3 adds `DocumentNotFoundError` to the exceptions hierarchy. The
documents route raises this for 404 responses.

Step 4 (vector store) adds `list_documents()` and
`find_document_by_id()` — called by the documents route in step 9.
These are the most complex new functions (Qdrant scroll pagination)
and must exist before the route.

Step 5 (context) adds `compute_confidence()` — a pure function with no
dependencies beyond `SourceChunk`. Called by the query route in step 8.

Step 6 (prompts) replaces `RAG_SYSTEM_PROMPT` with the OCR-aware
version that includes `HIGHLIGHT[N]:` instructions. The chain (step 7)
parses these markers.

Step 7 (chain) adds highlight extraction to `answer()`. Depends on the
new prompt (step 6) to produce `HIGHLIGHT[N]:` markers. Returns
highlights alongside the answer.

Step 8 (query route) adds `confidence` to the `QueryResponse`. Depends
on `compute_confidence` (step 5) and the updated `answer()` return
value (step 7).

Step 9 (documents route) wires up `GET /documents` and
`DELETE /documents/{document_id}`. Depends on vector store functions
(step 4) and schemas (step 2).

Step 10 registers the documents router in `app/main.py`. One-line
change, but must come after the route exists (step 9).

Step 11 (frontend) depends on all API endpoints being functional. It is
the consumer of every backend change.

Step 12 (tests) last — references final public interfaces.

---

## 2. Scaffold Changes

### 2.1 New file: `app/api/routes/documents.py`

Create the file with a router stub. The `GET` and `DELETE` endpoints
are implemented in step 9.

### 2.2 `app/api/schemas.py` — three new models, two field additions

**Add three new models** after `ErrorResponse`:

```python
class DocumentMetadataResponse(BaseModel):
    document_id: str
    filename: str
    sha256: str
    chunk_count: int
    pages: int

class DocumentListResponse(BaseModel):
    documents: list[DocumentMetadataResponse]
    total: int

class DeleteDocumentResponse(BaseModel):
    document_id: str
    message: str = "Document deleted successfully"
```

**Add `highlight` to `SourceChunk`:**

```python
class SourceChunk(BaseModel):
    document_id: str
    filename: str
    page: int
    text: str
    score: float
    highlight: str | None = None
```

The `highlight` field defaults to `None` so all existing code that
constructs `SourceChunk` without a highlight continues to work.

**Add `confidence` to `QueryResponse`:**

```python
class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    found: bool
    confidence: str
    conversation_history: list[ConversationTurn]
```

`confidence` is a required `str` field — the route must always set it.
Values are `"high"`, `"medium"`, or `"low"`.

**Test impact:** The existing Phase 2 query tests mock `answer()` and
build `QueryResponse` directly. Adding `confidence` as a required field
means the route must supply it. The tests that mock `is_not_found` to
return `True` exercise the not-found path where the route builds
`QueryResponse` directly — these must now include `confidence`. The
tests that mock `answer()` and `is_not_found` to return `False`
exercise the found path — these must also include `confidence`. Since
the tests patch functions at the route module level, the route code
computes confidence from the mocked chunks. Mock chunk scores of 0.85
yield `"high"`, scores of 0.0 (not-found with empty sources) yield
`"low"`.

To avoid breaking existing Phase 2 tests, `confidence` can be given
a default of `"low"` on the schema:

```python
confidence: str = "low"
```

This way existing tests that don't supply `confidence` still construct
valid `QueryResponse` objects. The route overrides it with the computed
value.

### 2.3 `app/ingestion/exceptions.py` — one new class

```python
class DocumentNotFoundError(IngestionError):
    """No chunks exist for the requested document_id."""
```

This maps to HTTP 404 in the documents route.

### 2.4 `app/main.py` — register the documents router

After the existing router registrations:

```python
from app.api.routes import ingest, query, documents

app.include_router(documents.router)
```

---

## 3. Implementation Strategy — Module by Module

### 3.1 Vector Store — `app/retrieval/vector_store.py`

Two new functions.

**`list_documents(client, settings) -> list[dict]`**

Uses the Qdrant scroll pagination loop (detailed in section 4 below) to
iterate through all points, group them by `document_id`, and return
deduplicated document metadata.

**`find_document_by_id(document_id, client, settings) -> bool`**

Uses `client.scroll()` with a payload filter on `document_id` and
`limit=1`. Returns `True` if the scroll returns any points, `False`
otherwise. Wraps in try/except for missing collection — returns `False`
if the collection doesn't exist.

Implementation pattern matches `find_existing_document` in
`app/ingestion/hasher.py` (scroll with filter, limit=1, try/except).

**Pitfalls:**

- `list_documents` must handle the case where the collection does not
  exist (first-ever startup). Wrap the scroll call in try/except and
  return an empty list.
- The `pages` field is `max(page_number)` across chunks for a given
  `document_id`. If a document has only one chunk on page 1, `pages=1`.

### 3.2 Context — `app/qa/context.py`

**`compute_confidence(chunks: list[SourceChunk]) -> str`**

```python
def compute_confidence(chunks: list[SourceChunk]) -> str:
    if not chunks:
        return "low"
    top_score = max(c.score for c in chunks)
    if top_score >= 0.025:
        return "high"
    if top_score >= 0.015:
        return "medium"
    return "low"
```

Pure function. No external dependencies. Thresholds calibrated to
RRF score range (max ~0.033 for rank 1 in both dense and sparse).

### 3.3 Prompts — `app/qa/prompts.py`

Replace `RAG_SYSTEM_PROMPT` entirely. Keep `RAG_HUMAN_TEMPLATE`
unchanged.

The new prompt adds:
1. OCR awareness — instructions to handle broken words, missing spaces,
   garbled characters from scanned PDFs.
2. HIGHLIGHT extraction — for each source used, the LLM extracts the
   single most relevant sentence or phrase (under 30 words) and emits
   it as `HIGHLIGHT[N]: <text>` on its own line.
3. Grounding rule preserved — answer only from context.

The `HIGHLIGHT[N]:` format uses 1-based source indices matching the
`[Source N]` labels in the context block.

**Pitfalls:**

- The prompt must instruct the LLM to put `HIGHLIGHT[N]:` markers on
  separate lines, not inline within the answer paragraph. This makes
  regex extraction reliable.
- The prompt must not change `RAG_HUMAN_TEMPLATE` — it stays as
  `"Question: {question}"`.

### 3.4 Chain — `app/qa/chain.py`

After the LLM returns its response, parse `HIGHLIGHT[N]:` markers.

**Regex:**

```python
import re
HIGHLIGHT_RE = re.compile(r"HIGHLIGHT\[(\d+)\]:\s*(.+)")
```

**Post-processing in `answer()`:**

```python
raw_answer = response.content
highlights: dict[int, str] = {}
clean_lines = []

for line in raw_answer.split("\n"):
    match = HIGHLIGHT_RE.match(line.strip())
    if match:
        source_idx = int(match.group(1))
        highlights[source_idx] = match.group(2).strip()
    else:
        clean_lines.append(line)

clean_answer = "\n".join(clean_lines).strip()

# Apply highlights to source chunks (1-indexed)
for i, chunk in enumerate(chunks, start=1):
    if i in highlights:
        chunk.highlight = highlights[i]
```

The `answer()` function returns `{"answer": clean_answer, "sources": chunks}`
where `clean_answer` has all `HIGHLIGHT[N]:` lines removed and each
`SourceChunk` in `chunks` has its `highlight` field set (or left as
`None`).

**Pitfalls:**

- The regex uses `match()` (anchored to start of line) on stripped
  lines. This avoids false matches on sentences that happen to contain
  the word "HIGHLIGHT" inside regular text.
- `SourceChunk` is a Pydantic model. Setting `.highlight = ...` works
  because Pydantic v2 models are mutable by default.
- The LLM may not emit highlights for every source. Sources without a
  matching `HIGHLIGHT[N]:` line keep `highlight = None`.
- The LLM may emit the marker mid-answer (not on its own line). The
  line-by-line split handles this — lines without the marker pass
  through to `clean_lines`.

### 3.5 Query Route — `app/api/routes/query.py`

Two changes:

1. Import `compute_confidence` from `app.qa.context`.
2. In the found path, after calling `answer()`, compute confidence
   from the source chunks and include it in the response:

```python
from app.qa.context import is_not_found, build_not_found_answer, compute_confidence

# In the found path:
confidence = compute_confidence(chunks_as_source)
return QueryResponse(
    answer=result["answer"],
    sources=sources,
    found=True,
    confidence=confidence,
    conversation_history=updated_history,
)

# In the not-found path:
return QueryResponse(
    answer=answer_text,
    sources=[],
    found=False,
    confidence="low",
    conversation_history=updated_history,
)
```

### 3.6 Documents Route — `app/api/routes/documents.py`

**Router prefix:** `/documents`

**`GET /documents`:**

```python
@router.get("", response_model=DocumentListResponse)
async def list_all_documents(
    settings: SettingsDep,
    qdrant: QdrantDep,
) -> DocumentListResponse:
    try:
        docs = list_documents(qdrant, settings)
        return DocumentListResponse(
            documents=[DocumentMetadataResponse(**d) for d in docs],
            total=len(docs),
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": "RETRIEVAL_ERROR", "detail": str(exc)},
        )
```

**`DELETE /documents/{document_id}`:**

```python
@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(
    document_id: str,
    settings: SettingsDep,
    qdrant: QdrantDep,
) -> DeleteDocumentResponse:
    try:
        if not find_document_by_id(document_id, qdrant, settings):
            return JSONResponse(
                status_code=404,
                content={"error": "DOCUMENT_NOT_FOUND",
                         "detail": f"No document found with id {document_id}"},
            )
        delete_by_document_id(document_id, qdrant, settings)
        return DeleteDocumentResponse(document_id=document_id)
    except StorageError as exc:
        return JSONResponse(
            status_code=500,
            content={"error": "STORAGE_ERROR", "detail": str(exc)},
        )
```

---

## 4. Qdrant Scroll Pagination Loop

`list_documents()` must iterate through *all* points in the collection,
not just the first page. Qdrant's `scroll()` API returns a tuple of
`(points, next_offset)`. The loop continues until `next_offset` is
`None`.

**Exact implementation:**

```python
from typing import TypedDict

class DocumentMetadata(TypedDict):
    document_id: str
    filename: str
    sha256: str
    chunk_count: int
    pages: int

def list_documents(
    client: QdrantClient,
    settings: Settings,
) -> list[DocumentMetadata]:
    """Scroll all points and return deduplicated document metadata."""
    try:
        docs: dict[str, dict] = {}     # document_id → accumulator
        offset = None                   # first call: no offset
        page_size = 1000

        while True:
            scroll_kwargs = {
                "collection_name": settings.qdrant_collection,
                "limit": page_size,
                "with_payload": True,
                "with_vectors": False,
            }
            if offset is not None:
                scroll_kwargs["offset"] = offset

            points, next_offset = client.scroll(**scroll_kwargs)

            for point in points:
                doc_id = point.payload["document_id"]
                if doc_id not in docs:
                    docs[doc_id] = {
                        "document_id": doc_id,
                        "filename": point.payload.get("filename", ""),
                        "sha256": point.payload.get("sha256", ""),
                        "chunk_count": 0,
                        "pages": 0,
                    }
                docs[doc_id]["chunk_count"] += 1
                page_num = point.payload.get("page_number", 0)
                if page_num > docs[doc_id]["pages"]:
                    docs[doc_id]["pages"] = page_num

            if next_offset is None:
                break
            offset = next_offset

        return list(docs.values())

    except Exception:
        return []
```

**Key details:**

- `with_vectors=False` — avoids transferring dense/sparse vectors we
  don't need. Cuts network traffic dramatically for large collections.
- `page_size=1000` — Qdrant's default max. Each scroll call returns up
  to 1000 points.
- The `offset` parameter is `None` for the first call. Qdrant returns a
  `next_offset` value (a point ID) to use for the next call. When
  `next_offset` is `None`, all points have been visited.
- Deduplication happens in the `docs` dict keyed by `document_id`.
- `pages` is the maximum `page_number` seen across all chunks for that
  document.
- The try/except returns an empty list if the collection does not exist
  (first-ever startup before any ingestion).

---

## 5. Highlight Extraction

### 5.1 Prompt format

The system prompt instructs the LLM to emit:

```
HIGHLIGHT[1]: The procurement threshold for direct contracting is VT 500,000.
HIGHLIGHT[3]: Annual leave entitlement shall be 20 working days.
```

The `[N]` index is 1-based, matching the `[Source N]` labels in the
context block. Not every source needs a highlight — only those the LLM
actually used in its answer.

### 5.2 Regex pattern

```python
HIGHLIGHT_RE = re.compile(r"HIGHLIGHT\[(\d+)\]:\s*(.+)")
```

- `\[(\d+)\]` — captures the 1-based source index
- `:\s*` — colon followed by optional whitespace
- `(.+)` — captures the rest of the line as the highlight text

Applied line-by-line on the LLM's raw response using
`match()` (anchored to start of stripped line).

### 5.3 Stripping from answer text

```python
clean_lines = []
for line in raw_answer.split("\n"):
    if not HIGHLIGHT_RE.match(line.strip()):
        clean_lines.append(line)
clean_answer = "\n".join(clean_lines).strip()
```

Lines matching the regex are removed entirely. All other lines are
preserved, including empty lines (to maintain paragraph structure).

### 5.4 Matching to source by index

```python
highlights: dict[int, str] = {}
for line in raw_answer.split("\n"):
    m = HIGHLIGHT_RE.match(line.strip())
    if m:
        highlights[int(m.group(1))] = m.group(2).strip()

for i, chunk in enumerate(chunks, start=1):
    if i in highlights:
        chunk.highlight = highlights[i]
```

- The chunks list is 1-indexed (matching `[Source 1]`, `[Source 2]`, …).
- If the LLM emits `HIGHLIGHT[5]:` but only 3 sources were provided,
  the index 5 is silently ignored (no `chunks[4]` to set).
- Sources without a matching highlight keep `highlight = None` (the
  schema default).

---

## 6. Confidence Thresholds

RRF scores are computed as `Σ 1/(K + rank)` with K=60. The score range
depends on the number of result lists (2: dense + sparse) and the
document's position in each list.

| Scenario | Score | Confidence |
|----------|-------|------------|
| Rank 1 in both lists | `2/(60+1) = 0.0328` | high (≥ 0.025) |
| Rank 1 dense, rank 5 sparse | `1/61 + 1/65 = 0.0318` | high |
| Rank 3 in both lists | `2/(60+3) = 0.0317` | high |
| Rank 1 in dense only (sparse miss) | `1/61 = 0.0164` | medium (≥ 0.015) |
| Rank 5 in both lists | `2/(60+5) = 0.0308` | high |
| Rank 10 in both lists | `2/(60+10) = 0.0286` | high |
| Rank 1 dense, no sparse match | `1/61 = 0.0164` | medium |
| Dense-only fallback, rank 1 | cosine score (0.0–1.0) | varies |
| No results | 0 | low |

The thresholds `0.025` (high) and `0.015` (medium) cover the expected
RRF range well. Most queries with at least one relevant chunk will
produce "high" or "medium" confidence.

**Dense-only fallback caveat:** When the sparse index doesn't exist
(pre-re-ingestion), `hybrid_search` falls back to dense-only and
returns raw cosine similarity scores (0.0–1.0). Cosine scores will
always exceed 0.025, so fallback queries always show "high" confidence.
This is acceptable — the confidence indicator is designed for RRF
scores post-re-ingestion.

**Implementation:**

```python
def compute_confidence(chunks: list[SourceChunk]) -> str:
    if not chunks:
        return "low"
    top_score = max(c.score for c in chunks)
    if top_score >= 0.025:
        return "high"
    if top_score >= 0.015:
        return "medium"
    return "low"
```

---

## 7. Frontend Architecture — `frontend/app.py`

### 7.1 Session state management

```python
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "documents" not in st.session_state:
    st.session_state.documents = []
```

- `conversation_history` — a list of `{"role": str, "content": str}`
  dicts. Updated from the `conversation_history` field in each
  `QueryResponse`. Cleared by the "Clear conversation" button.
- `documents` — cached list from `GET /documents`. Refreshed on page
  load, after each ingest, and after each delete.

### 7.2 API calls

All API calls use the `requests` library (already in
`Dockerfile.frontend`). The base URL is read from the `API_URL`
environment variable (set in `docker-compose.yml` as
`http://api:8001`).

```python
import os, requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

def api_get(path):    return requests.get(f"{API_URL}{path}", timeout=30)
def api_post(path, **kwargs):    return requests.post(f"{API_URL}{path}", timeout=60, **kwargs)
def api_delete(path):    return requests.delete(f"{API_URL}{path}", timeout=30)
```

### 7.3 Error handling

- Wrap every API call in try/except `requests.RequestException`.
- On connection failure: show `st.error("Cannot reach the API. Is the
  server running?")`.
- On HTTP error: show `st.error(f"Error: {response.json()['detail']}")`.
- Never let an unhandled exception crash the page — always degrade to
  an error message.

### 7.4 Layout structure

```python
st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")

# Sidebar
with st.sidebar:
    st.header("📄 Document Library")
    # file uploader + ingest button
    # document list with delete buttons
    st.divider()
    st.header("💬 Conversation")
    # history display + clear button

# Main
st.title("🔍 Ask a Question")
# question input + scope dropdown + ask button
# answer display + confidence badge + sources
```

### 7.5 Ingestion flow

```python
uploaded = st.file_uploader("Upload PDF", type=["pdf"])
if st.button("Ingest") and uploaded:
    with st.spinner("Ingesting..."):
        resp = api_post("/ingest", files={"file": (uploaded.name, uploaded.read(), "application/pdf")})
    if resp.status_code == 201:
        data = resp.json()
        st.success(f"Ingested {data['pages']} pages, {data['chunks']} chunks")
        refresh_documents()
    else:
        st.error(resp.json().get("detail", "Ingestion failed"))
```

### 7.6 Document list and delete

```python
for doc in st.session_state.documents:
    cols = st.columns([4, 1, 1, 1])
    cols[0].write(doc["filename"])
    cols[1].write(f"{doc['pages']}p")
    cols[2].write(f"{doc['chunk_count']}c")
    if cols[3].button("🗑", key=f"del-{doc['document_id']}"):
        resp = api_delete(f"/documents/{doc['document_id']}")
        if resp.status_code == 200:
            refresh_documents()
            st.rerun()
```

### 7.7 Query flow

```python
question = st.text_input("Question")
scope = st.selectbox("Scope", ["All documents"] + [d["filename"] for d in st.session_state.documents])

if st.button("Ask") and question:
    payload = {"question": question, "conversation_history": st.session_state.conversation_history}
    if scope != "All documents":
        doc = next(d for d in st.session_state.documents if d["filename"] == scope)
        payload["filters"] = {"document_id": doc["document_id"]}

    with st.spinner("Thinking..."):
        resp = api_post("/query", json=payload)

    if resp.status_code == 200:
        data = resp.json()
        st.session_state.conversation_history = data["conversation_history"]
        display_answer(data)
```

### 7.8 Answer display

```python
def display_answer(data):
    st.markdown(data["answer"])

    # Confidence badge
    conf = data["confidence"]
    if conf == "high":
        st.success("Confidence: ●●● High")
    elif conf == "medium":
        st.warning("Confidence: ●● Medium")
    else:
        st.error("Confidence: ● Low")

    if not data["found"]:
        st.info("No direct answer found. Try rephrasing your question.")
        return

    # Sources
    for src in data["sources"]:
        with st.expander(f"📄 {src['filename']}, page {src['page']}  (score: {src['score']})"):
            if src.get("highlight"):
                st.markdown(f"> **{src['highlight']}**")
            st.text(src["text"])
```

---

## 8. Mocking Strategy for Tests

### 8.1 Principles

Same as Phases 1 and 2: mock at the boundary (Qdrant, Ollama), run
real code for pure functions. Frontend tests are manual (AC-UI).

### 8.2 Per-module mocking table

| Test file | Module under test | What is mocked | How |
|-----------|-------------------|----------------|-----|
| `test_retrieval.py` | `list_documents` | `QdrantClient` | `client.scroll.side_effect` returning paginated results then `([], None)`. Points have payloads with `document_id`, `filename`, `sha256`, `page_number` |
| `test_retrieval.py` | `find_document_by_id` | `QdrantClient` | `client.scroll.return_value = ([point], None)` for found, `([], None)` for not found |
| `test_context.py` | `compute_confidence` | Nothing | Pure function — pass `SourceChunk` objects with known scores |
| `test_api.py` | `GET /documents` | `list_documents` | `@patch("app.api.routes.documents.list_documents")` returning known dicts |
| `test_api.py` | `DELETE /documents/{id}` | `find_document_by_id`, `delete_by_document_id` | Patch at route module level. Return `True`/`False` for existence check |
| `test_query_chain.py` | highlight extraction | `_get_llm` | Mock LLM returns response with `HIGHLIGHT[1]: ...` lines. Assert highlights are extracted and answer is clean |

### 8.3 Existing test compatibility

- `SourceChunk` gains `highlight: str | None = None` — default `None`
  means existing tests constructing `SourceChunk` without `highlight`
  still work.
- `QueryResponse` gains `confidence: str = "low"` — default `"low"`
  means existing tests building responses without `confidence` still
  work. Tests that check response fields should continue to pass
  because `confidence` is additional, not replacing existing fields.
- Existing Phase 2 query API tests patch `answer()` which returns
  `{"answer": ..., "sources": ...}`. The route now also computes
  confidence from `chunks_as_source`. Since the mock chunks have
  `score=0.85`, `compute_confidence` returns `"high"`. This is fine —
  the tests don't currently assert on `confidence`.
- The `test_query_found_false_when_not_relevant` test returns empty
  sources and mocks `is_not_found` to return `True`. The route sets
  `confidence="low"` for the not-found path. No test break.

---

## 9. Acceptance Criteria Traceability

### Document listing (AC-LIST)

| AC | Test function | File |
|----|---------------|------|
| AC-LIST-01 | `test_list_documents_returns_200` | `tests/test_api.py` |
| AC-LIST-02 | `test_list_documents_includes_all_fields` | `tests/test_api.py` |
| AC-LIST-03 | `test_list_documents_empty_collection` | `tests/test_api.py` |
| AC-LIST-04 | `test_list_documents_after_two_ingests` | `tests/test_api.py` |

### Document deletion (AC-DEL)

| AC | Test function | File |
|----|---------------|------|
| AC-DEL-01 | `test_delete_document_returns_200` | `tests/test_api.py` |
| AC-DEL-02 | `test_delete_document_removed_from_list` | `tests/test_api.py` |
| AC-DEL-03 | `test_delete_unknown_document_returns_404` | `tests/test_api.py` |
| AC-DEL-04 | `test_delete_document_query_returns_not_found` | `tests/test_api.py` |

### Answer quality (AC-QUAL)

| AC | Test function | File |
|----|---------------|------|
| AC-QUAL-01 | `test_query_response_includes_confidence` | `tests/test_api.py` |
| AC-QUAL-02 | `test_confidence_high_for_top_score_above_025` | `tests/test_context.py` |
| AC-QUAL-03 | `test_confidence_medium_for_top_score_015_to_024` | `tests/test_context.py` |
| AC-QUAL-04 | `test_confidence_low_for_top_score_below_015` | `tests/test_context.py` |
| AC-QUAL-05 | `test_source_chunk_includes_highlight` | `tests/test_query_chain.py` |
| AC-QUAL-06 | `test_highlight_is_none_when_no_marker` | `tests/test_query_chain.py` |
| AC-QUAL-07 | `test_answer_text_has_no_raw_highlight_markers` | `tests/test_query_chain.py` |

### Unit tests — vector store

| AC | Test function | File |
|----|---------------|------|
| — | `test_list_documents_returns_metadata` | `tests/test_retrieval.py` |
| — | `test_list_documents_empty_collection` | `tests/test_retrieval.py` |
| — | `test_list_documents_deduplicates_by_document_id` | `tests/test_retrieval.py` |
| — | `test_find_document_by_id_returns_true_for_known` | `tests/test_retrieval.py` |
| — | `test_find_document_by_id_returns_false_for_unknown` | `tests/test_retrieval.py` |

### Frontend (AC-UI)

| AC | Test function | File |
|----|---------------|------|
| AC-UI-01 | Manual test | — |
| AC-UI-02 | Manual test | — |
| AC-UI-03 | Manual test | — |
| AC-UI-04 | Manual test | — |
| AC-UI-05 | Manual test | — |
| AC-UI-06 | Manual test | — |
| AC-UI-07 | Manual test | — |
| AC-UI-08 | Manual test | — |

AC-UI tests are manual — verified by launching the frontend and
exercising each flow visually.

---

## 10. Definition of Done Checklist

- [ ] **All new files created** — `app/api/routes/documents.py`
- [ ] **All existing tests pass** — `pytest tests/ -v` shows 81
  pre-existing Phase 1+2 tests still passing
- [ ] **All 20 Phase 3 automated acceptance criteria pass** — 20
  new tests covering AC-LIST, AC-DEL, AC-QUAL, and vector store
  unit tests
- [ ] **Frontend loads** — `http://localhost:8501` shows the full
  application without errors (AC-UI-01)
- [ ] **Frontend E2E smoke test:**
  ```
  1. Upload a PDF → sidebar shows document with page/chunk counts
  2. Ask a question → answer appears with confidence badge and sources
  3. Sources show highlights (if LLM returns them)
  4. Conversation history appears in sidebar
  5. Delete document → removed from sidebar
  6. Clear conversation → history reset
  ```
- [ ] **API smoke test:**
  ```bash
  # List documents
  curl -s http://localhost:8001/documents | python -m json.tool
  # → 200, documents array with metadata

  # Delete a document
  curl -X DELETE http://localhost:8001/documents/{document_id}
  # → 200, document deleted

  # Query with confidence
  curl -s -X POST http://localhost:8001/query \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the budget?"}' | python -m json.tool
  # → 200, response includes confidence and highlights
  ```
- [ ] **CI green** — GitHub Actions passes
- [ ] **Lint clean** — `ruff check app/ tests/ frontend/`
- [ ] **No Phase 1/2 regressions** — all 81 existing tests pass
