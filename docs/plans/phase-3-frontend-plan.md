# Phase 3 — Frontend & Quality Improvements Implementation Plan

**Spec:** `docs/specs/phase-3-frontend-spec.md` v1.0
**Date:** 2026-05-11

---

## 1. Build Order

```
Step  1  Schemas              app/api/schemas.py
Step  2  Exceptions           app/ingestion/exceptions.py
Step  3  Vector store         app/retrieval/vector_store.py
Step  4  Context              app/qa/context.py
Step  5  Prompts              app/qa/prompts.py
Step  6  Chain                app/qa/chain.py
Step  7  Query route update   app/api/routes/query.py
Step  8  Documents route      app/api/routes/documents.py  (new)
Step  9  Main router          app/main.py
Step 10  Frontend             frontend/app.py
Step 11  Tests                all test files
```

**Why this order:**

Step 1 (schemas) comes first because every downstream module imports
the new models. `DocumentMetadataResponse`, `DocumentListResponse`, and
`DeleteDocumentResponse` are used by the documents route (step 8).
`SourceChunk` gains `highlight: str | None = None` — the default `None`
keeps all 81 existing tests passing. `QueryResponse` gains
`confidence: str = "low"` — the default keeps existing Phase 2 query
tests passing without modification.

Step 2 adds `DocumentNotFoundError` to the exception hierarchy. The
documents route (step 8) raises this for 404 responses.

Step 3 (vector store) adds `list_documents()` and
`find_document_by_id()`. Both are called by the documents route in
step 8. `list_documents()` is the most complex new function — it uses
Qdrant's scroll pagination loop (detailed in section 4).

Step 4 (context) adds `compute_confidence()` — a pure function. Called
by the query route in step 7.

Step 5 (prompts) replaces `RAG_SYSTEM_PROMPT` with the OCR-aware
version that instructs the LLM to emit `HIGHLIGHT[N]:` markers.

Step 6 (chain) adds highlight extraction to `answer()`. Parses
`HIGHLIGHT[N]:` markers from the LLM response, strips them from the
visible answer text, and sets `chunk.highlight` on each source.
Depends on the new prompt (step 5).

Step 7 (query route) adds `confidence` to the `QueryResponse` for
both found and not-found paths. Depends on `compute_confidence`
(step 4) and the updated `answer()` return (step 6).

Step 8 (documents route) wires up `GET /documents` and
`DELETE /documents/{document_id}`. Depends on vector store functions
(step 3), schemas (step 1), and exceptions (step 2).

Step 9 registers the documents router in `app/main.py`. Must come
after the route file exists (step 8).

Step 10 (frontend) depends on all API endpoints being functional.

Step 11 (tests) last — references final public interfaces.

---

## 2. Scaffold Changes

### 2.1 `app/api/schemas.py` — three new models, two field additions

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

Default `None` means every existing test that constructs `SourceChunk`
without `highlight` continues to work unchanged.

**Add `confidence` to `QueryResponse`:**

```python
class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    found: bool
    confidence: str = "low"
    conversation_history: list[ConversationTurn]
```

Default `"low"` means existing Phase 2 tests that construct
`QueryResponse` without `confidence` still pass. The route overrides
it with the computed value.

### 2.2 `app/ingestion/exceptions.py` — one new class

```python
class DocumentNotFoundError(IngestionError):
    """No chunks exist for the requested document_id."""
```

Added after the existing Phase 2 exceptions. Maps to HTTP 404 in
the documents route.

### 2.3 `app/main.py` — register the documents router

Add one import and one `include_router` call. The exact location is
after the existing `query.router` registration:

```python
from app.api.routes import ingest, query, documents
# ...
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(documents.router)   # NEW
```

### 2.4 New file: `app/api/routes/documents.py`

Created in step 8. Contains `GET /documents` and
`DELETE /documents/{document_id}` with router prefix `/documents`.

---

## 3. Implementation Strategy — Module by Module

### 3.1 Vector Store — `app/retrieval/vector_store.py`

Two new public functions.

**`list_documents(client, settings) -> list[dict]`**

Scrolls all points in the collection, groups by `document_id`, and
returns deduplicated metadata. Full implementation in section 4.

**`find_document_by_id(document_id, client, settings) -> bool`**

Uses `client.scroll()` with a payload filter on `document_id` and
`limit=1`. Returns `True` if any point is found, `False` otherwise.

```python
def find_document_by_id(
    document_id: str,
    client: QdrantClient,
    settings: Settings,
) -> bool:
    try:
        points, _ = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="document_id",
                                     match=MatchValue(value=document_id))]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(points) > 0
    except Exception:
        return False
```

The pattern mirrors `find_existing_document` in
`app/ingestion/hasher.py`. The try/except returns `False` if the
collection does not exist yet.

**Pitfalls:**

- `with_payload=False` and `with_vectors=False` minimise data transfer.
  We only need to know if a point exists, not its content.
- The `_build_filter` helper already exists and handles
  `FieldCondition` construction. However, `find_document_by_id` uses
  `scroll_filter` (not `query_filter`), so the filter is built inline.

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

Pure function. No external dependencies. Threshold calibration
detailed in section 6.

### 3.3 Prompts — `app/qa/prompts.py`

Replace `RAG_SYSTEM_PROMPT` entirely. Keep `RAG_HUMAN_TEMPLATE`
unchanged (`"Question: {question}"`).

The new prompt adds three things:
1. OCR awareness — instructs the LLM to handle broken words, missing
   spaces, garbled characters from scanned PDFs.
2. HIGHLIGHT extraction — for each source used, the LLM emits
   `HIGHLIGHT[N]: <sentence>` on its own line, where N matches the
   `[Source N]` label in the context block.
3. Grounding rule preserved — answer only from context.

The exact prompt text is specified in spec section 5.4.

**Pitfalls:**

- The `HIGHLIGHT[N]:` instruction must tell the LLM to put each
  marker on its own line. Inline markers break the regex parser.
- The `{context}` placeholder must remain at the end of the prompt
  (same position as the current prompt) — `build_context()` in
  `context.py` injects the formatted chunks here.

### 3.4 Chain — `app/qa/chain.py`

After the LLM returns its response, parse `HIGHLIGHT[N]:` markers
from the response text, strip them from the visible answer, and set
`chunk.highlight` on each source. Full logic in section 5.

The `answer()` function signature does **not** change — it still
returns `{"answer": str, "sources": list[SourceChunk]}`. The
difference is that `answer` is now the cleaned text (markers removed)
and each chunk in `sources` may have its `.highlight` field set.

**Pitfalls:**

- `SourceChunk` is a Pydantic model. Setting `.highlight = "..."` on
  an existing instance works because Pydantic v2 models are mutable
  by default. No need to reconstruct the object.
- The LLM may not emit highlights for every source. Unmatched sources
  keep `highlight = None` (the schema default).
- The LLM may emit markers that reference source indices beyond the
  actual source list. These are silently ignored.

### 3.5 Query Route — `app/api/routes/query.py`

Two changes:

1. Import `compute_confidence` from `app.qa.context`.
2. In both the found path and the not-found path, set the
   `confidence` field on `QueryResponse`.

Found path — confidence is computed from the source chunks:

```python
confidence = compute_confidence(chunks_as_source)
return QueryResponse(
    ...,
    confidence=confidence,
    ...
)
```

Not-found path — confidence is always `"low"`:

```python
return QueryResponse(
    ...,
    confidence="low",
    ...
)
```

### 3.6 Documents Route — `app/api/routes/documents.py`

Router prefix: `/documents`. Two endpoints.

**`GET /documents`** — calls `list_documents()`, wraps results in
`DocumentListResponse`:

```python
@router.get("", response_model=DocumentListResponse)
async def list_all_documents(settings: SettingsDep, qdrant: QdrantDep):
    try:
        docs = list_documents(qdrant, settings)
        return DocumentListResponse(
            documents=[DocumentMetadataResponse(**d) for d in docs],
            total=len(docs),
        )
    except Exception as exc:
        return JSONResponse(status_code=500,
                            content={"error": "RETRIEVAL_ERROR",
                                     "detail": str(exc)})
```

**`DELETE /documents/{document_id}`** — checks existence, then deletes:

```python
@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(document_id: str, settings: SettingsDep,
                          qdrant: QdrantDep):
    try:
        if not find_document_by_id(document_id, qdrant, settings):
            return JSONResponse(status_code=404,
                                content={"error": "DOCUMENT_NOT_FOUND",
                                         "detail": f"No document with id {document_id}"})
        delete_by_document_id(document_id, qdrant, settings)
        return DeleteDocumentResponse(document_id=document_id)
    except StorageError as exc:
        return JSONResponse(status_code=500,
                            content={"error": "STORAGE_ERROR",
                                     "detail": str(exc)})
```

Uses the same `JSONResponse` error pattern as the ingest and query
routes.

---

## 4. Qdrant Scroll Pagination Loop

`list_documents()` must iterate through *all* points in the collection.
Qdrant's `scroll()` returns `(points, next_offset)`. The loop continues
until `next_offset` is `None`.

```python
def list_documents(
    client: QdrantClient,
    settings: Settings,
) -> list[dict]:
    try:
        docs: dict[str, dict] = {}
        offset = None
        page_size = 1000

        while True:
            scroll_kwargs: dict = {
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

- `with_vectors=False` avoids transferring dense and sparse vectors.
  For a collection with 500 chunks of 768-float vectors, this saves
  ~1.5 MB of unnecessary network transfer per scroll page.
- `page_size=1000` is the Qdrant default maximum for scroll.
- The `offset` parameter is `None` for the first call (Qdrant starts
  from the beginning). After each call, Qdrant returns a
  `next_offset` value (a point ID). When `next_offset` is `None`, all
  points have been visited.
- The `offset` kwarg must be omitted entirely (not passed as `None`)
  on the first call. Some Qdrant client versions reject `offset=None`.
  The `if offset is not None` guard handles this.
- Deduplication uses a dict keyed by `document_id`. Each point adds
  to `chunk_count` and updates `pages` if its `page_number` is higher.
- The try/except returns an empty list if the collection does not exist
  (first-ever startup before any ingestion).

---

## 5. Highlight Extraction

### 5.1 Prompt format

The system prompt (section 3.3) instructs the LLM to emit:

```
HIGHLIGHT[1]: The procurement threshold for direct contracting is VT 500,000.
HIGHLIGHT[3]: Annual leave entitlement shall be 20 working days.
```

The `[N]` index is 1-based, matching the `[Source N]` labels in the
context block built by `build_context()`.

### 5.2 Regex pattern

```python
import re
HIGHLIGHT_RE = re.compile(r"HIGHLIGHT\[(\d+)\]:\s*(.+)")
```

- `HIGHLIGHT\[` — literal prefix and opening bracket
- `(\d+)` — captures the 1-based source index
- `\]:\s*` — closing bracket, colon, optional whitespace
- `(.+)` — captures the rest of the line as highlight text

Applied per-line using `HIGHLIGHT_RE.match(line.strip())` — anchored
to the start of each stripped line. This prevents false matches on
sentences that happen to contain the word "HIGHLIGHT" mid-text.

### 5.3 Stripping from answer text

Inside `answer()` in `chain.py`, after `response = llm.invoke(...)`:

```python
raw_answer = response.content
highlights: dict[int, str] = {}
clean_lines: list[str] = []

for line in raw_answer.split("\n"):
    m = HIGHLIGHT_RE.match(line.strip())
    if m:
        source_idx = int(m.group(1))
        highlights[source_idx] = m.group(2).strip()
    else:
        clean_lines.append(line)

clean_answer = "\n".join(clean_lines).strip()
```

Lines matching the regex are consumed into the `highlights` dict.
All other lines pass through to `clean_lines`, preserving paragraph
structure including blank lines.

### 5.4 Matching to source by index

```python
for i, chunk in enumerate(chunks, start=1):
    if i in highlights:
        chunk.highlight = highlights[i]
```

- The chunks list is enumerated 1-based (matching `[Source 1]`,
  `[Source 2]`, etc.).
- If the LLM emits `HIGHLIGHT[5]:` but only 3 sources exist, index 5
  is silently ignored — no `IndexError`.
- Sources without a matching marker keep `highlight = None`.

### 5.5 Updated return value

```python
return {
    "answer": clean_answer,
    "sources": chunks,
}
```

The `answer` field is the cleaned text (markers removed). The `sources`
list contains the same `SourceChunk` objects with `highlight` set where
applicable.

---

## 6. Confidence Thresholds

RRF scores are computed as `sum(1 / (K + rank))` with K=60 and two
result lists (dense + sparse).

| Scenario | Score | Confidence |
|----------|-------|------------|
| Rank 1 in both lists | 2 / 61 = 0.0328 | high (>= 0.025) |
| Rank 1 dense, rank 5 sparse | 1/61 + 1/65 = 0.0318 | high |
| Rank 3 in both lists | 2 / 63 = 0.0317 | high |
| Rank 5 in both lists | 2 / 65 = 0.0308 | high |
| Rank 10 in both lists | 2 / 70 = 0.0286 | high |
| Rank 1 dense only (no sparse match) | 1/61 = 0.0164 | medium (>= 0.015) |
| Rank 5 dense only | 1/65 = 0.0154 | medium |
| Rank 6 dense only | 1/66 = 0.0152 | medium |
| Rank 7 dense only | 1/67 = 0.0149 | low (< 0.015) |
| No results | 0 | low |

The thresholds `0.025` (high) and `0.015` (medium) cover the expected
RRF range well:
- **High** — document appeared in the top 10 of both dense and sparse,
  or near the top of at least one list.
- **Medium** — document appeared only in one list, at a middling rank.
- **Low** — very poor matches or no results at all.

**Dense-only fallback caveat:** When `hybrid_search` falls back to
dense-only (sparse index absent), raw cosine scores (0.0–1.0) are
returned. These always exceed 0.025, so fallback queries show "high"
confidence. This is acceptable — it only happens before re-ingestion.

---

## 7. Frontend Architecture — `frontend/app.py`

### 7.1 Session state management

```python
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None
```

- `conversation_history` — list of `{"role": str, "content": str}`
  dicts. Updated from the `conversation_history` field in each
  `QueryResponse`. Cleared by the "Clear conversation" button.
- `documents` — cached list from `GET /documents`. Refreshed on load,
  after ingest, and after delete.
- `last_response` — the most recent query response dict, used to
  render the answer panel without re-querying on rerun.

### 7.2 API calls

All API calls use `requests` (already in `Dockerfile.frontend`). The
base URL is `os.getenv("API_URL", "http://localhost:8000")`.

```python
def api_get(path):
    return requests.get(f"{API_URL}{path}", timeout=30)

def api_post_json(path, payload):
    return requests.post(f"{API_URL}{path}", json=payload, timeout=60)

def api_post_file(path, files):
    return requests.post(f"{API_URL}{path}", files=files, timeout=120)

def api_delete(path):
    return requests.delete(f"{API_URL}{path}", timeout=30)

def refresh_documents():
    try:
        resp = api_get("/documents")
        if resp.status_code == 200:
            st.session_state.documents = resp.json()["documents"]
    except requests.RequestException:
        st.session_state.documents = []
```

### 7.3 Error handling

- Every API call is wrapped in try/except `requests.RequestException`.
- On connection failure: `st.error("Cannot reach the API.")`.
- On HTTP error: `st.error(response.json().get("detail", "..."))`.
- Never crash the page — always degrade to an inline error message.

### 7.4 Layout structure

```python
st.set_page_config(
    page_title="Document Q&A — Vanuatu Gov",
    page_icon="📄",
    layout="wide",
)

with st.sidebar:
    # Document Library section
    # Conversation History section

# Main panel
# Query interface
# Answer display
```

### 7.5 Sidebar — Document Library

```python
st.header("📄 Document Library")
uploaded = st.file_uploader("Upload PDF", type=["pdf"])
if st.button("Ingest") and uploaded:
    with st.spinner("Ingesting..."):
        files = {"file": (uploaded.name, uploaded.read(), "application/pdf")}
        resp = api_post_file("/ingest", files=files)
    if resp.status_code == 201:
        d = resp.json()
        st.success(f"✓ {d['pages']} pages, {d['chunks']} chunks")
        refresh_documents()
    else:
        st.error(resp.json().get("detail", "Ingestion failed"))

st.divider()
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

### 7.6 Sidebar — Conversation History

```python
st.divider()
st.header("💬 Conversation")
for i in range(0, len(st.session_state.conversation_history), 2):
    turn = st.session_state.conversation_history[i]
    st.caption(f"Q: {turn['content'][:80]}...")
    if i + 1 < len(st.session_state.conversation_history):
        ans = st.session_state.conversation_history[i + 1]
        st.caption(f"A: {ans['content'][:100]}...")

if st.button("Clear conversation"):
    st.session_state.conversation_history = []
    st.session_state.last_response = None
    st.rerun()
```

### 7.7 Main panel — Query interface

```python
st.title("🔍 Ask a Question")
question = st.text_input("Question")
doc_names = ["All documents"] + [d["filename"] for d in st.session_state.documents]
scope = st.selectbox("Scope", doc_names)

if st.button("Ask") and question:
    payload = {
        "question": question,
        "conversation_history": st.session_state.conversation_history,
    }
    if scope != "All documents":
        doc = next(d for d in st.session_state.documents if d["filename"] == scope)
        payload["filters"] = {"document_id": doc["document_id"]}

    with st.spinner("Thinking..."):
        resp = api_post_json("/query", payload)

    if resp.status_code == 200:
        data = resp.json()
        st.session_state.conversation_history = data["conversation_history"]
        st.session_state.last_response = data
    else:
        st.error(resp.json().get("detail", "Query failed"))
```

### 7.8 Main panel — Answer display

```python
data = st.session_state.last_response
if data:
    st.markdown(data["answer"])

    conf = data.get("confidence", "low")
    if conf == "high":
        st.success("Confidence: ●●● High")
    elif conf == "medium":
        st.warning("Confidence: ●● Medium")
    else:
        st.error("Confidence: ● Low")

    if not data["found"]:
        st.info("No direct answer found. Try rephrasing.")
        return

    st.subheader("Sources")
    for src in data["sources"]:
        label = f"📄 {src['filename']}, page {src['page']} (score: {src['score']})"
        with st.expander(label):
            if src.get("highlight"):
                st.markdown(f"> **{src['highlight']}**")
            st.text(src["text"])
```

---

## 8. Mocking Strategy for Tests

### 8.1 Principles

Mock at the boundary (Qdrant, Ollama). Run real code for pure functions
(compute_confidence, highlight parsing). Frontend (AC-UI) tests are
manual.

### 8.2 Per-module mocking table

| Test file | Module under test | What is mocked | How |
|-----------|-------------------|----------------|-----|
| `test_retrieval.py` | `list_documents` | `QdrantClient` | `client.scroll.side_effect` returns paginated results then `([], None)`. Points have payloads with `document_id`, `filename`, `sha256`, `page_number` |
| `test_retrieval.py` | `find_document_by_id` | `QdrantClient` | `client.scroll.return_value = ([point], None)` for found; `([], None)` for not found |
| `test_context.py` | `compute_confidence` | Nothing | Pure function — pass `SourceChunk` objects with known scores |
| `test_query_chain.py` | highlight extraction | `_get_llm` | Mock LLM returns response with `HIGHLIGHT[1]: ...` lines. Assert highlights are extracted and answer text is clean |
| `test_api.py` | `GET /documents` | `list_documents` | `@patch("app.api.routes.documents.list_documents")` returning known dicts |
| `test_api.py` | `DELETE /documents/{id}` | `find_document_by_id`, `delete_by_document_id` | Patch at route module level. Return `True`/`False` for existence check |
| `test_api.py` | `confidence` in query | Existing query mocks | Existing mock chunks have `score=0.85`; `compute_confidence` returns `"high"`. Assert `confidence` appears in response |

### 8.3 Existing test compatibility

- `SourceChunk` gains `highlight: str | None = None` — default `None`
  means all existing `SourceChunk` constructors still work.
- `QueryResponse` gains `confidence: str = "low"` — default `"low"`
  means existing `QueryResponse` constructors still work.
- Existing Phase 2 query API tests that patch `answer()` return
  `{"answer": ..., "sources": ...}`. The route now also calls
  `compute_confidence(chunks_as_source)`. Mock chunk scores of 0.85
  yield `"high"`. No test break — tests don't currently assert on
  `confidence`.
- `test_query_found_false_when_not_relevant` — the route sets
  `confidence="low"` in the not-found path. No test break.

---

## 9. Acceptance Criteria Traceability

### Document listing (AC-LIST)

| AC | Test function | File |
|----|---------------|------|
| AC-LIST-01 | `test_list_documents_returns_200` | `tests/test_api.py` |
| AC-LIST-02 | `test_list_documents_includes_all_fields` | `tests/test_api.py` |
| AC-LIST-03 | `test_list_documents_empty_collection` | `tests/test_api.py` |
| AC-LIST-04 | `test_list_documents_multiple_documents` | `tests/test_api.py` |

### Document deletion (AC-DEL)

| AC | Test function | File |
|----|---------------|------|
| AC-DEL-01 | `test_delete_document_returns_200` | `tests/test_api.py` |
| AC-DEL-02 | `test_delete_document_removed_from_list` | `tests/test_api.py` |
| AC-DEL-03 | `test_delete_unknown_document_returns_404` | `tests/test_api.py` |
| AC-DEL-04 | `test_delete_then_query_returns_not_found` | `tests/test_api.py` |

### Answer quality (AC-QUAL)

| AC | Test function | File |
|----|---------------|------|
| AC-QUAL-01 | `test_query_response_includes_confidence` | `tests/test_api.py` |
| AC-QUAL-02 | `test_confidence_high_when_score_above_025` | `tests/test_context.py` |
| AC-QUAL-03 | `test_confidence_medium_when_score_015_to_024` | `tests/test_context.py` |
| AC-QUAL-04 | `test_confidence_low_when_score_below_015` | `tests/test_context.py` |
| AC-QUAL-05 | `test_source_chunk_has_highlight_field` | `tests/test_query_chain.py` |
| AC-QUAL-06 | `test_highlight_is_none_when_no_marker` | `tests/test_query_chain.py` |
| AC-QUAL-07 | `test_answer_text_has_no_highlight_markers` | `tests/test_query_chain.py` |

### Unit tests — vector store

| Test function | File |
|---------------|------|
| `test_list_documents_returns_metadata` | `tests/test_retrieval.py` |
| `test_list_documents_empty_collection` | `tests/test_retrieval.py` |
| `test_list_documents_deduplicates` | `tests/test_retrieval.py` |
| `test_list_documents_pagination` | `tests/test_retrieval.py` |
| `test_find_document_by_id_found` | `tests/test_retrieval.py` |
| `test_find_document_by_id_not_found` | `tests/test_retrieval.py` |

### Frontend (AC-UI) — manual tests

| AC | Verification |
|----|-------------|
| AC-UI-01 | Frontend loads at `http://localhost:8501` |
| AC-UI-02 | Upload PDF triggers ingestion, shows counts |
| AC-UI-03 | Document list updates after ingestion |
| AC-UI-04 | Delete button removes document from list |
| AC-UI-05 | Question displays answer with confidence badge |
| AC-UI-06 | Sources show filename, page, highlight |
| AC-UI-07 | Conversation history in sidebar |
| AC-UI-08 | Clear conversation resets history |

---

## 10. Definition of Done Checklist

- [ ] **All new files created** — `app/api/routes/documents.py`
- [ ] **All 81 existing tests pass** — `pytest tests/ -v`
- [ ] **All Phase 3 automated tests pass** — new tests covering
  AC-LIST, AC-DEL, AC-QUAL, and vector store unit tests
- [ ] **Frontend loads** — `http://localhost:8501` shows the full
  application (AC-UI-01)
- [ ] **Frontend E2E smoke test:**
  1. Upload a PDF — sidebar shows document with page/chunk counts
  2. Ask a question — answer with confidence badge and sources
  3. Sources show highlights (when LLM returns them)
  4. Conversation history in sidebar
  5. Delete document — removed from sidebar list
  6. Clear conversation — history reset
- [ ] **API smoke test:**
  ```bash
  # List documents
  curl -s http://localhost:8001/documents | python -m json.tool

  # Delete a document
  curl -X DELETE http://localhost:8001/documents/{id}

  # Query with confidence
  curl -s -X POST http://localhost:8001/query \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the budget?"}' | python -m json.tool
  # → includes confidence, sources with highlights

  # Delete unknown
  curl -s -X DELETE http://localhost:8001/documents/nonexistent
  # → 404
  ```
- [ ] **CI green** — GitHub Actions passes
- [ ] **Lint clean** — `ruff check app/ tests/ frontend/`
- [ ] **No regressions** — all 81 Phase 1+2 tests pass unchanged
