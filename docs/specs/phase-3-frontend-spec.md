# Phase 3 — Frontend & Quality Improvements Specification

**Status:** Approved  
**Version:** 1.0  
**Date:** 2026-05-08  
**Author:** Joe Livu  

---

## 1. Overview

Phase 3 delivers three things:

1. **Two new API endpoints** — document listing (`GET /documents`) and
   document deletion (`DELETE /documents/{document_id}`) — needed to
   support the frontend document management features.

2. **Answer quality improvements** — an improved system prompt for OCR
   text, a confidence indicator derived from source scores, and
   highlighting of the specific text within each source chunk that most
   directly answers the question.

3. **A professional Streamlit frontend** — a polished, demo-ready UI
   with PDF upload, query interface with conversation history sidebar,
   document list with delete, and source citation display.

---

## 2. Scope

### In scope

- `GET /documents` — list all ingested documents with metadata
- `DELETE /documents/{document_id}` — remove a document and all its chunks
- Improved RAG system prompt tuned for OCR-extracted government text
- Confidence level (`high` / `medium` / `low`) derived from top source score
- Relevant text highlighting — the most relevant sentence or phrase within
  each source chunk, extracted by the LLM
- Full Streamlit frontend rewrite — professional layout, sidebar, document
  management, conversation history

### Out of scope (future phases)

- User authentication
- Multi-user document namespacing
- Streaming responses
- PDF viewer (embedded in-page)
- Export conversation to PDF or Word
- Answer feedback / rating

---

## 3. Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Document list API | `GET /documents` returns deduplicated docs from Qdrant payload scroll | No separate document store needed — Qdrant is the source of truth |
| Document delete API | `DELETE /documents/{document_id}` calls existing `delete_by_document_id` | Already implemented in Phase 1 vector store |
| Confidence indicator | Derived from top RRF score — thresholds tuned to RRF range | No additional LLM call needed |
| Relevant text highlight | LLM extracts the most relevant sentence per chunk in the same call | Avoids a second LLM call; adds minimal prompt overhead |
| Frontend framework | Streamlit — already in requirements, consistent with Phase 1 scaffold | No new dependency |
| Frontend state | Streamlit `st.session_state` for conversation history and document list | Stateless server — no backend session management needed |

---

## 4. New API endpoints

### 4.1 `GET /documents`

Returns a deduplicated list of all documents currently ingested in Qdrant.
Deduplication is by `document_id` — each document appears once regardless
of how many chunks it has.

**Endpoint:**

```
GET /documents
```

**Success response — HTTP 200:**

```json
{
  "documents": [
    {
      "document_id": "dbfd1a95-bef5-4925-852d-3d2c9f336a2e",
      "filename": "GOV_Contracts_and_Tenders_Regulation_2021.pdf",
      "sha256": "0b453dad9311bb790a5ce43bc539b555d995d053f00562be43d8cb05fca9d682",
      "chunk_count": 90,
      "pages": 23
    }
  ],
  "total": 1
}
```

| Field | Type | Description |
|---|---|---|
| `documents` | array | List of ingested documents |
| `document_id` | string (UUID) | Unique document identifier |
| `filename` | string | Original filename |
| `sha256` | string | SHA-256 digest of the file |
| `chunk_count` | integer | Number of chunks stored for this document |
| `pages` | integer | Page count — derived from max `page_number` across chunks |
| `total` | integer | Total number of distinct documents |

**Empty collection — HTTP 200:**

```json
{
  "documents": [],
  "total": 0
}
```

**Error responses:**

| HTTP status | Code | Condition |
|---|---|---|
| 500 Internal Server Error | `RETRIEVAL_ERROR` | Qdrant scroll failed |

---

### 4.2 `DELETE /documents/{document_id}`

Removes all chunks belonging to the specified `document_id` from Qdrant.

**Endpoint:**

```
DELETE /documents/{document_id}
```

**Success response — HTTP 200:**

```json
{
  "document_id": "dbfd1a95-bef5-4925-852d-3d2c9f336a2e",
  "message": "Document deleted successfully"
}
```

**Error responses:**

| HTTP status | Code | Condition |
|---|---|---|
| 404 Not Found | `DOCUMENT_NOT_FOUND` | No chunks exist for this `document_id` |
| 500 Internal Server Error | `STORAGE_ERROR` | Qdrant delete failed |

The route must verify the document exists before deleting — call
`find_document_by_id()` first and return 404 if not found.

---

## 5. Module specifications

### 5.1 Document store — new functions in `app/retrieval/vector_store.py`

**`list_documents(client, settings) -> list[DocumentMetadata]`**

Scrolls through all Qdrant points and builds a deduplicated document list.

```python
class DocumentMetadata(TypedDict):
    document_id: str
    filename: str
    sha256: str
    chunk_count: int
    pages: int
```

Implementation:
- Use `client.scroll()` with `limit=1000` and paginate using the `offset`
  parameter until no more points are returned
- Group points by `document_id`
- For each group: `chunk_count = len(points)`,
  `pages = max(p.payload["page_number"] for p in points)`
- Return one `DocumentMetadata` per unique `document_id`
- Return empty list if collection does not exist — do not raise

**`find_document_by_id(document_id, client, settings) -> bool`**

Returns `True` if at least one chunk with this `document_id` exists in
Qdrant, `False` otherwise. Used by the delete route to check existence
before deletion.

---

### 5.2 New routes — `app/api/routes/documents.py` (new file)

Houses both new endpoints. Router prefix: `/documents`.

**`GET /documents`** calls `list_documents()` and returns `DocumentListResponse`.

**`DELETE /documents/{document_id}`**:
1. Calls `find_document_by_id()` — returns 404 if `False`
2. Calls `delete_by_document_id()` — returns 500 on `StorageError`
3. Returns `DeleteDocumentResponse`

Register this router in `app/main.py`.

---

### 5.3 New schemas in `app/api/schemas.py`

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

---

### 5.4 Answer quality — improved system prompt in `app/qa/prompts.py`

The current prompt does not account for OCR artefacts. Replace
`RAG_SYSTEM_PROMPT` with a version that:

- Explicitly instructs the LLM to handle OCR noise (broken words,
  missing spaces, garbled characters)
- Instructs the LLM to reconstruct fragmented text before answering
- Asks the LLM to extract the single most relevant sentence or phrase
  from each source chunk that directly supports the answer
- Maintains the grounding rule — answer only from context

**New prompt structure:**

```
You are a precise assistant answering questions about government 
documents. The documents were scanned and OCR-processed, so the 
text may contain: broken words (e.g. "Govern[/1ENT"), missing 
spaces, garbled characters, or split lines. Reconstruct the 
intended text using context before answering.

Rules:
1. Answer ONLY using the context provided. Do not use prior knowledge.
2. If the answer is not in the context, say exactly:
   "I could not find an answer to that question in the provided documents."
3. For each source you use, identify the single most relevant sentence 
   or short phrase (under 30 words) that most directly answers the 
   question. Return it verbatim in your response as:
   HIGHLIGHT[N]: <sentence or phrase>
   where N matches the source number.
4. Cite sources by their [Source N] label.
5. Be concise — prefer a direct answer over a long explanation.

Context:
{context}
```

The `HIGHLIGHT[N]:` markers are parsed from the LLM response by the
chain module and returned as a `highlight` field on each `SourceChunk`.

---

### 5.5 Answer quality — confidence indicator in `app/qa/context.py`

Add a `compute_confidence(chunks) -> str` function:

```python
def compute_confidence(chunks: list[SourceChunk]) -> str:
    """Derive a confidence level from the top source RRF score.
    
    RRF scores range from ~0.008 (rank 60+) to ~0.033 (rank 1).
    Thresholds are calibrated to this range.
    """
```

| Top score | Confidence |
|---|---|
| ≥ 0.025 | `"high"` |
| ≥ 0.015 | `"medium"` |
| < 0.015 | `"low"` |

Add `confidence: str` to `QueryResponse` schema.

---

### 5.6 Answer quality — highlight extraction in `app/qa/chain.py`

After the LLM returns its response, parse `HIGHLIGHT[N]:` markers
from the response text using a regex:

```python
import re
HIGHLIGHT_RE = re.compile(r"HIGHLIGHT\[(\d+)\]:\s*(.+)")
```

- Strip the `HIGHLIGHT[N]:` lines from the visible answer text
- Match each highlight to its source by index (1-based)
- Add `highlight: str | None` field to `SourceChunk` schema
- If no highlight is found for a source, set `highlight = None`

---

### 5.7 Schema update — `SourceChunk`

Add two new optional fields:

```python
class SourceChunk(BaseModel):
    document_id: str
    filename: str
    page: int
    text: str
    score: float
    highlight: str | None = None    # most relevant sentence from this chunk
```

Add `confidence` to `QueryResponse`:

```python
class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    found: bool
    confidence: str                  # "high", "medium", or "low"
    conversation_history: list[ConversationTurn]
```

---

### 5.8 Frontend — `frontend/app.py` (full rewrite)

A professional, demo-ready Streamlit application. Full layout:

```
┌─────────────────────────────────────────────────────────────┐
│  SIDEBAR                    │  MAIN PANEL                   │
│                             │                               │
│  📄 Document Library        │  🔍 Ask a Question            │
│  ─────────────────          │  ───────────────────          │
│  [Upload PDF button]        │  [Question input]             │
│                             │  [Scope dropdown]             │
│  doc1.pdf  23p  90c  [🗑]   │  [Ask button]                 │
│  doc2.pdf  12p  44c  [🗑]   │                               │
│                             │  ┌─────────────────────────┐  │
│  ─────────────────          │  │ Answer                  │  │
│  💬 Conversation            │  │ ─────                   │  │
│  ─────────────────          │  │ [answer text]           │  │
│  Q: What is...              │  │                         │  │
│  A: The regulation...       │  │ Confidence: ●●● High    │  │
│  Q: How does...             │  │                         │  │
│  A: According to...         │  │ Sources                 │  │
│                             │  │ ─────                   │  │
│  [Clear conversation]       │  │ ▶ doc.pdf, page 4       │  │
│                             │  │   "highlighted text"    │  │
│                             │  │   [full chunk text]     │  │
└─────────────────────────────┴─────────────────────────────┘
```

**Sidebar — Document Library:**
- `st.file_uploader` for PDF upload with an Ingest button
- Progress spinner during ingestion showing "Parsing… Embedding… Storing…"
- On success: show chunk count and page count
- Document list fetched from `GET /documents` on load and after each
  ingest or delete
- Each document shown as: filename, page count, chunk count, delete button
- Delete confirmation dialog before calling `DELETE /documents/{id}`
- Error messages inline if ingest or delete fails

**Sidebar — Conversation History:**
- Scrollable list of prior turns in this session
- Each turn shows the question and a truncated answer (first 100 chars)
- "Clear conversation" button resets `st.session_state`

**Main panel — Query interface:**
- Text input for the question
- Optional scope dropdown: "All documents" or specific document by filename
- "Ask" button — disabled while a request is in flight
- Spinner while awaiting response

**Main panel — Answer display:**
- Answer text rendered as markdown
- Confidence badge: green dot(s) for high, amber for medium, red for low
- Expandable source citations — each shows filename, page, highlight text
  in a coloured callout box, and the full chunk text in a collapsed expander
- "not found" state shown with an info banner and suggested topics

**Styling guidelines:**
- Use Streamlit's native theming — no custom CSS injection
- Use `st.columns`, `st.expander`, `st.badge`, `st.info`, `st.success`,
  `st.error` for structure
- Consistent spacing using `st.divider()` and `st.empty()`
- Page config: wide layout, custom title "Document Q&A — Vanuatu Gov",
  page icon 📄

---

## 6. Acceptance criteria

### Document listing (AC-LIST)

- **AC-LIST-01** — `GET /documents` returns HTTP 200 with a list of documents
- **AC-LIST-02** — Each document entry includes `document_id`, `filename`,
  `sha256`, `chunk_count`, `pages`
- **AC-LIST-03** — `GET /documents` returns `total: 0` and empty list when
  no documents are ingested
- **AC-LIST-04** — After ingesting two documents, `GET /documents` returns
  `total: 2`

### Document deletion (AC-DEL)

- **AC-DEL-01** — `DELETE /documents/{id}` returns HTTP 200 for a known document
- **AC-DEL-02** — After deletion, `GET /documents` no longer includes that document
- **AC-DEL-03** — `DELETE /documents/{id}` returns HTTP 404 for an unknown id
- **AC-DEL-04** — After deletion, querying with `document_id` filter returns
  `found: false`

### Answer quality (AC-QUAL)

- **AC-QUAL-01** — `QueryResponse` includes a `confidence` field with value
  `"high"`, `"medium"`, or `"low"`
- **AC-QUAL-02** — `confidence` is `"high"` when top source score ≥ 0.025
- **AC-QUAL-03** — `confidence` is `"medium"` when top score is 0.015–0.024
- **AC-QUAL-04** — `confidence` is `"low"` when top score < 0.015
- **AC-QUAL-05** — `SourceChunk` includes a `highlight` field
- **AC-QUAL-06** — `highlight` is `None` when the LLM returns no highlight
  marker for that source
- **AC-QUAL-07** — The visible answer text does not contain raw
  `HIGHLIGHT[N]:` markers

### Frontend (AC-UI)

- **AC-UI-01** — Frontend loads at `http://localhost:8501` without errors
- **AC-UI-02** — Uploading a PDF triggers ingestion and shows chunk count
- **AC-UI-03** — Document list updates after ingestion
- **AC-UI-04** — Deleting a document removes it from the list
- **AC-UI-05** — Asking a question displays an answer with confidence badge
- **AC-UI-06** — Sources are shown with filename, page, and highlight
- **AC-UI-07** — Conversation history sidebar shows prior turns
- **AC-UI-08** — Clear conversation resets history

---

## 7. New and modified files

| File | Action |
|---|---|
| `app/api/routes/documents.py` | Create |
| `app/api/schemas.py` | Add `DocumentMetadataResponse`, `DocumentListResponse`, `DeleteDocumentResponse`; add `highlight` to `SourceChunk`; add `confidence` to `QueryResponse` |
| `app/retrieval/vector_store.py` | Add `list_documents()`, `find_document_by_id()` |
| `app/qa/prompts.py` | Replace `RAG_SYSTEM_PROMPT` with OCR-aware version |
| `app/qa/chain.py` | Add highlight extraction, `confidence` computation |
| `app/qa/context.py` | Add `compute_confidence()` |
| `app/main.py` | Register documents router |
| `frontend/app.py` | Full rewrite |

---

## 8. Notes for the plan

- `GET /documents` uses `client.scroll()` with pagination — the plan must
  detail the pagination loop carefully. Qdrant scroll returns a tuple of
  `(points, next_offset)`. Loop until `next_offset` is `None`.
- `find_document_by_id()` should use `scroll(limit=1, filter=...)` —
  it only needs to know if one point exists, not retrieve all of them.
- The `HIGHLIGHT[N]:` parsing regex must handle the case where the LLM
  returns the marker mid-sentence or with extra whitespace.
- The confidence thresholds (0.025 / 0.015) are based on RRF scores
  for a `top_k=5` search. If `top_k` changes significantly, the thresholds
  may need retuning.
- The frontend must handle API unavailability gracefully — show an error
  banner if the API is unreachable rather than crashing.
- `st.session_state` is reset on page refresh — the conversation history
  is session-scoped only, not persistent.
- The `Dockerfile.frontend` may need updating if new Python packages are
  required for the frontend. Check before writing the plan.
- `DeleteDocumentResponse` is a new schema — `app/api/routes/documents.py`
  must import it before use.
- Registering the new router in `main.py` requires adding one
  `app.include_router(documents.router)` line — the plan should note
  the exact location.
