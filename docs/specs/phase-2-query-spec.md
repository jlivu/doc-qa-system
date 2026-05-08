# Phase 2 — Query Pipeline Specification

**Status:** Approved  
**Version:** 1.0  
**Date:** 2026-05-08  
**Author:** Joe Livu  

---

## 1. Overview

Phase 2 implements the query pipeline — the process by which a user submits
a natural language question, the system retrieves the most relevant document
chunks from Qdrant using hybrid search, passes them to a local LLM
(qwen2.5:7b via Ollama), and returns a grounded answer with source citations.

The pipeline supports multi-turn conversation — the user can ask follow-up
questions and the system maintains context across turns within a session.

The pipeline is exposed as a single HTTP endpoint: `POST /query`. It is
synchronous — the client waits for the full response before receiving a reply.

### Pipeline stages

```
HTTP request (question + optional filters + conversation history)
    │
    ▼
[1] Validate         — question length, filter schema, history format
    │
    ▼
[2] Embed query      — convert question to vector using nomic-embed-text
    │
    ▼
[3] Hybrid search    — vector similarity + BM25 keyword search in Qdrant
    │
    ▼
[4] Filter & rank    — apply metadata filters, merge and rank results
    │
    ▼
[5] Build context    — format retrieved chunks into prompt context block
    │
    ▼
[6] Generate answer  — call qwen2.5:7b via Ollama with context + history
    │
    ▼
[7] Handle not-found — if no relevant chunks found, suggest related topics
    │
    ▼
[8] Respond          — return answer + sources + updated conversation history
```

---

## 2. Scope

### In scope

- Natural language question answering grounded in ingested documents
- Hybrid search combining dense vector similarity and sparse BM25 keyword search
- Multi-turn conversation with history maintained by the client
- Flexible search scoping — all documents, by document_id, or by filename
- Graceful not-found handling with related topic suggestions
- Source citations — full chunk text, filename, page number, relevance score
- Structured JSON response

### Out of scope (future phases)

- Streaming responses (token-by-token output)
- Server-side conversation session management
- Re-ranking with a cross-encoder model
- Query rewriting or decomposition
- Document summarisation endpoint
- User authentication and authorisation
- Answer feedback / rating endpoint

---

## 3. Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Retrieval | Hybrid search (vector + BM25) | Government documents use precise terminology where exact keyword matching matters alongside semantic similarity |
| Conversation | Client-managed history | Simpler server — client sends full history each request; server is stateless |
| Not found | Suggest related topics | Honest and useful — avoids hallucination risk of falling back to LLM general knowledge |
| Scoping | Flexible filters | document_id, filename, or no filter (all documents) |
| Sources | Full chunk + metadata + score | Maximum transparency for government document use case |
| LLM | qwen2.5:7b via Ollama | Local inference, no API key, no cost, adequate quality for document Q&A |
| Embedding | nomic-embed-text via Ollama | Consistent with Phase 1 ingestion — same model for query and storage |

---

## 4. API contract

### Endpoint

```
POST /query
Content-Type: application/json
```

### Request

```json
{
  "question": "What is the procurement threshold for direct contracting?",
  "filters": {
    "document_id": "9ceca0eb-5af0-4001-b245-6f9f6a86e13c",
    "filename": null
  },
  "top_k": 5,
  "conversation_history": [
    {
      "role": "user",
      "content": "What does this regulation cover?"
    },
    {
      "role": "assistant",
      "content": "This regulation covers the rules and procedures for..."
    }
  ]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `question` | string | Yes | Natural language question (3–1000 chars) |
| `filters` | object | No | Optional scoping filters |
| `filters.document_id` | string (UUID) | No | Restrict search to a specific document |
| `filters.filename` | string | No | Restrict search to documents with this filename |
| `top_k` | integer (1–20) | No | Number of chunks to retrieve. Defaults to `settings.retrieval_top_k` |
| `conversation_history` | array | No | Prior turns in this conversation. Empty list or omitted for first question |

**`conversation_history` entry:**

| Field | Type | Values |
|---|---|---|
| `role` | string | `"user"` or `"assistant"` |
| `content` | string | Message text |

### Success response — HTTP 200

```json
{
  "answer": "The procurement threshold for direct contracting is VT 500,000...",
  "sources": [
    {
      "document_id": "9ceca0eb-5af0-4001-b245-6f9f6a86e13c",
      "filename": "GOV_Contracts_and_Tenders_Regulation_2021.pdf",
      "page": 7,
      "text": "Direct contracting may be used where the estimated value...",
      "score": 0.8923
    }
  ],
  "found": true,
  "conversation_history": [
    {
      "role": "user",
      "content": "What is the procurement threshold for direct contracting?"
    },
    {
      "role": "assistant",
      "content": "The procurement threshold for direct contracting is VT 500,000..."
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `answer` | string | LLM-generated answer grounded in retrieved chunks |
| `sources` | array | Chunks used to generate the answer, ordered by score descending |
| `found` | boolean | `true` if relevant chunks were found; `false` if not |
| `conversation_history` | array | Updated history including this turn — client stores and sends back next request |

### Not-found response — HTTP 200

When no relevant chunks are found (all scores below threshold), `found` is
`false` and `answer` describes what related topics were found instead:

```json
{
  "answer": "I could not find a direct answer to your question in the available documents. The documents do contain information about: procurement procedures, tender evaluation criteria, and contract award processes. Try rephrasing your question around one of these topics.",
  "sources": [],
  "found": false,
  "conversation_history": [...]
}
```

### Error responses

| HTTP status | Code | Condition |
|---|---|---|
| 422 Unprocessable Entity | `INVALID_QUESTION` | Question is fewer than 3 or more than 1000 characters |
| 422 Unprocessable Entity | `INVALID_FILTERS` | Both `document_id` and `filename` provided simultaneously |
| 422 Unprocessable Entity | `INVALID_HISTORY` | History entry has invalid role or missing content |
| 500 Internal Server Error | `EMBEDDING_ERROR` | Ollama embedding call failed |
| 500 Internal Server Error | `RETRIEVAL_ERROR` | Qdrant search failed |
| 500 Internal Server Error | `GENERATION_ERROR` | Ollama LLM call failed |

All error responses follow this structure:

```json
{
  "error": "INVALID_QUESTION",
  "detail": "Question must be between 3 and 1000 characters"
}
```

---

## 5. Module specifications

### 5.1 Query validator

**Location:** `app/query/validator.py` (new file)

**Purpose:** Validate the incoming query request before any expensive
operations begin.

**Interface:**

```python
def validate_query_request(
    question: str,
    filters: QueryFilters | None,
    history: list[ConversationTurn] | None,
) -> None:
```

**Behaviour:**

- Raises `InvalidQuestionError` if `len(question.strip()) < 3`
- Raises `InvalidQuestionError` if `len(question) > 1000`
- Raises `InvalidFiltersError` if both `filters.document_id` and
  `filters.filename` are provided — only one scope filter is allowed
  per query
- Raises `InvalidHistoryError` if any history entry has a `role` value
  other than `"user"` or `"assistant"`
- Raises `InvalidHistoryError` if any history entry has an empty `content`
- Returns `None` on success

---

### 5.2 Query embedder

**Location:** `app/retrieval/retriever.py` (modify existing)

**Purpose:** Embed the user's question using the same model used during
ingestion (`nomic-embed-text`). Produces the dense vector used for the
vector component of hybrid search.

**Interface:**

```python
def embed_query(question: str, settings: Settings) -> list[float]:
```

**Behaviour:**

- Calls Ollama embeddings API at `settings.ollama_base_url`
- Uses `settings.ollama_embedding_model`
- Returns a list of 768 floats
- Raises `EmbeddingError` on failure — do not retry (Ollama is local)

---

### 5.3 Hybrid search

**Location:** `app/retrieval/vector_store.py` (modify existing)

**Purpose:** Perform hybrid search combining dense vector similarity search
with sparse BM25 keyword search. Qdrant's native sparse vector support is
used for BM25.

**Interface:**

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

**Behaviour:**

- Performs two searches in parallel:
  - Dense search: cosine similarity on the `nomic-embed-text` vectors
  - Sparse search: BM25 keyword matching on chunk text
- Merges results using Reciprocal Rank Fusion (RRF) — a standard
  score-fusion technique that does not require tuning weight parameters
- Applies metadata filters (`document_id` or `filename`) to both searches
- Returns top-k results ordered by merged RRF score descending
- Each `SearchResult` carries the RRF-merged score as its `score` field
- Falls back to dense-only search if sparse index does not exist

**RRF formula:**

```
rrf_score(d) = Σ 1 / (k + rank(d))
```

Where `k = 60` (standard constant) and `rank(d)` is the 1-based position
of document `d` in each result list.

---

### 5.4 Context builder

**Location:** `app/qa/context.py` (new file)

**Purpose:** Format retrieved chunks into a structured context block for
the LLM prompt. Also detects the not-found condition and generates the
related-topics suggestion.

**Interface:**

```python
def build_context(chunks: list[SourceChunk]) -> str:
def is_not_found(chunks: list[SourceChunk], threshold: float = 0.3) -> bool:
def build_not_found_answer(chunks: list[SourceChunk]) -> str:
```

**Behaviour of `build_context`:**

- Formats each chunk as:
  ```
  [Source N] {filename}, page {page} (score: {score})
  {text}
  ```
- Separates chunks with `\n\n---\n\n`
- Returns the full formatted string

**Behaviour of `is_not_found`:**

- Returns `True` if `chunks` is empty
- Returns `True` if the highest score across all chunks is below `threshold`
- Returns `False` otherwise
- Default threshold of 0.3 reflects RRF scores which are lower than raw
  cosine scores

**Behaviour of `build_not_found_answer`:**

- Called only when `is_not_found` returns `True`
- If `chunks` is empty: returns a generic not-found message
- If chunks exist but scores are below threshold: extracts the filenames
  from the chunks and generates a message listing the topics those documents
  cover, inviting the user to rephrase

---

### 5.5 RAG chain

**Location:** `app/qa/chain.py` (modify existing)

**Purpose:** Build the LangChain chain that passes context and conversation
history to the LLM and returns a grounded answer.

**Interface:**

```python
def answer(
    question: str,
    chunks: list[SourceChunk],
    history: list[ConversationTurn],
    settings: Settings,
) -> dict:
```

**Behaviour:**

- Builds the system prompt using `RAG_SYSTEM_PROMPT` with context injected
- Constructs the message list:
  - System message with context
  - All prior conversation turns from `history` as alternating
    `HumanMessage` / `AIMessage` pairs
  - Current question as a final `HumanMessage`
- Calls `ChatOllama` with the full message list
- Returns `{"answer": str, "sources": list[SourceChunk]}`
- Raises `GenerationError` on LLM failure

**Conversation history handling:**

History entries with `role == "user"` become `HumanMessage`.
History entries with `role == "assistant"` become `AIMessage`.
History is injected between the system message and the current question,
preserving the full conversational context for the LLM.

---

### 5.6 Query route

**Location:** `app/api/routes/query.py` (rewrite existing skeleton)

**Purpose:** Orchestrate the full query pipeline. Validate → embed →
hybrid search → check not-found → build context → generate → respond.

**Behaviour:**

```
1. validate_query_request(question, filters, history)
2. query_vector = embed_query(question, settings)
3. chunks = hybrid_search(query_vector, question, qdrant, settings, top_k, ...)
4. if is_not_found(chunks):
       answer_text = build_not_found_answer(chunks)
       return QueryResponse(answer=answer_text, sources=[], found=False, ...)
5. result = answer(question, chunks, history, settings)
6. updated_history = history + [user turn] + [assistant turn]
7. return QueryResponse(answer=result["answer"], sources=result["sources"],
                        found=True, conversation_history=updated_history)
```

**Error handling — exception-to-HTTP mapping:**

| Exception | HTTP status | Error code |
|---|---|---|
| `InvalidQuestionError` | 422 | `INVALID_QUESTION` |
| `InvalidFiltersError` | 422 | `INVALID_FILTERS` |
| `InvalidHistoryError` | 422 | `INVALID_HISTORY` |
| `EmbeddingError` | 500 | `EMBEDDING_ERROR` |
| `RetrievalError` | 500 | `RETRIEVAL_ERROR` |
| `GenerationError` | 500 | `GENERATION_ERROR` |

---

## 6. New exceptions

Add to `app/ingestion/exceptions.py`:

```python
class InvalidQuestionError(IngestionError): pass
class InvalidFiltersError(IngestionError): pass
class InvalidHistoryError(IngestionError): pass
class RetrievalError(IngestionError): pass
class GenerationError(IngestionError): pass
```

Note: `EmbeddingError` already exists from Phase 1.

---

## 7. Schema updates

### New types in `app/api/schemas.py`

```python
class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)

class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    filters: QueryFilters | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    found: bool
    conversation_history: list[ConversationTurn]
```

The existing `QueryRequest` and `QueryResponse` in schemas.py are replaced
entirely with these definitions.

---

## 8. Sparse vector index

For hybrid search, Qdrant requires a sparse vector index on the collection.
The sparse vectors are computed from chunk text using BM25 tokenisation.

**Implementation approach:**

Rather than using a separate sparse encoder library, use Qdrant's built-in
`models.SparseVector` with a simple TF-IDF-style tokenisation:

```python
from qdrant_client.models import SparseVector

def text_to_sparse_vector(text: str) -> SparseVector:
    """Convert text to a sparse vector using term frequency."""
```

This avoids adding a heavy dependency (e.g. `fastembed`) for Phase 2.
Phase 4 can upgrade to a proper sparse encoder if retrieval quality
needs improvement.

The collection must be updated to support sparse vectors alongside the
existing dense vectors. `_ensure_collection` in `vector_store.py` is
updated to create both vector configurations.

**Important:** Existing ingested documents do not have sparse vectors.
After the Phase 2 code is deployed, documents must be re-ingested to
populate the sparse index. The smoke test should re-ingest the sample
document before testing the query endpoint.

---

## 9. Data flow

```
QueryRequest (question, filters, top_k, history)
    │
validate_query_request()
    │
embed_query(question) → query_vector [768 floats]
    │
hybrid_search(query_vector, question, filters, top_k)
    │
    ├── is_not_found(chunks) == True
    │       │
    │       └── build_not_found_answer(chunks)
    │               │
    │               └── QueryResponse(found=False, sources=[])
    │
    └── is_not_found(chunks) == False
            │
            answer(question, chunks, history)
                │
                └── QueryResponse(found=True, sources=chunks,
                                  conversation_history=updated)
```

---

## 10. Acceptance criteria

### Validation (AC-QVAL)

- **AC-QVAL-01** — A question shorter than 3 characters returns HTTP 422 with `INVALID_QUESTION`
- **AC-QVAL-02** — A question longer than 1000 characters returns HTTP 422 with `INVALID_QUESTION`
- **AC-QVAL-03** — Providing both `document_id` and `filename` filters returns HTTP 422 with `INVALID_FILTERS`
- **AC-QVAL-04** — A history entry with an invalid role returns HTTP 422 with `INVALID_HISTORY`
- **AC-QVAL-05** — A valid request with no filters and no history passes validation

### Embedding (AC-QEMB)

- **AC-QEMB-01** — `embed_query` returns a list of 768 floats
- **AC-QEMB-02** — `embed_query` raises `EmbeddingError` on Ollama failure

### Hybrid search (AC-HSEARCH)

- **AC-HSEARCH-01** — Results are returned ordered by RRF score descending
- **AC-HSEARCH-02** — `document_id` filter restricts results to that document only
- **AC-HSEARCH-03** — `filename` filter restricts results to matching documents only
- **AC-HSEARCH-04** — No filter returns results from all documents
- **AC-HSEARCH-05** — An empty collection returns an empty list without raising
- **AC-HSEARCH-06** — Falls back to dense-only search if sparse index absent

### Context builder (AC-CTX)

- **AC-CTX-01** — `build_context` formats each chunk with source label, filename, page, score
- **AC-CTX-02** — `is_not_found` returns `True` for an empty chunk list
- **AC-CTX-03** — `is_not_found` returns `True` when all scores are below threshold
- **AC-CTX-04** — `is_not_found` returns `False` when at least one score meets threshold
- **AC-CTX-05** — `build_not_found_answer` returns a string mentioning related topics when low-score chunks exist
- **AC-CTX-06** — `build_not_found_answer` returns a generic message when chunk list is empty

### RAG chain (AC-CHAIN)

- **AC-CHAIN-01** — Answer is a non-empty string
- **AC-CHAIN-02** — Returned sources match the input chunks
- **AC-CHAIN-03** — Conversation history is included in the LLM message list
- **AC-CHAIN-04** — `GenerationError` is raised on LLM failure

### End-to-end route (AC-ROUTE)

- **AC-ROUTE-01** — A valid question against an ingested document returns HTTP 200
- **AC-ROUTE-02** — Response contains `answer`, `sources`, `found`, `conversation_history`
- **AC-ROUTE-03** — `found` is `true` when relevant chunks are retrieved
- **AC-ROUTE-04** — `found` is `false` and `sources` is empty when nothing relevant is found
- **AC-ROUTE-05** — Sources include `document_id`, `filename`, `page`, `text`, `score`
- **AC-ROUTE-06** — `conversation_history` in response includes the current turn appended
- **AC-ROUTE-07** — A follow-up question with prior history returns a contextually aware answer
- **AC-ROUTE-08** — `document_id` filter returns only sources from that document
- **AC-ROUTE-09** — An invalid question returns HTTP 422

---

## 11. Notes for the plan

- `app/query/` is a new directory — `validator.py` lives here alongside
  a new `__init__.py`
- `app/qa/context.py` is a new file
- `ConversationTurn` schema must use `Literal["user", "assistant"]` — 
  requires `from typing import Literal`
- The sparse vector index requires updating `_ensure_collection` in
  `vector_store.py` — the existing collection must be deleted and
  re-created, or re-ingestion handles the update
- Re-ingestion of the sample document is required after Phase 2 is
  deployed to populate sparse vectors
- `app/api/routes/query.py` currently raises HTTP 501 — it will be
  fully rewritten in Phase 2
- The existing `QueryRequest` and `QueryResponse` in `schemas.py` will
  be replaced — existing `test_api.py` query tests must be updated
  accordingly
- `embed_query` in `retriever.py` already exists from Phase 1 — verify
  it uses the Ollama base URL correctly before the plan is written
