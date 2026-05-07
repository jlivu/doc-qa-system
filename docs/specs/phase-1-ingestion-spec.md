# Phase 1 — Ingestion Pipeline Specification

**Status:** Approved  
**Version:** 1.0  
**Date:** 2026-05-08  
**Author:** Joe Livu  

---

## 1. Overview

Phase 1 implements the ingestion pipeline — the process by which a PDF document
is uploaded, parsed, split into chunks, embedded, and stored in the Qdrant vector
database so it can later be searched.

The pipeline is exposed as a single HTTP endpoint: `POST /ingest`. It is
synchronous — the client waits for the full pipeline to complete before receiving
a response.

### Pipeline stages

```
HTTP upload
    │
    ▼
[1] Validate         — file type, file size, content integrity
    │
    ▼
[2] Deduplicate      — hash the file; replace existing document if already ingested
    │
    ▼
[3] Parse            — extract text from each page using PyMuPDF
    │
    ▼
[4] Chunk            — split page text into overlapping segments
    │
    ▼
[5] Embed            — convert each chunk to a dense vector via OpenAI
    │
    ▼
[6] Store            — upsert vectors and metadata into Qdrant
    │
    ▼
[7] Respond          — return document ID, page count, chunk count
```

---

## 2. Scope

### In scope

- Uploading and ingesting a single PDF file per request
- Text-native PDFs (digitally created, selectable text)
- Scanned PDFs (image-based pages requiring OCR)
- Mixed PDFs (some pages text-native, some scanned)
- Hash-based deduplication with automatic replacement
- File size enforcement
- Structured JSON response with ingestion metrics
- Storing chunk metadata (document ID, filename, page number, SHA-256 hash)

### Out of scope (future phases)

- Batch ingestion of multiple files in one request
- Non-PDF file types (DOCX, XLSX, TXT)
- Asynchronous / background processing
- User authentication and authorisation
- Per-user document namespacing
- Document deletion endpoint
- Ingestion status polling

---

## 3. Decisions

| Decision | Choice | Rationale |
|---|---|---|
| PDF types | Text-native, scanned, mixed | Government documents span all three types |
| OCR engine | PyMuPDF built-in (`get_text("blocks")`) with fallback to pytesseract | PyMuPDF handles most cases; pytesseract for image-only pages |
| Duplicate handling | Re-ingest and replace (SHA-256 hash) | Documents are revised regularly; stale chunks degrade retrieval |
| File size limit | 50MB | Covers all realistic government documents with headroom |
| Embedding model | OpenAI `text-embedding-3-small` | Fast, low cost, 1536 dimensions, configured via environment |
| Processing mode | Synchronous | Simpler implementation; adequate for Phase 1 document sizes |
| Chunk size | 800 characters | Balances context preservation with retrieval precision |
| Chunk overlap | 100 characters | Prevents context loss at chunk boundaries |

---

## 4. API contract

### Endpoint

```
POST /ingest
Content-Type: multipart/form-data
```

### Request

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | The PDF file to ingest |

### Success response — HTTP 201

```json
{
  "document_id": "3f7a2b1c-...",
  "filename": "vanuatu_budget_2024.pdf",
  "sha256": "e3b0c44298fc...",
  "pages": 42,
  "chunks": 187,
  "replaced": false,
  "message": "Document ingested successfully"
}
```

| Field | Type | Description |
|---|---|---|
| `document_id` | string (UUID) | Unique identifier assigned to this document |
| `filename` | string | Original filename from the upload |
| `sha256` | string | SHA-256 hex digest of the file content |
| `pages` | integer | Number of pages parsed (including blank) |
| `chunks` | integer | Number of chunks stored in Qdrant |
| `replaced` | boolean | True if a previous version was replaced |
| `message` | string | Human-readable status message |

### Error responses

| HTTP status | Code | Condition |
|---|---|---|
| 415 Unsupported Media Type | `INVALID_FILE_TYPE` | File is not `application/pdf` |
| 413 Content Too Large | `FILE_TOO_LARGE` | File exceeds 50MB |
| 422 Unprocessable Entity | `INVALID_PDF` | File has PDF MIME type but cannot be opened as a valid PDF |
| 422 Unprocessable Entity | `EMPTY_PDF` | PDF has zero pages or all pages are blank |
| 500 Internal Server Error | `EMBEDDING_ERROR` | OpenAI API call failed after retries |
| 500 Internal Server Error | `STORAGE_ERROR` | Qdrant upsert failed |

All error responses follow this structure:

```json
{
  "error": "INVALID_FILE_TYPE",
  "detail": "Expected application/pdf, received text/plain"
}
```

---

## 5. Module specifications

### 5.1 Validator

**Location:** `app/ingestion/validator.py`

**Purpose:** Enforce file type and size constraints before any processing begins.
Fast, cheap checks that fail early to avoid wasted compute.

**Interface:**

```python
def validate_upload(filename: str, content_type: str, size_bytes: int) -> None:
    ...
```

**Behaviour:**

- Raises `InvalidFileTypeError` if `content_type` is not `application/pdf`
- Raises `FileTooLargeError` if `size_bytes` exceeds `MAX_FILE_SIZE_BYTES` (50 × 1024 × 1024)
- Returns `None` on success — any failure raises immediately

**Constants:**

```python
MAX_FILE_SIZE_BYTES = 52_428_800  # 50MB
ALLOWED_CONTENT_TYPE = "application/pdf"
```

---

### 5.2 Hasher

**Location:** `app/ingestion/hasher.py`

**Purpose:** Compute a SHA-256 digest of the raw PDF bytes and check whether
that digest already exists in Qdrant. Used for deduplication.

**Interface:**

```python
def compute_sha256(data: bytes) -> str:
    ...

def find_existing_document(sha256: str, client: QdrantClient, settings: Settings) -> str | None:
    ...
```

**Behaviour of `compute_sha256`:**

- Accepts raw file bytes
- Returns a lowercase hex string of the SHA-256 digest
- Is deterministic — same bytes always produce the same digest

**Behaviour of `find_existing_document`:**

- Searches Qdrant for any point whose payload contains `sha256 == sha256`
- Returns the `document_id` string if found, `None` if not found
- Does not raise on an empty collection

---

### 5.3 Parser

**Location:** `app/ingestion/parser.py`

**Purpose:** Extract text from every page of a PDF. Handles text-native pages
directly via PyMuPDF. Falls back to pytesseract OCR for image-only pages.

**Interface:**

```python
def extract_text(pdf_bytes: bytes) -> list[PageDict]:
    ...
```

**`PageDict` type:**

```python
class PageDict(TypedDict):
    page_number: int    # 1-indexed
    text: str           # extracted plain text, stripped
    char_count: int     # len(text) — used to detect blank pages
    ocr_used: bool      # True if pytesseract was used for this page
```

**Behaviour:**

- Raises `InvalidPDFError` if `pdf_bytes` cannot be opened as a PDF
- Raises `EmptyPDFError` if the document has zero pages
- For each page, attempts `page.get_text("text")` first
- If the extracted text has fewer than 20 characters, treats the page as
  image-based and runs pytesseract OCR on a rasterised image of the page
- Always returns one `PageDict` per page, even for blank pages
- Does not raise if individual pages are blank — blank pages are returned
  with `char_count == 0` and filtered out by the chunker
- Closes the PyMuPDF document in a `finally` block

**OCR rasterisation settings:**

- DPI: 300 (balance of accuracy and speed)
- Colour mode: grayscale
- pytesseract language: `eng` (English)

---

### 5.4 Chunker

**Location:** `app/ingestion/chunker.py`

**Purpose:** Split page text into overlapping chunks suitable for embedding.
Adds document metadata to every chunk for storage and citation.

**Interface:**

```python
def chunk_pages(
    pages: list[PageDict],
    document_id: str,
    filename: str,
    sha256: str,
    settings: Settings,
) -> list[ChunkDict]:
    ...
```

**`ChunkDict` type:**

```python
class ChunkDict(TypedDict):
    chunk_id: str       # UUID4 — unique identifier for this chunk
    document_id: str    # UUID of the parent document
    filename: str       # original filename
    sha256: str         # SHA-256 of the parent document
    page_number: int    # source page (1-indexed)
    text: str           # chunk text
    char_count: int     # len(text)
```

**Behaviour:**

- Skips pages where `char_count < 20` (blank or near-blank)
- Uses `RecursiveCharacterTextSplitter` with separators
  `["\n\n", "\n", ". ", " ", ""]` in that priority order
- `chunk_size` and `chunk_overlap` read from `settings`
- Each chunk receives a freshly generated UUID4 as `chunk_id`
- Chunks with fewer than 10 characters after stripping are discarded
- Returns an empty list if all pages are blank — does not raise
- The same `document_id` and `sha256` are applied to every chunk

---

### 5.5 Embedder

**Location:** `app/ingestion/embedder.py`

**Purpose:** Convert chunk text into dense vector embeddings via the OpenAI
Embeddings API. Processes chunks in batches to respect API limits.

**Interface:**

```python
def embed_chunks(chunks: list[ChunkDict], settings: Settings) -> list[EmbeddedChunk]:
    ...
```

**`EmbeddedChunk` type:**

```python
class EmbeddedChunk(ChunkDict):
    vector: list[float]     # dense embedding — length determined by model
```

**Behaviour:**

- Processes chunks in batches of 100 (OpenAI limit is 2048; 100 is safe)
- Uses the model specified in `settings.embedding_model`
- Retries up to 3 times on transient API errors using exponential backoff
  (2s, 4s, 8s)
- Raises `EmbeddingError` if all retries are exhausted
- Preserves the original order of chunks in the output
- Returns a list of the same length as the input

---

### 5.6 Vector store — upsert

**Location:** `app/retrieval/vector_store.py`

**Purpose:** Write embedded chunks to Qdrant. Creates the collection if it does
not exist. Deletes existing chunks for a `document_id` before upserting when
replacing a document.

**Interface:**

```python
def upsert_chunks(
    chunks: list[EmbeddedChunk],
    client: QdrantClient,
    settings: Settings,
) -> int:
    ...

def delete_by_document_id(
    document_id: str,
    client: QdrantClient,
    settings: Settings,
) -> None:
    ...
```

**Behaviour of `upsert_chunks`:**

- Creates the Qdrant collection if it does not already exist
- Vector size is inferred from `len(chunks[0]["vector"])`
- Distance metric: Cosine
- Each point payload stores: `document_id`, `filename`, `sha256`,
  `page_number`, `text`
- Returns the number of points upserted
- Returns 0 without error if `chunks` is empty
- Raises `StorageError` on Qdrant client failure

**Behaviour of `delete_by_document_id`:**

- Deletes all Qdrant points whose payload `document_id` matches
- Does not raise if no matching points exist
- Used by the ingest route when replacing an existing document

---

### 5.7 Ingest route

**Location:** `app/api/routes/ingest.py`

**Purpose:** Orchestrate the full pipeline. Validate → hash → deduplicate →
parse → chunk → embed → store → respond.

**Behaviour:**

- Reads the uploaded file into memory as bytes
- Calls `validate_upload()` — raises HTTP 415 or 413 on failure
- Calls `compute_sha256()` on the raw bytes
- Calls `find_existing_document()` to check for a duplicate
  - If found: calls `delete_by_document_id()` to remove old chunks,
    sets `replaced = True`, reuses the existing `document_id`
  - If not found: generates a new UUID4 as `document_id`,
    sets `replaced = False`
- Calls `extract_text()` — raises HTTP 422 on `InvalidPDFError`
  or `EmptyPDFError`
- Calls `chunk_pages()`
- Calls `embed_chunks()` — raises HTTP 500 on `EmbeddingError`
- Calls `upsert_chunks()` — raises HTTP 500 on `StorageError`
- Returns HTTP 201 with `IngestResponse`

---

## 6. Custom exceptions

All custom exceptions live in `app/ingestion/exceptions.py`.

| Exception | Raised by | Meaning |
|---|---|---|
| `InvalidFileTypeError` | Validator | File is not a PDF |
| `FileTooLargeError` | Validator | File exceeds 50MB |
| `InvalidPDFError` | Parser | Bytes are not a valid PDF |
| `EmptyPDFError` | Parser | PDF has no pages or all pages are blank |
| `EmbeddingError` | Embedder | OpenAI API failed after retries |
| `StorageError` | Vector store | Qdrant upsert failed |

---

## 7. Data flow

```
validate_upload(filename, content_type, size_bytes)
    │
compute_sha256(pdf_bytes) → sha256
    │
find_existing_document(sha256) → document_id | None
    │
    ├── Found → delete_by_document_id(document_id)
    │            replaced = True
    │
    └── Not found → document_id = uuid4()
                    replaced = False
    │
extract_text(pdf_bytes) → list[PageDict]
    │
chunk_pages(pages, document_id, filename, sha256, settings) → list[ChunkDict]
    │
embed_chunks(chunks, settings) → list[EmbeddedChunk]
    │
upsert_chunks(embedded, qdrant, settings) → int
    │
IngestResponse(document_id, filename, sha256, pages, chunks, replaced)
```

---

## 8. Acceptance criteria

These criteria map directly to test cases. Every criterion must pass before
Phase 1 is considered complete.

### Validation (AC-VAL)

- **AC-VAL-01** — Uploading a non-PDF file returns HTTP 415 with error code `INVALID_FILE_TYPE`
- **AC-VAL-02** — Uploading a file larger than 50MB returns HTTP 413 with error code `FILE_TOO_LARGE`
- **AC-VAL-03** — A valid PDF under 50MB passes validation without error

### Hashing (AC-HASH)

- **AC-HASH-01** — `compute_sha256` returns a 64-character lowercase hex string
- **AC-HASH-02** — The same bytes always produce the same hash
- **AC-HASH-03** — Different bytes always produce different hashes
- **AC-HASH-04** — `find_existing_document` returns `None` for an unknown hash
- **AC-HASH-05** — `find_existing_document` returns the `document_id` for a known hash

### Parsing (AC-PARSE)

- **AC-PARSE-01** — A text-native PDF returns one `PageDict` per page
- **AC-PARSE-02** — Page numbers are 1-indexed
- **AC-PARSE-03** — `char_count` equals `len(text)` for every page
- **AC-PARSE-04** — Invalid bytes raise `InvalidPDFError`
- **AC-PARSE-05** — A page with fewer than 20 characters of native text triggers OCR
- **AC-PARSE-06** — `ocr_used` is `True` for OCR-processed pages, `False` otherwise
- **AC-PARSE-07** — Blank pages are returned with `char_count == 0`, not omitted

### Chunking (AC-CHUNK)

- **AC-CHUNK-01** — Chunks from a multi-page document carry correct `page_number`
- **AC-CHUNK-02** — Every chunk has a unique `chunk_id`
- **AC-CHUNK-03** — `document_id`, `filename`, and `sha256` are identical across all chunks from the same document
- **AC-CHUNK-04** — Pages with `char_count < 20` produce no chunks
- **AC-CHUNK-05** — No chunk has fewer than 10 characters
- **AC-CHUNK-06** — An all-blank document returns an empty list without raising

### Embedding (AC-EMBED)

- **AC-EMBED-01** — Output list length equals input list length
- **AC-EMBED-02** — Every `EmbeddedChunk` has a `vector` field that is a non-empty list of floats
- **AC-EMBED-03** — All vectors from the same model have the same length
- **AC-EMBED-04** — `EmbeddingError` is raised after 3 failed API attempts

### Storage (AC-STORE)

- **AC-STORE-01** — `upsert_chunks` returns the correct count of stored points
- **AC-STORE-02** — `upsert_chunks` returns 0 for an empty input without error
- **AC-STORE-03** — After `delete_by_document_id`, no points with that `document_id` remain
- **AC-STORE-04** — `delete_by_document_id` does not raise for an unknown `document_id`

### End-to-end route (AC-ROUTE)

- **AC-ROUTE-01** — A valid PDF upload returns HTTP 201
- **AC-ROUTE-02** — The response includes `document_id`, `filename`, `sha256`, `pages`, `chunks`, `replaced`
- **AC-ROUTE-03** — Uploading the same PDF twice returns `replaced: true` on the second upload
- **AC-ROUTE-04** — Uploading the same PDF twice results in the same number of chunks as a single upload (no duplication)
- **AC-ROUTE-05** — A corrupt PDF returns HTTP 422 with error code `INVALID_PDF`
- **AC-ROUTE-06** — A PDF with all blank pages returns HTTP 422 with error code `EMPTY_PDF`

---

## 9. Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pymupdf` | 1.24.x | PDF parsing and page rasterisation |
| `pytesseract` | 0.3.x | OCR fallback for image-based pages |
| `Pillow` | 10.x | Image handling for pytesseract |
| `openai` | 1.47.x | Embeddings API |
| `qdrant-client` | 1.11.x | Vector store |
| `tenacity` | 9.x | Retry logic for embedding API calls |

System dependency: `tesseract-ocr` must be installed in the Docker image.

---

## 10. Notes for the plan

- `pytesseract` and `tesseract-ocr` are not in the current `requirements.txt`
  or `Dockerfile` — both must be added before implementation begins
- The `IngestResponse` schema in `app/api/schemas.py` needs two new fields:
  `sha256` and `replaced` — update before writing tests
- A new file `app/ingestion/exceptions.py` is required — not in the current scaffold
- A new file `app/ingestion/validator.py` is required — not in the current scaffold
- A new file `app/ingestion/hasher.py` is required — not in the current scaffold
