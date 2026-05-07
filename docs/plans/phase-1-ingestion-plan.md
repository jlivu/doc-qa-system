# Phase 1 — Ingestion Pipeline Implementation Plan

**Spec:** `docs/specs/phase-1-ingestion.md` v1.0
**Date:** 2026-05-08

---

## 1. Scaffold Gap Analysis

Before defining build order, here is every delta between the spec and the
current scaffold. Each gap is tagged with the step that resolves it.

### New files required

| File | Purpose | Step |
|------|---------|------|
| `app/ingestion/exceptions.py` | 6 custom exception classes + base class | 3 |
| `app/ingestion/validator.py` | `validate_upload()` — type & size checks | 4 |
| `app/ingestion/hasher.py` | `compute_sha256()`, `find_existing_document()` | 5 |
| `tests/test_validator.py` | Unit tests for validator | 10 |
| `tests/test_hasher.py` | Unit tests for hasher | 10 |

### Existing files with required changes

| File | Gap | Step |
|------|-----|------|
| `requirements.txt` | Missing `pytesseract`, `Pillow` | 1 |
| `Dockerfile` | Missing `tesseract-ocr` system package | 1 |
| `.github/workflows/ci.yml` | Missing `tesseract-ocr` in CI runner | 1 |
| `app/api/schemas.py` | `IngestResponse` missing `sha256`, `replaced`; no `ErrorResponse` model | 2 |
| `app/ingestion/parser.py` | `PageDict` missing `ocr_used`; raises `ValueError` not `InvalidPDFError`; no `EmptyPDFError`; no OCR fallback; no `finally` block | 6 |
| `app/ingestion/chunker.py` | `ChunkDict` missing `sha256`, `char_count`; function missing `sha256` param; blank-page threshold is 10 not 20; no chunk-minimum filter | 7 |
| `app/ingestion/embedder.py` | Doesn't wrap `RetryError` → `EmbeddingError` | 8 |
| `app/retrieval/vector_store.py` | Function named `delete_document` not `delete_by_document_id`; payload missing `sha256`; no `StorageError` wrapping | 9 |
| `app/api/routes/ingest.py` | Entire pipeline is a skeleton | 10 |
| `tests/test_ingestion.py` | Parser test expects `ValueError`; chunker calls missing `sha256`; no OCR/EmptyPDF/new-field tests | 11 |
| `tests/test_api.py` | Fixture doesn't mock pipeline; missing dedup, error-code, size-limit tests | 11 |
| `tests/test_retrieval.py` | Imports `delete_document`; missing delete tests | 11 |

---

## 2. Build Order

Steps are ordered so that every dependency is satisfied before the code
that imports it. Each step is independently testable before moving on.

```
Step  1  Infrastructure         requirements.txt, Dockerfile, ci.yml
Step  2  Schemas                app/api/schemas.py
Step  3  Exceptions             app/ingestion/exceptions.py          (new)
Step  4  Validator              app/ingestion/validator.py            (new)
Step  5  Hasher                 app/ingestion/hasher.py               (new)
Step  6  Parser                 app/ingestion/parser.py               (modify)
Step  7  Chunker                app/ingestion/chunker.py              (modify)
Step  8  Embedder               app/ingestion/embedder.py             (modify)
Step  9  Vector store           app/retrieval/vector_store.py         (modify)
Step 10  Ingest route           app/api/routes/ingest.py              (rewrite)
Step 11  Tests                  all test files
```

**Why this order:**

- **Step 1 first** — pytesseract won't import without `Pillow`; parser
  changes in step 6 depend on both packages being installed. CI will fail
  without `tesseract-ocr` on the runner.
- **Step 2 before step 10** — the route returns `IngestResponse`; the schema
  must have `sha256` and `replaced` fields before the route can be written.
- **Step 3 before steps 4-9** — every module imports its exception class from
  `exceptions.py`.
- **Steps 4-5 are leaf modules** — no internal dependencies; can be built in
  either order.
- **Step 6 before step 7** — the chunker imports `PageDict` from the parser;
  `PageDict` gains `ocr_used` in step 6.
- **Step 7 before step 8** — the embedder imports `ChunkDict` from the
  chunker; `ChunkDict` gains `sha256` and `char_count` in step 7.
- **Step 8 before step 9** — the vector store imports `EmbeddedChunk` from
  the embedder.
- **Step 10 last among production code** — the route orchestrates all modules;
  every import must already exist.
- **Step 11 last** — tests reference the final public interfaces of every
  module.

---

## 3. Scaffold Changes (before any module code)

### 3.1 `requirements.txt`

Add two lines after the `pymupdf==1.24.10` line:

```
pytesseract==0.3.13
Pillow==10.4.0
```

`pytesseract` is the Python wrapper; `Pillow` provides the `Image` class
that pytesseract requires as input. Both are listed in spec section 9.

### 3.2 `Dockerfile`

Insert a system-package install in the **runtime stage** (between the
`FROM python:3.11-slim AS runtime` line and the `addgroup` line):

```dockerfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*
```

This is only needed in the runtime stage. The builder stage installs Python
wheels which do not link against Tesseract at build time. The `rm` cleans up
the apt cache to keep the image small.

### 3.3 `.github/workflows/ci.yml`

Add a step in the `test` job, **before** "Install dependencies":

```yaml
      - name: Install system dependencies
        run: sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
```

Without this the `import pytesseract` line will succeed (it's a pure-Python
package) but any test that actually calls `pytesseract.image_to_string()`
will fail with `TesseractNotFoundError`.

### 3.4 `app/api/schemas.py`

Add two fields to `IngestResponse`:

```python
class IngestResponse(BaseModel):
    document_id: str = Field(...)
    filename: str = Field(...)
    sha256: str = Field(description="SHA-256 hex digest of the file content")
    pages: int = Field(...)
    chunks: int = Field(...)
    replaced: bool = Field(default=False, description="True if a previous version was replaced")
    message: str = Field(default="Document ingested successfully")
```

Add an `ErrorResponse` model (used to document the error shape in OpenAPI,
and optionally to serialise error bodies in the route):

```python
class ErrorResponse(BaseModel):
    error: str = Field(description="Machine-readable error code")
    detail: str = Field(description="Human-readable error description")
```

---

## 4. Implementation Strategy — Module by Module

### 4.1 Exceptions — `app/ingestion/exceptions.py` (new file)

**What to build:**

A common base class `IngestionError(Exception)` and six subclasses:

| Class | Inherits from |
|-------|--------------|
| `InvalidFileTypeError` | `IngestionError` |
| `FileTooLargeError` | `IngestionError` |
| `InvalidPDFError` | `IngestionError` |
| `EmptyPDFError` | `IngestionError` |
| `EmbeddingError` | `IngestionError` |
| `StorageError` | `IngestionError` |

**Key decisions:**

- All are plain subclasses with no custom `__init__` — the message is passed
  via the standard `Exception(msg)` constructor. No extra fields needed.
- The shared `IngestionError` base allows the route to catch all pipeline
  errors in a single `except IngestionError` clause if needed, while still
  mapping individual subclasses to specific HTTP status codes.

**Pitfalls:** None — this is a trivial module. Just make sure every class name
matches the spec exactly, as they are referenced by name in exception handlers.

---

### 4.2 Validator — `app/ingestion/validator.py` (new file)

**What to build:**

Two module-level constants and one function:

```python
MAX_FILE_SIZE_BYTES = 52_428_800  # 50 * 1024 * 1024
ALLOWED_CONTENT_TYPE = "application/pdf"

def validate_upload(filename: str, content_type: str, size_bytes: int) -> None:
```

**Internal logic:**

1. If `content_type != ALLOWED_CONTENT_TYPE` → raise `InvalidFileTypeError`
   with message `f"Expected application/pdf, received {content_type}"`.
2. If `size_bytes > MAX_FILE_SIZE_BYTES` → raise `FileTooLargeError` with
   message including both actual and max size.
3. Otherwise return `None`.

**Key decisions:**

- The `filename` parameter is accepted (spec mandates the signature) but
  not used in Phase 1. Future phases may validate the extension.
- The size check uses `>` (strictly greater), not `>=`. A file of exactly
  50 MB (52,428,800 bytes) is allowed.
- The function does not read or open the file — it validates metadata only.
  The route must read the file into memory first, then pass `len(pdf_bytes)`.

**Pitfalls:**

- FastAPI's `UploadFile.size` may be `None` in some contexts. The route must
  use `len(await file.read())` instead of `file.size` to get a reliable byte
  count.
- Do not compare `content_type` case-insensitively — MIME types in HTTP are
  case-insensitive by spec, but FastAPI normalises them to lowercase. A
  simple `!=` comparison is safe.

---

### 4.3 Hasher — `app/ingestion/hasher.py` (new file)

**What to build:**

Two functions:

```python
def compute_sha256(data: bytes) -> str: ...
def find_existing_document(sha256: str, client: QdrantClient, settings: Settings) -> str | None: ...
```

**Internal logic — `compute_sha256`:**

```python
import hashlib
return hashlib.sha256(data).hexdigest()
```

Pure function, no side effects, stdlib only.

**Internal logic — `find_existing_document`:**

Use Qdrant's `scroll()` API with a payload filter to find any point whose
`sha256` field matches the given hash. `scroll()` is correct here because we
are filtering by metadata, not doing a vector similarity search.

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

points, _ = client.scroll(
    collection_name=settings.qdrant_collection,
    scroll_filter=Filter(
        must=[FieldCondition(key="sha256", match=MatchValue(value=sha256))]
    ),
    limit=1,
    with_payload=True,
    with_vectors=False,
)
```

- If `points` is non-empty, return `points[0].payload["document_id"]`.
- If empty, return `None`.

**Key decisions:**

- `limit=1` — we only need to know *if* a duplicate exists and retrieve its
  `document_id`. We don't need all matching points.
- `with_vectors=False` — avoids transferring 1536 floats we don't need.
- The function must handle the case where the collection doesn't exist yet
  (first-ever ingestion). Qdrant raises an exception on `scroll()` for a
  non-existent collection. Wrap in `try/except` and return `None`.

**Pitfalls:**

- Do not use `client.search()` — it requires a query vector and performs ANN,
  which is semantically wrong for an exact metadata lookup.
- Do not use `client.get_points()` — it fetches by point ID, not by payload.

---

### 4.4 Parser — `app/ingestion/parser.py` (modify existing)

**Current state:** Working text extraction with PyMuPDF. Missing `ocr_used`
field, OCR fallback, custom exceptions, `finally` cleanup, and zero-page
check.

**Changes:**

1. **Add imports:**
   ```python
   from PIL import Image
   import pytesseract
   from app.ingestion.exceptions import InvalidPDFError, EmptyPDFError
   ```

2. **Add `ocr_used: bool` to `PageDict`:**
   ```python
   class PageDict(TypedDict):
       page_number: int
       text: str
       char_count: int
       ocr_used: bool      # NEW
   ```

3. **Replace `ValueError` with `InvalidPDFError`** in the `except` block.

4. **Add zero-page check** after opening the document:
   ```python
   if doc.page_count == 0:
       doc.close()
       raise EmptyPDFError("PDF has zero pages")
   ```

5. **Restructure with `finally`:**
   ```python
   try:
       doc = fitz.open(stream=pdf_bytes, filetype="pdf")
   except Exception as exc:
       raise InvalidPDFError(...) from exc

   try:
       # zero-page check
       # page iteration loop
       return pages
   finally:
       doc.close()
   ```

6. **Add OCR fallback** inside the page loop:
   ```python
   text = page.get_text("text").strip()
   ocr_used = False

   if len(text) < 20:
       pix = page.get_pixmap(dpi=300, colorspace=fitz.csGRAY)
       img = Image.frombytes("L", [pix.width, pix.height], pix.samples)
       ocr_text = pytesseract.image_to_string(img, lang="eng").strip()
       if ocr_text:
           text = ocr_text
           ocr_used = True
   ```

**Key decisions:**

- The 20-character threshold applies to the *native* text extracted by
  PyMuPDF. If a page has, say, just a page number ("12"), OCR is triggered
  to check if there's image-based content the native extractor missed.
- OCR replaces the native text entirely when triggered (not appended).
  If OCR also returns nothing, the page stays blank (`ocr_used` remains
  `False`) — this is by design: a truly blank page produces
  `char_count == 0` and `ocr_used == False`.
- `fitz.csGRAY` produces a grayscale pixmap. Grayscale is passed to
  pytesseract because OCR accuracy on grayscale is comparable to colour,
  and the image is ~3x smaller (faster processing).
- DPI 300 is the de facto standard for OCR rasterisation. Lower values
  degrade accuracy; higher values are slower with minimal benefit.

**Pitfalls:**

- `page.get_pixmap()` returns pixel data as a flat `bytes` object via
  `pix.samples`. The `Image.frombytes()` call must use mode `"L"` (8-bit
  grayscale) and size `[pix.width, pix.height]` — getting either wrong
  produces a corrupted image.
- Must strip the OCR text. pytesseract often appends a trailing `\x0c`
  (form feed) character.
- Do not catch exceptions from pytesseract inside the page loop — let them
  propagate. A broken Tesseract installation should fail fast, not silently
  skip OCR.

---

### 4.5 Chunker — `app/ingestion/chunker.py` (modify existing)

**Current state:** Working chunking with `RecursiveCharacterTextSplitter`.
Missing `sha256` and `char_count` fields, wrong blank-page threshold, no
minimum-chunk filter.

**Changes:**

1. **Add fields to `ChunkDict`:**
   ```python
   class ChunkDict(TypedDict):
       chunk_id: str
       document_id: str
       filename: str
       sha256: str         # NEW
       page_number: int
       text: str
       char_count: int     # NEW
   ```

2. **Add `sha256` parameter** to `chunk_pages()` — between `filename` and
   `settings`:
   ```python
   def chunk_pages(
       pages: list[PageDict],
       document_id: str,
       filename: str,
       sha256: str,        # NEW
       settings: Settings,
   ) -> list[ChunkDict]:
   ```

3. **Change blank-page threshold** from `< 10` to `< 20`:
   ```python
   if page["char_count"] < 20:
       continue
   ```

4. **Replace empty-string check with minimum-length filter:**
   ```python
   stripped = text.strip()
   if len(stripped) < 10:
       continue
   ```

5. **Propagate new fields** in the `ChunkDict` construction:
   ```python
   chunks.append(ChunkDict(
       chunk_id=str(uuid.uuid4()),
       document_id=document_id,
       filename=filename,
       sha256=sha256,
       page_number=page["page_number"],
       text=stripped,
       char_count=len(stripped),
   ))
   ```

**Key decisions:**

- The blank-page threshold (20) and minimum-chunk threshold (10) are
  intentionally different numbers. The page threshold skips pages that
  are essentially blank (page numbers, headers). The chunk threshold
  catches degenerate splits that would embed poorly.
- `char_count` is computed from the *stripped* chunk text, matching
  `len(text)` since `text` is already stripped.

**Pitfalls:**

- The `sha256` parameter is inserted between `filename` and `settings`.
  All existing call sites (tests and the ingest route) must be updated.
  Since `settings` is the last positional arg, accidentally omitting
  `sha256` would pass `settings` as `sha256` — a type error at runtime,
  not a silent bug, but still confusing. The tests must be updated in
  step 11.

---

### 4.6 Embedder — `app/ingestion/embedder.py` (modify existing)

**Current state:** Working batch embedding with retry. The only gap is that
when all retries fail, tenacity raises its own `RetryError` instead of the
spec-required `EmbeddingError`.

**Changes:**

1. **Add imports:**
   ```python
   from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
   from app.ingestion.exceptions import EmbeddingError
   ```

2. **Wrap the batch call** in `embed_chunks()`:
   ```python
   try:
       vectors = _embed_batch(texts, client, settings.embedding_model)
   except RetryError as exc:
       raise EmbeddingError(
           f"Embedding API failed after 3 attempts: {exc}"
       ) from exc
   ```

**Key decisions:**

- The `@retry` decorator on `_embed_batch` is left as-is. The wrapping
  happens in the *caller* (`embed_chunks`), not in the retried function
  itself. This keeps the retry logic clean and the exception translation
  in one place.
- `EmbeddedChunk` inherits from `ChunkDict` via `total=False`. When
  `ChunkDict` gains `sha256` and `char_count` in step 7, `EmbeddedChunk`
  automatically inherits them. The `{**chunk, "vector": vector}` spread
  on the existing line propagates the new fields with no code change.

**Pitfalls:**

- `RetryError` must be imported from `tenacity`, not from some other
  package. tenacity's `RetryError` wraps the last underlying exception
  in `.last_attempt.result()`.

---

### 4.7 Vector Store — `app/retrieval/vector_store.py` (modify existing)

**Current state:** Working `upsert_chunks`, `search`, and `delete_document`.
Gaps: wrong function name, missing `sha256` in payload, no error wrapping.

**Changes:**

1. **Add import:**
   ```python
   from app.ingestion.exceptions import StorageError
   ```

2. **Rename `delete_document` → `delete_by_document_id`**. Update the
   module docstring (line 9) to match. The function body is unchanged.

3. **Add `sha256` to the point payload** in `upsert_chunks`:
   ```python
   payload={
       "document_id": chunk["document_id"],
       "filename": chunk["filename"],
       "sha256": chunk["sha256"],       # NEW
       "page_number": chunk["page_number"],
       "text": chunk["text"],
   },
   ```

4. **Wrap Qdrant exceptions in `StorageError`** in `upsert_chunks`:
   ```python
   try:
       client.upsert(collection_name=settings.qdrant_collection, points=points)
   except Exception as exc:
       raise StorageError(f"Qdrant upsert failed: {exc}") from exc
   ```

   And in `delete_by_document_id`:
   ```python
   try:
       client.delete(...)
   except Exception as exc:
       raise StorageError(f"Qdrant delete failed: {exc}") from exc
   ```

**Key decisions:**

- `search()` and `SearchResult` are **not modified**. They belong to the
  Phase 2 retrieval path and must remain stable.
- The `_ensure_collection` helper is also left unchanged — it already
  handles idempotent collection creation.

**Pitfalls:**

- The rename from `delete_document` to `delete_by_document_id` breaks
  nothing in the current scaffold because the function is never called
  (ingest route is a skeleton, no test imports it). However, if
  `retriever.py` or any future code references `delete_document`, it
  would break. A grep confirms there are no other call sites.

---

### 4.8 Ingest Route — `app/api/routes/ingest.py` (rewrite)

**Current state:** Skeleton that only checks content type and returns zeros.

**Strategy:** Replace the entire route body with the full pipeline
orchestration. Use a try/except block to catch custom exceptions and return
`JSONResponse` with the spec's error format.

**Imports needed:**

```python
import uuid

from fastapi import APIRouter, UploadFile, File, status
from fastapi.responses import JSONResponse

from app.api.schemas import IngestResponse
from app.core.dependencies import QdrantDep, SettingsDep
from app.ingestion.exceptions import (
    InvalidFileTypeError, FileTooLargeError,
    InvalidPDFError, EmptyPDFError,
    EmbeddingError, StorageError,
    IngestionError,
)
from app.ingestion.validator import validate_upload
from app.ingestion.hasher import compute_sha256, find_existing_document
from app.ingestion.parser import extract_text
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import embed_chunks
from app.retrieval.vector_store import upsert_chunks, delete_by_document_id
```

**Internal logic:**

```
1. pdf_bytes = await file.read()
2. validate_upload(file.filename, file.content_type, len(pdf_bytes))
3. sha256 = compute_sha256(pdf_bytes)
4. existing_id = find_existing_document(sha256, qdrant, settings)
5. if existing_id:
       delete_by_document_id(existing_id, qdrant, settings)
       document_id = existing_id
       replaced = True
   else:
       document_id = str(uuid.uuid4())
       replaced = False
6. pages = extract_text(pdf_bytes)
7. chunks = chunk_pages(pages, document_id, filename, sha256, settings)
8. if not chunks and pages:
       raise EmptyPDFError("All pages are blank")
9. embedded = embed_chunks(chunks, settings)
10. stored = upsert_chunks(embedded, qdrant, settings)
11. return IngestResponse(...)
```

**Error handling — exception-to-HTTP mapping:**

```python
_STATUS_MAP = {
    InvalidFileTypeError: (415, "INVALID_FILE_TYPE"),
    FileTooLargeError:    (413, "FILE_TOO_LARGE"),
    InvalidPDFError:      (422, "INVALID_PDF"),
    EmptyPDFError:        (422, "EMPTY_PDF"),
    EmbeddingError:       (500, "EMBEDDING_ERROR"),
    StorageError:         (500, "STORAGE_ERROR"),
}
```

The try/except catches `IngestionError` (the base class), looks up the
concrete type in `_STATUS_MAP`, and returns:

```python
JSONResponse(
    status_code=status_code,
    content={"error": error_code, "detail": str(exc)},
)
```

**Key decisions:**

- Use `JSONResponse`, not `HTTPException`. FastAPI's `HTTPException` returns
  `{"detail": "..."}`, but the spec requires `{"error": "CODE", "detail": "..."}`.
  `JSONResponse` gives full control over the response body.
- Read the file **before** validating. `UploadFile.size` is unreliable in
  some ASGI servers; `len(await file.read())` is authoritative.
- The "all blank pages" check (line 8) lives in the route, not in the parser
  or chunker. The parser returns blank pages by design (one `PageDict` per
  page). The chunker filters them out and returns an empty list without
  raising. The *route* is the right place to decide that an empty chunk list
  from a non-empty page list is an error worth reporting to the user.
- On dedup, the route **reuses the original `document_id`**. This preserves
  any external references to the document across re-ingestion.

**Pitfalls:**

- `file.filename` can be `None`. Default to `"unknown.pdf"`.
- `file.content_type` can be `None`. Default to `""` so the validator rejects
  it cleanly with `InvalidFileTypeError`.
- The `response_model=IngestResponse` on the decorator causes FastAPI to
  validate the happy-path return. The `JSONResponse` error returns bypass
  this validation, which is correct — they have a different shape.

---

## 5. Mocking Strategy for Tests

### 5.1 Principles

- **Mock at the boundary, not inside the module.** External services (OpenAI,
  Qdrant) are always mocked. Internal pure functions (hasher, validator,
  parser, chunker) run real code.
- **Use `unittest.mock.patch`** to replace module-level references in the
  code under test. Use `MagicMock` for Qdrant client instances.
- **Use FastAPI `dependency_overrides`** for API-level tests to inject mock
  settings and a mock Qdrant client.

### 5.2 Per-module mocking table

| Test file | Module under test | What is mocked | How |
|-----------|-------------------|----------------|-----|
| `test_validator.py` | validator | Nothing | Pure function — no external calls |
| `test_hasher.py` | hasher — `compute_sha256` | Nothing | Pure function — stdlib only |
| `test_hasher.py` | hasher — `find_existing_document` | `QdrantClient` | Pass a `MagicMock()` as the `client` parameter. Configure `client.scroll()` return value to simulate found / not-found / missing-collection scenarios |
| `test_ingestion.py` | parser — text PDFs | Nothing | Uses `fitz.open()` on in-memory PDFs; no network. The existing `_make_pdf()` helper creates real PDFs |
| `test_ingestion.py` | parser — OCR path | `pytesseract.image_to_string` | `@patch("app.ingestion.parser.pytesseract.image_to_string")` — return a known string. This avoids requiring Tesseract to be installed for unit tests, and makes assertions deterministic |
| `test_ingestion.py` | chunker | Nothing | Pure function — uses real `RecursiveCharacterTextSplitter` on synthetic page dicts. No network calls |
| `test_ingestion.py` | embedder | `OpenAI` client | `@patch("app.ingestion.embedder.OpenAI")` — configure the mock so `client.embeddings.create()` returns fake embeddings (lists of floats). To test retry exhaustion, make `embeddings.create` raise `openai.APIError` on every call |
| `test_retrieval.py` | vector_store | `QdrantClient` | Pass a `MagicMock()` as the `client` parameter (existing pattern). Verify that `client.upsert`, `client.delete` are called with the correct arguments |
| `test_api.py` | ingest route (E2E) | `embed_chunks`, `find_existing_document`, `QdrantClient` | Use `dependency_overrides` for settings and qdrant. Use `@patch("app.api.routes.ingest.embed_chunks")` to return fake embeddings. Use `@patch("app.api.routes.ingest.find_existing_document")` to control dedup behaviour. Let the real parser and chunker run on in-memory PDFs |

### 5.3 Shared test helpers

Reuse the existing helpers in `tests/test_ingestion.py`:

- `_make_pdf(pages: list[str]) -> bytes` — generates a minimal in-memory PDF.
  Also useful in `test_api.py` (already duplicated there as `_make_pdf_bytes`).
- `_make_settings(**overrides) -> Settings` — returns test-friendly Settings.

Consider extracting these into `tests/conftest.py` if duplication becomes
unwieldy, but this is not strictly required for Phase 1.

### 5.4 OCR test strategy

Testing the OCR path in the parser without requiring Tesseract installed:

1. **Unit tests** — mock `pytesseract.image_to_string` at the module level.
   Create a PDF with a page containing fewer than 20 characters of native
   text. Assert that `pytesseract.image_to_string` was called, and that the
   returned `PageDict` has `ocr_used == True`.

2. **Integration tests** (CI only) — since `.github/workflows/ci.yml` installs
   Tesseract, one test can exercise the real OCR path. Create a PDF with an
   embedded image of text (use `page.insert_image()` with a small PNG
   containing a word). Assert that OCR extracts some text. Mark this test
   with `@pytest.mark.skipif(not shutil.which("tesseract"), reason="tesseract not installed")`.

---

## 6. Acceptance Criteria Traceability

Every AC from spec section 8 is mapped to a specific test function and file.

### Validation (AC-VAL)

| AC | Test function | File |
|----|---------------|------|
| AC-VAL-01 | `test_validate_rejects_non_pdf` | `tests/test_validator.py` |
| AC-VAL-02 | `test_validate_rejects_oversized_file` | `tests/test_validator.py` |
| AC-VAL-03 | `test_validate_accepts_valid_pdf` | `tests/test_validator.py` |

### Hashing (AC-HASH)

| AC | Test function | File |
|----|---------------|------|
| AC-HASH-01 | `test_sha256_returns_64_char_lowercase_hex` | `tests/test_hasher.py` |
| AC-HASH-02 | `test_sha256_deterministic` | `tests/test_hasher.py` |
| AC-HASH-03 | `test_sha256_different_inputs_differ` | `tests/test_hasher.py` |
| AC-HASH-04 | `test_find_existing_returns_none_for_unknown` | `tests/test_hasher.py` |
| AC-HASH-05 | `test_find_existing_returns_document_id_for_known` | `tests/test_hasher.py` |

### Parsing (AC-PARSE)

| AC | Test function | File |
|----|---------------|------|
| AC-PARSE-01 | `test_extract_text_returns_one_dict_per_page` | `tests/test_ingestion.py` (existing) |
| AC-PARSE-02 | `test_extract_text_page_numbers_are_one_indexed` | `tests/test_ingestion.py` (existing) |
| AC-PARSE-03 | `test_extract_text_char_count_matches_text_length` | `tests/test_ingestion.py` (existing) |
| AC-PARSE-04 | `test_extract_text_raises_invalid_pdf_error` | `tests/test_ingestion.py` (update existing) |
| AC-PARSE-05 | `test_extract_text_triggers_ocr_on_short_text` | `tests/test_ingestion.py` (new) |
| AC-PARSE-06 | `test_extract_text_ocr_used_flag` | `tests/test_ingestion.py` (new) |
| AC-PARSE-07 | `test_extract_text_blank_page_not_omitted` | `tests/test_ingestion.py` (new) |

### Chunking (AC-CHUNK)

| AC | Test function | File |
|----|---------------|------|
| AC-CHUNK-01 | `test_chunk_pages_correct_page_number` | `tests/test_ingestion.py` (update existing `test_chunk_pages_metadata_propagated`) |
| AC-CHUNK-02 | `test_chunk_ids_are_unique` | `tests/test_ingestion.py` (existing) |
| AC-CHUNK-03 | `test_chunk_pages_document_metadata_identical` | `tests/test_ingestion.py` (new) |
| AC-CHUNK-04 | `test_chunk_pages_skips_near_blank_pages` | `tests/test_ingestion.py` (update existing `test_chunk_pages_skips_blank_pages`) |
| AC-CHUNK-05 | `test_chunk_pages_discards_tiny_chunks` | `tests/test_ingestion.py` (new) |
| AC-CHUNK-06 | `test_chunk_pages_all_blank_returns_empty` | `tests/test_ingestion.py` (new) |

### Embedding (AC-EMBED)

| AC | Test function | File |
|----|---------------|------|
| AC-EMBED-01 | `test_embed_output_length_matches_input` | `tests/test_ingestion.py` (new) |
| AC-EMBED-02 | `test_embed_chunks_have_vector_field` | `tests/test_ingestion.py` (new) |
| AC-EMBED-03 | `test_embed_vectors_same_length` | `tests/test_ingestion.py` (new) |
| AC-EMBED-04 | `test_embed_raises_embedding_error_after_retries` | `tests/test_ingestion.py` (new) |

### Storage (AC-STORE)

| AC | Test function | File |
|----|---------------|------|
| AC-STORE-01 | `test_upsert_returns_chunk_count` | `tests/test_retrieval.py` (existing) |
| AC-STORE-02 | `test_upsert_empty_list_returns_zero` | `tests/test_retrieval.py` (existing) |
| AC-STORE-03 | `test_delete_by_document_id_removes_points` | `tests/test_retrieval.py` (new) |
| AC-STORE-04 | `test_delete_by_document_id_no_error_for_unknown` | `tests/test_retrieval.py` (new) |

### End-to-end route (AC-ROUTE)

| AC | Test function | File |
|----|---------------|------|
| AC-ROUTE-01 | `test_ingest_valid_pdf_returns_201` | `tests/test_api.py` (update existing) |
| AC-ROUTE-02 | `test_ingest_response_includes_all_fields` | `tests/test_api.py` (new) |
| AC-ROUTE-03 | `test_ingest_duplicate_returns_replaced_true` | `tests/test_api.py` (new) |
| AC-ROUTE-04 | `test_ingest_duplicate_same_chunk_count` | `tests/test_api.py` (new) |
| AC-ROUTE-05 | `test_ingest_corrupt_pdf_returns_422` | `tests/test_api.py` (new) |
| AC-ROUTE-06 | `test_ingest_blank_pdf_returns_422` | `tests/test_api.py` (new) |

---

## 7. Test Update Details

### `tests/test_ingestion.py` — changes to existing tests

1. **`test_extract_text_raises_on_invalid_bytes`** — change
   `pytest.raises(ValueError, ...)` to `pytest.raises(InvalidPDFError, ...)`.
   Add `from app.ingestion.exceptions import InvalidPDFError, EmptyPDFError`.

2. **All 5 `chunk_pages()` calls** — insert `sha256` argument. Each call like
   `chunk_pages(pages, "doc-1", "test.pdf", settings)` becomes
   `chunk_pages(pages, "doc-1", "test.pdf", "abc123", settings)`.

### `tests/test_api.py` — fixture changes

The current `client` fixture uses a `MagicMock` for Qdrant and overrides
settings. Once the route is wired, additional mocking is needed:

- `@patch("app.api.routes.ingest.embed_chunks")` — return a list of dicts
  with `"vector": [0.1] * 1536` appended to each input chunk. This avoids
  real OpenAI calls.
- `@patch("app.api.routes.ingest.find_existing_document")` — return `None`
  by default (no duplicate). Tests for AC-ROUTE-03/04 override this to
  return an existing `document_id`.
- The `mock_qdrant.scroll.return_value` should be set to `([], None)` for
  the default case.
- The mock Qdrant's `upsert` should be a no-op (default MagicMock behaviour).

### `tests/test_retrieval.py` — import fix

Change `from app.retrieval.vector_store import ... delete_document` to
`... delete_by_document_id`.

---

## 8. Definition of Done Checklist

Phase 1 is complete when **all** of the following are true:

- [ ] **Dependencies installed** — `pip install -r requirements.txt` succeeds;
  `python -c "import pytesseract; import PIL"` succeeds
- [ ] **Docker builds** — `docker compose build` completes without errors;
  `docker compose up` starts all three services; `tesseract --version` runs
  inside the API container
- [ ] **All 30 acceptance criteria pass** — `pytest tests/ -v` shows 30+
  passing tests covering every AC from spec section 8 (see traceability table
  above)
- [ ] **No existing tests broken** — every test that existed before Phase 1
  continues to pass (health check, query 501, query validation, search tests)
- [ ] **CI green** — the GitHub Actions workflow passes on a PR branch with
  all Phase 1 changes
- [ ] **Lint clean** — `ruff check app/ tests/` reports zero violations
- [ ] **Manual smoke test** passes:
  ```bash
  # Start services
  docker compose up -d

  # Happy path — ingest a real PDF
  curl -s -X POST http://localhost:8000/ingest \
    -F "file=@sample.pdf" | python -m json.tool
  # → HTTP 201, response has document_id, filename, sha256, pages > 0,
  #   chunks > 0, replaced == false

  # Dedup — re-ingest the same file
  curl -s -X POST http://localhost:8000/ingest \
    -F "file=@sample.pdf" | python -m json.tool
  # → HTTP 201, replaced == true, same document_id, same chunk count

  # Error — non-PDF
  curl -s -w "\n%{http_code}\n" -X POST http://localhost:8000/ingest \
    -F "file=@README.md;type=text/plain"
  # → HTTP 415, {"error": "INVALID_FILE_TYPE", ...}

  # Error — corrupt PDF
  curl -s -w "\n%{http_code}\n" -X POST http://localhost:8000/ingest \
    -F "file=@/dev/null;type=application/pdf;filename=bad.pdf"
  # → HTTP 422, {"error": "INVALID_PDF", ...}
  ```
- [ ] **Error response format** matches spec — every error returns
  `{"error": "CODE", "detail": "..."}`, not FastAPI's default
  `{"detail": "..."}`
- [ ] **No regressions to Phase 2 code** — `app/retrieval/retriever.py`,
  `app/qa/chain.py`, `app/qa/prompts.py`, and `app/api/routes/query.py` are
  unmodified
