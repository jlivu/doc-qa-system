"""API-level tests using FastAPI's TestClient.

Phase 1: ingest endpoint tests (embed_chunks and find_existing_document mocked).
Phase 2: query endpoint tests (embed_query, hybrid_search, answer mocked).

External services are mocked so no Qdrant or Ollama calls are made.
"""

import io
from unittest.mock import MagicMock, patch

import fitz
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.dependencies import get_qdrant_client
from app.core.config import get_settings, Settings


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_settings_override():
    return Settings(openai_api_key="sk-test")


def _make_pdf_bytes() -> bytes:
    """A single-page PDF with enough text (> 20 chars) to skip OCR."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Vanuatu National Budget Report 2024.")
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _make_blank_pdf_bytes() -> bytes:
    """A single-page PDF with no text at all."""
    doc = fitz.open()
    doc.new_page()  # blank — no text inserted
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _fake_embed_chunks(chunks, settings):
    """Side-effect for the embed_chunks mock — adds a fake vector to each chunk."""
    return [{**c, "vector": [0.1] * 1536} for c in chunks]


def _make_mock_source_chunk(
    score=0.85,
    document_id="doc-1",
    filename="budget.pdf",
    page=3,
    text="The total expenditure for FY2024.",
):
    """Build a MagicMock matching the SourceChunk schema."""
    chunk = MagicMock()
    chunk.document_id = document_id
    chunk.filename = filename
    chunk.page = page
    chunk.text = text
    chunk.score = score
    return chunk


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    mock_qdrant = MagicMock()
    mock_qdrant.get_collections.return_value.collections = []
    mock_qdrant.scroll.return_value = ([], None)   # find_existing_document -> None
    app.dependency_overrides[get_qdrant_client] = lambda: mock_qdrant
    app.dependency_overrides[get_settings] = _make_settings_override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Ingest tests
# ══════════════════════════════════════════════════════════════════════════════

def test_ingest_rejects_non_pdf(client):
    response = client.post(
        "/ingest",
        files={"file": ("doc.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 415
    assert response.json()["error"] == "INVALID_FILE_TYPE"


# AC-ROUTE-01 — A valid PDF upload returns HTTP 201
@patch("app.api.routes.ingest.find_existing_document", return_value=None)
@patch("app.api.routes.ingest.embed_chunks", side_effect=_fake_embed_chunks)
def test_ingest_valid_pdf_returns_201(mock_embed, mock_find, client):
    pdf_bytes = _make_pdf_bytes()
    response = client.post(
        "/ingest",
        files={"file": ("budget.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 201


# AC-ROUTE-02 — Response includes document_id, filename, sha256, pages, chunks, replaced
@patch("app.api.routes.ingest.find_existing_document", return_value=None)
@patch("app.api.routes.ingest.embed_chunks", side_effect=_fake_embed_chunks)
def test_ingest_response_includes_all_fields(mock_embed, mock_find, client):
    pdf_bytes = _make_pdf_bytes()
    response = client.post(
        "/ingest",
        files={"file": ("budget.pdf", pdf_bytes, "application/pdf")},
    )
    body = response.json()
    assert "document_id" in body
    assert body["filename"] == "budget.pdf"
    assert isinstance(body["sha256"], str) and len(body["sha256"]) == 64
    assert body["pages"] > 0
    assert body["chunks"] > 0
    assert body["replaced"] is False
    assert "message" in body


# AC-ROUTE-03 — Same PDF twice returns replaced: true on the second upload
@patch("app.api.routes.ingest.find_existing_document", side_effect=[None, "existing-doc-id"])
@patch("app.api.routes.ingest.embed_chunks", side_effect=_fake_embed_chunks)
def test_ingest_duplicate_returns_replaced_true(mock_embed, mock_find, client):
    pdf_bytes = _make_pdf_bytes()

    resp1 = client.post("/ingest", files={"file": ("budget.pdf", pdf_bytes, "application/pdf")})
    assert resp1.json()["replaced"] is False

    resp2 = client.post("/ingest", files={"file": ("budget.pdf", pdf_bytes, "application/pdf")})
    assert resp2.json()["replaced"] is True


# AC-ROUTE-04 — Same PDF twice produces the same chunk count (no duplication)
@patch("app.api.routes.ingest.find_existing_document", side_effect=[None, "existing-doc-id"])
@patch("app.api.routes.ingest.embed_chunks", side_effect=_fake_embed_chunks)
def test_ingest_duplicate_same_chunk_count(mock_embed, mock_find, client):
    pdf_bytes = _make_pdf_bytes()

    resp1 = client.post("/ingest", files={"file": ("budget.pdf", pdf_bytes, "application/pdf")})
    resp2 = client.post("/ingest", files={"file": ("budget.pdf", pdf_bytes, "application/pdf")})

    assert resp1.json()["chunks"] == resp2.json()["chunks"]
    assert resp1.json()["chunks"] > 0


# AC-ROUTE-05 — A corrupt PDF returns HTTP 422 with error code INVALID_PDF
def test_ingest_corrupt_pdf_returns_422(client):
    response = client.post(
        "/ingest",
        files={"file": ("bad.pdf", b"not a valid pdf", "application/pdf")},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "INVALID_PDF"


# AC-ROUTE-06 — A PDF with all blank pages returns HTTP 422 with error code EMPTY_PDF
@patch("app.ingestion.parser.pytesseract.image_to_string", return_value="")
def test_ingest_blank_pdf_returns_422(mock_ocr, client):
    blank_pdf = _make_blank_pdf_bytes()
    response = client.post(
        "/ingest",
        files={"file": ("blank.pdf", blank_pdf, "application/pdf")},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "EMPTY_PDF"


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Query validation (still valid after Phase 2 schema update)
# ══════════════════════════════════════════════════════════════════════════════

def test_query_rejects_empty_question(client):
    response = client.post("/query", json={"question": "ab"})
    assert response.status_code == 422  # Pydantic min_length=3


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Query route tests
#
# These tests use inline imports for Phase 2 modules so that Phase 1 tests
# above are not broken when the Phase 2 production code does not yet exist.
# Each test imports from app.query.validator (which doesn't exist until
# Phase 2 Step 4) — this triggers ModuleNotFoundError cleanly.
# ══════════════════════════════════════════════════════════════════════════════

# AC-ROUTE-01 (query) — A valid question returns HTTP 200
def test_query_valid_question_returns_200(client):
    from app.query.validator import validate_query_request  # noqa: F401

    with patch("app.api.routes.query.embed_query", return_value=[0.1] * 768), \
         patch("app.api.routes.query.hybrid_search") as mock_search, \
         patch("app.api.routes.query.answer") as mock_answer, \
         patch("app.api.routes.query.is_not_found", return_value=False):
        mock_search.return_value = [_make_mock_source_chunk()]
        mock_answer.return_value = {"answer": "The answer.", "sources": [_make_mock_source_chunk()]}
        response = client.post("/query", json={"question": "What is the total expenditure?"})
    assert response.status_code == 200


# AC-ROUTE-02 (query) — Response contains answer, sources, found, conversation_history
def test_query_response_includes_all_fields(client):
    from app.query.validator import validate_query_request  # noqa: F401

    with patch("app.api.routes.query.embed_query", return_value=[0.1] * 768), \
         patch("app.api.routes.query.hybrid_search") as mock_search, \
         patch("app.api.routes.query.answer") as mock_answer, \
         patch("app.api.routes.query.is_not_found", return_value=False):
        chunk = _make_mock_source_chunk()
        mock_search.return_value = [chunk]
        mock_answer.return_value = {"answer": "The answer.", "sources": [chunk]}
        response = client.post("/query", json={"question": "What is the budget?"})
    body = response.json()
    assert "answer" in body
    assert "sources" in body
    assert "found" in body
    assert "conversation_history" in body


# AC-ROUTE-03 (query) — found is true when relevant chunks are retrieved
def test_query_found_true_when_relevant(client):
    from app.query.validator import validate_query_request  # noqa: F401

    with patch("app.api.routes.query.embed_query", return_value=[0.1] * 768), \
         patch("app.api.routes.query.hybrid_search") as mock_search, \
         patch("app.api.routes.query.answer") as mock_answer, \
         patch("app.api.routes.query.is_not_found", return_value=False):
        mock_search.return_value = [_make_mock_source_chunk(score=0.85)]
        mock_answer.return_value = {"answer": "Found it.", "sources": [_make_mock_source_chunk()]}
        response = client.post("/query", json={"question": "What is the budget?"})
    assert response.json()["found"] is True


# AC-ROUTE-04 (query) — found is false and sources empty when not relevant
def test_query_found_false_when_not_relevant(client):
    from app.query.validator import validate_query_request  # noqa: F401

    with patch("app.api.routes.query.embed_query", return_value=[0.1] * 768), \
         patch("app.api.routes.query.hybrid_search", return_value=[]), \
         patch("app.api.routes.query.is_not_found", return_value=True), \
         patch("app.api.routes.query.build_not_found_answer", return_value="Not found."):
        response = client.post("/query", json={"question": "What is the speed of light?"})
    body = response.json()
    assert body["found"] is False
    assert body["sources"] == []


# AC-ROUTE-05 (query) — Sources include document_id, filename, page, text, score
def test_query_sources_include_all_metadata(client):
    from app.query.validator import validate_query_request  # noqa: F401

    chunk = _make_mock_source_chunk(
        document_id="doc-99", filename="report.pdf", page=7,
        text="Direct contracting threshold.", score=0.89,
    )
    with patch("app.api.routes.query.embed_query", return_value=[0.1] * 768), \
         patch("app.api.routes.query.hybrid_search", return_value=[chunk]), \
         patch("app.api.routes.query.answer") as mock_answer, \
         patch("app.api.routes.query.is_not_found", return_value=False):
        mock_answer.return_value = {"answer": "Answer.", "sources": [chunk]}
        response = client.post("/query", json={"question": "What is the threshold?"})
    source = response.json()["sources"][0]
    assert source["document_id"] == "doc-99"
    assert source["filename"] == "report.pdf"
    assert source["page"] == 7
    assert "threshold" in source["text"].lower()
    assert source["score"] == 0.89


# AC-ROUTE-06 (query) — conversation_history includes the current turn appended
def test_query_history_appended_in_response(client):
    from app.query.validator import validate_query_request  # noqa: F401

    with patch("app.api.routes.query.embed_query", return_value=[0.1] * 768), \
         patch("app.api.routes.query.hybrid_search") as mock_search, \
         patch("app.api.routes.query.answer") as mock_answer, \
         patch("app.api.routes.query.is_not_found", return_value=False):
        mock_search.return_value = [_make_mock_source_chunk()]
        mock_answer.return_value = {"answer": "The answer.", "sources": [_make_mock_source_chunk()]}
        response = client.post("/query", json={"question": "What is the budget?"})
    history = response.json()["conversation_history"]
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "What is the budget?"
    assert history[1]["role"] == "assistant"


# AC-ROUTE-07 (query) — Follow-up with prior history returns contextually aware answer
def test_query_followup_with_history(client):
    from app.query.validator import validate_query_request  # noqa: F401

    prior_history = [
        {"role": "user", "content": "What is the total expenditure?"},
        {"role": "assistant", "content": "The total expenditure is 12B vatu."},
    ]
    with patch("app.api.routes.query.embed_query", return_value=[0.1] * 768), \
         patch("app.api.routes.query.hybrid_search") as mock_search, \
         patch("app.api.routes.query.answer") as mock_answer, \
         patch("app.api.routes.query.is_not_found", return_value=False):
        mock_search.return_value = [_make_mock_source_chunk()]
        mock_answer.return_value = {"answer": "Compared to last year...", "sources": [_make_mock_source_chunk()]}
        response = client.post("/query", json={
            "question": "How does that compare?",
            "conversation_history": prior_history,
        })
    history = response.json()["conversation_history"]
    # Prior 2 turns + current question + current answer = 4 entries
    assert len(history) == 4


# AC-ROUTE-08 (query) — document_id filter returns only sources from that document
def test_query_document_id_filter(client):
    from app.query.validator import validate_query_request  # noqa: F401

    chunk = _make_mock_source_chunk(document_id="doc-42")
    with patch("app.api.routes.query.embed_query", return_value=[0.1] * 768), \
         patch("app.api.routes.query.hybrid_search") as mock_search, \
         patch("app.api.routes.query.answer") as mock_answer, \
         patch("app.api.routes.query.is_not_found", return_value=False):
        mock_search.return_value = [chunk]
        mock_answer.return_value = {"answer": "Filtered.", "sources": [chunk]}
        response = client.post("/query", json={
            "question": "What is the budget?",
            "filters": {"document_id": "doc-42"},
        })
    # Verify hybrid_search was called with the document_id filter
    call_kwargs = mock_search.call_args
    assert call_kwargs.kwargs.get("document_id") == "doc-42" or \
           (len(call_kwargs.args) > 5 and call_kwargs.args[5] == "doc-42")


# AC-ROUTE-09 (query) — An invalid question returns HTTP 422
def test_query_invalid_question_returns_422(client):
    from app.query.validator import validate_query_request  # noqa: F401

    response = client.post("/query", json={"question": "   "})  # whitespace only
    assert response.status_code == 422
