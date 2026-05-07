"""API-level tests using FastAPI's TestClient (Phase 1).

These tests hit the HTTP layer directly. External services are mocked:
- Qdrant: via dependency_overrides (MagicMock)
- OpenAI embeddings: via @patch("app.api.routes.ingest.embed_chunks")
- Dedup lookup: via @patch("app.api.routes.ingest.find_existing_document")
- OCR: via @patch("app.ingestion.parser.pytesseract.image_to_string")

The real parser and chunker run on in-memory PDFs — no network calls.
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


# ── Ingest — validation errors ───────────────────────────────────────────────

def test_ingest_rejects_non_pdf(client):
    response = client.post(
        "/ingest",
        files={"file": ("doc.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 415
    assert response.json()["error"] == "INVALID_FILE_TYPE"


# ── Ingest — happy path ──────────────────────────────────────────────────────

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


# ── Ingest — deduplication ────────────────────────────────────────────────────

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


# ── Ingest — error paths ─────────────────────────────────────────────────────

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


# ── Query ─────────────────────────────────────────────────────────────────────

def test_query_returns_501_before_phase2(client):
    response = client.post(
        "/query",
        json={"question": "What is the total expenditure?"},
    )
    assert response.status_code == 501


def test_query_rejects_empty_question(client):
    response = client.post("/query", json={"question": "ab"})
    assert response.status_code == 422  # Pydantic min_length=3
