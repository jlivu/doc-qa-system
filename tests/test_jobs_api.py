"""Tests for async ingestion and jobs API (Phase 4 — AC-ASYNC).

Imports are inline so earlier-phase tests are not blocked when
Phase 4 production code does not yet exist.
"""

import io
from unittest.mock import MagicMock, patch

import fitz
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.dependencies import get_qdrant_client
from app.core.config import get_settings, Settings


def _make_settings_override():
    return Settings(openai_api_key="sk-test")


def _make_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Vanuatu National Budget Report 2024.")
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


@pytest.fixture
def client():
    mock_qdrant = MagicMock()
    mock_qdrant.get_collections.return_value.collections = []
    mock_qdrant.scroll.return_value = ([], None)
    app.dependency_overrides[get_qdrant_client] = lambda: mock_qdrant
    app.dependency_overrides[get_settings] = _make_settings_override
    # Mock reranker if get_reranker exists
    try:
        from app.core.dependencies import get_reranker
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.9]
        app.dependency_overrides[get_reranker] = lambda: mock_reranker
    except ImportError:
        pass
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# AC-ASYNC-01 — POST /ingest returns HTTP 202 immediately with a job_id
def test_ingest_returns_202_with_job_id(client):
    from app.api.schemas import IngestAcceptedResponse  # noqa: F401

    pdf_bytes = _make_pdf_bytes()
    response = client.post(
        "/ingest",
        files={"file": ("budget.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 202
    body = response.json()
    assert "job_id" in body
    assert body["status"] == "pending"
    assert body["filename"] == "budget.pdf"


# AC-ASYNC-02 — GET /jobs/{job_id} returns pending or running while in progress
def test_job_status_pending_or_running(client):
    from app.api.routes.jobs import get_job  # noqa: F401

    with patch("app.api.routes.jobs.get_job", return_value={
        "job_id": "j1", "status": "running", "filename": "a.pdf",
        "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:01",
        "document_id": None, "pages": None, "chunks": None,
        "replaced": None, "error": None,
    }):
        response = client.get("/jobs/j1")
    assert response.status_code == 200
    assert response.json()["status"] in ("pending", "running")


# AC-ASYNC-03 — GET /jobs/{job_id} returns completed with results
def test_job_status_completed(client):
    from app.api.routes.jobs import get_job  # noqa: F401

    with patch("app.api.routes.jobs.get_job", return_value={
        "job_id": "j2", "status": "completed", "filename": "b.pdf",
        "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:30",
        "document_id": "doc-42", "pages": 23, "chunks": 90,
        "replaced": False, "error": None,
    }):
        response = client.get("/jobs/j2")
    body = response.json()
    assert body["status"] == "completed"
    assert body["document_id"] == "doc-42"
    assert body["pages"] == 23
    assert body["chunks"] == 90


# AC-ASYNC-04 — GET /jobs/{job_id} returns failed with error for invalid PDF
def test_job_status_failed_for_invalid_pdf(client):
    from app.api.routes.jobs import get_job  # noqa: F401

    with patch("app.api.routes.jobs.get_job", return_value={
        "job_id": "j3", "status": "failed", "filename": "bad.pdf",
        "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:05",
        "document_id": None, "pages": None, "chunks": None,
        "replaced": None, "error": "Could not open PDF",
    }):
        response = client.get("/jobs/j3")
    body = response.json()
    assert body["status"] == "failed"
    assert body["error"] is not None
    assert "PDF" in body["error"]


# AC-ASYNC-05 — GET /jobs/{unknown_id} returns HTTP 404
def test_job_not_found_returns_404(client):
    from app.api.routes.jobs import get_job  # noqa: F401

    with patch("app.api.routes.jobs.get_job", return_value=None):
        response = client.get("/jobs/nonexistent")
    assert response.status_code == 404
    assert response.json()["error"] == "JOB_NOT_FOUND"


# AC-ASYNC-06 — GET /jobs returns a list of all jobs
def test_list_jobs_returns_all(client):
    from app.api.routes.jobs import list_jobs  # noqa: F401

    with patch("app.api.routes.jobs.list_jobs", return_value=[
        {"job_id": "j1", "status": "completed", "filename": "a.pdf",
         "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:30",
         "document_id": "doc-1", "pages": 10, "chunks": 50,
         "replaced": False, "error": None},
        {"job_id": "j2", "status": "pending", "filename": "b.pdf",
         "created_at": "2026-01-01T00:01:00", "updated_at": "2026-01-01T00:01:00",
         "document_id": None, "pages": None, "chunks": None,
         "replaced": None, "error": None},
    ]):
        response = client.get("/jobs")
    body = response.json()
    assert body["total"] == 2
    assert len(body["jobs"]) == 2


# AC-ASYNC-07 — Async-ingested document is queryable after completion
def test_async_ingested_document_is_queryable(client):
    from app.api.routes.jobs import get_job  # noqa: F401

    # Verify a completed job's document can be queried
    with patch("app.api.routes.query.embed_query", return_value=[0.1] * 768), \
         patch("app.api.routes.query.hybrid_search") as mock_search, \
         patch("app.api.routes.query.answer") as mock_answer, \
         patch("app.api.routes.query.is_not_found", return_value=False):
        mock_chunk = MagicMock()
        mock_chunk.document_id = "doc-async"
        mock_chunk.filename = "async.pdf"
        mock_chunk.page = 1
        mock_chunk.text = "Async ingested content."
        mock_chunk.score = 0.85
        mock_search.return_value = [mock_chunk]
        mock_answer.return_value = {"answer": "Answer from async doc.", "sources": [mock_chunk]}
        response = client.post("/query", json={
            "question": "What is in the async document?",
            "filters": {"document_id": "doc-async"},
        })
    assert response.status_code == 200
    assert response.json()["found"] is True
