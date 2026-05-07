"""API-level tests using FastAPI's TestClient.

These tests hit the HTTP layer directly and mock the underlying services,
so no Qdrant or OpenAI calls are made.
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
    app.dependency_overrides[get_qdrant_client] = lambda: mock_qdrant
    app.dependency_overrides[get_settings] = _make_settings_override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ── Health ─────────────────────────────────────────────────────────────────────

def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ── Ingest ─────────────────────────────────────────────────────────────────────

def test_ingest_rejects_non_pdf(client):
    response = client.post(
        "/ingest",
        files={"file": ("doc.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 415


def test_ingest_accepts_pdf_returns_201(client):
    pdf_bytes = _make_pdf_bytes()
    response = client.post(
        "/ingest",
        files={"file": ("budget.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 201
    body = response.json()
    assert "document_id" in body
    assert body["filename"] == "budget.pdf"


# ── Query ──────────────────────────────────────────────────────────────────────

def test_query_returns_501_before_phase2(client):
    response = client.post(
        "/query",
        json={"question": "What is the total expenditure?"},
    )
    assert response.status_code == 501


def test_query_rejects_empty_question(client):
    response = client.post("/query", json={"question": "ab"})
    assert response.status_code == 422  # Pydantic min_length=3
