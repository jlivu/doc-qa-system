"""Tests for the retrieval layer (Phase 2).

Qdrant calls are mocked so these tests run without a running Qdrant instance.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.retrieval.vector_store import SearchResult, search, upsert_chunks
from app.core.config import Settings


def _make_settings() -> Settings:
    return Settings(openai_api_key="sk-test")


def _make_mock_hit(score: float = 0.9) -> MagicMock:
    hit = MagicMock()
    hit.id = "chunk-uuid-1"
    hit.score = score
    hit.payload = {
        "document_id": "doc-1",
        "filename": "budget.pdf",
        "page_number": 3,
        "text": "The total expenditure for FY2024 was 12 billion vatu.",
    }
    return hit


# ── upsert_chunks ─────────────────────────────────────────────────────────────

def test_upsert_returns_chunk_count():
    client = MagicMock()
    client.get_collections.return_value.collections = []
    settings = _make_settings()

    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "filename": "f.pdf",
            "page_number": 1,
            "text": "hello",
            "vector": [0.1] * 1536,
        }
    ]
    count = upsert_chunks(chunks, client, settings)
    assert count == 1


def test_upsert_empty_list_returns_zero():
    client = MagicMock()
    settings = _make_settings()
    assert upsert_chunks([], client, settings) == 0


# ── search ────────────────────────────────────────────────────────────────────

def test_search_returns_search_results():
    client = MagicMock()
    client.search.return_value = [_make_mock_hit(0.95)]
    settings = _make_settings()

    results = search([0.1] * 1536, client, settings)
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)


def test_search_result_fields_populated():
    client = MagicMock()
    client.search.return_value = [_make_mock_hit(0.88)]
    settings = _make_settings()

    result = search([0.1] * 1536, client, settings)[0]
    assert result.filename == "budget.pdf"
    assert result.page_number == 3
    assert result.score == 0.88
    assert "expenditure" in result.text


def test_search_passes_top_k_to_client():
    client = MagicMock()
    client.search.return_value = []
    settings = _make_settings()

    search([0.1] * 1536, client, settings, top_k=7)
    call_kwargs = client.search.call_args.kwargs
    assert call_kwargs["limit"] == 7
