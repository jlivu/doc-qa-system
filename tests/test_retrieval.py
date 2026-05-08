"""Tests for the retrieval layer and vector store.

Phase 1: upsert_chunks, delete_by_document_id, search (Qdrant mocked).
Phase 2: embed_query, hybrid_search, text_to_sparse_vector.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.retrieval.vector_store import (
    SearchResult,
    search,
    upsert_chunks,
    delete_by_document_id,
)
from app.core.config import Settings


def _make_settings() -> Settings:
    return Settings(openai_api_key="sk-test")


def _make_mock_hit(
    score: float = 0.9,
    chunk_id: str = "chunk-uuid-1",
    document_id: str = "doc-1",
    filename: str = "budget.pdf",
    page_number: int = 3,
    text: str = "The total expenditure for FY2024 was 12 billion vatu.",
) -> MagicMock:
    hit = MagicMock()
    hit.id = chunk_id
    hit.score = score
    hit.payload = {
        "document_id": document_id,
        "filename": filename,
        "page_number": page_number,
        "text": text,
    }
    return hit


# ── upsert_chunks (Phase 1) ─────────────────────────────────────────────────

# AC-STORE-01 — upsert_chunks returns the correct count of stored points
def test_upsert_returns_chunk_count():
    client = MagicMock()
    client.get_collections.return_value.collections = []
    settings = _make_settings()

    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "filename": "f.pdf",
            "sha256": "abc123",
            "page_number": 1,
            "text": "hello",
            "char_count": 5,
            "vector": [0.1] * 1536,
        }
    ]
    count = upsert_chunks(chunks, client, settings)
    assert count == 1


# AC-STORE-02 — upsert_chunks returns 0 for an empty input without error
def test_upsert_empty_list_returns_zero():
    client = MagicMock()
    settings = _make_settings()
    assert upsert_chunks([], client, settings) == 0


# ── delete_by_document_id (Phase 1) ─────────────────────────────────────────

# AC-STORE-03 — After delete_by_document_id, client.delete was called
def test_delete_by_document_id_removes_points():
    client = MagicMock()
    settings = _make_settings()

    delete_by_document_id("doc-1", client, settings)

    client.delete.assert_called_once()
    call_kwargs = client.delete.call_args.kwargs
    assert call_kwargs["collection_name"] == settings.qdrant_collection


# AC-STORE-04 — delete_by_document_id does not raise for an unknown document_id
def test_delete_by_document_id_no_error_for_unknown():
    client = MagicMock()
    settings = _make_settings()
    delete_by_document_id("nonexistent-doc", client, settings)  # should not raise


# ── search (Phase 1) ────────────────────────────────────────────────────────

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


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 tests — embed_query, text_to_sparse_vector, hybrid_search
#
# These use inline imports so Phase 1 tests above are not broken when the
# Phase 2 production code does not yet exist.
# ══════════════════════════════════════════════════════════════════════════════

# ── embed_query ──────────────────────────────────────────────────────────────

# AC-QEMB-01 — embed_query returns a list of 768 floats
@patch("app.retrieval.retriever.OpenAI")
def test_embed_query_returns_float_list(mock_openai_cls):
    from app.retrieval.retriever import embed_query

    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1] * 768
    mock_response = MagicMock()
    mock_response.data = [mock_embedding]
    mock_client.embeddings.create.return_value = mock_response

    settings = _make_settings()
    result = embed_query("What is the budget?", settings)

    assert isinstance(result, list)
    assert len(result) == 768
    assert all(isinstance(v, float) for v in result)


# AC-QEMB-02 — embed_query raises EmbeddingError on Ollama failure
@patch("app.retrieval.retriever.OpenAI")
def test_embed_query_raises_embedding_error(mock_openai_cls):
    from app.retrieval.retriever import embed_query
    from app.ingestion.exceptions import EmbeddingError

    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.embeddings.create.side_effect = Exception("Connection refused")

    settings = _make_settings()
    with pytest.raises(EmbeddingError):
        embed_query("What is the budget?", settings)


# ── text_to_sparse_vector ────────────────────────────────────────────────────

def test_text_to_sparse_vector_returns_sparse_vector():
    from app.retrieval.vector_store import text_to_sparse_vector

    result = text_to_sparse_vector("The total expenditure was 12 billion vatu.")
    assert hasattr(result, "indices")
    assert hasattr(result, "values")
    assert len(result.indices) > 0
    assert len(result.values) > 0
    assert len(result.indices) == len(result.values)
    assert all(isinstance(v, float) for v in result.values)


def test_text_to_sparse_vector_same_text_same_output():
    from app.retrieval.vector_store import text_to_sparse_vector

    text = "procurement threshold direct contracting"
    a = text_to_sparse_vector(text)
    b = text_to_sparse_vector(text)
    assert a.indices == b.indices
    assert a.values == b.values


# ── hybrid_search ────────────────────────────────────────────────────────────

# AC-HSEARCH-01 — Results are returned ordered by RRF score descending
def test_hybrid_search_ordered_by_rrf_score():
    from app.retrieval.vector_store import hybrid_search

    client = MagicMock()
    hit_a = _make_mock_hit(score=0.9, chunk_id="a")
    hit_b = _make_mock_hit(score=0.8, chunk_id="b")
    hit_c = _make_mock_hit(score=0.7, chunk_id="c")
    # Dense returns [a, b], sparse returns [c, a]
    client.search.side_effect = [[hit_a, hit_b], [hit_c, hit_a]]

    settings = _make_settings()
    results = hybrid_search([0.1] * 768, "budget", client, settings, top_k=3)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    # 'a' appears in both lists → highest RRF score
    assert results[0].chunk_id == "a"


# AC-HSEARCH-02 — document_id filter restricts results to that document only
def test_hybrid_search_document_id_filter():
    from app.retrieval.vector_store import hybrid_search

    client = MagicMock()
    client.search.side_effect = [
        [_make_mock_hit(document_id="doc-1")],
        [_make_mock_hit(document_id="doc-1")],
    ]

    settings = _make_settings()
    hybrid_search([0.1] * 768, "budget", client, settings, top_k=5,
                  document_id="doc-1")

    # Both search calls should include a filter
    for call in client.search.call_args_list:
        assert call.kwargs.get("query_filter") is not None


# AC-HSEARCH-03 — filename filter restricts results to matching documents only
def test_hybrid_search_filename_filter():
    from app.retrieval.vector_store import hybrid_search

    client = MagicMock()
    client.search.side_effect = [
        [_make_mock_hit(filename="budget.pdf")],
        [_make_mock_hit(filename="budget.pdf")],
    ]

    settings = _make_settings()
    hybrid_search([0.1] * 768, "budget", client, settings, top_k=5,
                  filename="budget.pdf")

    for call in client.search.call_args_list:
        assert call.kwargs.get("query_filter") is not None


# AC-HSEARCH-04 — No filter returns results from all documents
def test_hybrid_search_no_filter_returns_all():
    from app.retrieval.vector_store import hybrid_search

    client = MagicMock()
    client.search.side_effect = [
        [_make_mock_hit(chunk_id="x")],
        [_make_mock_hit(chunk_id="y")],
    ]

    settings = _make_settings()
    results = hybrid_search([0.1] * 768, "budget", client, settings, top_k=5)
    assert len(results) > 0

    for call in client.search.call_args_list:
        assert call.kwargs.get("query_filter") is None


# AC-HSEARCH-05 — An empty collection returns an empty list without raising
def test_hybrid_search_empty_collection():
    from app.retrieval.vector_store import hybrid_search

    client = MagicMock()
    client.search.side_effect = [[], []]

    settings = _make_settings()
    results = hybrid_search([0.1] * 768, "budget", client, settings, top_k=5)
    assert results == []


# AC-HSEARCH-06 — Falls back to dense-only search if sparse index absent
def test_hybrid_search_falls_back_to_dense_only():
    from app.retrieval.vector_store import hybrid_search

    client = MagicMock()
    dense_hit = _make_mock_hit(score=0.9, chunk_id="dense-only")
    # Dense search succeeds, sparse search raises
    client.search.side_effect = [[dense_hit], Exception("sparse index not found")]

    settings = _make_settings()
    results = hybrid_search([0.1] * 768, "budget", client, settings, top_k=5)
    assert len(results) >= 1
    assert results[0].chunk_id == "dense-only"
