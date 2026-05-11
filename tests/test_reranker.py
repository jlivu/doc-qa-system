"""Tests for the reranker module (Phase 4 — AC-RERANK).

CrossEncoder is mocked via @patch to avoid downloading the model.
Imports are inline so earlier-phase tests are not blocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.api.schemas import SourceChunk


def _make_chunk(text: str = "Some text.", score: float = 0.02) -> SourceChunk:
    return SourceChunk(
        document_id="doc-1",
        filename="budget.pdf",
        page=3,
        text=text,
        score=score,
    )


# AC-RERANK-01 — The reranker model loads at startup without error
@patch("app.retrieval.reranker.CrossEncoder")
def test_load_reranker_returns_model(mock_ce_cls):
    from app.retrieval.reranker import load_reranker

    mock_ce_cls.return_value = MagicMock()
    model = load_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    mock_ce_cls.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert model is not None


# AC-RERANK-02 — rerank() returns exactly top_k chunks when input > top_k
def test_rerank_returns_top_k():
    from app.retrieval.reranker import rerank

    reranker = MagicMock()
    reranker.predict.return_value = [0.9, 0.7, 0.5, 0.3, 0.1, -0.2, -0.5]
    chunks = [_make_chunk(text=f"chunk {i}") for i in range(7)]
    result = rerank("question?", chunks, top_k=5, reranker=reranker)
    assert len(result) == 5


# AC-RERANK-03 — rerank() returns all chunks when input < top_k
def test_rerank_returns_all_when_fewer_than_top_k():
    from app.retrieval.reranker import rerank

    reranker = MagicMock()
    reranker.predict.return_value = [0.9, 0.5]
    chunks = [_make_chunk(text="a"), _make_chunk(text="b")]
    result = rerank("question?", chunks, top_k=5, reranker=reranker)
    assert len(result) == 2


# AC-RERANK-04 — Returned chunks are ordered by cross-encoder score descending
def test_rerank_ordered_by_score_descending():
    from app.retrieval.reranker import rerank

    reranker = MagicMock()
    reranker.predict.return_value = [0.1, 0.9, 0.5]
    chunks = [_make_chunk(text="low"), _make_chunk(text="high"), _make_chunk(text="mid")]
    result = rerank("question?", chunks, top_k=3, reranker=reranker)
    scores = [c.score for c in result]
    assert scores == sorted(scores, reverse=True)
    assert result[0].text == "high"


# AC-RERANK-05 — rerank() returns empty list for empty input
def test_rerank_empty_input():
    from app.retrieval.reranker import rerank

    reranker = MagicMock()
    result = rerank("question?", [], top_k=5, reranker=reranker)
    assert result == []
    reranker.predict.assert_not_called()


# AC-RERANK-06 — SourceChunk.score reflects cross-encoder score, not RRF
def test_rerank_replaces_rrf_score():
    from app.retrieval.reranker import rerank

    reranker = MagicMock()
    reranker.predict.return_value = [4.2]
    chunks = [_make_chunk(score=0.02)]  # original RRF score
    result = rerank("question?", chunks, top_k=5, reranker=reranker)
    assert result[0].score == pytest.approx(4.2)
