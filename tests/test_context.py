"""Tests for the context builder (Phase 2).

No mocking needed — all functions are pure, operating on SourceChunk objects.
Imports are inline so Phase 1 test collection is not blocked.
"""

import pytest

from app.api.schemas import SourceChunk


def _make_chunk(
    score: float = 0.85,
    filename: str = "budget.pdf",
    page: int = 3,
    text: str = "The total expenditure for FY2024 was 12 billion vatu.",
) -> SourceChunk:
    return SourceChunk(
        document_id="doc-1",
        filename=filename,
        page=page,
        text=text,
        score=score,
    )


# ── build_context ────────────────────────────────────────────────────────────

# AC-CTX-01 — build_context formats each chunk with source label, filename, page, score
def test_build_context_format():
    from app.qa.context import build_context

    chunks = [
        _make_chunk(score=0.9, filename="report.pdf", page=5, text="Some text."),
        _make_chunk(score=0.7, filename="budget.pdf", page=2, text="Other text."),
    ]
    result = build_context(chunks)
    assert "[Source 1] report.pdf, page 5 (score: 0.9)" in result
    assert "Some text." in result
    assert "[Source 2] budget.pdf, page 2 (score: 0.7)" in result
    assert "Other text." in result
    assert "\n\n---\n\n" in result


# ── is_not_found ─────────────────────────────────────────────────────────────

# AC-CTX-02 — is_not_found returns True for an empty chunk list
def test_is_not_found_empty_chunks():
    from app.qa.context import is_not_found

    assert is_not_found([]) is True


# AC-CTX-03 — is_not_found returns True when all scores are below threshold
def test_is_not_found_low_scores():
    from app.qa.context import is_not_found

    chunks = [_make_chunk(score=0.1), _make_chunk(score=0.05)]
    assert is_not_found(chunks, threshold=0.3) is True


# AC-CTX-04 — is_not_found returns False when at least one score meets threshold
def test_is_not_found_above_threshold():
    from app.qa.context import is_not_found

    chunks = [_make_chunk(score=0.5), _make_chunk(score=0.1)]
    assert is_not_found(chunks, threshold=0.3) is False


# ── build_not_found_answer ───────────────────────────────────────────────────

# AC-CTX-05 — build_not_found_answer mentions related topics when low-score chunks exist
def test_not_found_answer_with_low_score_chunks():
    from app.qa.context import build_not_found_answer

    chunks = [
        _make_chunk(score=0.1, filename="contracts.pdf"),
        _make_chunk(score=0.05, filename="tenders.pdf"),
    ]
    result = build_not_found_answer(chunks)
    assert isinstance(result, str)
    assert "contracts.pdf" in result or "tenders.pdf" in result
    assert "rephras" in result.lower()


# AC-CTX-06 — build_not_found_answer returns a generic message when chunk list is empty
def test_not_found_answer_empty_chunks():
    from app.qa.context import build_not_found_answer

    result = build_not_found_answer([])
    assert isinstance(result, str)
    assert len(result) > 0
    assert "could not find" in result.lower() or "no relevant" in result.lower()
