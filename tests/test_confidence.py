"""Tests for compute_confidence (Phase 3 — AC-QUAL-02 through AC-QUAL-04).

No mocking needed — compute_confidence is a pure function.
Imports are inline so Phase 1+2 test collection is not blocked.
"""

import pytest

from app.api.schemas import SourceChunk


def _make_chunk(score: float = 0.85) -> SourceChunk:
    return SourceChunk(
        document_id="doc-1",
        filename="budget.pdf",
        page=3,
        text="Some text.",
        score=score,
    )


# AC-QUAL-02 — confidence is "high" when top source score >= 0.025
def test_confidence_high_when_score_above_025():
    from app.qa.context import compute_confidence

    chunks = [_make_chunk(score=0.030), _make_chunk(score=0.010)]
    assert compute_confidence(chunks) == "high"


# AC-QUAL-03 — confidence is "medium" when top score is 0.015–0.024
def test_confidence_medium_when_score_015_to_024():
    from app.qa.context import compute_confidence

    chunks = [_make_chunk(score=0.020), _make_chunk(score=0.005)]
    assert compute_confidence(chunks) == "medium"


# AC-QUAL-04 — confidence is "low" when top score < 0.015
def test_confidence_low_when_score_below_015():
    from app.qa.context import compute_confidence

    chunks = [_make_chunk(score=0.010), _make_chunk(score=0.002)]
    assert compute_confidence(chunks) == "low"


def test_confidence_low_when_no_chunks():
    from app.qa.context import compute_confidence

    assert compute_confidence([]) == "low"
