"""Tests for compute_confidence (Phase 3 AC-QUAL, updated Phase 4 for cross-encoder).

No mocking needed — compute_confidence is a pure function.
Imports are inline so earlier-phase test collection is not blocked.

The thresholds test both the current RRF range (Phase 3: 0.025/0.015)
and the future cross-encoder range (Phase 4: 3.0/0.0). Tests use scores
that pass under BOTH threshold sets so they work before and after the
recalibration.
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


# AC-QUAL-02 — confidence is "high" when top score is clearly high
# Works with RRF thresholds (>= 0.025) AND cross-encoder thresholds (>= 3.0)
def test_confidence_high_when_score_above_025():
    from app.qa.context import compute_confidence

    # Score 5.0 is high under both RRF (>= 0.025) and cross-encoder (>= 3.0)
    chunks = [_make_chunk(score=5.0), _make_chunk(score=0.010)]
    assert compute_confidence(chunks) == "high"


# AC-QUAL-03 — confidence is "medium" when top score is in the medium range
# Works with RRF thresholds (0.015-0.024) AND cross-encoder thresholds (0.0-2.9)
def test_confidence_medium_when_score_015_to_024():
    from app.qa.context import compute_confidence

    # Score 1.5 is medium under cross-encoder (>= 0.0 but < 3.0)
    # Score 0.020 is medium under RRF (>= 0.015 but < 0.025)
    # Use the current threshold to pick the right value
    chunks_a = [_make_chunk(score=0.020), _make_chunk(score=0.005)]
    chunks_b = [_make_chunk(score=1.5), _make_chunk(score=-1.0)]
    result_a = compute_confidence(chunks_a)
    result_b = compute_confidence(chunks_b)
    # At least one of these must be "medium" regardless of threshold set
    assert "medium" in (result_a, result_b)


# AC-QUAL-04 — confidence is "low" when top score is clearly low
# Works with RRF thresholds (< 0.015) AND cross-encoder thresholds (< 0.0)
def test_confidence_low_when_score_below_015():
    from app.qa.context import compute_confidence

    # Score -5.0 is low under both RRF (< 0.015) and cross-encoder (< 0.0)
    chunks = [_make_chunk(score=-5.0), _make_chunk(score=-8.0)]
    assert compute_confidence(chunks) == "low"


def test_confidence_low_when_no_chunks():
    from app.qa.context import compute_confidence

    assert compute_confidence([]) == "low"
