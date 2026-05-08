"""Tests for the query validator (Phase 2).

No mocking needed — validate_query_request is a pure function.
Imports are inline so Phase 1 test collection is not blocked.
"""

import pytest


# AC-QVAL-01 — A question shorter than 3 characters returns INVALID_QUESTION
def test_validate_rejects_short_question():
    from app.query.validator import validate_query_request
    from app.ingestion.exceptions import InvalidQuestionError

    with pytest.raises(InvalidQuestionError):
        validate_query_request("ab", None, None)


# AC-QVAL-02 — A question longer than 1000 characters returns INVALID_QUESTION
def test_validate_rejects_long_question():
    from app.query.validator import validate_query_request
    from app.ingestion.exceptions import InvalidQuestionError

    with pytest.raises(InvalidQuestionError):
        validate_query_request("x" * 1001, None, None)


# AC-QVAL-03 — Both document_id and filename filters returns INVALID_FILTERS
def test_validate_rejects_dual_filters():
    from app.query.validator import validate_query_request
    from app.api.schemas import QueryFilters
    from app.ingestion.exceptions import InvalidFiltersError

    filters = QueryFilters(document_id="doc-1", filename="budget.pdf")
    with pytest.raises(InvalidFiltersError):
        validate_query_request("What is the budget?", filters, None)


# AC-QVAL-04 — A history entry with an invalid role returns INVALID_HISTORY
def test_validate_rejects_invalid_history_role():
    from app.query.validator import validate_query_request
    from app.api.schemas import ConversationTurn
    from app.ingestion.exceptions import InvalidHistoryError

    bad_turn = ConversationTurn(role="system", content="hello")
    with pytest.raises(InvalidHistoryError):
        validate_query_request("What is the budget?", None, [bad_turn])


# AC-QVAL-05 — A valid request with no filters and no history passes validation
def test_validate_accepts_valid_request():
    from app.query.validator import validate_query_request

    result = validate_query_request("What is the total expenditure?", None, None)
    assert result is None
