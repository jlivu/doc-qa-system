"""Query request validation."""

from app.api.schemas import QueryFilters, ConversationTurn
from app.ingestion.exceptions import (
    InvalidQuestionError,
    InvalidFiltersError,
    InvalidHistoryError,
)


def validate_query_request(
    question: str,
    filters: QueryFilters | None,
    history: list[ConversationTurn] | None,
) -> None:
    """Validate query parameters before expensive operations begin.

    Raises:
        InvalidQuestionError: Question too short (stripped) or too long.
        InvalidFiltersError: Both document_id and filename provided.
        InvalidHistoryError: History entry has bad role or empty content.
    """
    if len(question.strip()) < 3:
        raise InvalidQuestionError(
            f"Question must be at least 3 non-whitespace characters, got {len(question.strip())}"
        )
    if len(question) > 1000:
        raise InvalidQuestionError(
            f"Question must be at most 1000 characters, got {len(question)}"
        )

    if filters is not None and filters.document_id is not None and filters.filename is not None:
        raise InvalidFiltersError(
            "Only one of document_id or filename may be provided, not both"
        )

    if history is not None:
        for entry in history:
            if entry.role not in ("user", "assistant"):
                raise InvalidHistoryError(
                    f"History role must be 'user' or 'assistant', got '{entry.role}'"
                )
            if not entry.content.strip():
                raise InvalidHistoryError("History entry content must not be empty")
