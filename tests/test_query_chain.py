"""Tests for the RAG chain answer() with conversation history (Phase 2).

ChatOllama is mocked via @patch("app.qa.chain._get_llm") to avoid
requiring a running Ollama instance.
Imports are inline so Phase 1 test collection is not blocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.api.schemas import SourceChunk
from app.core.config import Settings


def _make_settings() -> Settings:
    return Settings()


def _make_chunk(text: str = "Relevant document text.") -> SourceChunk:
    return SourceChunk(
        document_id="doc-1",
        filename="budget.pdf",
        page=3,
        text=text,
        score=0.85,
    )


# AC-CHAIN-01 — Answer is a non-empty string
@patch("app.qa.chain._get_llm")
def test_answer_returns_non_empty_string(mock_get_llm):
    from app.qa.chain import answer
    from app.api.schemas import ConversationTurn  # noqa: F401 — Phase 2

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="The answer is 42.")
    mock_get_llm.return_value = mock_llm

    result = answer("What is it?", [_make_chunk()], [], _make_settings())
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0


# AC-CHAIN-02 — Returned sources match the input chunks
@patch("app.qa.chain._get_llm")
def test_answer_returns_input_chunks_as_sources(mock_get_llm):
    from app.qa.chain import answer
    from app.api.schemas import ConversationTurn  # noqa: F401

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Answer text.")
    mock_get_llm.return_value = mock_llm

    chunks = [_make_chunk("First chunk."), _make_chunk("Second chunk.")]
    result = answer("Question?", chunks, [], _make_settings())
    assert result["sources"] == chunks


# AC-CHAIN-03 — Conversation history is included in the LLM message list
@patch("app.qa.chain._get_llm")
def test_answer_includes_history_in_messages(mock_get_llm):
    from app.qa.chain import answer
    from app.api.schemas import ConversationTurn

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Follow-up answer.")
    mock_get_llm.return_value = mock_llm

    history = [
        ConversationTurn(role="user", content="What is this about?"),
        ConversationTurn(role="assistant", content="This is about budgets."),
    ]
    answer("Follow-up question?", [_make_chunk()], history, _make_settings())

    messages = mock_llm.invoke.call_args[0][0]
    # Messages: SystemMessage, HumanMessage (history), AIMessage (history), HumanMessage (question)
    assert len(messages) >= 4
    contents = [m.content for m in messages]
    assert "What is this about?" in contents
    assert "This is about budgets." in contents


# AC-CHAIN-04 — GenerationError is raised on LLM failure
@patch("app.qa.chain._get_llm")
def test_answer_raises_generation_error(mock_get_llm):
    from app.qa.chain import answer
    from app.ingestion.exceptions import GenerationError

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("Ollama down")
    mock_get_llm.return_value = mock_llm

    with pytest.raises(GenerationError):
        answer("Question?", [_make_chunk()], [], _make_settings())


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Highlight extraction tests (AC-QUAL-05, AC-QUAL-06, AC-QUAL-07)
# ══════════════════════════════════════════════════════════════════════════════

# AC-QUAL-05 — SourceChunk includes a highlight field
@patch("app.qa.chain._get_llm")
def test_source_chunk_has_highlight_field(mock_get_llm):
    from app.qa.chain import answer

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="The answer is here.\nHIGHLIGHT[1]: The key sentence from source one."
    )
    mock_get_llm.return_value = mock_llm

    chunks = [_make_chunk("Source one text.")]
    result = answer("Question?", chunks, [], _make_settings())
    assert result["sources"][0].highlight == "The key sentence from source one."


# AC-QUAL-06 — highlight is None when the LLM returns no highlight marker
@patch("app.qa.chain._get_llm")
def test_highlight_is_none_when_no_marker(mock_get_llm):
    from app.qa.chain import answer

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Just a plain answer.")
    mock_get_llm.return_value = mock_llm

    chunks = [_make_chunk("Source text.")]
    result = answer("Question?", chunks, [], _make_settings())
    assert result["sources"][0].highlight is None


# AC-QUAL-07 — The visible answer text does not contain raw HIGHLIGHT[N]: markers
@patch("app.qa.chain._get_llm")
def test_answer_text_has_no_highlight_markers(mock_get_llm):
    from app.qa.chain import answer

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="The budget is 12B vatu.\nHIGHLIGHT[1]: Budget is 12B vatu.\nHIGHLIGHT[2]: Revenue was 10B."
    )
    mock_get_llm.return_value = mock_llm

    chunks = [_make_chunk("First."), _make_chunk("Second.")]
    result = answer("Question?", chunks, [], _make_settings())
    assert "HIGHLIGHT[" not in result["answer"]
    assert "The budget is 12B vatu." in result["answer"]
