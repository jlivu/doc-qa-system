"""LangChain RAG chain with conversation history and highlight extraction.

answer() passes context and conversation history to the LLM, parses
HIGHLIGHT[N]: markers from the response, strips them from the visible
answer, and sets chunk.highlight on each source.
"""

import re
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.api.schemas import ConversationTurn, SourceChunk
from app.core.config import Settings
from app.ingestion.exceptions import GenerationError
from app.qa.context import build_context
from app.qa.prompts import RAG_HUMAN_TEMPLATE, RAG_SYSTEM_PROMPT

HIGHLIGHT_RE = re.compile(r"HIGHLIGHT\[(\d+)\]:\s*(.+)")


@lru_cache
def _get_llm(model: str, base_url: str) -> ChatOllama:
    """Return a cached LLM instance."""
    return ChatOllama(model=model, base_url=base_url, temperature=0)


def answer(
    question: str,
    chunks: list[SourceChunk],
    history: list[ConversationTurn],
    settings: Settings,
) -> dict:
    """Run the RAG chain and return the answer with sources.

    Parses HIGHLIGHT[N]: markers from the LLM response, strips them
    from the visible answer text, and sets chunk.highlight on each
    matching source.

    Returns:
        dict with keys 'answer' (str) and 'sources' (list[SourceChunk]).

    Raises:
        GenerationError: If the LLM call fails.
    """
    context = build_context(chunks)
    llm = _get_llm(settings.ollama_llm_model, settings.ollama_base_url)

    messages: list = [
        SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context)),
    ]

    for turn in history:
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        else:
            messages.append(AIMessage(content=turn.content))

    messages.append(HumanMessage(content=RAG_HUMAN_TEMPLATE.format(question=question)))

    try:
        response = llm.invoke(messages)
    except Exception as exc:
        raise GenerationError(f"LLM generation failed: {exc}") from exc

    # Parse HIGHLIGHT[N]: markers and strip them from the answer
    raw_answer = response.content
    highlights: dict[int, str] = {}
    clean_lines: list[str] = []

    for line in raw_answer.split("\n"):
        m = HIGHLIGHT_RE.match(line.strip())
        if m:
            source_idx = int(m.group(1))
            highlights[source_idx] = m.group(2).strip()
        else:
            clean_lines.append(line)

    clean_answer = "\n".join(clean_lines).strip()
    if not clean_answer:
        clean_answer = "I was unable to generate an answer. Please try rephrasing your question."

    # Apply highlights to source chunks (1-indexed)
    for i, chunk in enumerate(chunks, start=1):
        if i in highlights:
            chunk.highlight = highlights[i]

    return {
        "answer": clean_answer,
        "sources": chunks,
    }
