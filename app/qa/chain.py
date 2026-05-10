"""LangChain RAG chain with conversation history support.

answer() passes context and conversation history to the LLM and returns
a grounded answer. Phase 3 can swap in a more sophisticated approach
(MapReduce, Re-rank + Refine) without changing the route.
"""

from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.api.schemas import ConversationTurn, SourceChunk
from app.core.config import Settings
from app.ingestion.exceptions import GenerationError
from app.qa.context import build_context
from app.qa.prompts import RAG_HUMAN_TEMPLATE, RAG_SYSTEM_PROMPT


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

    Args:
        question: The user's question.
        chunks: Retrieved chunks from the retriever.
        history: Prior conversation turns.
        settings: App settings (ollama_llm_model, ollama_base_url).

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

    return {
        "answer": response.content,
        "sources": chunks,
    }
