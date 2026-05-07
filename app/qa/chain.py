"""LangChain RAG chain.

build_chain() returns a callable that accepts a question and a list of
SourceChunks and returns a dict with 'answer' and 'sources'.

The chain is intentionally simple for Phase 2: stuff the retrieved chunks
into the prompt and call the LLM. Phase 3 can swap in a more sophisticated
approach (MapReduce, Re-rank + Refine) without changing the route.
"""

from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.api.schemas import SourceChunk
from app.core.config import Settings
from app.qa.prompts import RAG_HUMAN_TEMPLATE, RAG_SYSTEM_PROMPT


def _build_context(chunks: list[SourceChunk]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}] {chunk.filename}, page {chunk.page}\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


@lru_cache
def _get_llm(model: str, api_key: str) -> ChatOpenAI:
    """Return a cached LLM instance. Re-creating on every request wastes resources."""
    return ChatOpenAI(model=model, api_key=api_key, temperature=0)


def answer(
    question: str,
    chunks: list[SourceChunk],
    settings: Settings,
) -> dict:
    """Run the RAG chain and return the answer with sources.

    Args:
        question: The user's question.
        chunks: Retrieved chunks from the retriever.
        settings: App settings (llm_model, openai_api_key).

    Returns:
        dict with keys 'answer' (str) and 'sources' (list[SourceChunk]).
    """
    if not chunks:
        return {
            "answer": "I could not find any relevant documents to answer that question.",
            "sources": [],
        }

    context = _build_context(chunks)
    llm = _get_llm(settings.llm_model, settings.openai_api_key)

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=RAG_HUMAN_TEMPLATE.format(question=question)),
    ]

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": chunks,
    }
