"""Context builder for the RAG prompt and not-found handling."""

from app.api.schemas import SourceChunk


def build_context(chunks: list[SourceChunk]) -> str:
    """Format retrieved chunks into a context block for the LLM prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}] {chunk.filename}, page {chunk.page} (score: {chunk.score})\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def is_not_found(chunks: list[SourceChunk], threshold: float = -5.0) -> bool:
    """Return True if no relevant chunks were found.

    After Phase 4 reranking, scores are cross-encoder logits (typically
    -10 to +10). A score below -5.0 means the retrieved chunks are
    genuinely irrelevant. The *threshold* parameter is kept for backward
    compatibility with tests that pass it explicitly.
    """
    if not chunks:
        return True
    return max(c.score for c in chunks) < threshold


def compute_confidence(chunks: list[SourceChunk]) -> str:
    """Derive a confidence level from the top cross-encoder score.

    Cross-encoder scores are raw logits (typically -10 to +10).
    A score above 0 generally indicates relevance.
    """
    if not chunks:
        return "low"
    top_score = max(c.score for c in chunks)
    if top_score >= 3.0:
        return "high"
    if top_score >= 0.0:
        return "medium"
    return "low"


def build_not_found_answer(chunks: list[SourceChunk]) -> str:
    """Build a helpful not-found message, optionally listing related topics."""
    if not chunks:
        return (
            "I could not find any relevant information in the available documents. "
            "Try rephrasing your question or uploading additional documents."
        )
    filenames = sorted({c.filename for c in chunks})
    names = ", ".join(filenames)
    return (
        "I could not find a direct answer to your question in the available documents. "
        f"The documents do contain information about: {names}. "
        "Try rephrasing your question around one of these topics."
    )
