"""Cross-encoder reranker for improving retrieval quality."""

from sentence_transformers import CrossEncoder

from app.api.schemas import SourceChunk


def load_reranker(model_name: str) -> CrossEncoder:
    """Load and return the cross-encoder model."""
    return CrossEncoder(model_name)


def rerank(
    question: str,
    chunks: list[SourceChunk],
    top_k: int,
    reranker: CrossEncoder,
) -> list[SourceChunk]:
    """Rerank chunks by cross-encoder score and return top_k."""
    if not chunks:
        return []
    pairs = [(question, c.text) for c in chunks]
    scores = reranker.predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)
    ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
    return ranked[:top_k]
