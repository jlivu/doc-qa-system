"""High-level retriever.

retrieve() is the single entry point called by the query route. It embeds
the question, calls the vector store, and returns structured results.

Phase 3 additions (not yet implemented):
    - Hybrid search: combine dense vector search with BM25 keyword scoring
    - Re-ranking: apply a cross-encoder to re-score the top-k candidates
"""

from openai import OpenAI
from qdrant_client import QdrantClient

from app.api.schemas import QueryFilters, SourceChunk
from app.core.config import Settings
from app.retrieval.vector_store import SearchResult, search


def _embed_query(question: str, settings: Settings) -> list[float]:
    """Embed a query string using the same model used during ingestion."""
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        input=question,
        model=settings.embedding_model,
    )
    return response.data[0].embedding


def retrieve(
    question: str,
    filters: QueryFilters | None,
    top_k: int,
    qdrant: QdrantClient,
    settings: Settings,
) -> list[SourceChunk]:
    """Embed a question and return the most relevant document chunks.

    Args:
        question: Natural language query from the user.
        filters: Optional document_id / filename scope.
        top_k: Number of chunks to return.
        qdrant: Shared QdrantClient instance.
        settings: App settings.

    Returns:
        List of SourceChunk, ordered by descending similarity score.
    """
    query_vector = _embed_query(question, settings)

    results: list[SearchResult] = search(
        query_vector=query_vector,
        client=qdrant,
        settings=settings,
        top_k=top_k,
        document_id=filters.document_id if filters else None,
        filename=filters.filename if filters else None,
    )

    return [
        SourceChunk(
            document_id=r.document_id,
            filename=r.filename,
            page=r.page_number,
            text=r.text,
            score=round(r.score, 4),
        )
        for r in results
    ]
