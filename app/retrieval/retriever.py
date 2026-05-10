"""High-level retriever.

embed_query() produces the dense vector for a user question.
retrieve() orchestrates embedding + hybrid search and returns SourceChunks.
"""

from openai import OpenAI
from qdrant_client import QdrantClient

from app.api.schemas import QueryFilters, SourceChunk
from app.core.config import Settings
from app.ingestion.exceptions import EmbeddingError
from app.retrieval.vector_store import SearchResult, hybrid_search


def embed_query(question: str, settings: Settings) -> list[float]:
    """Embed a query string using Ollama's OpenAI-compatible endpoint.

    Args:
        question: The user's natural language question.
        settings: App settings (ollama_base_url, ollama_embedding_model).

    Returns:
        List of floats — the dense embedding vector.

    Raises:
        EmbeddingError: On any failure (no retry — Ollama is local).
    """
    try:
        client = OpenAI(
            base_url=settings.ollama_base_url + "/v1",
            api_key="ollama",
        )
        response = client.embeddings.create(
            input=question,
            model=settings.ollama_embedding_model,
        )
        return response.data[0].embedding
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError(f"Query embedding failed: {exc}") from exc


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
        List of SourceChunk, ordered by descending score.
    """
    query_vector = embed_query(question, settings)

    results: list[SearchResult] = hybrid_search(
        query_vector=query_vector,
        query_text=question,
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
