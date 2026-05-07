"""Qdrant vector store wrapper.

All Qdrant-specific code lives here. To swap to a different vector database
(pgvector, Milvus, etc.), replace this file only — the interface stays the same.

Public functions:
    upsert_chunks()       — write embedded chunks to Qdrant
    search()              — nearest-neighbour search with optional metadata filtering
    delete_by_document_id()— remove all chunks belonging to a document_id
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.config import Settings
from app.ingestion.embedder import EmbeddedChunk
from app.ingestion.exceptions import StorageError


class SearchResult:
    """Thin wrapper around a Qdrant search hit."""

    def __init__(self, point):
        self.chunk_id: str = str(point.id)
        self.score: float = point.score
        self.document_id: str = point.payload.get("document_id", "")
        self.filename: str = point.payload.get("filename", "")
        self.page_number: int = point.payload.get("page_number", 0)
        self.text: str = point.payload.get("text", "")


def _ensure_collection(client: QdrantClient, settings: Settings, vector_size: int = 1536) -> None:
    """Create the Qdrant collection if it does not already exist."""
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def upsert_chunks(
    chunks: list[EmbeddedChunk],
    client: QdrantClient,
    settings: Settings,
) -> int:
    """Write embedded chunks to Qdrant.

    Each chunk is stored as a point: the vector is the dense embedding,
    and the payload carries all metadata needed for citation.

    Args:
        chunks: Output of embed_chunks().
        client: Shared QdrantClient instance.
        settings: App settings (collection name).

    Returns:
        Number of points upserted.

    Raises:
        StorageError: If the Qdrant upsert fails.
    """
    if not chunks:
        return 0

    vector_size = len(chunks[0]["vector"])
    _ensure_collection(client, settings, vector_size)

    points = [
        PointStruct(
            id=chunk["chunk_id"],
            vector=chunk["vector"],
            payload={
                "document_id": chunk["document_id"],
                "filename": chunk["filename"],
                "sha256": chunk["sha256"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
            },
        )
        for chunk in chunks
    ]

    try:
        client.upsert(collection_name=settings.qdrant_collection, points=points)
    except Exception as exc:
        raise StorageError(f"Qdrant upsert failed: {exc}") from exc
    return len(points)


def search(
    query_vector: list[float],
    client: QdrantClient,
    settings: Settings,
    top_k: int | None = None,
    document_id: str | None = None,
    filename: str | None = None,
) -> list[SearchResult]:
    """Nearest-neighbour search with optional metadata filtering.

    Args:
        query_vector: Embedded query vector (same model as ingestion).
        client: Shared QdrantClient instance.
        settings: App settings.
        top_k: Number of results to return. Defaults to settings.retrieval_top_k.
        document_id: Filter results to a specific document.
        filename: Filter results to documents with this filename.

    Returns:
        List of SearchResult ordered by descending similarity score.
    """
    k = top_k or settings.retrieval_top_k

    must_conditions = []
    if document_id:
        must_conditions.append(
            FieldCondition(key="document_id", match=MatchValue(value=document_id))
        )
    if filename:
        must_conditions.append(
            FieldCondition(key="filename", match=MatchValue(value=filename))
        )

    query_filter = Filter(must=must_conditions) if must_conditions else None

    hits = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=k,
        with_payload=True,
    )

    return [SearchResult(hit) for hit in hits]


def delete_by_document_id(
    document_id: str,
    client: QdrantClient,
    settings: Settings,
) -> None:
    """Remove all chunks belonging to a document_id from Qdrant."""
    try:
        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
        )
    except Exception as exc:
        raise StorageError(f"Qdrant delete failed: {exc}") from exc
