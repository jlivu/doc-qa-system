"""Qdrant vector store wrapper.

All Qdrant-specific code lives here. To swap to a different vector database
(pgvector, Milvus, etc.), replace this file only — the interface stays the same.

Public functions:
    upsert_chunks()       — write embedded chunks to Qdrant
    search()              — dense-only nearest-neighbour search
    hybrid_search()       — dense + sparse BM25 search with RRF fusion
    delete_by_document_id()— remove all chunks belonging to a document_id
    text_to_sparse_vector()— convert text to a sparse BM25-style vector
"""

import logging
import math
import re
from collections import Counter

logger = logging.getLogger(__name__)

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    NamedSparseVector,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.core.config import Settings
from app.ingestion.embedder import EmbeddedChunk
from app.ingestion.exceptions import StorageError

RRF_K = 60


class SearchResult:
    """Thin wrapper around a Qdrant search hit."""

    def __init__(self, point):
        self.chunk_id: str = str(point.id)
        self.score: float = point.score
        self.document_id: str = point.payload.get("document_id", "")
        self.filename: str = point.payload.get("filename", "")
        self.page_number: int = point.payload.get("page_number", 0)
        self.text: str = point.payload.get("text", "")


# ── Sparse vector helpers ────────────────────────────────────────────────────

def text_to_sparse_vector(text: str) -> SparseVector:
    """Convert text to a sparse BM25-style vector using term frequency.

    Each token is mapped to an integer index via hash(token) % 2^31.
    Weights use log(1 + tf) for sublinear term-frequency scaling.
    """
    tokens = re.findall(r"\w+", text.lower())
    tf = Counter(tokens)
    indices = []
    values = []
    for token, count in tf.items():
        indices.append(hash(token) % (2**31))
        values.append(math.log(1 + count))
    return SparseVector(indices=indices, values=values)


# ── Collection management ────────────────────────────────────────────────────

def _ensure_collection(client: QdrantClient, settings: Settings, vector_size: int = 1536) -> None:
    """Create the Qdrant collection with dense + sparse vectors if it does not exist."""
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                "dense": VectorParams(size=vector_size, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )


# ── Filter builder ───────────────────────────────────────────────────────────

def _build_filter(
    document_id: str | None = None,
    filename: str | None = None,
) -> Filter | None:
    must_conditions = []
    if document_id:
        must_conditions.append(
            FieldCondition(key="document_id", match=MatchValue(value=document_id))
        )
    if filename:
        must_conditions.append(
            FieldCondition(key="filename", match=MatchValue(value=filename))
        )
    return Filter(must=must_conditions) if must_conditions else None


# ── Upsert ───────────────────────────────────────────────────────────────────

def upsert_chunks(
    chunks: list[EmbeddedChunk],
    client: QdrantClient,
    settings: Settings,
) -> int:
    """Write embedded chunks to Qdrant with dense + sparse vectors."""
    if not chunks:
        return 0

    vector_size = len(chunks[0]["vector"])
    _ensure_collection(client, settings, vector_size)

    points = [
        PointStruct(
            id=chunk["chunk_id"],
            vector={
                "dense": chunk["vector"],
                "sparse": text_to_sparse_vector(chunk["text"]),
            },
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


# ── Dense-only search (Phase 1 legacy) ───────────────────────────────────────

def search(
    query_vector: list[float],
    client: QdrantClient,
    settings: Settings,
    top_k: int | None = None,
    document_id: str | None = None,
    filename: str | None = None,
) -> list[SearchResult]:
    """Nearest-neighbour search using dense vectors only."""
    k = top_k or settings.retrieval_top_k
    query_filter = _build_filter(document_id, filename)

    hits = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=("dense", query_vector),
        query_filter=query_filter,
        limit=k,
        with_payload=True,
    )

    return [SearchResult(hit) for hit in hits]


# ── RRF merge ────────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """Merge two ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    results_by_id: dict[str, SearchResult] = {}

    for rank, r in enumerate(dense_results, start=1):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1.0 / (RRF_K + rank)
        results_by_id[r.chunk_id] = r

    for rank, r in enumerate(sparse_results, start=1):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1.0 / (RRF_K + rank)
        results_by_id[r.chunk_id] = r

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]

    merged = []
    for cid in sorted_ids:
        result = results_by_id[cid]
        result.score = round(scores[cid], 4)
        merged.append(result)
    return merged


# ── Hybrid search ────────────────────────────────────────────────────────────

def hybrid_search(
    query_vector: list[float],
    query_text: str,
    client: QdrantClient,
    settings: Settings,
    top_k: int,
    document_id: str | None = None,
    filename: str | None = None,
) -> list[SearchResult]:
    """Dense + sparse hybrid search with RRF fusion.

    Falls back to dense-only if the sparse index does not exist.
    """
    query_filter = _build_filter(document_id, filename)
    fetch_limit = top_k * 2

    # Dense search
    dense_hits = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=("dense", query_vector),
        query_filter=query_filter,
        limit=fetch_limit,
        with_payload=True,
    )
    dense_results = [SearchResult(h) for h in dense_hits]
    logger.debug("hybrid_search: dense returned %d results", len(dense_results))

    # Sparse search — fall back to dense-only on failure
    try:
        sparse_hits = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=NamedSparseVector(
                name="sparse",
                vector=text_to_sparse_vector(query_text),
            ),
            query_filter=query_filter,
            limit=fetch_limit,
            with_payload=True,
        )
        sparse_results = [SearchResult(h) for h in sparse_hits]
        logger.debug("hybrid_search: sparse returned %d results", len(sparse_results))
    except Exception:
        logger.debug("hybrid_search: sparse search failed, falling back to dense-only")
        for r in dense_results:
            r.score = round(r.score, 4)
        return dense_results[:top_k]

    return _reciprocal_rank_fusion(dense_results, sparse_results, top_k)


# ── Delete ───────────────────────────────────────────────────────────────────

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
