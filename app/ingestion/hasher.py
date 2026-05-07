"""SHA-256 hashing and deduplication lookup."""

import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.core.config import Settings


def compute_sha256(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def find_existing_document(
    sha256: str,
    client: QdrantClient,
    settings: Settings,
) -> str | None:
    """Search Qdrant for a point whose payload sha256 matches.

    Returns the document_id if found, None otherwise.
    Does not raise if the collection does not exist yet.
    """
    try:
        points, _ = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="sha256", match=MatchValue(value=sha256))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        return None

    if points:
        return points[0].payload["document_id"]
    return None
