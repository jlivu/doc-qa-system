from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from qdrant_client import QdrantClient

from app.core.config import Settings, get_settings

_reranker_instance = None


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Return a cached Qdrant client.

    A single client is shared across all requests rather than
    opening a new connection on every call.
    """
    return QdrantClient(url=get_settings().qdrant_url)


def set_reranker(model) -> None:
    """Store the reranker model at startup. Called from lifespan."""
    global _reranker_instance
    _reranker_instance = model


def get_reranker():
    """Return the reranker model, or None if it failed to load."""
    return _reranker_instance


# Convenience type aliases for use in route signatures
SettingsDep = Annotated[Settings, Depends(get_settings)]
QdrantDep = Annotated[QdrantClient, Depends(get_qdrant_client)]
RerankerDep = Annotated[object, Depends(get_reranker)]
