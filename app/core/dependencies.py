from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from qdrant_client import QdrantClient

from app.core.config import Settings, get_settings


@lru_cache
def get_qdrant_client(settings: Settings = get_settings()) -> QdrantClient:
    """Return a cached Qdrant client.

    A single client is shared across all requests rather than
    opening a new connection on every call.
    """
    return QdrantClient(url=settings.qdrant_url)


# Convenience type aliases for use in route signatures
SettingsDep = Annotated[Settings, Depends(get_settings)]
QdrantDep = Annotated[QdrantClient, Depends(get_qdrant_client)]
