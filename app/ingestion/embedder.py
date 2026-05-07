"""Embedding chunks via the OpenAI embeddings API.

embed_chunks() enriches each ChunkDict with a `vector` field. Chunks are
sent in batches to stay within API rate limits and reduce latency.
"""

from typing import TypedDict

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import Settings
from app.ingestion.chunker import ChunkDict

BATCH_SIZE = 100  # OpenAI allows up to 2048 inputs per request


class EmbeddedChunk(ChunkDict, total=False):
    vector: list[float]   # Dense embedding vector added by this module


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def _embed_batch(texts: list[str], client: OpenAI, model: str) -> list[list[float]]:
    """Embed a single batch of texts, with retry on transient errors."""
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def embed_chunks(
    chunks: list[ChunkDict],
    settings: Settings,
) -> list[EmbeddedChunk]:
    """Add an embedding vector to every chunk.

    Processes chunks in batches of BATCH_SIZE. The OpenAI client is
    instantiated here using the API key from settings.

    Args:
        chunks: Output of chunk_pages().
        settings: App settings (openai_api_key, embedding_model).

    Returns:
        List of EmbeddedChunk — same as input but with `vector` added.
    """
    client = OpenAI(api_key=settings.openai_api_key)
    embedded: list[EmbeddedChunk] = []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vectors = _embed_batch(texts, client, settings.embedding_model)

        for chunk, vector in zip(batch, vectors):
            embedded.append({**chunk, "vector": vector})  # type: ignore[misc]

    return embedded
