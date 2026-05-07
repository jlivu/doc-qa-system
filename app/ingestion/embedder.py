"""Embedding chunks via Ollama's OpenAI-compatible embeddings API.

embed_chunks() enriches each ChunkDict with a `vector` field. Chunks are
sent in batches to stay within rate limits and reduce latency.
"""

from typing import TypedDict

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

from app.core.config import Settings
from app.ingestion.exceptions import EmbeddingError
from app.ingestion.chunker import ChunkDict

BATCH_SIZE = 100


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

    Processes chunks in batches of BATCH_SIZE. Uses Ollama's
    OpenAI-compatible endpoint for embeddings.

    Args:
        chunks: Output of chunk_pages().
        settings: App settings (ollama_base_url, ollama_embedding_model).

    Returns:
        List of EmbeddedChunk — same as input but with `vector` added.
    """
    client = OpenAI(
        base_url=settings.ollama_base_url + "/v1",
        api_key="ollama",
    )
    embedded: list[EmbeddedChunk] = []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        try:
            vectors = _embed_batch(texts, client, settings.ollama_embedding_model)
        except RetryError as exc:
            raise EmbeddingError(
                f"Embedding API failed after 3 attempts: {exc}"
            ) from exc

        for chunk, vector in zip(batch, vectors):
            embedded.append({**chunk, "vector": vector})  # type: ignore[misc]

    return embedded
