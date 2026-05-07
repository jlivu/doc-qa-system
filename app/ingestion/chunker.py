"""Text chunking.

chunk_pages() splits page text into overlapping chunks suitable for embedding.
Using RecursiveCharacterTextSplitter means splits prefer paragraph and sentence
boundaries over hard character cuts.
"""

import uuid
from typing import TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings
from app.ingestion.parser import PageDict


class ChunkDict(TypedDict):
    chunk_id: str       # UUID for this individual chunk
    document_id: str    # UUID of the parent document
    filename: str       # Original filename — stored as metadata in Qdrant
    sha256: str         # SHA-256 of the parent document
    page_number: int    # Source page (1-indexed)
    text: str           # Chunk text to embed
    char_count: int     # len(text)


def chunk_pages(
    pages: list[PageDict],
    document_id: str,
    filename: str,
    sha256: str,
    settings: Settings,
) -> list[ChunkDict]:
    """Split parsed pages into overlapping chunks.

    Pages with fewer than 20 characters are skipped. Chunks shorter
    than 10 characters after stripping are discarded.

    Args:
        pages: Output of extract_text().
        document_id: UUID of the parent document.
        filename: Original filename, stored for citation.
        sha256: SHA-256 hex digest of the parent document.
        settings: App settings (chunk_size, chunk_overlap).

    Returns:
        List of ChunkDict ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[ChunkDict] = []
    for page in pages:
        if page["char_count"] < 20:
            continue  # skip blank / near-blank pages

        texts = splitter.split_text(page["text"])
        for text in texts:
            stripped = text.strip()
            if len(stripped) < 10:
                continue
            chunks.append(
                ChunkDict(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    filename=filename,
                    sha256=sha256,
                    page_number=page["page_number"],
                    text=stripped,
                    char_count=len(stripped),
                )
            )

    return chunks
