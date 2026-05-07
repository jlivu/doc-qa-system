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
    page_number: int    # Source page (1-indexed)
    text: str           # Chunk text to embed


def chunk_pages(
    pages: list[PageDict],
    document_id: str,
    filename: str,
    settings: Settings,
) -> list[ChunkDict]:
    """Split parsed pages into overlapping chunks.

    Blank pages are skipped. Each chunk carries the document_id, filename,
    and source page number as metadata for downstream filtering and citation.

    Args:
        pages: Output of extract_text().
        document_id: UUID of the parent document.
        filename: Original filename, stored for citation.
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
        if page["char_count"] < 10:
            continue  # skip blank / near-blank pages

        texts = splitter.split_text(page["text"])
        for text in texts:
            if not text.strip():
                continue
            chunks.append(
                ChunkDict(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    filename=filename,
                    page_number=page["page_number"],
                    text=text.strip(),
                )
            )

    return chunks
