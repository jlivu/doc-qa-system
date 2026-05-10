from typing import Literal

from pydantic import BaseModel, Field


# ── Ingestion ────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    document_id: str = Field(description="UUID assigned to the ingested document")
    filename: str = Field(description="Original filename of the uploaded PDF")
    sha256: str = Field(description="SHA-256 hex digest of the file content")
    pages: int = Field(description="Number of pages parsed")
    chunks: int = Field(description="Number of chunks stored in Qdrant")
    replaced: bool = Field(default=False, description="True if a previous version was replaced")
    message: str = Field(default="Document ingested successfully")


class ErrorResponse(BaseModel):
    error: str = Field(description="Machine-readable error code")
    detail: str = Field(description="Human-readable error description")


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryFilters(BaseModel):
    """Optional metadata filters to narrow the vector search."""
    document_id: str | None = Field(
        default=None,
        description="Restrict search to a specific document"
    )
    filename: str | None = Field(
        default=None,
        description="Restrict search to documents with this filename"
    )


class ConversationTurn(BaseModel):
    role: str = Field(description="Must be 'user' or 'assistant'")
    content: str = Field(min_length=1)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    filters: QueryFilters | None = Field(default=None)
    top_k: int | None = Field(default=None, ge=1, le=20)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)


class SourceChunk(BaseModel):
    """A retrieved document chunk returned alongside the answer."""
    document_id: str
    filename: str
    page: int
    text: str = Field(description="The chunk text used as context")
    score: float = Field(description="Similarity score")


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    found: bool
    conversation_history: list[ConversationTurn]


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    qdrant: str = Field(description="'ok' or error message")
