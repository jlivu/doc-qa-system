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


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000,
        description="The natural language question to answer"
    )
    filters: QueryFilters | None = Field(
        default=None,
        description="Optional metadata filters"
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of chunks to retrieve. Overrides server default."
    )


class SourceChunk(BaseModel):
    """A retrieved document chunk returned alongside the answer."""
    document_id: str
    filename: str
    page: int
    text: str = Field(description="The chunk text used as context")
    score: float = Field(description="Cosine similarity score (0–1)")


class QueryResponse(BaseModel):
    answer: str = Field(description="LLM-generated answer grounded in retrieved chunks")
    sources: list[SourceChunk] = Field(description="Chunks used to generate the answer")
    question: str = Field(description="The original question, echoed back")


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    qdrant: str = Field(description="'ok' or error message")
