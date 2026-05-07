import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File, status

from app.api.schemas import IngestResponse
from app.core.dependencies import QdrantDep, SettingsDep

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a PDF document",
    description=(
        "Upload a PDF file. The system parses the text, splits it into overlapping "
        "chunks, embeds each chunk, and stores the vectors in Qdrant. Returns a "
        "document_id that can be used to scope future queries to this document."
    ),
)
async def ingest_document(
    settings: SettingsDep,
    qdrant: QdrantDep,
    file: UploadFile = File(..., description="PDF file to ingest"),
) -> IngestResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Expected application/pdf, got {file.content_type}",
        )

    document_id = str(uuid.uuid4())
    pdf_bytes = await file.read()

    # ── Phase 1 implementation goes here ────────────────────────────────────
    # from app.ingestion.parser import extract_text
    # from app.ingestion.chunker import chunk_pages
    # from app.ingestion.embedder import embed_chunks
    # from app.retrieval.vector_store import upsert_chunks
    #
    # pages      = extract_text(pdf_bytes)
    # chunks     = chunk_pages(pages, document_id, file.filename, settings)
    # embedded   = embed_chunks(chunks, settings)
    # upsert_chunks(embedded, qdrant, settings)
    # ────────────────────────────────────────────────────────────────────────

    # Skeleton response — replace with real values once ingestion is wired up
    return IngestResponse(
        document_id=document_id,
        filename=file.filename or "unknown.pdf",
        pages=0,
        chunks=0,
        message="[Skeleton] Ingestion pipeline not yet implemented",
    )
