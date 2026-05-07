"""POST /ingest — full ingestion pipeline."""

import uuid

from fastapi import APIRouter, UploadFile, File, status
from fastapi.responses import JSONResponse

from app.api.schemas import IngestResponse
from app.core.dependencies import QdrantDep, SettingsDep
from app.ingestion.exceptions import (
    InvalidFileTypeError,
    FileTooLargeError,
    InvalidPDFError,
    EmptyPDFError,
    EmbeddingError,
    StorageError,
    IngestionError,
)
from app.ingestion.validator import validate_upload
from app.ingestion.hasher import compute_sha256, find_existing_document
from app.ingestion.parser import extract_text
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import embed_chunks
from app.retrieval.vector_store import upsert_chunks, delete_by_document_id

router = APIRouter(prefix="/ingest", tags=["ingestion"])

_STATUS_MAP = {
    InvalidFileTypeError: (415, "INVALID_FILE_TYPE"),
    FileTooLargeError:    (413, "FILE_TOO_LARGE"),
    InvalidPDFError:      (422, "INVALID_PDF"),
    EmptyPDFError:        (422, "EMPTY_PDF"),
    EmbeddingError:       (500, "EMBEDDING_ERROR"),
    StorageError:         (500, "STORAGE_ERROR"),
}


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
    try:
        # 1. Read file
        pdf_bytes = await file.read()
        filename = file.filename or "unknown.pdf"

        # 2. Validate
        validate_upload(filename, file.content_type or "", len(pdf_bytes))

        # 3. Hash
        sha256 = compute_sha256(pdf_bytes)

        # 4. Deduplicate
        existing_id = find_existing_document(sha256, qdrant, settings)
        if existing_id:
            delete_by_document_id(existing_id, qdrant, settings)
            document_id = existing_id
            replaced = True
        else:
            document_id = str(uuid.uuid4())
            replaced = False

        # 5. Parse
        pages = extract_text(pdf_bytes)

        # 6. Chunk
        chunks = chunk_pages(pages, document_id, filename, sha256, settings)

        # 7. Check for all-blank PDF
        if not chunks and pages:
            raise EmptyPDFError("All pages are blank")

        # 8. Embed
        embedded = embed_chunks(chunks, settings)

        # 9. Store
        stored = upsert_chunks(embedded, qdrant, settings)

        # 10. Respond
        return IngestResponse(
            document_id=document_id,
            filename=filename,
            sha256=sha256,
            pages=len(pages),
            chunks=stored,
            replaced=replaced,
        )

    except IngestionError as exc:
        status_code, error_code = _STATUS_MAP[type(exc)]
        return JSONResponse(
            status_code=status_code,
            content={"error": error_code, "detail": str(exc)},
        )
