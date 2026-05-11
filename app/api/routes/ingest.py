"""POST /ingest — async ingestion pipeline.

PDF validation, parsing, and chunking run synchronously (fast, catches
errors immediately). Embedding and storage run in a background task.
"""

import uuid

from fastapi import APIRouter, BackgroundTasks, UploadFile, File, status
from fastapi.responses import JSONResponse

from app.api.schemas import IngestAcceptedResponse
from app.core.dependencies import QdrantDep, SettingsDep
from app.ingestion.exceptions import (
    InvalidFileTypeError,
    FileTooLargeError,
    InvalidPDFError,
    EmptyPDFError,
    IngestionError,
)
from app.ingestion.validator import validate_upload
from app.ingestion.hasher import compute_sha256, find_existing_document
from app.ingestion.parser import extract_text
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import embed_chunks
from app.ingestion.job_store import create_job, update_job
from app.retrieval.vector_store import upsert_chunks, delete_by_document_id

router = APIRouter(prefix="/ingest", tags=["ingestion"])

_STATUS_MAP = {
    InvalidFileTypeError: (415, "INVALID_FILE_TYPE"),
    FileTooLargeError:    (413, "FILE_TOO_LARGE"),
    InvalidPDFError:      (422, "INVALID_PDF"),
    EmptyPDFError:        (422, "EMPTY_PDF"),
}


def _run_ingestion(
    job_id: str,
    document_id: str,
    filename: str,
    sha256: str,
    replaced: bool,
    chunks: list,
    page_count: int,
    qdrant,
    settings,
) -> None:
    """Run embedding + storage in a background thread."""
    update_job(job_id, status="running")
    try:
        embedded = embed_chunks(chunks, settings)
        chunk_count = upsert_chunks(embedded, qdrant, settings)
        update_job(
            job_id,
            status="completed",
            document_id=document_id,
            pages=page_count,
            chunks=chunk_count,
            replaced=replaced,
        )
    except Exception as exc:
        update_job(job_id, status="failed", error=str(exc))


@router.post(
    "",
    response_model=IngestAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a PDF document (async)",
    description=(
        "Upload a PDF file. Validation, parsing and chunking run immediately. "
        "Embedding and storage run in the background. Returns a job_id to poll."
    ),
)
async def ingest_document(
    settings: SettingsDep,
    qdrant: QdrantDep,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to ingest"),
) -> IngestAcceptedResponse:
    try:
        # 1. Read file
        pdf_bytes = await file.read()
        filename = file.filename or "unknown.pdf"

        # 2. Validate type + size (sync — returns 415/413 immediately)
        validate_upload(filename, file.content_type or "", len(pdf_bytes))

        # 3. Hash + dedup (sync)
        sha256 = compute_sha256(pdf_bytes)
        existing_id = find_existing_document(sha256, qdrant, settings)
        if existing_id:
            delete_by_document_id(existing_id, qdrant, settings)
            document_id = existing_id
            replaced = True
        else:
            document_id = str(uuid.uuid4())
            replaced = False

        # 4. Parse + chunk (sync — catches InvalidPDFError, EmptyPDFError)
        pages = extract_text(pdf_bytes)
        chunks = chunk_pages(pages, document_id, filename, sha256, settings)
        if not chunks and pages:
            raise EmptyPDFError("All pages are blank")

        # 5. Create job + launch background task for embedding + storage
        job_id = str(uuid.uuid4())
        create_job(job_id, filename)

        background_tasks.add_task(
            _run_ingestion,
            job_id, document_id, filename, sha256, replaced,
            chunks, len(pages), qdrant, settings,
        )

        return IngestAcceptedResponse(
            job_id=job_id,
            filename=filename,
            status="pending",
            message="Ingestion started. Poll GET /jobs/{job_id} for status.",
        )

    except IngestionError as exc:
        status_code, error_code = _STATUS_MAP.get(
            type(exc), (500, "INGESTION_ERROR")
        )
        return JSONResponse(
            status_code=status_code,
            content={"error": error_code, "detail": str(exc)},
        )
