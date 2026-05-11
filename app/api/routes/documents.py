"""GET /documents and DELETE /documents/{document_id}."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.schemas import (
    DocumentListResponse,
    DocumentMetadataResponse,
    DeleteDocumentResponse,
)
from app.core.dependencies import QdrantDep, SettingsDep
from app.ingestion.exceptions import StorageError
from app.retrieval.vector_store import (
    list_documents,
    find_document_by_id,
    delete_by_document_id,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
async def list_all_documents(
    settings: SettingsDep,
    qdrant: QdrantDep,
) -> DocumentListResponse:
    """Return a deduplicated list of all ingested documents."""
    try:
        docs = list_documents(qdrant, settings)
        return DocumentListResponse(
            documents=[DocumentMetadataResponse(**d) for d in docs],
            total=len(docs),
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": "RETRIEVAL_ERROR", "detail": str(exc)},
        )


@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(
    document_id: str,
    settings: SettingsDep,
    qdrant: QdrantDep,
) -> DeleteDocumentResponse:
    """Remove all chunks belonging to a document."""
    try:
        if not find_document_by_id(document_id, qdrant, settings):
            return JSONResponse(
                status_code=404,
                content={
                    "error": "DOCUMENT_NOT_FOUND",
                    "detail": f"No document found with id {document_id}",
                },
            )
        delete_by_document_id(document_id, qdrant, settings)
        return DeleteDocumentResponse(document_id=document_id)
    except StorageError as exc:
        return JSONResponse(
            status_code=500,
            content={"error": "STORAGE_ERROR", "detail": str(exc)},
        )
