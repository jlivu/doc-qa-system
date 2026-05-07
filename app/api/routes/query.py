from fastapi import APIRouter, HTTPException, status

from app.api.schemas import QueryRequest, QueryResponse
from app.core.dependencies import QdrantDep, SettingsDep

router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "",
    response_model=QueryResponse,
    summary="Query the document knowledge base",
    description=(
        "Ask a natural language question. The system embeds the question, retrieves "
        "the most relevant document chunks from Qdrant, and passes them to an LLM "
        "to generate a grounded answer with source citations."
    ),
)
async def query_documents(
    request: QueryRequest,
    settings: SettingsDep,
    qdrant: QdrantDep,
) -> QueryResponse:

    # ── Phase 2 implementation goes here ────────────────────────────────────
    # from app.retrieval.retriever import retrieve
    # from app.qa.chain import get_chain
    #
    # top_k   = request.top_k or settings.retrieval_top_k
    # chunks  = retrieve(request.question, request.filters, top_k, qdrant, settings)
    # chain   = get_chain(settings)
    # result  = chain.invoke({"question": request.question, "chunks": chunks})
    #
    # return QueryResponse(
    #     answer=result["answer"],
    #     sources=result["sources"],
    #     question=request.question,
    # )
    # ────────────────────────────────────────────────────────────────────────

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Query pipeline not yet implemented — coming in Phase 2",
    )
