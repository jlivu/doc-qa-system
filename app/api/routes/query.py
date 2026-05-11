"""POST /query — full query pipeline."""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.api.schemas import QueryRequest, QueryResponse, ConversationTurn, SourceChunk
from app.core.dependencies import QdrantDep, SettingsDep
from app.ingestion.exceptions import (
    InvalidQuestionError,
    InvalidFiltersError,
    InvalidHistoryError,
    EmbeddingError,
    RetrievalError,
    GenerationError,
    IngestionError,
)
from app.query.validator import validate_query_request
from app.retrieval.retriever import embed_query
from app.retrieval.vector_store import hybrid_search, SearchResult
from app.qa.chain import answer
from app.qa.context import is_not_found, build_not_found_answer, compute_confidence

router = APIRouter(prefix="/query", tags=["query"])

_STATUS_MAP = {
    InvalidQuestionError: (422, "INVALID_QUESTION"),
    InvalidFiltersError:  (422, "INVALID_FILTERS"),
    InvalidHistoryError:  (422, "INVALID_HISTORY"),
    EmbeddingError:       (500, "EMBEDDING_ERROR"),
    RetrievalError:       (500, "RETRIEVAL_ERROR"),
    GenerationError:      (500, "GENERATION_ERROR"),
}


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query the document knowledge base",
    description=(
        "Ask a natural language question. The system embeds the question, retrieves "
        "the most relevant document chunks from Qdrant using hybrid search, and "
        "passes them to a local LLM to generate a grounded answer with citations."
    ),
)
async def query_documents(
    payload: QueryRequest,
    settings: SettingsDep,
    qdrant: QdrantDep,
) -> QueryResponse:
    try:
        history = payload.conversation_history or []

        # 1. Validate
        validate_query_request(payload.question, payload.filters, history)

        # 2. Embed query
        top_k = payload.top_k or settings.retrieval_top_k
        query_vector = embed_query(payload.question, settings)

        # 3. Hybrid search
        source_chunks = hybrid_search(
            query_vector=query_vector,
            query_text=payload.question,
            client=qdrant,
            settings=settings,
            top_k=top_k,
            document_id=payload.filters.document_id if payload.filters else None,
            filename=payload.filters.filename if payload.filters else None,
        )

        # 4. Not-found check
        if is_not_found(source_chunks):
            answer_text = build_not_found_answer(source_chunks)
            updated_history = list(history) + [
                ConversationTurn(role="user", content=payload.question),
                ConversationTurn(role="assistant", content=answer_text),
            ]
            return QueryResponse(
                answer=answer_text,
                sources=[],
                found=False,
                confidence="low",
                conversation_history=updated_history,
            )

        # 5. Convert SearchResult → SourceChunk
        chunks_as_source = [
            s if isinstance(s, SourceChunk) else SourceChunk(
                document_id=s.document_id,
                filename=s.filename,
                page=s.page_number if isinstance(s, SearchResult) else s.page,
                text=s.text,
                score=round(s.score, 4),
            )
            for s in source_chunks
        ]

        # 6. Generate answer
        result = answer(payload.question, chunks_as_source, history, settings)

        # 7. Normalise sources to SourceChunk
        sources = [
            s if isinstance(s, SourceChunk) else SourceChunk(
                document_id=s.document_id,
                filename=s.filename,
                page=s.page_number if isinstance(s, SearchResult) else s.page,
                text=s.text,
                score=round(s.score, 4),
            )
            for s in result["sources"]
        ]

        # 8. Guard against empty answer
        answer_text = result["answer"]
        if not answer_text or not answer_text.strip():
            answer_text = "I was unable to generate an answer. Please try rephrasing your question."

        # 9. Build updated history
        updated_history = list(history) + [
            ConversationTurn(role="user", content=payload.question),
            ConversationTurn(role="assistant", content=answer_text),
        ]

        confidence = compute_confidence(chunks_as_source)

        return QueryResponse(
            answer=answer_text,
            sources=sources,
            found=True,
            confidence=confidence,
            conversation_history=updated_history,
        )

    except IngestionError as exc:
        status_code, error_code = _STATUS_MAP[type(exc)]
        return JSONResponse(
            status_code=status_code,
            content={"error": error_code, "detail": str(exc)},
        )
