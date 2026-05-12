import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

logger = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import ingest, query, documents, jobs
from app.api.schemas import HealthResponse
from app.core.config import get_settings
from app.core.dependencies import get_qdrant_client

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load reranker and verify Qdrant."""
    # Load reranker singleton
    try:
        from app.retrieval.reranker import load_reranker
        app.state.reranker = load_reranker(settings.reranker_model)
    except Exception as exc:
        logger.warning("Reranker failed to load: %s. Queries will run without reranking.", exc)
        app.state.reranker = None

    # Verify Qdrant
    try:
        client = get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]
        if settings.qdrant_collection not in collections:
            from qdrant_client.models import Distance, VectorParams
            client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
    except Exception:
        pass
    yield


app = FastAPI(
    title="Intelligent Document Q&A System",
    description=(
        "A RAG-powered API that ingests PDF documents and answers natural language "
        "questions grounded in the document content, with source citations."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(jobs.router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health_check() -> HealthResponse:
    """Liveness probe — verifies the API and Qdrant are reachable."""
    try:
        client = get_qdrant_client()
        client.get_collections()
        qdrant_status = "ok"
    except Exception as exc:
        qdrant_status = str(exc)

    return HealthResponse(status="ok", qdrant=qdrant_status)
