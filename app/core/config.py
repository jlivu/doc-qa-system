from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Ollama
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_llm_model: str = "qwen2.5:7b"
    ollama_embedding_model: str = "nomic-embed-text"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"

    # Ingestion
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Retrieval
    retrieval_top_k: int = 10

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5

    # API
    cors_origins: list[str] = ["http://localhost:8501"]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Using lru_cache means the .env file is read once per process,
    not on every request.
    """
    return Settings()
