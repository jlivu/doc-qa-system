"""Custom exceptions for the ingestion pipeline."""


class IngestionError(Exception):
    """Base class for all ingestion pipeline errors."""


class InvalidFileTypeError(IngestionError):
    """File is not application/pdf."""


class FileTooLargeError(IngestionError):
    """File exceeds the maximum allowed size."""


class InvalidPDFError(IngestionError):
    """Bytes do not represent a valid PDF document."""


class EmptyPDFError(IngestionError):
    """PDF has zero pages or all pages are blank."""


class EmbeddingError(IngestionError):
    """OpenAI embedding API failed after all retries."""


class StorageError(IngestionError):
    """Qdrant upsert or delete operation failed."""
