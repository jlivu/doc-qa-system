"""Upload validation — file type and size checks."""

from app.ingestion.exceptions import InvalidFileTypeError, FileTooLargeError

MAX_FILE_SIZE_BYTES = 52_428_800  # 50 * 1024 * 1024
ALLOWED_CONTENT_TYPE = "application/pdf"


def validate_upload(filename: str, content_type: str, size_bytes: int) -> None:
    """Validate file metadata before any processing begins.

    Args:
        filename: Original filename (accepted for future use, not checked).
        content_type: MIME type reported by the client.
        size_bytes: File size in bytes.

    Raises:
        InvalidFileTypeError: If content_type is not application/pdf.
        FileTooLargeError: If size_bytes exceeds 50 MB.
    """
    if content_type != ALLOWED_CONTENT_TYPE:
        raise InvalidFileTypeError(
            f"Expected application/pdf, received {content_type}"
        )
    if size_bytes > MAX_FILE_SIZE_BYTES:
        raise FileTooLargeError(
            f"File size {size_bytes} bytes exceeds maximum {MAX_FILE_SIZE_BYTES} bytes"
        )
