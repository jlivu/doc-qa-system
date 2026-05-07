"""Tests for the upload validator (Phase 1).

No mocking needed — validate_upload is a pure function with no external calls.
"""

import pytest

from app.ingestion.validator import validate_upload, MAX_FILE_SIZE_BYTES
from app.ingestion.exceptions import InvalidFileTypeError, FileTooLargeError


# AC-VAL-01 — Uploading a non-PDF file returns HTTP 415 / INVALID_FILE_TYPE
def test_validate_rejects_non_pdf():
    with pytest.raises(InvalidFileTypeError, match="text/plain"):
        validate_upload("doc.txt", "text/plain", 100)


# AC-VAL-02 — Uploading a file larger than 50 MB returns HTTP 413 / FILE_TOO_LARGE
def test_validate_rejects_oversized_file():
    with pytest.raises(FileTooLargeError):
        validate_upload("big.pdf", "application/pdf", MAX_FILE_SIZE_BYTES + 1)


# AC-VAL-03 — A valid PDF under 50 MB passes validation without error
def test_validate_accepts_valid_pdf():
    assert validate_upload("ok.pdf", "application/pdf", 1000) is None


def test_validate_accepts_exactly_max_size():
    """Boundary: a file of exactly 50 MB (52 428 800 bytes) is allowed."""
    assert validate_upload("max.pdf", "application/pdf", MAX_FILE_SIZE_BYTES) is None
