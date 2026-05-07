"""Tests for the ingestion pipeline (Phase 1).

Parser and chunker tests use minimal in-memory PDFs — no external services.
Embedder tests mock the OpenAI client via @patch("app.ingestion.embedder.OpenAI").
OCR tests mock pytesseract via @patch("app.ingestion.parser.pytesseract.image_to_string").
"""

import io
from unittest.mock import MagicMock, patch

import pytest
import fitz  # PyMuPDF
import tenacity

from app.ingestion.parser import extract_text
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import embed_chunks
from app.ingestion.exceptions import InvalidPDFError, EmptyPDFError, EmbeddingError
from app.core.config import Settings


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pdf(pages: list[str]) -> bytes:
    """Create a minimal in-memory PDF with one text block per page."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _make_settings(**overrides) -> Settings:
    """Return a Settings instance with test-friendly defaults."""
    defaults = dict(
        openai_api_key="sk-test",
        chunk_size=200,
        chunk_overlap=20,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _make_chunk(**overrides) -> dict:
    """Return a minimal ChunkDict for embedder tests."""
    defaults = dict(
        chunk_id="c1",
        document_id="d1",
        filename="f.pdf",
        sha256="abc123",
        page_number=1,
        text="Some text for embedding.",
        char_count=24,
    )
    defaults.update(overrides)
    return defaults


def _mock_openai_response(vectors: list[list[float]]) -> MagicMock:
    """Build a mock OpenAI embeddings API response."""
    response = MagicMock()
    response.data = []
    for v in vectors:
        item = MagicMock()
        item.embedding = v
        response.data.append(item)
    return response


# ── Parser tests ──────────────────────────────────────────────────────────────

# AC-PARSE-01 — A text-native PDF returns one PageDict per page
def test_extract_text_returns_one_dict_per_page():
    pdf = _make_pdf(["Page one content.", "Page two content."])
    pages = extract_text(pdf)
    assert len(pages) == 2


# AC-PARSE-02 — Page numbers are 1-indexed
def test_extract_text_page_numbers_are_one_indexed():
    pdf = _make_pdf(["Hello world."])
    pages = extract_text(pdf)
    assert pages[0]["page_number"] == 1


# AC-PARSE-03 — char_count equals len(text) for every page
def test_extract_text_char_count_matches_text_length():
    pdf = _make_pdf(["Exactly this text."])
    pages = extract_text(pdf)
    assert pages[0]["char_count"] == len(pages[0]["text"])


# AC-PARSE-04 — Invalid bytes raise InvalidPDFError
def test_extract_text_raises_invalid_pdf_error():
    with pytest.raises(InvalidPDFError, match="Could not open PDF"):
        extract_text(b"this is not a pdf")


# AC-PARSE-05 — A page with fewer than 20 native chars triggers OCR
@patch("app.ingestion.parser.pytesseract.image_to_string")
def test_extract_text_triggers_ocr_on_short_text(mock_ocr):
    mock_ocr.return_value = "OCR extracted text from scanned page."
    pdf = _make_pdf(["Hi"])  # 2 characters — below the 20-char threshold
    pages = extract_text(pdf)
    mock_ocr.assert_called_once()
    assert "OCR extracted text" in pages[0]["text"]


# AC-PARSE-06 — ocr_used is True for OCR pages, False for text-native pages
@patch("app.ingestion.parser.pytesseract.image_to_string")
def test_extract_text_ocr_used_flag(mock_ocr):
    mock_ocr.return_value = "OCR text from scan."
    pdf = _make_pdf([
        "Short",                                                # < 20 chars -> OCR
        "This page has more than twenty characters of text.",    # >= 20 -> no OCR
    ])
    pages = extract_text(pdf)
    assert pages[0]["ocr_used"] is True
    assert pages[1]["ocr_used"] is False


# AC-PARSE-07 — Blank pages are returned with char_count == 0, not omitted
@patch("app.ingestion.parser.pytesseract.image_to_string")
def test_extract_text_blank_page_not_omitted(mock_ocr):
    mock_ocr.return_value = ""  # OCR also finds nothing
    pdf = _make_pdf(["Real content here repeated. " * 5, ""])
    pages = extract_text(pdf)
    assert len(pages) == 2           # blank page is NOT omitted
    assert pages[1]["char_count"] == 0


def test_extract_text_captures_content():
    pdf = _make_pdf(["Vanuatu national budget 2024."])
    pages = extract_text(pdf)
    assert "Vanuatu" in pages[0]["text"]


def test_extract_text_raises_empty_pdf_error():
    """A PDF with zero pages raises EmptyPDFError."""
    # PyMuPDF refuses to save a zero-page document, so build minimal
    # PDF bytes by hand: a valid PDF whose /Pages tree has Count 0.
    zero_page_pdf = (
        b"%PDF-1.0\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
        b"xref\n0 3\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"trailer<</Size 3/Root 1 0 R>>\nstartxref\n109\n%%EOF"
    )
    with pytest.raises(EmptyPDFError):
        extract_text(zero_page_pdf)


# ── Chunker tests ─────────────────────────────────────────────────────────────

# AC-CHUNK-01 — Chunks from a multi-page document carry correct page_number
def test_chunk_pages_correct_page_number():
    pdf = _make_pdf(["First page content. " * 20, "Second page content. " * 20])
    pages = extract_text(pdf)
    settings = _make_settings()
    chunks = chunk_pages(pages, "doc-1", "test.pdf", "hash123", settings)
    page_numbers = {c["page_number"] for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers
    for chunk in chunks:
        assert chunk["page_number"] in (1, 2)


# AC-CHUNK-02 — Every chunk has a unique chunk_id
def test_chunk_ids_are_unique():
    pdf = _make_pdf(["Content. " * 100])
    pages = extract_text(pdf)
    settings = _make_settings()
    chunks = chunk_pages(pages, "doc-1", "test.pdf", "hash123", settings)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


# AC-CHUNK-03 — document_id, filename, and sha256 identical across all chunks
def test_chunk_pages_document_metadata_identical():
    pdf = _make_pdf(["Some content. " * 30, "More content. " * 30])
    pages = extract_text(pdf)
    settings = _make_settings()
    chunks = chunk_pages(pages, "doc-abc", "budget.pdf", "deadbeef", settings)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk["document_id"] == "doc-abc"
        assert chunk["filename"] == "budget.pdf"
        assert chunk["sha256"] == "deadbeef"


# AC-CHUNK-04 — Pages with char_count < 20 produce no chunks
def test_chunk_pages_skips_near_blank_pages():
    pdf = _make_pdf(["Real content here. " * 20, "   ", "More content. " * 20])
    pages = extract_text(pdf)
    settings = _make_settings()
    chunks = chunk_pages(pages, "doc-1", "test.pdf", "hash123", settings)
    page_numbers = {c["page_number"] for c in chunks}
    assert 2 not in page_numbers  # blank / near-blank page skipped


# AC-CHUNK-05 — No chunk has fewer than 10 characters
def test_chunk_pages_discards_tiny_chunks():
    pdf = _make_pdf(["This is a longer document. " * 20])
    pages = extract_text(pdf)
    settings = _make_settings()
    chunks = chunk_pages(pages, "doc-1", "test.pdf", "hash123", settings)
    for chunk in chunks:
        assert len(chunk["text"]) >= 10


# AC-CHUNK-06 — An all-blank document returns an empty list without raising
def test_chunk_pages_all_blank_returns_empty():
    pdf = _make_pdf(["", "   "])
    pages = extract_text(pdf)
    settings = _make_settings()
    chunks = chunk_pages(pages, "doc-1", "test.pdf", "hash123", settings)
    assert chunks == []


def test_chunk_pages_returns_chunks():
    pdf = _make_pdf(["This is a longer document. " * 20])
    pages = extract_text(pdf)
    settings = _make_settings()
    chunks = chunk_pages(pages, "doc-1", "test.pdf", "hash123", settings)
    assert len(chunks) > 0


def test_chunk_pages_respects_chunk_size():
    pdf = _make_pdf(["Word " * 200])
    pages = extract_text(pdf)
    settings = _make_settings(chunk_size=100, chunk_overlap=10)
    chunks = chunk_pages(pages, "doc-1", "test.pdf", "hash123", settings)
    for chunk in chunks:
        assert len(chunk["text"]) <= 150  # allow some splitter headroom


# ── Embedder tests ────────────────────────────────────────────────────────────

# AC-EMBED-01 — Output list length equals input list length
@patch("app.ingestion.embedder.OpenAI")
def test_embed_output_length_matches_input(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    vectors = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
    mock_client.embeddings.create.return_value = _mock_openai_response(vectors)

    settings = _make_settings()
    chunks = [_make_chunk(chunk_id=f"c{i}", text=f"text number {i}") for i in range(3)]
    result = embed_chunks(chunks, settings)
    assert len(result) == len(chunks)


# AC-EMBED-02 — Every EmbeddedChunk has a vector field (non-empty list of floats)
@patch("app.ingestion.embedder.OpenAI")
def test_embed_chunks_have_vector_field(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.embeddings.create.return_value = _mock_openai_response([[0.5] * 1536])

    settings = _make_settings()
    result = embed_chunks([_make_chunk()], settings)
    assert "vector" in result[0]
    assert isinstance(result[0]["vector"], list)
    assert len(result[0]["vector"]) > 0
    assert all(isinstance(v, float) for v in result[0]["vector"])


# AC-EMBED-03 — All vectors from the same model have the same length
@patch("app.ingestion.embedder.OpenAI")
def test_embed_vectors_same_length(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    vectors = [[0.1] * 1536, [0.2] * 1536]
    mock_client.embeddings.create.return_value = _mock_openai_response(vectors)

    settings = _make_settings()
    chunks = [_make_chunk(chunk_id="c1"), _make_chunk(chunk_id="c2")]
    result = embed_chunks(chunks, settings)
    lengths = {len(r["vector"]) for r in result}
    assert len(lengths) == 1  # all vectors have the same dimension


# AC-EMBED-04 — EmbeddingError is raised after 3 failed API attempts
@patch("app.ingestion.embedder.OpenAI")
def test_embed_raises_embedding_error_after_retries(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.embeddings.create.side_effect = Exception("API unavailable")

    # Disable retry wait to avoid ~6 s delay from exponential backoff
    from app.ingestion.embedder import _embed_batch
    original_wait = _embed_batch.retry.wait
    _embed_batch.retry.wait = tenacity.wait_none()
    try:
        settings = _make_settings()
        with pytest.raises(EmbeddingError):
            embed_chunks([_make_chunk()], settings)
        assert mock_client.embeddings.create.call_count == 3
    finally:
        _embed_batch.retry.wait = original_wait
