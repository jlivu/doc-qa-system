"""Tests for the ingestion pipeline (Phase 1).

These tests use a minimal in-memory PDF so they run without any
external services or API keys.
"""

import io
import pytest
import fitz  # PyMuPDF

from app.ingestion.parser import extract_text
from app.ingestion.chunker import chunk_pages
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


# ── Parser tests ──────────────────────────────────────────────────────────────

def test_extract_text_returns_one_dict_per_page():
    pdf = _make_pdf(["Page one content.", "Page two content."])
    pages = extract_text(pdf)
    assert len(pages) == 2


def test_extract_text_page_numbers_are_one_indexed():
    pdf = _make_pdf(["Hello world."])
    pages = extract_text(pdf)
    assert pages[0]["page_number"] == 1


def test_extract_text_captures_content():
    pdf = _make_pdf(["Vanuatu national budget 2024."])
    pages = extract_text(pdf)
    assert "Vanuatu" in pages[0]["text"]


def test_extract_text_raises_on_invalid_bytes():
    with pytest.raises(ValueError, match="Could not open PDF"):
        extract_text(b"this is not a pdf")


def test_extract_text_char_count_matches_text_length():
    pdf = _make_pdf(["Exactly this text."])
    pages = extract_text(pdf)
    assert pages[0]["char_count"] == len(pages[0]["text"])


# ── Chunker tests ─────────────────────────────────────────────────────────────

def test_chunk_pages_returns_chunks():
    pdf = _make_pdf(["This is a longer document. " * 20])
    pages = extract_text(pdf)
    settings = _make_settings()
    chunks = chunk_pages(pages, "doc-1", "test.pdf", settings)
    assert len(chunks) > 0


def test_chunk_pages_metadata_propagated():
    pdf = _make_pdf(["Some content. " * 30])
    pages = extract_text(pdf)
    settings = _make_settings()
    chunks = chunk_pages(pages, "doc-abc", "budget.pdf", settings)
    for chunk in chunks:
        assert chunk["document_id"] == "doc-abc"
        assert chunk["filename"] == "budget.pdf"
        assert chunk["page_number"] == 1


def test_chunk_pages_respects_chunk_size():
    pdf = _make_pdf(["Word " * 200])
    pages = extract_text(pdf)
    settings = _make_settings(chunk_size=100, chunk_overlap=10)
    chunks = chunk_pages(pages, "doc-1", "test.pdf", settings)
    for chunk in chunks:
        assert len(chunk["text"]) <= 150  # allow some splitter headroom


def test_chunk_pages_skips_blank_pages():
    pdf = _make_pdf(["Real content here. " * 20, "   ", "More content. " * 20])
    pages = extract_text(pdf)
    settings = _make_settings()
    chunks = chunk_pages(pages, "doc-1", "test.pdf", settings)
    page_numbers = {c["page_number"] for c in chunks}
    assert 2 not in page_numbers  # blank page skipped


def test_chunk_ids_are_unique():
    pdf = _make_pdf(["Content. " * 100])
    pages = extract_text(pdf)
    settings = _make_settings()
    chunks = chunk_pages(pages, "doc-1", "test.pdf", settings)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
