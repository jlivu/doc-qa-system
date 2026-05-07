"""Tests for the hasher module (Phase 1).

compute_sha256 is a pure function — no mocks needed.
find_existing_document receives a QdrantClient — pass a MagicMock.
"""

import re
from unittest.mock import MagicMock

import pytest

from app.ingestion.hasher import compute_sha256, find_existing_document
from app.core.config import Settings


def _make_settings() -> Settings:
    return Settings(openai_api_key="sk-test")


# ── compute_sha256 ───────────────────────────────────────────────────────────

# AC-HASH-01 — Returns a 64-character lowercase hex string
def test_sha256_returns_64_char_lowercase_hex():
    result = compute_sha256(b"hello")
    assert len(result) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", result)


# AC-HASH-02 — Same bytes always produce the same hash
def test_sha256_deterministic():
    assert compute_sha256(b"same") == compute_sha256(b"same")


# AC-HASH-03 — Different bytes always produce different hashes
def test_sha256_different_inputs_differ():
    assert compute_sha256(b"a") != compute_sha256(b"b")


# ── find_existing_document ───────────────────────────────────────────────────

# AC-HASH-04 — Returns None for an unknown hash
def test_find_existing_returns_none_for_unknown():
    client = MagicMock()
    client.scroll.return_value = ([], None)
    settings = _make_settings()

    assert find_existing_document("nonexistent_hash", client, settings) is None


# AC-HASH-05 — Returns the document_id for a known hash
def test_find_existing_returns_document_id_for_known():
    point = MagicMock()
    point.payload = {"document_id": "doc-123", "sha256": "abc"}

    client = MagicMock()
    client.scroll.return_value = ([point], None)
    settings = _make_settings()

    assert find_existing_document("abc", client, settings) == "doc-123"


def test_find_existing_handles_missing_collection():
    """When the collection does not exist yet, return None instead of raising."""
    client = MagicMock()
    client.scroll.side_effect = Exception("Collection not found")
    settings = _make_settings()

    assert find_existing_document("any_hash", client, settings) is None
