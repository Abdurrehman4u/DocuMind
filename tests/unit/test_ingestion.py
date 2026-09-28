"""
Unit tests for Phase 2 services:
    - services/chunking.py
    - utils/file_parsers.py
"""

import io
import pytest

from app.services.chunking import chunk_text, TextChunk
from app.utils.file_parsers import parse_file, parse_text, SUPPORTED_EXTENSIONS


# ── Chunking ──────────────────────────────────────────────────────────────


def test_chunk_text_returns_text_chunks():
    text = "Hello world. " * 200  # ~2600 chars
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(isinstance(c, TextChunk) for c in chunks)
    assert all(c.text for c in chunks)


def test_chunk_indexes_are_sequential():
    text = "paragraph\n\n" * 50
    chunks = chunk_text(text)
    for i, c in enumerate(chunks):
        assert c.index == i


def test_no_chunk_exceeds_max_size():
    text = "word " * 1000
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=30)
    for c in chunks:
        assert len(c.text) <= 300 + 50  # small tolerance for separator


def test_short_text_is_single_chunk():
    text = "Short doc."
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].text == "Short doc."


def test_empty_text_returns_no_chunks():
    chunks = chunk_text("", chunk_size=500, chunk_overlap=50)
    assert chunks == []


def test_chunk_overlap_present():
    """Verifies that content from the end of chunk N appears at the start of chunk N+1."""
    # Build text with clear paragraphs so overlap has something to work with
    text = "\n\n".join(["paragraph " + str(i) + " content " * 10 for i in range(20)])
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=100)
    if len(chunks) > 1:
        # The end of chunk 0 should share some content with chunk 1
        end_of_first = chunks[0].text[-50:]
        start_of_second = chunks[1].text[:200]
        # At least some word from the end should appear in the next chunk
        words = [w for w in end_of_first.split() if len(w) > 3]
        assert any(w in start_of_second for w in words), (
            "Expected overlap between consecutive chunks"
        )


# ── File Parsers ───────────────────────────────────────────────────────────


def test_parse_txt_file():
    content = b"Hello, this is a plain text document."
    result = parse_file("readme.txt", content)
    assert "Hello" in result


def test_parse_md_file():
    content = b"# Heading\n\nSome **markdown** content."
    result = parse_file("notes.md", content)
    assert "Heading" in result


def test_parse_pdf_file():
    """
    Generate a minimal valid PDF in-memory and verify the parser
    extracts text without raising.
    """
    from pypdf import PdfWriter
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    # add_blank_page returns a page we can't easily write text to without
    # reportlab/pdfrw, so we just verify the parser handles a real PDF
    buf = io.BytesIO()
    writer.write(buf)
    pdf_bytes = buf.getvalue()

    result = parse_file("doc.pdf", pdf_bytes)
    # Blank PDF has no text — result should be a string (possibly empty)
    assert isinstance(result, str)


def test_unsupported_extension_raises():
    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_file("data.csv", b"a,b,c")


def test_supported_extensions_set():
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".txt" in SUPPORTED_EXTENSIONS
    assert ".md" in SUPPORTED_EXTENSIONS
