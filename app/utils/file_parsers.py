"""
File parsing utilities.

Converts uploaded files (PDF, .txt, .md) to plain UTF-8 text.
All functions are synchronous / pure-Python — they are called from
the Celery worker, not from an async FastAPI endpoint.
"""

import io
from pathlib import Path

from pypdf import PdfReader


def parse_pdf(file_bytes: bytes) -> str:
    """Extract plain text from a PDF byte string using pypdf."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def parse_text(file_bytes: bytes) -> str:
    """Decode plain-text or Markdown bytes to str (UTF-8 with fallback to latin-1)."""
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def parse_file(file_path: str | Path, file_bytes: bytes) -> str:
    """
    Dispatch to the correct parser based on the file extension.

    Parameters
    ----------
    file_path:
        Original filename or full path — used only to determine extension.
    file_bytes:
        Raw bytes of the uploaded file.

    Returns
    -------
    str
        Plain text content of the file.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_bytes)
    elif ext in {".txt", ".md"}:
        return parse_text(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
