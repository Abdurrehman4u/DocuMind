"""
Ingestion service.

Orchestrates the full document ingestion pipeline:

    parse file  →  chunk text  →  embed chunks
    →  upsert vectors into ChromaDB
    →  write Chunk rows to Postgres
    →  update Document.status

All heavy I/O (disk read, embedding model inference, DB writes) happens
inside the Celery worker via workers/tasks.py.  This service function is
framework-agnostic: it receives a DB session and document_id, and does
the rest.  It never imports FastAPI or Celery directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.chunk import Chunk
from app.services.chunking import chunk_text
from app.services.embedding_service import embed_texts
from app.services import vector_store_service
from app.utils.file_parsers import parse_file
from app.config import settings

logger = logging.getLogger(__name__)


def run_ingestion(document_id: str, db: Session) -> None:
    """
    Execute the ingestion pipeline for a single document.

    Parameters
    ----------
    document_id:
        Primary key of the Document row to process.
    db:
        SQLAlchemy session (provided by the Celery task — *not* a
        FastAPI request-scoped session).

    Side-effects
    ------------
    - Sets Document.status to "processing" at the start.
    - Creates Chunk rows linked to the document.
    - Upserts chunk vectors into ChromaDB.
    - Sets Document.status to "ready" on success, "failed" on any error.
    """
    doc: Document | None = db.query(Document).filter(Document.id == document_id).first()
    if doc is None:
        logger.error("ingestion: document %s not found — skipping", document_id)
        return

    # Mark as in-progress
    doc.status = "processing"
    db.commit()

    try:
        # 1. Read raw file from disk
        file_path = Path(settings.UPLOAD_DIR) / doc.user_id / doc.filename
        if not file_path.exists():
            raise FileNotFoundError(f"Upload not found: {file_path}")

        raw_bytes = file_path.read_bytes()

        # 2. Parse to plain text
        plain_text = parse_file(doc.filename, raw_bytes)
        if not plain_text.strip():
            raise ValueError("Parsed document produced empty text — nothing to index.")

        # 3. Chunk the text
        chunks = chunk_text(plain_text)
        if not chunks:
            raise ValueError("Chunking produced zero chunks.")

        logger.info(
            "ingestion: document %s → %d chunks",
            document_id, len(chunks),
        )

        # 4. Embed all chunks in one batch call
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)

        # 5. Persist Chunk rows (clear any stale rows first for idempotency)
        db.query(Chunk).filter(Chunk.document_id == document_id).delete()
        db.flush()

        chunk_records: list[Chunk] = []
        for c, embedding in zip(chunks, embeddings):
            chunk_row = Chunk(
                document_id=document_id,
                chunk_index=c.index,
                text=c.text,
                # vector_id will be set to the chunk's own PK after flush
            )
            db.add(chunk_row)
            chunk_records.append(chunk_row)

        db.flush()  # assigns UUIDs to chunk_row.id

        # 6. Set vector_id = chunk.id (same UUID used as ChromaDB entry ID)
        for chunk_row in chunk_records:
            chunk_row.vector_id = chunk_row.id

        # 7. Upsert into ChromaDB
        vector_store_service.add_chunks(
            chunk_ids=[cr.id for cr in chunk_records],
            texts=[cr.text for cr in chunk_records],
            embeddings=embeddings,
            user_id=doc.user_id,
            document_id=document_id,
        )

        # 8. Mark complete
        doc.status = "ready"
        doc.error_message = None
        db.commit()
        logger.info("ingestion: document %s completed successfully", document_id)

    except Exception as exc:  # noqa: BLE001
        db.rollback()
        # Re-fetch doc after rollback
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = "failed"
            doc.error_message = str(exc)
            db.commit()
        logger.exception("ingestion: document %s failed: %s", document_id, exc)
