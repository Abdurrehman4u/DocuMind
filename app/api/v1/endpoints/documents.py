"""
Document management endpoints.

POST   /documents/upload         — upload a file, enqueue ingestion
GET    /documents/               — list the current user's documents
GET    /documents/{document_id}  — fetch a single document
GET    /documents/{document_id}/status — lightweight status poll
DELETE /documents/{document_id}  — delete doc, chunks, and vectors

All routes are auth-protected via get_current_user.
"""

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session

from app.config import settings
from app.core.dependencies import get_current_user, get_db
from app.models.document import Document
from app.models.user import User
from app.schemas.document import DocumentResponse, DocumentStatusResponse
from app.services import vector_store_service
from app.utils.file_parsers import SUPPORTED_EXTENSIONS
from app.workers.tasks import ingest_document_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB guard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_user_upload_dir(user_id: str) -> Path:
    """Return (and create) the per-user upload directory."""
    upload_dir = Path(settings.UPLOAD_DIR) / user_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def _assert_doc_ownership(doc: Document | None, user_id: str) -> Document:
    """Raise 404 if doc is None or belongs to a different user."""
    if doc is None or doc.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return doc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document and enqueue background ingestion",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF, .txt, or .md file to ingest"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Accept a file upload, persist it to disk, create a Document row with
    status *pending*, and immediately enqueue the Celery ingestion task.
    Returns the Document before ingestion completes — poll
    `/documents/{id}/status` to track progress.
    """
    # Validate extension
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported file type '{file_ext}'. "
                f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    # Save file to disk (streaming to avoid loading entire file into RAM)
    user_dir = _get_user_upload_dir(current_user.id)
    dest_path = user_dir / (file.filename or "upload")
    try:
        with dest_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file.",
        ) from exc

    # Guard against absurdly large files (50 MB limit)
    if dest_path.stat().st_size > _MAX_FILE_SIZE_BYTES:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds the 50 MB size limit.",
        )

    # Create Document row
    doc = Document(
        user_id=current_user.id,
        filename=file.filename or dest_path.name,
        source_type="upload",
        status="pending",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Create Task tracking row & enqueue Celery ingestion task
    from app.models.task import Task
    task_row = Task(
        user_id=current_user.id,
        type="ingestion",
        status="pending",
    )
    db.add(task_row)
    db.commit()
    db.refresh(task_row)

    async_res = ingest_document_task.delay(doc.id, task_record_id=task_row.id)
    task_row.celery_task_id = async_res.id
    db.commit()

    logger.info("Document %s uploaded by user %s — ingestion enqueued (task_id=%s)", doc.id, current_user.id, task_row.id)

    return doc


@router.get(
    "/",
    response_model=list[DocumentResponse],
    summary="List all documents belonging to the current user",
)
def list_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return every Document row owned by the authenticated user."""
    return (
        db.query(Document)
        .filter(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
        .all()
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get full details of a single document",
)
def get_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Fetch a document by ID. Returns 404 if not found or owned by another user."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    return _assert_doc_ownership(doc, current_user.id)


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Poll the ingestion status of a document",
)
def get_document_status(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Lightweight endpoint for polling ingestion progress.
    Status values: pending | processing | ready | failed
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    return _assert_doc_ownership(doc, current_user.id)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document, its chunks, and its vectors",
)
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Remove the document from Postgres (cascades to Chunk rows),
    delete its vectors from ChromaDB, and remove the file from disk.
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    _assert_doc_ownership(doc, current_user.id)

    # Remove vectors from ChromaDB first (before DB row is gone)
    try:
        vector_store_service.delete_document_chunks(
            document_id=document_id, user_id=current_user.id
        )
    except Exception as exc:
        logger.warning(
            "Failed to remove ChromaDB vectors for document %s: %s", document_id, exc
        )

    # Remove file from disk
    file_path = Path(settings.UPLOAD_DIR) / current_user.id / doc.filename
    file_path.unlink(missing_ok=True)

    # Delete Document (cascades to Chunk rows via DB foreign key)
    db.delete(doc)
    db.commit()
    logger.info("Document %s deleted by user %s", document_id, current_user.id)
