"""
Agent Tool: query_structured_data

Safely queries structured relational document metadata using whitelisted,
parameterized queries. Arbitrary SQL execution is strictly forbidden.
"""

from __future__ import annotations

from typing import Any
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.document import Document
from app.models.chunk import Chunk

ALLOWED_QUERY_TYPES = {
    "document_summary",
    "document_counts_by_status",
    "recent_documents",
}


def query_structured_data(
    query_type: str,
    user_id: str,
    db: Session,
    limit: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Execute a whitelisted structured metadata query scoped to user_id.

    Parameters
    ----------
    query_type:
        One of 'document_summary', 'document_counts_by_status', 'recent_documents'.
    user_id:
        Owner user id to enforce tenant isolation.
    db:
        Active SQLAlchemy database session.
    limit:
        Max records for list queries.
    """
    if query_type not in ALLOWED_QUERY_TYPES:
        return {
            "error": f"Invalid query_type '{query_type}'. Allowed types are: {sorted(list(ALLOWED_QUERY_TYPES))}",
            "allowed_query_types": sorted(list(ALLOWED_QUERY_TYPES)),
        }

    limit = min(max(1, limit), 50)

    if query_type == "document_summary":
        total_docs = db.query(func.count(Document.id)).filter(Document.user_id == user_id).scalar() or 0
        total_chunks = (
            db.query(func.count(Chunk.id))
            .join(Document, Chunk.document_id == Document.id)
            .filter(Document.user_id == user_id)
            .scalar()
            or 0
        )
        status_rows = (
            db.query(Document.status, func.count(Document.id))
            .filter(Document.user_id == user_id)
            .group_by(Document.status)
            .all()
        )
        status_breakdown = {status: count for status, count in status_rows}

        return {
            "query_type": query_type,
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "status_breakdown": status_breakdown,
        }

    elif query_type == "document_counts_by_status":
        status_rows = (
            db.query(Document.status, func.count(Document.id))
            .filter(Document.user_id == user_id)
            .group_by(Document.status)
            .all()
        )
        return {
            "query_type": query_type,
            "counts": {status: count for status, count in status_rows},
        }

    elif query_type == "recent_documents":
        docs = (
            db.query(Document)
            .filter(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .all()
        )
        doc_list = []
        for d in docs:
            chunk_count = db.query(func.count(Chunk.id)).filter(Chunk.document_id == d.id).scalar() or 0
            doc_list.append({
                "id": d.id,
                "filename": d.filename,
                "status": d.status,
                "chunk_count": chunk_count,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            })
        return {
            "query_type": query_type,
            "count": len(doc_list),
            "documents": doc_list,
        }

    return {"error": "Unhandled query_type"}
