"""
Agent Tool: search_knowledge_base

Searches the user's uploaded document chunks using semantic similarity.
"""

from __future__ import annotations

from typing import Any
from app.services import retrieval_service


def search_knowledge_base(query: str, user_id: str, k: int = 4, **kwargs: Any) -> list[dict[str, Any]]:
    """
    Search the knowledge base for relevant passages.

    Returns list of dicts with keys: id, document_id, text, distance.
    """
    chunks = retrieval_service.retrieve_context(
        query=query,
        user_id=user_id,
        k=k,
    )
    # Simplify chunk text for LLM tool consumption
    return [
        {
            "chunk_id": c.get("id"),
            "document_id": c.get("document_id"),
            "text": c.get("text"),
        }
        for c in chunks
    ]
