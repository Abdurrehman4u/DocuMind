"""
Retrieval service — top-k context retrieval scoped to the authenticated user.

Bridges embedding_service and vector_store_service.
"""

from __future__ import annotations

from typing import Sequence

from app.services import embedding_service, vector_store_service


def retrieve_context(
    query: str,
    user_id: str,
    k: int = 5,
    document_ids: Sequence[str] | None = None,
) -> list[dict]:
    """
    Retrieve top-k relevant chunks for a user's query.

    Parameters
    ----------
    query:
        The raw question / query string.
    user_id:
        UUID of the user (tenant isolation filter).
    k:
        Number of context chunks to retrieve.
    document_ids:
        Optional sequence of document IDs to restrict retrieval to.

    Returns
    -------
    list[dict]:
        List of dicts: {"id": str, "text": str, "distance": float, "document_id": str}
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    doc_ids_list = list(document_ids) if document_ids else None

    # 1. Embed query
    query_vector = embedding_service.embed_query(cleaned_query)

    # 2. Query vector store
    results = vector_store_service.query_chunks(
        query_embedding=query_vector,
        user_id=user_id,
        k=k,
        document_ids=doc_ids_list,
    )

    return results
