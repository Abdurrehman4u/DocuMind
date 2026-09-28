"""
Vector store service — ChromaDB wrapper.

Design decisions:
- Single ChromaDB collection named "documind_chunks".
- Per-user scoping is enforced via metadata filters ({"user_id": ...})
  on every query and delete call. This avoids the overhead of one
  collection per user while still guaranteeing data isolation.
- The ChromaDB client is a lazy singleton loaded once per process.
- IDs stored in ChromaDB match the Postgres Chunk.vector_id field
  (= Chunk.id — the UUID primary key), so we can cross-reference easily.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import chromadb
from chromadb import Collection

from app.config import settings

COLLECTION_NAME = "documind_chunks"


# ---------------------------------------------------------------------------
# Singleton client / collection
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_client() -> chromadb.ClientAPI:
    """Return (and cache) the ChromaDB persistent client."""
    return chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)


def _get_collection() -> Collection:
    """Return (or create) the shared chunks collection."""
    client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_chunks(
    chunk_ids: Sequence[str],
    texts: Sequence[str],
    embeddings: Sequence[list[float]],
    user_id: str,
    document_id: str,
) -> None:
    """
    Upsert a batch of chunk vectors into the ChromaDB collection.

    Parameters
    ----------
    chunk_ids:
        Postgres Chunk UUIDs — used as ChromaDB document IDs.
    texts:
        The raw text for each chunk (stored as ChromaDB document content).
    embeddings:
        Pre-computed embedding vectors (one per chunk).
    user_id:
        Owner's User.id — stored in metadata for isolation filtering.
    document_id:
        Parent Document.id — stored in metadata for deletion by document.
    """
    collection = _get_collection()
    metadatas = [
        {"user_id": user_id, "document_id": document_id}
        for _ in chunk_ids
    ]
    collection.upsert(
        ids=list(chunk_ids),
        documents=list(texts),
        embeddings=list(embeddings),
        metadatas=metadatas,
    )


def query_chunks(
    query_embedding: list[float],
    user_id: str,
    k: int = 5,
    document_ids: list[str] | None = None,
) -> list[dict]:
    """
    Retrieve the top-k most similar chunks for *user_id*.

    Parameters
    ----------
    query_embedding:
        The embedded query vector.
    user_id:
        Only chunks belonging to this user are considered.
    k:
        Number of results to return.
    document_ids:
        Optional list of Document IDs to restrict search to a subset of
        the user's documents. If None, all user documents are searched.

    Returns
    -------
    list[dict]
        Each dict has keys: ``id``, ``text``, ``distance``, ``document_id``.
    """
    collection = _get_collection()

    # Build where clause — always scope to user
    if document_ids:
        where: dict = {
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"document_id": {"$in": document_ids}},
            ]
        }
    else:
        where = {"user_id": {"$eq": user_id}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # Flatten the nested lists (one query → one list per field)
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {
            "id": ids[i],
            "text": documents[i],
            "distance": distances[i],
            "document_id": metadatas[i].get("document_id"),
        }
        for i in range(len(ids))
    ]


def delete_document_chunks(document_id: str, user_id: str) -> None:
    """
    Remove all vectors belonging to *document_id* owned by *user_id*.

    Called when a document is deleted from the API.
    """
    collection = _get_collection()
    # Fetch IDs for this document first, then delete
    results = collection.get(
        where={"$and": [
            {"document_id": {"$eq": document_id}},
            {"user_id": {"$eq": user_id}},
        ]},
        include=[],
    )
    ids_to_delete = results.get("ids", [])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)


def delete_user_chunks(user_id: str) -> None:
    """Remove ALL vectors belonging to *user_id*. Used for account deletion."""
    collection = _get_collection()
    results = collection.get(
        where={"user_id": {"$eq": user_id}},
        include=[],
    )
    ids_to_delete = results.get("ids", [])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
