"""
Unit tests for retrieval service (app.services.retrieval_service).
"""

from unittest.mock import MagicMock, patch

from app.services.retrieval_service import retrieve_context


def test_retrieve_context_empty_query():
    """Empty or whitespace-only query returns empty list without calling vector store."""
    results = retrieve_context("", user_id="user-123")
    assert results == []

    results = retrieve_context("   \n\t ", user_id="user-123")
    assert results == []


@patch("app.services.retrieval_service.vector_store_service.query_chunks")
@patch("app.services.retrieval_service.embedding_service.embed_query")
def test_retrieve_context_success(mock_embed, mock_query_chunks):
    """retrieve_context embeds the query and scopes vector store query to user_id."""
    mock_embed.return_value = [0.1, 0.2, 0.3]
    mock_query_chunks.return_value = [
        {
            "id": "chunk-1",
            "text": "DocuMind is an intelligent document assistant.",
            "distance": 0.05,
            "document_id": "doc-abc",
        },
        {
            "id": "chunk-2",
            "text": "It supports grounding and citations.",
            "distance": 0.12,
            "document_id": "doc-abc",
        },
    ]

    results = retrieve_context(
        query="What is DocuMind?",
        user_id="user-123",
        k=2,
        document_ids=["doc-abc"],
    )

    mock_embed.assert_called_once_with("What is DocuMind?")
    mock_query_chunks.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        user_id="user-123",
        k=2,
        document_ids=["doc-abc"],
    )

    assert len(results) == 2
    assert results[0]["id"] == "chunk-1"
    assert results[0]["document_id"] == "doc-abc"
