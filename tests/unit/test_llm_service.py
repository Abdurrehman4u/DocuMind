"""
Unit tests for LLM service (app.services.llm_service).
"""

from unittest.mock import MagicMock, patch
import pytest

from app.services.llm_service import (
    LLMService,
    build_rag_prompt,
    extract_citations,
)


def test_build_rag_prompt_with_chunks():
    chunks = [
        {"id": "chunk-1", "document_id": "doc-1", "text": "First chunk text"},
        {"id": "chunk-2", "document_id": "doc-2", "text": "Second chunk text"},
    ]
    prompt = build_rag_prompt("What is in chunk 1?", chunks)
    assert "[Chunk ID: chunk-1 | Document ID: doc-1]" in prompt
    assert "First chunk text" in prompt
    assert "[Chunk ID: chunk-2 | Document ID: doc-2]" in prompt
    assert "Question: What is in chunk 1?" in prompt


def test_build_rag_prompt_empty_chunks():
    prompt = build_rag_prompt("What is here?", [])
    assert "No document excerpts found." in prompt
    assert "Question: What is here?" in prompt


def test_extract_citations():
    answer = (
        "DocuMind supports semantic search [chunk:chunk-abc-123] and vector indexing [chunk:chunk-xyz-456]. "
        "Repeated reference [chunk:chunk-abc-123] should be deduplicated."
    )
    context_chunks = [
        {"id": "chunk-abc-123", "document_id": "doc-1", "text": "Semantic search details."},
        {"id": "chunk-xyz-456", "document_id": "doc-2", "text": "Vector indexing details."},
        {"id": "chunk-unused", "document_id": "doc-3", "text": "Unused text."},
    ]

    citations = extract_citations(answer, context_chunks)
    assert len(citations) == 2
    assert citations[0]["chunk_id"] == "chunk-abc-123"
    assert citations[0]["document_id"] == "doc-1"
    assert citations[0]["text_snippet"] == "Semantic search details."
    assert citations[1]["chunk_id"] == "chunk-xyz-456"
    assert citations[1]["document_id"] == "doc-2"


def test_llm_service_empty_context_refusal():
    """When context_chunks is empty, the service immediately returns honest refusal."""
    service = LLMService(provider="gemini", api_key="fake-key", model="gemini-2.5-flash")
    answer, citations = service.generate_rag_answer(
        query="What is the revenue?",
        context_chunks=[],
    )
    assert "I cannot find the answer to that in the provided documents." in answer
    assert citations == []


def test_llm_service_missing_api_key():
    """Missing API key raises ValueError when generating answer with context."""
    service = LLMService(provider="gemini", api_key="", model="gemini-2.5-flash")
    chunks = [{"id": "c1", "document_id": "d1", "text": "Sample text"}]

    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        service.generate_rag_answer(query="Sample question", context_chunks=chunks)


@patch("app.services.llm_service.genai.Client")
def test_llm_service_gemini_generation(mock_genai_client_cls):
    """Verifies that Gemini generate_content is invoked and citations are parsed."""
    mock_client = MagicMock()
    mock_genai_client_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.text = "DocuMind stores vectors in ChromaDB [chunk:chunk-42]."
    mock_client.models.generate_content.return_value = mock_response

    service = LLMService(provider="gemini", api_key="test-api-key", model="gemini-2.5-flash")
    chunks = [{"id": "chunk-42", "document_id": "doc-99", "text": "ChromaDB stores embeddings."}]

    answer, citations = service.generate_rag_answer(
        query="Where does DocuMind store vectors?",
        context_chunks=chunks,
        chat_history=[{"role": "user", "content": "Hi"}],
    )

    assert answer == "DocuMind stores vectors in ChromaDB [chunk:chunk-42]."
    assert len(citations) == 1
    assert citations[0]["chunk_id"] == "chunk-42"
    assert citations[0]["document_id"] == "doc-99"
    assert mock_client.models.generate_content.called
