"""
Unit tests for evaluation service (app.services.eval_service).
"""

from unittest.mock import MagicMock, patch
from app.models.message import Message
from app.models.chunk import Chunk
from app.services.eval_service import (
    evaluate_message_groundedness,
    _heuristic_groundedness,
)


def test_heuristic_groundedness_supported():
    context = "DocuMind utilizes ChromaDB vector store and PostgreSQL database for persistent storage."
    answer = "DocuMind uses ChromaDB and PostgreSQL database for storage."
    score, flagged, notes = _heuristic_groundedness(answer, context)
    assert score >= 0.7
    assert flagged is False
    assert "grounded" in notes


def test_heuristic_groundedness_unsupported():
    context = "DocuMind is built with Python and FastAPI."
    answer = "DocuMind was written in Java using Spring Boot microservices with Kubernetes."
    score, flagged, notes = _heuristic_groundedness(answer, context)
    assert score < 0.7
    assert flagged is True
    assert "flagged" in notes


def test_heuristic_groundedness_empty_context():
    score, flagged, notes = _heuristic_groundedness("Some factual claim", "")
    assert score == 0.0
    assert flagged is True


def test_evaluate_message_honest_refusal():
    mock_db = MagicMock()
    msg = Message(
        id="msg-1",
        content="I cannot find the answer to that in the provided documents.",
        citations=[],
    )
    result = evaluate_message_groundedness(msg, mock_db)
    assert result.groundedness_score == 1.0
    assert result.flagged is False
    assert "truthfully acknowledges" in result.notes
    assert mock_db.add.called
    assert mock_db.commit.called


def test_evaluate_message_no_context_cited():
    mock_db = MagicMock()
    msg = Message(
        id="msg-2",
        content="The quarterly revenue grew by 45 percent last year.",
        citations=[],
    )
    result = evaluate_message_groundedness(msg, mock_db)
    assert result.groundedness_score == 0.0
    assert result.flagged is True
    assert "No context excerpts were cited" in result.notes


def test_evaluate_message_unknown_chunk_ids_rejected():
    """Verify that citations with unknown chunk IDs are rejected as valid evidence."""
    mock_db = MagicMock()
    mock_query = MagicMock()
    mock_db.query.return_value = mock_query
    mock_query.join.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.all.return_value = []

    msg = Message(
        id="msg-fake-chunk",
        content="DocuMind was founded in 1990 [chunk:non-existent-chunk].",
        citations=[{"chunk_id": "non-existent-chunk", "document_id": "doc-99"}],
    )

    result = evaluate_message_groundedness(msg, mock_db)
    assert result.groundedness_score == 0.0
    assert result.flagged is True
    assert "unknown chunk IDs or chunks not owned by the user" in result.notes


def test_evaluate_message_grounded_with_db_chunks():
    mock_db = MagicMock()
    mock_query = MagicMock()
    mock_db.query.return_value = mock_query
    mock_query.join.return_value = mock_query
    mock_query.filter.return_value = mock_query
    chunk = Chunk(
        id="chunk-abc",
        document_id="doc-1",
        text="DocuMind supports semantic chunking and embedding retrieval.",
    )
    mock_query.all.return_value = [chunk]

    msg = Message(
        id="msg-3",
        content="DocuMind supports semantic chunking and embedding retrieval [chunk:chunk-abc].",
        citations=[{"chunk_id": "chunk-abc", "document_id": "doc-1"}],
    )

    result = evaluate_message_groundedness(msg, mock_db)
    assert result.groundedness_score >= 0.7
    assert result.flagged is False


def test_evaluate_message_unowned_chunk_rejected():
    """Verify that chunks belonging to another user are rejected as valid evidence."""
    mock_db = MagicMock()
    mock_query = MagicMock()
    mock_db.query.return_value = mock_query
    mock_query.join.return_value = mock_query
    mock_query.filter.return_value = mock_query
    # When filtered by Document.user_id == current_user, no chunks match
    mock_query.all.return_value = []

    msg = Message(
        id="msg-4",
        content="Unauthorized document claim [chunk:other-user-chunk].",
        citations=[{"chunk_id": "other-user-chunk", "document_id": "doc-other"}],
    )

    result = evaluate_message_groundedness(msg, mock_db, user_id="user-123")
    assert result.groundedness_score == 0.0
    assert result.flagged is True
    assert "not owned by the user" in result.notes


def test_evaluate_message_llm_as_judge_path():
    """Verify LLM-as-judge path when API key is present and Gemini returns valid JSON."""
    mock_db = MagicMock()
    mock_query = MagicMock()
    mock_db.query.return_value = mock_query
    mock_query.join.return_value = mock_query
    mock_query.filter.return_value = mock_query
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        text="DocuMind architecture uses FastAPI and PostgreSQL.",
    )
    mock_query.all.return_value = [chunk]

    msg = Message(
        id="msg-llm-judge",
        content="DocuMind runs on FastAPI [chunk:chunk-1].",
        citations=[{"chunk_id": "chunk-1", "document_id": "doc-1"}],
    )

    mock_llm = MagicMock()
    mock_llm.api_key = "mock-key-for-test"
    mock_llm.model = "gemini-2.5-flash"
    mock_client = MagicMock()
    mock_llm._get_gemini_client.return_value = mock_client

    mock_response = MagicMock()
    mock_response.text = '{"score": 0.95, "flagged": false, "notes": "Claims fully supported."}'
    mock_client.models.generate_content.return_value = mock_response

    with patch("app.services.eval_service.get_llm_service", return_value=mock_llm):
        result = evaluate_message_groundedness(msg, mock_db)

    assert result.groundedness_score == 0.95
    assert result.flagged is False
    assert result.notes == "Claims fully supported."
    assert mock_client.models.generate_content.called


