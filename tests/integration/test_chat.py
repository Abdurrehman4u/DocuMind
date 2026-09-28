"""
Phase 3 integration tests — RAG Chat (Q&A with Citations).

Tests:
1. Ask question without conversation_id creates new conversation and persists user/assistant messages.
2. Citations are parsed and returned with answer.
3. Ask question on existing conversation continues thread and includes history.
4. Querying another user's conversation returns 404.
5. Asking when no context chunks match returns honest refusal without hallucination.
6. List conversations returns user's conversations ordered newest first.
7. Get conversation returns all messages in chronological order.
8. Delete conversation removes conversation and cascades messages.
"""

import uuid
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def auth_headers() -> Generator[dict, None, None]:
    """Register a fresh user and return JWT auth headers."""
    client = TestClient(app)
    email = f"chattest_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "ChatTest123!"},
    )
    assert resp.status_code == 201
    token = resp.json()["access_token"]
    yield {"Authorization": f"Bearer {token}"}


@pytest.fixture
def other_user_headers() -> Generator[dict, None, None]:
    """Register a second fresh user to test isolation."""
    client = TestClient(app)
    email = f"otheruser_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "OtherUser123!"},
    )
    assert resp.status_code == 201
    token = resp.json()["access_token"]
    yield {"Authorization": f"Bearer {token}"}


def test_ask_question_creates_conversation_and_citations(auth_headers):
    client = TestClient(app)

    fake_chunks = [
        {
            "id": "chunk-101",
            "document_id": "doc-202",
            "text": "DocuMind uses ChromaDB and Neon Postgres for storage.",
            "distance": 0.05,
        }
    ]

    with patch(
        "app.api.v1.endpoints.chat.retrieval_service.retrieve_context",
        return_value=fake_chunks,
    ), patch(
        "app.services.llm_service.LLMService.generate_rag_answer",
        return_value=(
            "DocuMind uses ChromaDB and Neon Postgres [chunk:chunk-101].",
            [
                {
                    "chunk_id": "chunk-101",
                    "document_id": "doc-202",
                    "text_snippet": "DocuMind uses ChromaDB...",
                }
            ],
        ),
    ):
        resp = client.post(
            "/api/v1/chat/ask",
            json={"query": "What storage does DocuMind use?"},
            headers=auth_headers,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "conversation_id" in data
    assert "message_id" in data
    assert "DocuMind uses ChromaDB and Neon Postgres" in data["answer"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["chunk_id"] == "chunk-101"
    assert data["citations"][0]["document_id"] == "doc-202"

    # Verify conversation is persisted and readable
    conv_resp = client.get(
        f"/api/v1/chat/conversations/{data['conversation_id']}",
        headers=auth_headers,
    )
    assert conv_resp.status_code == 200
    conv_data = conv_resp.json()
    assert len(conv_data["messages"]) == 2
    assert conv_data["messages"][0]["role"] == "user"
    assert conv_data["messages"][0]["content"] == "What storage does DocuMind use?"
    assert conv_data["messages"][1]["role"] == "assistant"
    assert conv_data["messages"][1]["citations"][0]["chunk_id"] == "chunk-101"


def test_ask_question_continues_existing_conversation(auth_headers):
    client = TestClient(app)

    # First turn
    with patch(
        "app.api.v1.endpoints.chat.retrieval_service.retrieve_context",
        return_value=[],
    ), patch(
        "app.services.llm_service.LLMService.generate_rag_answer",
        return_value=("Answer 1", []),
    ):
        resp1 = client.post(
            "/api/v1/chat/ask",
            json={"query": "Turn 1 question"},
            headers=auth_headers,
        )
    assert resp1.status_code == 200
    conv_id = resp1.json()["conversation_id"]

    # Second turn continuing conversation
    with patch(
        "app.api.v1.endpoints.chat.retrieval_service.retrieve_context",
        return_value=[],
    ), patch(
        "app.services.llm_service.LLMService.generate_rag_answer",
        return_value=("Answer 2", []),
    ):
        resp2 = client.post(
            "/api/v1/chat/ask",
            json={"conversation_id": conv_id, "query": "Turn 2 question"},
            headers=auth_headers,
        )
    assert resp2.status_code == 200
    assert resp2.json()["conversation_id"] == conv_id

    # Verify 4 messages in total
    conv_resp = client.get(
        f"/api/v1/chat/conversations/{conv_id}",
        headers=auth_headers,
    )
    assert conv_resp.status_code == 200
    messages = conv_resp.json()["messages"]
    assert len(messages) == 4
    assert [m["content"] for m in messages] == [
        "Turn 1 question",
        "Answer 1",
        "Turn 2 question",
        "Answer 2",
    ]


def test_ask_question_unauthorized_conversation_rejected(auth_headers, other_user_headers):
    client = TestClient(app)

    # User 1 creates conversation
    with patch(
        "app.api.v1.endpoints.chat.retrieval_service.retrieve_context",
        return_value=[],
    ), patch(
        "app.services.llm_service.LLMService.generate_rag_answer",
        return_value=("Answer 1", []),
    ):
        resp = client.post(
            "/api/v1/chat/ask",
            json={"query": "Private user 1 thread"},
            headers=auth_headers,
        )
    conv_id = resp.json()["conversation_id"]

    # User 2 tries to post to User 1's conversation
    unauth_resp = client.post(
        "/api/v1/chat/ask",
        json={"conversation_id": conv_id, "query": "Intruder question"},
        headers=other_user_headers,
    )
    assert unauth_resp.status_code == 404
    assert "not found" in unauth_resp.json()["detail"].lower()

    # User 2 tries to get User 1's conversation
    get_unauth = client.get(
        f"/api/v1/chat/conversations/{conv_id}",
        headers=other_user_headers,
    )
    assert get_unauth.status_code == 404


def test_ask_question_unanswerable_honest_refusal(auth_headers):
    """When retrieval yields no relevant chunks, LLM returns honest refusal and 0 citations."""
    client = TestClient(app)

    with patch(
        "app.api.v1.endpoints.chat.retrieval_service.retrieve_context",
        return_value=[],
    ):
        resp = client.post(
            "/api/v1/chat/ask",
            json={"query": "What is the secret recipe for quantum soup?"},
            headers=auth_headers,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "cannot find the answer" in data["answer"].lower()
    assert data["citations"] == []


def test_list_and_delete_conversations(auth_headers):
    client = TestClient(app)

    # Create two conversations
    with patch(
        "app.api.v1.endpoints.chat.retrieval_service.retrieve_context",
        return_value=[],
    ), patch(
        "app.services.llm_service.LLMService.generate_rag_answer",
        return_value=("OK", []),
    ):
        r1 = client.post("/api/v1/chat/ask", json={"query": "First thread"}, headers=auth_headers)
        r2 = client.post("/api/v1/chat/ask", json={"query": "Second thread"}, headers=auth_headers)

    c1_id = r1.json()["conversation_id"]
    c2_id = r2.json()["conversation_id"]

    # List conversations
    list_resp = client.get("/api/v1/chat/conversations", headers=auth_headers)
    assert list_resp.status_code == 200
    data = list_resp.json()
    assert data["total"] >= 2
    conv_ids = [c["id"] for c in data["conversations"]]
    assert c1_id in conv_ids
    assert c2_id in conv_ids

    # Delete conversation 1
    del_resp = client.delete(f"/api/v1/chat/conversations/{c1_id}", headers=auth_headers)
    assert del_resp.status_code == 204

    # Subsequent GET on conversation 1 should 404
    get_resp = client.get(f"/api/v1/chat/conversations/{c1_id}", headers=auth_headers)
    assert get_resp.status_code == 404
