"""
Phase 5 integration tests — Evaluation harness endpoints (/api/v1/eval).
"""

import uuid
from typing import Generator
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.db import session as db_session_module
from app.models.document import Document
from app.models.chunk import Chunk
from app.models.message import Message
from app.models.conversation import Conversation


@pytest.fixture
def auth_headers() -> Generator[dict, None, None]:
    client = TestClient(app)
    email = f"evaltest_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "EvalPass123!"},
    )
    assert resp.status_code == 201
    token = resp.json()["access_token"]
    yield {"Authorization": f"Bearer {token}"}


@pytest.fixture
def other_user_headers() -> Generator[dict, None, None]:
    client = TestClient(app)
    email = f"othereval_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "OtherEval123!"},
    )
    assert resp.status_code == 201
    token = resp.json()["access_token"]
    yield {"Authorization": f"Bearer {token}"}


def test_eval_run_and_retrieve_result(auth_headers):
    client = TestClient(app)

    # 1. Ask a question to generate a Message
    with patch(
        "app.api.v1.endpoints.chat.retrieval_service.retrieve_context",
        return_value=[
            {
                "id": "chunk-grounded-1",
                "document_id": "doc-1",
                "text": "DocuMind utilizes semantic search to find document answers.",
            }
        ],
    ), patch(
        "app.services.llm_service.LLMService.generate_rag_answer",
        return_value=(
            "DocuMind utilizes semantic search to find document answers [chunk:chunk-grounded-1].",
            [
                {
                    "chunk_id": "chunk-grounded-1",
                    "document_id": "doc-1",
                    "text_snippet": "DocuMind utilizes semantic search to find document answers.",
                }
            ],
        ),
    ):
        ask_resp = client.post(
            "/api/v1/chat/ask",
            json={"query": "How does DocuMind find answers?"},
            headers=auth_headers,
        )

    assert ask_resp.status_code == 200
    message_id = ask_resp.json()["message_id"]

    # Insert verified Document and Chunk in DB so evidence validation succeeds
    db = db_session_module.SessionLocal()
    msg = db.query(Message).filter(Message.id == message_id).first()
    conv = db.query(Conversation).filter(Conversation.id == msg.conversation_id).first()
    doc = Document(id="doc-1", user_id=conv.user_id, filename="answers.txt", status="ready")
    chunk = Chunk(id="chunk-grounded-1", document_id="doc-1", chunk_index=0, text="DocuMind utilizes semantic search to find document answers.")
    db.add(doc)
    db.add(chunk)
    db.commit()
    db.close()

    # 2. Trigger evaluation
    eval_run_resp = client.post(
        "/api/v1/eval/run",
        json={"message_id": message_id},
        headers=auth_headers,
    )
    assert eval_run_resp.status_code == 202
    eval_data = eval_run_resp.json()
    assert "task_id" in eval_data
    assert eval_data["message_id"] == message_id

    # 3. Retrieve evaluation by message
    evals_for_msg = client.get(
        f"/api/v1/eval/message/{message_id}",
        headers=auth_headers,
    )
    assert evals_for_msg.status_code == 200
    results_list = evals_for_msg.json()
    assert len(results_list) >= 1
    eval_id = results_list[0]["id"]
    assert results_list[0]["groundedness_score"] >= 0.70
    assert results_list[0]["flagged"] is False

    # 4. Retrieve evaluation by UUID
    single_eval = client.get(f"/api/v1/eval/{eval_id}", headers=auth_headers)
    assert single_eval.status_code == 200
    assert single_eval.json()["id"] == eval_id


def test_eval_unsupported_answer_flagged(auth_headers):
    client = TestClient(app)

    # Generate assistant message with unsupported answer
    with patch(
        "app.api.v1.endpoints.chat.retrieval_service.retrieve_context",
        return_value=[
            {
                "id": "chunk-doc-2",
                "document_id": "doc-2",
                "text": "DocuMind is an NLP document search engine written in Python.",
            }
        ],
    ), patch(
        "app.services.llm_service.LLMService.generate_rag_answer",
        return_value=(
            "DocuMind was written in C++ and CUDA for quantum computing [chunk:chunk-doc-2].",
            [
                {
                    "chunk_id": "chunk-doc-2",
                    "document_id": "doc-2",
                    "text_snippet": "DocuMind is an NLP document search engine...",
                }
            ],
        ),
    ):
        ask_resp = client.post(
            "/api/v1/chat/ask",
            json={"query": "What language is DocuMind in?"},
            headers=auth_headers,
        )

    message_id = ask_resp.json()["message_id"]

    db = db_session_module.SessionLocal()
    msg = db.query(Message).filter(Message.id == message_id).first()
    conv = db.query(Conversation).filter(Conversation.id == msg.conversation_id).first()
    doc = Document(id="doc-2", user_id=conv.user_id, filename="code.txt", status="ready")
    chunk = Chunk(id="chunk-doc-2", document_id="doc-2", chunk_index=0, text="DocuMind is an NLP document search engine written in Python.")
    db.add(doc)
    db.add(chunk)
    db.commit()
    db.close()

    eval_run_resp = client.post(
        "/api/v1/eval/run",
        json={"message_id": message_id},
        headers=auth_headers,
    )
    assert eval_run_resp.status_code == 202

    evals = client.get(f"/api/v1/eval/message/{message_id}", headers=auth_headers).json()
    assert len(evals) >= 1
    assert evals[0]["groundedness_score"] < 0.70
    assert evals[0]["flagged"] is True


def test_eval_rejects_user_role_message(auth_headers):
    client = TestClient(app)

    with patch(
        "app.api.v1.endpoints.chat.retrieval_service.retrieve_context",
        return_value=[],
    ), patch(
        "app.services.llm_service.LLMService.generate_rag_answer",
        return_value=("Answer", []),
    ):
        ask_resp = client.post(
            "/api/v1/chat/ask",
            json={"query": "Hello"},
            headers=auth_headers,
        )

    conv_id = ask_resp.json()["conversation_id"]
    db = db_session_module.SessionLocal()
    user_msg = db.query(Message).filter(Message.conversation_id == conv_id, Message.role == "user").first()
    db.close()

    # Attempting to evaluate user message returns 400
    bad_resp = client.post("/api/v1/eval/run", json={"message_id": user_msg.id}, headers=auth_headers)
    assert bad_resp.status_code == 400
    assert "assistant messages" in bad_resp.json()["detail"].lower()


def test_eval_cross_tenant_isolation(auth_headers, other_user_headers):
    client = TestClient(app)

    # User 1 asks a question
    with patch(
        "app.api.v1.endpoints.chat.retrieval_service.retrieve_context",
        return_value=[],
    ), patch(
        "app.services.llm_service.LLMService.generate_rag_answer",
        return_value=("Honest refusal answer", []),
    ):
        ask_resp = client.post(
            "/api/v1/chat/ask",
            json={"query": "Secret question"},
            headers=auth_headers,
        )

    user1_msg_id = ask_resp.json()["message_id"]

    # User 2 tries to evaluate User 1's message -> 404
    unauth_eval = client.post(
        "/api/v1/eval/run",
        json={"message_id": user1_msg_id},
        headers=other_user_headers,
    )
    assert unauth_eval.status_code == 404


def test_eval_rejects_other_user_chunks_as_evidence(auth_headers, other_user_headers):
    """
    Verify that an assistant message citing chunks belonging to another user's documents
    has those chunks rejected as valid evidence, resulting in score 0.0 and flagged=True.
    """
    client = TestClient(app)

    # 1. User 1 asks a question and receives a message
    with patch(
        "app.api.v1.endpoints.chat.retrieval_service.retrieve_context",
        return_value=[],
    ), patch(
        "app.services.llm_service.LLMService.generate_rag_answer",
        return_value=(
            "Answer citing someone else's document [chunk:chunk-user2-secret].",
            [{"chunk_id": "chunk-user2-secret", "document_id": "doc-user2"}],
        ),
    ):
        ask_resp = client.post(
            "/api/v1/chat/ask",
            json={"query": "Snooping question"},
            headers=auth_headers,
        )
    assert ask_resp.status_code == 200
    user1_msg_id = ask_resp.json()["message_id"]

    # 2. Insert User 2's document and chunk into DB
    db = db_session_module.SessionLocal()
    # Find User 2's ID by decoding other_user_headers or creating a doc
    # We can create a user 2 doc with a unique user_id
    foreign_user_id = "foreign-user-id-" + uuid.uuid4().hex[:6]
    doc2 = Document(id="doc-user2", user_id=foreign_user_id, filename="confidential.txt", status="ready")
    chunk2 = Chunk(id="chunk-user2-secret", document_id="doc-user2", chunk_index=0, text="Super confidential secret data.")
    db.add(doc2)
    db.add(chunk2)
    db.commit()
    db.close()

    # 3. User 1 evaluates their message citing User 2's chunk
    eval_resp = client.post(
        "/api/v1/eval/run",
        json={"message_id": user1_msg_id},
        headers=auth_headers,
    )
    assert eval_resp.status_code == 202

    # 4. Result must have rejected the foreign chunk
    evals = client.get(f"/api/v1/eval/message/{user1_msg_id}", headers=auth_headers).json()
    assert len(evals) >= 1
    assert evals[0]["groundedness_score"] == 0.0
    assert evals[0]["flagged"] is True
    assert "not owned by the user" in evals[0]["notes"]

