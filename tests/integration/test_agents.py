"""
Phase 4 integration tests — Agent & Tool-Calling Layer (/api/v1/agents/run).
"""

import uuid
from typing import Generator
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.agent import AgentRunResponse, ToolCallRecord


@pytest.fixture
def auth_headers() -> Generator[dict, None, None]:
    """Register a fresh user and return JWT auth headers."""
    client = TestClient(app)
    email = f"agenttest_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "AgentPass123!"},
    )
    assert resp.status_code == 201
    token = resp.json()["access_token"]
    yield {"Authorization": f"Bearer {token}"}


def test_agent_run_requires_authentication():
    client = TestClient(app)
    resp = client.post("/api/v1/agents/run", json={"task": "Find documents"})
    assert resp.status_code == 401


def test_agent_run_endpoint_success(auth_headers):
    client = TestClient(app)

    mock_agent_response = AgentRunResponse(
        task="Find onboarding docs and summarize into a report",
        final_answer="Here is the final onboarding report synthesized from your documents.",
        steps_taken=2,
        trace=[
            ToolCallRecord(
                step=1,
                tool="search_knowledge_base",
                arguments={"query": "onboarding"},
                result=[{"chunk_id": "c1", "text": "Step 1: Setup laptop"}],
            ),
            ToolCallRecord(
                step=2,
                tool="generate_report",
                arguments={"title": "Onboarding", "summary": "Guide", "sections": []},
                result={"markdown_report": "# Onboarding\nGuide"},
            ),
        ],
    )

    with patch(
        "app.api.v1.endpoints.agents.get_agent_service"
    ) as mock_get_svc:
        mock_svc = MagicMock()
        mock_svc.run_agent.return_value = mock_agent_response
        mock_get_svc.return_value = mock_svc

        resp = client.post(
            "/api/v1/agents/run",
            json={
                "task": "Find onboarding docs and summarize into a report",
                "max_steps": 5,
            },
            headers=auth_headers,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["task"] == "Find onboarding docs and summarize into a report"
    assert "onboarding report" in data["final_answer"]
    assert data["steps_taken"] == 2
    assert len(data["trace"]) == 2
    assert data["trace"][0]["step"] == 1
    assert data["trace"][0]["tool"] == "search_knowledge_base"
    assert data["trace"][1]["step"] == 2
    assert data["trace"][1]["tool"] == "generate_report"
