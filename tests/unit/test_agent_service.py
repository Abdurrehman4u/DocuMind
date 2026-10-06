"""
Unit tests for agent service and tool registry.
"""

from unittest.mock import MagicMock, patch
from sqlalchemy.orm import Session
import pytest

from app.services.tool_registry import execute_tool, get_gemini_tools, TOOL_DEFINITIONS
from app.services.agent_service import AgentService


def test_tool_registry_definitions():
    assert len(TOOL_DEFINITIONS) == 3
    tool_names = {t["name"] for t in TOOL_DEFINITIONS}
    assert "search_knowledge_base" in tool_names
    assert "query_structured_data" in tool_names
    assert "generate_report" in tool_names


def test_get_gemini_tools():
    tools = get_gemini_tools()
    assert len(tools) == 1
    assert len(tools[0].function_declarations) == 3


def test_execute_tool_unrecognized():
    mock_db = MagicMock(spec=Session)
    res = execute_tool("unknown_tool", {}, user_id="u1", db=mock_db)
    assert "error" in res
    assert "not recognized" in res["error"]


@patch("app.services.agent_service.genai.Client")
def test_agent_service_single_tool_execution(mock_client_cls):
    """Test agent execution with one tool call followed by final synthesis."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_db = MagicMock(spec=Session)

    # Turn 1: Model asks to call search_knowledge_base
    fc_mock = MagicMock()
    fc_mock.name = "search_knowledge_base"
    fc_mock.args = {"query": "project status"}

    resp1 = MagicMock()
    resp1.function_calls = [fc_mock]
    resp1.candidates = [MagicMock(content=MagicMock(parts=[]))]
    resp1.text = None

    # Turn 2: Model returns final answer
    resp2 = MagicMock()
    resp2.function_calls = None
    resp2.candidates = [MagicMock(content=MagicMock(parts=[]))]
    resp2.text = "Based on the search results, the project is on track."

    mock_client.models.generate_content.side_effect = [resp1, resp2]

    with patch("app.services.agent_service.execute_tool") as mock_exec:
        mock_exec.return_value = [{"chunk_id": "c1", "text": "Status: on track"}]

        svc = AgentService(api_key="test-key", model="gemini-2.5-flash")
        result = svc.run_agent(task="What is the project status?", user_id="u1", db=mock_db, max_steps=5)

    assert result.task == "What is the project status?"
    assert result.steps_taken == 1
    assert len(result.trace) == 1
    assert result.trace[0].tool == "search_knowledge_base"
    assert result.trace[0].arguments == {"query": "project status"}
    assert "on track" in result.final_answer


@patch("app.services.agent_service.genai.Client")
def test_agent_service_multi_step_chain(mock_client_cls):
    """Test agent chaining 2 tool calls in sequence before providing final answer."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_db = MagicMock(spec=Session)

    # Step 1: search_knowledge_base
    fc1 = MagicMock()
    fc1.name = "search_knowledge_base"
    fc1.args = {"query": "onboarding"}
    resp1 = MagicMock(function_calls=[fc1], candidates=[MagicMock(content=MagicMock(parts=[]))], text=None)

    # Step 2: generate_report
    fc2 = MagicMock()
    fc2.name = "generate_report"
    fc2.args = {
        "title": "Onboarding Summary",
        "summary": "Step by step onboarding",
        "sections": [{"heading": "Overview", "content": "Welcome guide"}],
    }
    resp2 = MagicMock(function_calls=[fc2], candidates=[MagicMock(content=MagicMock(parts=[]))], text=None)

    # Step 3: Final synthesis
    resp3 = MagicMock(
        function_calls=None,
        candidates=[MagicMock(content=MagicMock(parts=[]))],
        text="# Onboarding Summary\nReport has been compiled successfully.",
    )

    mock_client.models.generate_content.side_effect = [resp1, resp2, resp3]

    with patch("app.services.agent_service.execute_tool") as mock_exec:
        mock_exec.side_effect = [
            [{"chunk_id": "c1", "text": "Onboarding steps..."}],
            {"markdown_report": "# Onboarding Summary..."},
        ]

        svc = AgentService(api_key="test-key", model="gemini-2.5-flash")
        result = svc.run_agent(
            task="Find onboarding docs and create a report",
            user_id="u1",
            db=mock_db,
            max_steps=5,
        )

    assert result.steps_taken == 2
    assert len(result.trace) == 2
    assert result.trace[0].tool == "search_knowledge_base"
    assert result.trace[1].tool == "generate_report"
    assert "Onboarding Summary" in result.final_answer
