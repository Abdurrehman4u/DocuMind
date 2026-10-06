"""
Agent Service — bounded multi-step autonomous agent loop with tool-calling.

Executes up to max_steps tool calls, records a full execution trace,
and returns the final synthesized answer.
"""

from __future__ import annotations

from typing import Any
from sqlalchemy.orm import Session
from google import genai
from google.genai import types

from app.config import settings
from app.schemas.agent import AgentRunResponse, ToolCallRecord
from app.services.tool_registry import execute_tool, get_gemini_tools

AGENT_SYSTEM_PROMPT = """You are DocuMind's Autonomous Document Intelligence Agent.
Your objective is to complete multi-step document research, metadata query, and reporting tasks.

Guidelines:
1. Break down the user's task logically.
2. Select and call the most relevant tool (e.g. search_knowledge_base, query_structured_data, generate_report).
3. Analyze the results from each tool call.
4. When you have gathered enough facts to fulfill the task, synthesize your final comprehensive response in Markdown without calling any additional tools.
5. If the request asks for a report, call the generate_report tool to format it before delivering the final answer.
"""


class AgentService:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self.model = model or settings.LLM_MODEL
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is not configured for the Agent service.")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def run_agent(
        self,
        task: str,
        user_id: str,
        db: Session,
        max_steps: int = 5,
    ) -> AgentRunResponse:
        """
        Execute the agent loop for the given task.

        Parameters
        ----------
        task:
            The user prompt or instruction.
        user_id:
            Tenant UUID to enforce data isolation.
        db:
            Database session for relational metadata tools.
        max_steps:
            Maximum number of sequential tool calls.
        """
        client = self._get_client()
        gemini_tools = get_gemini_tools()

        contents: list[types.Content] = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=task)],
            )
        ]

        config = types.GenerateContentConfig(
            system_instruction=AGENT_SYSTEM_PROMPT,
            temperature=0.2,
            tools=gemini_tools,
        )

        trace: list[ToolCallRecord] = []
        final_answer = ""

        for step in range(1, max_steps + 1):
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

            # Check for function calls
            function_calls = getattr(response, "function_calls", None)
            if not function_calls and hasattr(response, "candidates") and response.candidates:
                # Fallback check inside candidates parts
                first_cand = response.candidates[0]
                if first_cand.content and first_cand.content.parts:
                    function_calls = [
                        p.function_call for p in first_cand.content.parts if getattr(p, "function_call", None)
                    ]

            # If no tool calls were made, the agent finished its reasoning
            if not function_calls:
                final_answer = response.text or "Task completed."
                break

            # Execute each requested tool call
            # Add the model's call message into history
            if response.candidates and response.candidates[0].content:
                contents.append(response.candidates[0].content)

            tool_response_parts: list[types.Part] = []

            for fc in function_calls:
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}

                # Execute tool
                tool_result = execute_tool(
                    name=tool_name,
                    arguments=tool_args,
                    user_id=user_id,
                    db=db,
                )

                # Record trace
                trace.append(
                    ToolCallRecord(
                        step=len(trace) + 1,
                        tool=tool_name,
                        arguments=tool_args,
                        result=tool_result,
                    )
                )

                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": tool_result},
                    )
                )

            # Feed tool outputs back to the model as role="tool"
            contents.append(
                types.Content(
                    role="tool",
                    parts=tool_response_parts,
                )
            )

            # If we reached max_steps, request a final synthesis
            if len(trace) >= max_steps:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(
                                text="You have reached the maximum allowed tool execution steps. Please provide your final answer now based on the information gathered so far."
                            )
                        ],
                    )
                )
                final_resp = client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=AGENT_SYSTEM_PROMPT,
                        temperature=0.2,
                    ),
                )
                final_answer = final_resp.text or "Max steps reached. Final answer synthesized from tool results."
                break

        if not final_answer and trace:
            final_answer = "Task executed with tool calls."

        return AgentRunResponse(
            task=task,
            final_answer=final_answer,
            steps_taken=len(trace),
            trace=trace,
        )


_agent_service: AgentService | None = None


def get_agent_service() -> AgentService:
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService()
    return _agent_service
