from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    task: str = Field(..., min_length=2, max_length=4000, description="Task or multi-step instruction for the agent")
    max_steps: int = Field(default=5, ge=1, le=10, description="Maximum number of tool execution steps")


class ToolCallRecord(BaseModel):
    step: int
    tool: str
    arguments: dict[str, Any]
    result: Any


class AgentRunResponse(BaseModel):
    task: str
    final_answer: str
    steps_taken: int
    trace: list[ToolCallRecord]
