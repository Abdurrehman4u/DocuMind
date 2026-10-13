from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class EvalRunRequest(BaseModel):
    message_id: str = Field(..., description="UUID of the assistant Message to evaluate")


class EvalResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    message_id: str
    groundedness_score: float
    flagged: bool
    notes: str | None = None
    created_at: datetime


class EvalRunResponse(BaseModel):
    task_id: str
    message_id: str
    status: str
    eval_result_id: str | None = None
