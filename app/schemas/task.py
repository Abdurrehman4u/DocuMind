from __future__ import annotations

from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class TaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    celery_task_id: str | None = None
    type: str  # ingestion, eval
    status: str  # pending, started, success, failure
    result: dict[str, Any] | None = None
    created_at: datetime


class TaskListResponse(BaseModel):
    tasks: list[TaskResponse]
    total: int
