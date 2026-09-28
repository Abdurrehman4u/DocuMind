"""
Pydantic schemas for Document-related request/response bodies.
"""

from datetime import datetime
from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    """Full document metadata returned on create, list, and detail endpoints."""
    id: str
    user_id: str
    filename: str
    source_type: str
    status: str
    error_message: str | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentStatusResponse(BaseModel):
    """Lightweight status-polling response."""
    id: str
    filename: str
    status: str
    error_message: str | None = None

    model_config = ConfigDict(from_attributes=True)
