from __future__ import annotations

from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    text_snippet: str | None = None


class ChatAskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="User question to answer")
    conversation_id: str | None = Field(default=None, description="Optional existing conversation UUID")
    document_ids: list[str] | None = Field(default=None, description="Optional list of document UUIDs to scope retrieval")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    role: str
    content: str
    citations: list[dict[str, Any]] | None = None
    created_at: datetime


class ChatAskResponse(BaseModel):
    conversation_id: str
    message_id: str
    answer: str
    citations: list[Citation]


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    title: str
    created_at: datetime
    messages: list[MessageResponse] | None = None


class ConversationListResponse(BaseModel):
    conversations: list[ConversationResponse]
    total: int
