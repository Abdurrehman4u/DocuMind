"""
Agent endpoints — autonomous multi-step document research and tool-calling.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, get_db
from app.models.user import User
from app.schemas.agent import AgentRunRequest, AgentRunResponse
from app.services.agent_service import get_agent_service

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/run", response_model=AgentRunResponse, status_code=status.HTTP_200_OK)
def run_agent_task(
    request: AgentRunRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AgentRunResponse:
    """
    Execute a multi-step autonomous agent task with tool calling.

    The agent can search documents, query structured database metadata, and generate reports.
    Returns the final synthesized answer and full tool execution trace.
    """
    try:
        agent_svc = get_agent_service()
        response = agent_svc.run_agent(
            task=request.task,
            user_id=current_user.id,
            db=db,
            max_steps=request.max_steps,
        )
        return response
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(val_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution encountered an error: {str(exc)}",
        )
