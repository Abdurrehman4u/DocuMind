"""
Task endpoints — query async task states (ingestion, evaluation).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, get_db
from app.models.task import Task
from app.models.user import User
from app.schemas.task import TaskListResponse, TaskResponse
from app.workers.celery_app import celery_app

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}", response_model=TaskResponse)
def get_task_status(
    task_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TaskResponse:
    """
    Get the status of a background task by Task UUID or Celery task ID.
    Mirrors and synchronizes state between Celery and the Task table.
    """
    task = (
        db.query(Task)
        .filter(
            (Task.id == task_id) | (Task.celery_task_id == task_id),
            Task.user_id == current_user.id,
        )
        .first()
    )
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    # If status is still pending/started, inspect Celery state if available
    if task.status in ("pending", "started") and task.celery_task_id:
        try:
            async_res = celery_app.AsyncResult(task.celery_task_id)
            if async_res.ready():
                if async_res.successful():
                    task.status = "success"
                    if not task.result:
                        task.result = async_res.result if isinstance(async_res.result, dict) else {"result": str(async_res.result)}
                elif async_res.failed():
                    task.status = "failure"
                    task.result = {"error": str(async_res.result)}
                db.commit()
                db.refresh(task)
        except Exception:
            # Fall back to DB state if broker is unreachable
            pass

    return TaskResponse.model_validate(task)


@router.get("", response_model=TaskListResponse)
def list_tasks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TaskListResponse:
    """List all async background tasks for the current user."""
    tasks = (
        db.query(Task)
        .filter(Task.user_id == current_user.id)
        .order_by(Task.created_at.desc())
        .all()
    )
    task_responses = [TaskResponse.model_validate(t) for t in tasks]
    return TaskListResponse(tasks=task_responses, total=len(task_responses))
