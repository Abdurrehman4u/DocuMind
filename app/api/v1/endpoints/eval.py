"""
Evaluation endpoints — trigger and query answer groundedness assessments.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, get_db
from app.models.conversation import Conversation
from app.models.eval_result import EvalResult
from app.models.message import Message
from app.models.task import Task
from app.models.user import User
from app.schemas.eval import EvalResultResponse, EvalRunRequest, EvalRunResponse
from app.workers.tasks import run_eval_task

router = APIRouter(prefix="/eval", tags=["eval"])


@router.post("/run", response_model=EvalRunResponse, status_code=status.HTTP_202_ACCEPTED)
def run_evaluation(
    request: EvalRunRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EvalRunResponse:
    """
    Enqueue an async groundedness evaluation for an assistant message.
    Returns a Task tracking ID immediately.
    """
    message = (
        db.query(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .filter(Message.id == request.message_id, Conversation.user_id == current_user.id)
        .first()
    )
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found",
        )

    if message.role != "assistant":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only assistant messages can be evaluated for groundedness.",
        )

    # Create Task tracking row
    task_row = Task(
        user_id=current_user.id,
        type="eval",
        status="pending",
    )
    db.add(task_row)
    db.commit()
    db.refresh(task_row)

    # Enqueue Celery task
    async_res = run_eval_task.delay(request.message_id, task_record_id=task_row.id)
    task_row.celery_task_id = async_res.id
    db.commit()
    db.refresh(task_row)

    eval_result_id = None
    if task_row.result and isinstance(task_row.result, dict):
        eval_result_id = task_row.result.get("eval_result_id")

    return EvalRunResponse(
        task_id=task_row.id,
        message_id=request.message_id,
        status=task_row.status,
        eval_result_id=eval_result_id,
    )


@router.get("/{id}", response_model=EvalResultResponse)
def get_evaluation_result(
    id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EvalResultResponse:
    """Retrieve an evaluation result by UUID."""
    eval_result = (
        db.query(EvalResult)
        .join(Message, EvalResult.message_id == Message.id)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .filter(EvalResult.id == id, Conversation.user_id == current_user.id)
        .first()
    )
    if not eval_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation result not found",
        )

    return EvalResultResponse.model_validate(eval_result)


@router.get("/message/{message_id}", response_model=list[EvalResultResponse])
def get_message_evaluations(
    message_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[EvalResultResponse]:
    """Retrieve all evaluation runs associated with a given message."""
    message = (
        db.query(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .filter(Message.id == message_id, Conversation.user_id == current_user.id)
        .first()
    )
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found",
        )

    eval_results = (
        db.query(EvalResult)
        .filter(EvalResult.message_id == message_id)
        .order_by(EvalResult.created_at.desc())
        .all()
    )
    return [EvalResultResponse.model_validate(e) for e in eval_results]
