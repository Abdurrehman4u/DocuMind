import logging
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="tasks.ingest_document", bind=True, max_retries=3)
def ingest_document_task(self, document_id: str, task_record_id: str | None = None):
    """
    Celery task: run the full ingestion pipeline for a single document.

    Workflow:
        parse file → chunk → embed → upsert ChromaDB → write Chunk rows
        → set Document.status = 'ready' (or 'failed' on error)
        → update Task table row if task_record_id provided
    """
    from app.db.session import SessionLocal
    from app.models.task import Task
    from app.services.ingestion_service import run_ingestion

    logger.info("Ingestion task started for document_id=%s", document_id)
    db = SessionLocal()
    task_row = None
    if task_record_id:
        task_row = db.query(Task).filter(Task.id == task_record_id).first()
    elif getattr(self.request, "id", None):
        task_row = db.query(Task).filter(Task.celery_task_id == self.request.id).first()

    if task_row:
        task_row.status = "started"
        db.commit()

    try:
        run_ingestion(document_id=document_id, db=db)
        logger.info("Ingestion task completed for document_id=%s", document_id)
        if task_row:
            task_row.status = "success"
            task_row.result = {"status": "success", "document_id": document_id}
            db.commit()
        return {"status": "success", "document_id": document_id}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingestion task crashed for document_id=%s: %s", document_id, exc)
        if task_row:
            task_row.status = "failure"
            task_row.result = {"error": str(exc), "document_id": document_id}
            db.commit()
        raise self.retry(exc=exc, countdown=10) from exc
    finally:
        db.close()


@celery_app.task(name="tasks.run_eval", bind=True)
def run_eval_task(self, message_id: str, task_record_id: str | None = None):
    """
    Background Celery task for answer groundedness evaluation.
    """
    from app.db.session import SessionLocal
    from app.models.message import Message
    from app.models.task import Task
    from app.services.eval_service import evaluate_message_groundedness

    logger.info("Evaluation task started for message_id=%s", message_id)
    db = SessionLocal()
    task_row = None
    if task_record_id:
        task_row = db.query(Task).filter(Task.id == task_record_id).first()
    elif getattr(self.request, "id", None):
        task_row = db.query(Task).filter(Task.celery_task_id == self.request.id).first()

    if task_row:
        task_row.status = "started"
        db.commit()

    try:
        message = db.query(Message).filter(Message.id == message_id).first()
        if not message:
            raise ValueError(f"Message {message_id} not found")

        eval_user_id = task_row.user_id if task_row else None
        eval_result = evaluate_message_groundedness(message=message, db=db, user_id=eval_user_id)
        res_data = {
            "status": "success",
            "eval_result_id": eval_result.id,
            "groundedness_score": eval_result.groundedness_score,
            "flagged": eval_result.flagged,
            "notes": eval_result.notes,
        }

        if task_row:
            task_row.status = "success"
            task_row.result = res_data
            db.commit()

        logger.info("Evaluation task finished for message_id=%s: score=%.2f, flagged=%s",
                    message_id, eval_result.groundedness_score, eval_result.flagged)
        return res_data
    except Exception as exc:  # noqa: BLE001
        logger.exception("Evaluation task crashed for message_id=%s: %s", message_id, exc)
        if task_row:
            task_row.status = "failure"
            task_row.result = {"error": str(exc), "message_id": message_id}
            db.commit()
        raise exc
    finally:
        db.close()
