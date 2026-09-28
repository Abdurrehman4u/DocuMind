import logging
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="tasks.ingest_document", bind=True, max_retries=3)
def ingest_document_task(self, document_id: str):
    """
    Celery task: run the full ingestion pipeline for a single document.

    Workflow:
        parse file → chunk → embed → upsert ChromaDB → write Chunk rows
        → set Document.status = 'ready' (or 'failed' on error)
    """
    from app.db.session import SessionLocal
    from app.services.ingestion_service import run_ingestion

    logger.info("Ingestion task started for document_id=%s", document_id)
    db = SessionLocal()
    try:
        run_ingestion(document_id=document_id, db=db)
        logger.info("Ingestion task completed for document_id=%s", document_id)
        return {"status": "success", "document_id": document_id}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingestion task crashed for document_id=%s: %s", document_id, exc)
        # ingestion_service already sets Document.status='failed' on most paths,
        # but if the session itself died we can't do much more here.
        raise self.retry(exc=exc, countdown=10) from exc
    finally:
        db.close()


@celery_app.task(name="tasks.run_eval", bind=True)
def run_eval_task(self, message_id: str):
    """[Phase 5 Placeholder] Background Celery task for answer groundedness evaluation."""
    logger.info("Evaluation task started for message_id=%s", message_id)
    # Task logic will be implemented in Phase 5
    return {"status": "success", "message_id": message_id}

