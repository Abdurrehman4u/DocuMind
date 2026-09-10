import logging
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="tasks.ingest_document", bind=True)
def ingest_document_task(self, document_id: str):
    """Background Celery task for document ingestion, chunking, and embedding."""
    logger.info(f"Starting ingestion task for document_id={document_id}")
    # Task logic will be implemented in Phase 2
    return {"status": "success", "document_id": document_id}


@celery_app.task(name="tasks.run_eval", bind=True)
def run_eval_task(self, message_id: str):
    """Background Celery task for answer groundedness evaluation."""
    logger.info(f"Starting evaluation task for message_id={message_id}")
    # Task logic will be implemented in Phase 5
    return {"status": "success", "message_id": message_id}
