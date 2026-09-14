from fastapi import APIRouter
from app.config import settings

router = APIRouter()


@router.get("/health", tags=["Health"])
async def health_check():
    """API boot and liveness check endpoint."""
    return {
        "status": "ok",
        "environment": settings.ENV,
        "llm_provider": settings.LLM_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL,
    }
