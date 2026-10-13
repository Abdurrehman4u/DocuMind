from fastapi import APIRouter
from app.api.v1.endpoints import health, auth, documents, chat, agents, tasks, eval

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)
api_router.include_router(agents.router)
api_router.include_router(tasks.router)
api_router.include_router(eval.router)



