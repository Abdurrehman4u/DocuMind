# Import Base and all models here so Alembic can import app.db.base to get Base.metadata
from app.db.base_class import Base
from app.models.user import User
from app.models.document import Document
from app.models.chunk import Chunk
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.task import Task
from app.models.eval_result import EvalResult

__all__ = [
    "Base",
    "User",
    "Document",
    "Chunk",
    "Conversation",
    "Message",
    "Task",
    "EvalResult",
]
