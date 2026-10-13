"""
Root test configuration and fixtures.

Ensures:
1. Lightweight, isolated in-memory SQLite database (no connection to Neon/remote Postgres).
2. Celery tasks execute eagerly in-memory (no connection to Upstash/Redis).
3. Embeddings are mocked by default so SentenceTransformer / PyTorch is never loaded into RAM.
4. ChromaDB is mocked in-memory so no disk I/O or background threads occur.
5. Google GenAI / Gemini calls never hit the network.
"""

from __future__ import annotations

import os

# ── Early environment variable setup (before app imports) ───────────────────
os.environ["ENV"] = "testing"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["SECRET_KEY"] = "test-secret-key-32-bytes-minimum-length!!"
os.environ["REFRESH_SECRET_KEY"] = "test-refresh-secret-key-32-bytes-minimum-length!!"
os.environ["GEMINI_API_KEY"] = ""
os.environ["CHROMA_PERSIST_DIR"] = ":memory:"

import shutil
import uuid
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# Import models and declarative base to build schema
from app.config import settings
settings.GEMINI_API_KEY = ""
from app.db.base_class import Base
import app.models  # noqa: F401
from app.core.dependencies import get_db
from app.main import app
import app.db.session as session_module
from app.workers.celery_app import celery_app
from app.services import embedding_service, vector_store_service


# ── Database fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_engine():
    """Create a single in-memory SQLite engine with StaticPool for the entire test session."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_module.engine = engine
    session_module.SessionLocal.configure(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def _isolate_db(test_engine) -> Generator[None, None, None]:
    """
    Ensure every test uses the in-memory test SQLite engine,
    and wipe all table contents between test runs for clean isolation.
    """
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    def _override_get_db() -> Generator[Session, None, None]:
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db

    orig_session_local = session_module.SessionLocal
    session_module.SessionLocal = TestingSessionLocal

    yield

    app.dependency_overrides.pop(get_db, None)
    session_module.SessionLocal = orig_session_local

    # Clean up table rows between tests
    with test_engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
        conn.commit()


# ── Celery fixture ───────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _configure_celery():
    """Configure Celery to run tasks inline without a Redis broker."""
    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
        broker_url="memory://",
        result_backend="cache+memory://",
    )
    yield
    celery_app.conf.update(
        task_always_eager=False,
        task_eager_propagates=False,
    )


# ── Mock Embedding Fixture ──────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _mock_embeddings_globally(monkeypatch):
    """
    Prevent loading SentenceTransformer and PyTorch during tests.
    Generates deterministic 384-dimensional zero vectors.
    """
    def _fake_embed_texts(texts):
        return [[0.0] * 384 for _ in texts]

    def _fake_embed_query(query):
        return [0.0] * 384

    monkeypatch.setattr(embedding_service, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(embedding_service, "embed_query", _fake_embed_query)


# ── Mock ChromaDB Fixture ────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _mock_chromadb_globally(monkeypatch):
    """
    Prevent ChromaDB from initializing disk persistence or telemetry threads during tests.
    """
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    mock_collection.get.return_value = {"ids": []}

    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    monkeypatch.setattr(vector_store_service, "_get_client", lambda: mock_client)
    monkeypatch.setattr(vector_store_service, "_get_collection", lambda: mock_collection)


# ── Guard Gemini Calls ────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _prevent_gemini_network_calls(monkeypatch):
    """
    Ensure no test can ever make an un-mocked external network call to Google Gemini.
    """
    from google import genai

    def _blocked_client(*args, **kwargs):
        raise RuntimeError(
            "Blocked un-mocked Google GenAI Client initialization in test environment! "
            "Mock the client explicitly or rely on GEMINI_API_KEY='' for offline fallback."
        )

    monkeypatch.setattr(genai, "Client", _blocked_client)


# ── Isolated Workspace-Local Upload Directory Fixtures ───────────────────────

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
PYTEST_TEMP_DIR = WORKSPACE_ROOT / ".pytest_temp"
UPLOAD_TEMP_BASE = PYTEST_TEMP_DIR / "uploads"


@pytest.fixture(scope="session", autouse=True)
def _ensure_project_temp_dir():
    """Ensure workspace-local temporary directory is created and cleaned safely."""
    UPLOAD_TEMP_BASE.mkdir(parents=True, exist_ok=True)
    yield
    try:
        shutil.rmtree(PYTEST_TEMP_DIR, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _isolate_upload_dir(monkeypatch) -> Generator[Path, None, None]:
    """
    Ensure all tests store uploads in a safe, project-local workspace temporary directory.
    Completely avoids Windows OS %TEMP% permission and junction/symlink issues.
    Cleans up safely after each test run.
    """
    from app.config import settings
    test_upload_dir = UPLOAD_TEMP_BASE / uuid.uuid4().hex
    test_upload_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(test_upload_dir))

    yield test_upload_dir

    try:
        shutil.rmtree(test_upload_dir, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def tmp_path(_isolate_upload_dir) -> Path:
    """
    Safe project-local replacement for pytest's built-in tmp_path fixture.
    Guarantees that any test or fixture requesting tmp_path uses the
    workspace-local test directory and avoids Windows temp permission locks.
    """
    return _isolate_upload_dir

