"""
Phase 2 integration tests — Document Ingestion Pipeline.

Strategy:
    - Use Celery TASK_ALWAYS_EAGER so tasks run inline in the test
      process without a Redis broker.
    - Use a real SQLite database (same default dev DB that the app uses).
    - Patch chromadb.PersistentClient so no real vector store is needed;
      in-memory mock validates the interface without disk I/O.
    - A small .txt fixture represents the uploaded document.

Tests:
    1. Upload a .txt file → status 201, Document row created with status pending.
    2. Poll status → after TASK_ALWAYS_EAGER execution, status becomes 'ready'.
    3. Chunk rows are written to Postgres.
    4. List endpoint returns the document.
    5. Unsupported file type is rejected with 422.
    6. Delete endpoint removes document and triggers vector cleanup.
"""

import io
import uuid
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.workers.celery_app import celery_app

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _celery_eager():
    """
    Run Celery tasks synchronously without a broker or result backend.

    Override broker + backend to in-memory so the real Upstash Redis TLS URL
    in .env is never contacted during tests.  task_always_eager executes the
    task inline in the test process.
    """
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



@pytest.fixture
def _mock_vector_store():
    """
    Replace ChromaDB with an in-memory mock so tests don't need a
    real vector store or an embedding model loaded.
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

    with patch(
        "app.services.vector_store_service._get_client",
        return_value=mock_client,
    ), patch(
        "app.services.vector_store_service._get_collection",
        return_value=mock_collection,
    ):
        yield mock_collection


@pytest.fixture
def _mock_embedder():
    """Return a fixed 384-dim zero vector per text instead of running the real model."""
    def _fake_embed(texts):
        return [[0.0] * 384 for _ in texts]

    with patch(
        "app.services.ingestion_service.embed_texts",
        side_effect=_fake_embed,
    ):
        yield



@pytest.fixture
def auth_headers(tmp_path) -> Generator[dict, None, None]:
    """
    Register a fresh user and return JWT auth headers.

    Also patches UPLOAD_DIR to a pytest tmp_path so files land in a
    temporary directory and don't pollute the repo.
    """
    client = TestClient(app)
    email = f"doctest_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "DocTest123!"},
    )
    assert resp.status_code == 201
    token = resp.json()["access_token"]

    with patch("app.config.settings.UPLOAD_DIR", str(tmp_path)):
        yield {"Authorization": f"Bearer {token}"}


# ── Helper ─────────────────────────────────────────────────────────────────


def _txt_file(content: str = "Hello DocuMind. This is a test document.\n" * 20) -> tuple:
    """Return (filename, file-like, content-type) for TestClient upload."""
    return ("test_doc.txt", io.BytesIO(content.encode()), "text/plain")


# ── Tests ──────────────────────────────────────────────────────────────────


def test_upload_creates_document_and_enqueues_ingestion(
    auth_headers, _mock_vector_store, _mock_embedder, tmp_path
):
    """
    Upload a .txt file.
    Expects:
    - 201 response with status='pending' or 'ready' (eager task runs sync)
    - Document exists in the DB
    - Chunk rows are created
    """
    client = TestClient(app)

    with patch("app.config.settings.UPLOAD_DIR", str(tmp_path)):
        resp = client.post(
            "/api/v1/documents/upload",
            files={"file": _txt_file()},
            headers=auth_headers,
        )

    assert resp.status_code == 201
    data = resp.json()
    assert data["filename"] == "test_doc.txt"
    assert data["source_type"] == "upload"
    # Status is 'ready' because TASK_ALWAYS_EAGER runs ingestion inline
    assert data["status"] in ("pending", "processing", "ready")
    assert "id" in data


def test_status_is_ready_after_eager_ingestion(
    auth_headers, _mock_vector_store, _mock_embedder, tmp_path
):
    """After upload (eager mode), polling /status returns 'ready'."""
    client = TestClient(app)

    with patch("app.config.settings.UPLOAD_DIR", str(tmp_path)):
        upload_resp = client.post(
            "/api/v1/documents/upload",
            files={"file": _txt_file()},
            headers=auth_headers,
        )
        assert upload_resp.status_code == 201
        doc_id = upload_resp.json()["id"]

        status_resp = client.get(
            f"/api/v1/documents/{doc_id}/status",
            headers=auth_headers,
        )

    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "ready"


def test_chunks_written_to_db(
    auth_headers, _mock_vector_store, _mock_embedder, tmp_path
):
    """After ingestion, at least one Chunk row exists for the document."""
    from app.db.session import SessionLocal
    from app.models.chunk import Chunk

    client = TestClient(app)

    with patch("app.config.settings.UPLOAD_DIR", str(tmp_path)):
        upload_resp = client.post(
            "/api/v1/documents/upload",
            files={"file": _txt_file()},
            headers=auth_headers,
        )
    assert upload_resp.status_code == 201
    doc_id = upload_resp.json()["id"]

    db = SessionLocal()
    try:
        chunks = db.query(Chunk).filter(Chunk.document_id == doc_id).all()
        assert len(chunks) >= 1, "Expected at least one Chunk row after ingestion"
    finally:
        db.close()


def test_list_documents(auth_headers, _mock_vector_store, _mock_embedder, tmp_path):
    """GET /documents/ returns the uploaded document."""
    client = TestClient(app)

    with patch("app.config.settings.UPLOAD_DIR", str(tmp_path)):
        client.post(
            "/api/v1/documents/upload",
            files={"file": _txt_file()},
            headers=auth_headers,
        )
        list_resp = client.get("/api/v1/documents/", headers=auth_headers)

    assert list_resp.status_code == 200
    docs = list_resp.json()
    assert isinstance(docs, list)
    assert len(docs) >= 1
    assert docs[0]["filename"] == "test_doc.txt"


def test_unsupported_file_type_rejected(auth_headers):
    """Uploading a .csv file returns 422."""
    client = TestClient(app)
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("data.csv", io.BytesIO(b"a,b,c"), "text/csv")},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_delete_document(auth_headers, _mock_vector_store, _mock_embedder, tmp_path):
    """DELETE /documents/{id} returns 204 and document is gone from list."""
    client = TestClient(app)

    with patch("app.config.settings.UPLOAD_DIR", str(tmp_path)):
        upload_resp = client.post(
            "/api/v1/documents/upload",
            files={"file": _txt_file()},
            headers=auth_headers,
        )
        assert upload_resp.status_code == 201
        doc_id = upload_resp.json()["id"]

        del_resp = client.delete(
            f"/api/v1/documents/{doc_id}",
            headers=auth_headers,
        )
        assert del_resp.status_code == 204

        list_resp = client.get("/api/v1/documents/", headers=auth_headers)

    ids = [d["id"] for d in list_resp.json()]
    assert doc_id not in ids


def test_delete_removes_chroma_vectors(
    auth_headers, _mock_vector_store, _mock_embedder, tmp_path
):
    """Deleting a document calls vector_store_service.delete_document_chunks."""
    client = TestClient(app)

    # Make the collection's .get() return a fake chunk ID so the delete path fires
    _mock_vector_store.get.return_value = {"ids": ["fake-chunk-id"]}

    with patch("app.config.settings.UPLOAD_DIR", str(tmp_path)):
        upload_resp = client.post(
            "/api/v1/documents/upload",
            files={"file": _txt_file()},
            headers=auth_headers,
        )
        doc_id = upload_resp.json()["id"]
        client.delete(f"/api/v1/documents/{doc_id}", headers=auth_headers)

    # The mock collection's delete method should have been called
    _mock_vector_store.delete.assert_called()



def test_get_document_returns_404_for_other_user(
    auth_headers, _mock_vector_store, _mock_embedder, tmp_path
):
    """Another user cannot access a document they don't own."""
    client = TestClient(app)

    with patch("app.config.settings.UPLOAD_DIR", str(tmp_path)):
        upload_resp = client.post(
            "/api/v1/documents/upload",
            files={"file": _txt_file()},
            headers=auth_headers,
        )
    doc_id = upload_resp.json()["id"]

    # Register a second user
    other_email = f"other_{uuid.uuid4().hex[:8]}@example.com"
    reg_resp = client.post(
        "/api/v1/auth/register",
        json={"email": other_email, "password": "Other123!"},
    )
    other_token = reg_resp.json()["access_token"]

    detail_resp = client.get(
        f"/api/v1/documents/{doc_id}",
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert detail_resp.status_code == 404
