"""
Phase 5 integration tests — Task status tracking (/api/v1/tasks).
"""

import io
import uuid
from typing import Generator
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def auth_headers() -> Generator[dict, None, None]:
    client = TestClient(app)
    email = f"tasktest_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "TaskPass123!"},
    )
    assert resp.status_code == 201
    token = resp.json()["access_token"]
    yield {"Authorization": f"Bearer {token}"}


@pytest.fixture
def other_user_headers() -> Generator[dict, None, None]:
    client = TestClient(app)
    email = f"othertask_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "OtherTask123!"},
    )
    assert resp.status_code == 201
    token = resp.json()["access_token"]
    yield {"Authorization": f"Bearer {token}"}


def test_upload_creates_trackable_task(auth_headers):
    client = TestClient(app)
    fake_txt = io.BytesIO(b"Task integration test content. " * 10)

    # Upload document
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("task_doc.txt", fake_txt, "text/plain")},
        headers=auth_headers,
    )
    assert resp.status_code == 201

    # List tasks
    tasks_resp = client.get("/api/v1/tasks", headers=auth_headers)
    assert tasks_resp.status_code == 200
    data = tasks_resp.json()
    assert data["total"] >= 1
    task_id = data["tasks"][0]["id"]
    assert data["tasks"][0]["type"] == "ingestion"

    # Query individual task
    single_task = client.get(f"/api/v1/tasks/{task_id}", headers=auth_headers)
    assert single_task.status_code == 200
    task_data = single_task.json()
    assert task_data["id"] == task_id
    assert task_data["status"] in ("pending", "started", "success")


def test_task_cross_user_isolation(auth_headers, other_user_headers):
    client = TestClient(app)
    fake_txt = io.BytesIO(b"Private document.")

    client.post(
        "/api/v1/documents/upload",
        files={"file": ("private.txt", fake_txt, "text/plain")},
        headers=auth_headers,
    )

    tasks_resp = client.get("/api/v1/tasks", headers=auth_headers)
    task_id = tasks_resp.json()["tasks"][0]["id"]

    # User 2 attempts to query User 1's task
    unauth_resp = client.get(f"/api/v1/tasks/{task_id}", headers=other_user_headers)
    assert unauth_resp.status_code == 404
