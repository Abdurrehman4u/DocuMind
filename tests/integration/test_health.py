from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_api_boot_liveness():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "environment" in data
    assert "llm_provider" in data
    assert "embedding_model" in data
