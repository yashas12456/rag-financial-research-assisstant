"""Integration tests for the API layer (health + validation paths that don't
require the live Qdrant/embedding models, which are mocked here)."""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    with patch("app.services.ingestion_service.get_vector_store") as mock_vs:
        mock_vs.return_value.search.return_value = []
        from app.db.session import Base, engine
        Base.metadata.create_all(bind=engine)
        from app.main import app
        with TestClient(app) as tc:
            yield tc


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "embedding_model" in body


def test_query_rejects_empty_question(client):
    response = client.post("/api/query", json={"question": "ab"})  # below min_length=3
    assert response.status_code == 422


def test_list_documents_empty(client):
    response = client.get("/api/documents")
    assert response.status_code == 200
    assert response.json()["total"] >= 0


def test_get_nonexistent_document_returns_404(client):
    import uuid
    response = client.get(f"/api/documents/{uuid.uuid4()}")
    assert response.status_code == 404
