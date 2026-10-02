# ============================================================
# tests/test_api.py - FastAPI Integration Tests
#
# Learning Note:
#   Integration tests verify multiple components working together.
#   FastAPI provides a TestClient that simulates HTTP requests
#   WITHOUT actually starting a server. This makes tests fast.
#
#   We mock the RAG pipeline and ingestion to isolate the API layer.
#   Run with: pytest tests/test_api.py -v
# ============================================================

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("ENVIRONMENT", "development")

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """
    Create a FastAPI test client.
    scope="module" means this fixture runs once per test module
    (faster than creating it per test).
    """
    # Mock heavy components so tests run without real models
    with patch("src.ingestion.embedder.EmbeddingModel._load_model"):
        with patch("src.retrieval.hybrid_retriever.SemanticReRanker._load_model"):
            from src.api.main import app
            return TestClient(app, raise_server_exceptions=False)


# ===== HEALTH ENDPOINT TESTS =====

class TestHealthEndpoints:
    """Tests for the health check API endpoints."""

    def test_liveness_returns_200(self, client):
        """Liveness probe must always return 200."""
        res = client.get("/api/v1/health/live")
        assert res.status_code == 200
        assert res.json()["status"] == "alive"

    def test_readiness_returns_json(self, client):
        """Readiness probe returns JSON with status field."""
        res = client.get("/api/v1/health/ready")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded")

    def test_stats_returns_config_info(self, client):
        """Stats endpoint returns system configuration."""
        res = client.get("/api/v1/health/stats")
        assert res.status_code == 200
        data = res.json()
        assert "environment" in data
        assert "guardrails" in data
        assert "evaluations" in data

    def test_root_returns_response(self, client):
        """Root URL should return a response (UI or JSON)."""
        res = client.get("/")
        assert res.status_code in (200, 404)

    def test_docs_available(self, client):
        """Swagger docs should be accessible."""
        res = client.get("/docs")
        assert res.status_code == 200


# ===== QUERY ENDPOINT TESTS =====

class TestQueryEndpoint:
    """Tests for the main RAG query endpoint."""

    @patch("src.graph.pipeline.RAGPipeline.run")
    def test_query_returns_200(self, mock_run, client):
        """Valid query should return 200 with answer."""
        mock_run.return_value = {
            "answer": "Machine learning is a subset of AI.",
            "sources": ["doc.pdf"],
            "true_data_chunks": [],
            "noisy_data_chunks": [],
            "eval_metrics": {"evaluated": False},
            "pipeline_steps": ["input_guardrails", "retrieval", "generation"],
            "query_intent": "factual",
            "rewritten_query": "What is machine learning?",
            "hallucination_score": 0.8,
            "model_used": "gemini-1.5-flash",
            "latency_ms": 1200,
            "input_safe": True,
            "output_safe": True,
            "error": None,
        }

        res = client.post("/api/v1/query", json={
            "question": "What is machine learning?",
            "session_id": "test-session",
            "conversation_history": [],
        })

        assert res.status_code == 200
        data = res.json()
        assert "answer" in data
        assert data["answer"] == "Machine learning is a subset of AI."
        assert "sources" in data
        assert "eval_metrics" in data
        assert "pipeline_steps" in data

    def test_empty_question_returns_422(self, client):
        """Empty question should fail Pydantic validation."""
        res = client.post("/api/v1/query", json={
            "question": "",
        })
        assert res.status_code == 422  # Validation error

    def test_missing_question_returns_422(self, client):
        """Missing required field returns 422."""
        res = client.post("/api/v1/query", json={})
        assert res.status_code == 422

    def test_query_with_history(self, client):
        """Query with conversation history should be accepted."""
        with patch("src.graph.pipeline.RAGPipeline.run") as mock_run:
            mock_run.return_value = {
                "answer": "It refers to BERT from the previous context.",
                "sources": [],
                "true_data_chunks": [],
                "noisy_data_chunks": [],
                "eval_metrics": {},
                "pipeline_steps": [],
                "query_intent": "conversational",
                "rewritten_query": "What are BERT limitations?",
                "hallucination_score": 0.5,
                "model_used": "gemini-1.5-flash",
                "latency_ms": 800,
                "input_safe": True,
                "output_safe": True,
                "error": None,
            }

            res = client.post("/api/v1/query", json={
                "question": "What are its limitations?",
                "session_id": "sess-123",
                "conversation_history": [
                    {"role": "user", "content": "Tell me about BERT"},
                    {"role": "assistant", "content": "BERT is a transformer model..."},
                ],
            })
            assert res.status_code == 200


# ===== INGEST ENDPOINT TESTS =====

class TestIngestEndpoint:
    """Tests for the document ingestion endpoint."""

    def test_ingest_unsupported_type_returns_400(self, client):
        """Uploading an unsupported file type should return 400."""
        res = client.post(
            "/api/v1/ingest/file",
            files={"file": ("test.exe", b"fake content", "application/octet-stream")},
        )
        assert res.status_code == 400
        assert "unsupported" in res.json()["detail"].lower()

    @patch("src.ingestion.ingestion_pipeline.IngestionPipeline.ingest_bytes")
    def test_ingest_txt_file_success(self, mock_ingest, client):
        """Valid TXT file upload should succeed."""
        mock_ingest.return_value = {
            "status": "success",
            "source": "test.txt",
            "chunks_added": 5,
        }

        res = client.post(
            "/api/v1/ingest/file",
            files={"file": ("test.txt", b"This is test content for ingestion.", "text/plain")},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["chunks_added"] == 5

    def test_ingest_stats_returns_200(self, client):
        """Stats endpoint returns 200."""
        with patch("src.ingestion.ingestion_pipeline.IngestionPipeline.get_stats") as mock_stats:
            mock_stats.return_value = {"total_chunks": 42}
            res = client.get("/api/v1/ingest/stats")
            assert res.status_code == 200
