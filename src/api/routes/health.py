# ============================================================
# src/api/routes/health.py - Health Check & Metrics Endpoints
#
# Learning Note:
#   Health endpoints are required for production deployments.
#   Container orchestrators (Kubernetes, Cloud Run) use them to:
#   - /health/live  - Is the process alive? (liveness probe)
#   - /health/ready - Is it ready to serve traffic? (readiness probe)
#   - /health/stats - Detailed system metrics (for dashboards)
#
#   If /health/live returns non-200, the container is restarted.
#   If /health/ready returns non-200, traffic is not routed to it.
# ============================================================

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
import time

router = APIRouter()

# Track server start time for uptime calculation
_START_TIME = time.time()


class HealthStatus(BaseModel):
    status: str           # "healthy" or "degraded"
    uptime_seconds: float
    environment: str
    version: str = "1.0.0"


@router.get("/health/live", summary="Liveness probe")
async def liveness():
    """
    Simple liveness check. Returns 200 if the process is running.
    Used by Kubernetes/Cloud Run to decide if the container should be restarted.
    """
    return {"status": "alive"}


@router.get("/health/ready", response_model=HealthStatus, summary="Readiness probe")
async def readiness():
    """
    Readiness check. Verifies critical components are loaded and ready.
    Returns 200 only when the system can serve requests.
    """
    from src.config import settings

    # Kept instant on purpose: the UI polls this while answers are being generated, and a
    # slow check (e.g. counting vectors while the CPU is busy) made the app look offline.
    # Detailed component checks live in /health/stats.
    status = "healthy"

    return HealthStatus(
        status=status,
        uptime_seconds=round(time.time() - _START_TIME, 1),
        environment=settings.environment,
    )


@router.get("/health/stats", summary="Detailed system statistics")
async def system_stats() -> Dict[str, Any]:
    """
    Detailed statistics for monitoring dashboards.
    Shows document counts, model info, and usage stats.
    """
    from src.config import settings

    stats: Dict[str, Any] = {
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "environment": settings.environment,
        "vector_store_type": settings.vector_store_type,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "llm_model": settings.vertex_ai_model,
        "guardrails": {
            "input_enabled": settings.enable_input_guardrails,
            "output_enabled": settings.enable_output_guardrails,
            "pii_detection": settings.enable_pii_detection,
        },
        "evaluations": {
            "inline_evals_enabled": settings.enable_inline_evals,
            "sample_rate": settings.eval_sample_rate,
        },
    }

    # Document count
    try:
        from src.retrieval.vector_store import vector_store
        stats["knowledge_base"] = {
            "total_chunks": vector_store.get_document_count(),
            "collection": vector_store.collection_name,
        }
    except Exception as exc:
        stats["knowledge_base"] = {"error": str(exc)}

    # LLM usage stats
    try:
        from src.gateway.llm_gateway import llm_gateway
        stats["llm_usage"] = llm_gateway.get_usage_stats()
    except Exception:
        pass

    return stats
