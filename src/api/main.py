# ============================================================
# src/api/main.py - FastAPI Application Entry Point
#
# Learning Note:
#   FastAPI is a modern Python web framework that:
#   - Generates interactive API docs automatically (at /docs)
#   - Validates request/response types using Pydantic
#   - Supports async/await natively (non-blocking I/O)
#   - Is production-ready (used by Netflix, Uber, Microsoft)
#
#   Our API has 4 main route groups:
#     /api/v1/query    - Ask questions (main RAG endpoint)
#     /api/v1/ingest   - Upload and ingest documents
#     /api/v1/health   - Health check and system status
#     /api/v1/stats    - Usage and evaluation metrics
# ============================================================

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from src.observability.logger import configure_logging, get_logger
from src.config import settings

# Configure structured logging before anything else
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Code before 'yield' runs at STARTUP.
    Code after 'yield' runs at SHUTDOWN.

    Learning Note:
        This replaces the old @app.on_event("startup") pattern.
        We use it to pre-load heavy models so the first request
        is not slow (warm-up vs cold-start).
    """
    # Startup: log ready immediately (models load lazily on first request)
    # Learning Note: We intentionally do NOT pre-load the 90MB embedding model
    # at startup. Instead, models load on first request (lazy loading).
    # This makes the server start in <1 second instead of waiting for downloads.
    # On first query, there will be a ~5 second delay for model init.
    logger.info("application_startup", environment=settings.environment)
    from src.sources.jobs import index_jobs
    index_jobs.recover_interrupted()
    logger.info("application_ready", host=settings.api_host, port=settings.api_port)

    yield  # Application runs here

    # Shutdown: flush observability data
    logger.info("application_shutdown")
    try:
        from src.observability.tracer import tracer
        tracer.flush()
    except Exception:
        pass


# Create FastAPI app
app = FastAPI(
    title="Advanced RAG System",
    description=(
        "Production-grade Retrieval-Augmented Generation with "
        "LangGraph, Guardrails, LLM Gateway, and Evaluations on GCP."
    ),
    version="1.0.0",
    docs_url="/docs",         # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc",       # ReDoc UI at http://localhost:8000/redoc
    lifespan=lifespan,
)

# ============================================================
# CORS Middleware
# Learning Note:
#   CORS (Cross-Origin Resource Sharing) allows the HTML UI
#   (served from a different port or domain) to call the API.
#   Without this, browsers block the requests for security.
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else ["https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Mount static files (HTML UI)
# ============================================================
ui_path = os.path.join(os.path.dirname(__file__), "..", "..", "ui")
if os.path.exists(ui_path):
    app.mount("/static", StaticFiles(directory=ui_path), name="static")

# Include routers
from src.api.routes import query, ingest, health
from src.api.routes.sources import documents as doc_sources
from src.api.routes.sources import bitbucket as bb_sources
from src.api.routes.sources import jira as jira_sources

app.include_router(query.router, prefix="/api/v1", tags=["Query"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(doc_sources.router, prefix="/api/v1")
app.include_router(bb_sources.router, prefix="/api/v1")
app.include_router(jira_sources.router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the HTML frontend at the root URL."""
    ui_index = os.path.join(ui_path, "index.html")
    if os.path.exists(ui_index):
        return FileResponse(ui_index)
    return {"message": "Advanced RAG API. Visit /docs for API documentation."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app", 
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )
