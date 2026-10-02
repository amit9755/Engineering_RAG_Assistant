# ============================================================
# src/api/routes/sources/jira.py - Jira source management
#
# Phase 1: Full CRUD + test connection stub.
# Phase 6 will add real Jira API connection testing and
# dynamic project discovery.
# Phase 7 will implement actual indexing and sync.
#
# Endpoints:
#   POST   /api/v1/sources/jira/test               test connection
#   GET    /api/v1/sources/jira/projects            list projects (stub, Phase 6)
#   POST   /api/v1/sources/jira                     add source
#   GET    /api/v1/sources/jira                     list all
#   GET    /api/v1/sources/jira/{source_id}         get one
#   PUT    /api/v1/sources/jira/{source_id}         update
#   DELETE /api/v1/sources/jira/{source_id}         remove
#   POST   /api/v1/sources/jira/{source_id}/index   (stub, Phase 7)
#   POST   /api/v1/sources/jira/{source_id}/sync    (stub, Phase 7)
# ============================================================

from typing import List, Optional
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.sources.credentials import credential_store
from src.sources.models import (
    AddJiraSourceRequest,
    IndexJobResponse,
    JiraSourceConfig,
    JiraSourceResponse,
    SourceStatus,
    TestJiraRequest,
    UpdateJiraSourceRequest,
)
from src.sources.jobs import index_jobs
from src.sources.registry import source_registry
from src.observability.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/sources/jira", tags=["Sources - Jira"])


# ============================================================
# Test connection
# ============================================================

class TestConnectionResponse(BaseModel):
    success: bool
    message: str
    base_url: Optional[str] = None
    display_name: Optional[str] = None
    projects_found: Optional[int] = None


@router.post("/test", response_model=TestConnectionResponse, summary="Test Jira connection")
async def test_jira_connection(request: TestJiraRequest):
    """
    Test a Jira connection using provided credentials.
    Returns success/failure without storing any credentials.

    Phase 1: Basic HTTP test against Jira REST API.
    Phase 6: Returns full project list for dynamic dropdown.
    """
    try:
        import requests as http_requests
        from requests.auth import HTTPBasicAuth

        auth = HTTPBasicAuth(request.email, request.token)
        # Test with the /myself endpoint which is lightweight
        url = f"{request.base_url.rstrip('/')}/rest/api/2/myself"
        response = http_requests.get(url, auth=auth, timeout=10)

        if response.status_code == 200:
            data = response.json()
            display_name = data.get("displayName", request.email)

            # Also get project count
            projects_url = f"{request.base_url.rstrip('/')}/rest/api/2/project"
            projects_response = http_requests.get(projects_url, auth=auth, timeout=10)
            projects_count = len(projects_response.json()) if projects_response.status_code == 200 else None

            return TestConnectionResponse(
                success=True,
                message="Connection successful",
                base_url=request.base_url,
                display_name=display_name,
                projects_found=projects_count,
            )
        elif response.status_code == 401:
            return TestConnectionResponse(
                success=False,
                message="Authentication failed: invalid email or API token",
            )
        elif response.status_code == 403:
            return TestConnectionResponse(
                success=False,
                message="Permission denied: check your API token scopes",
            )
        else:
            return TestConnectionResponse(
                success=False,
                message=f"Jira returned status {response.status_code}",
            )

    except Exception as exc:
        logger.warning("jira_connection_test_failed", error=str(exc))
        return TestConnectionResponse(
            success=False,
            message=f"Connection error: {str(exc)}",
        )


# ============================================================
# Project discovery (Phase 6 stub)
# ============================================================

@router.get("/projects", summary="List Jira projects (Phase 6)")
async def list_jira_projects(base_url: str, email: str, token: str):
    """
    Dynamically list Jira projects for a connection.
    Phase 1 stub - full implementation in Phase 6.
    """
    return {
        "status": "not_implemented",
        "message": "Project discovery will be available in Phase 6",
        "base_url": base_url,
    }


# ============================================================
# CRUD operations
# ============================================================

@router.post("", response_model=JiraSourceResponse, summary="Add a Jira source")
async def add_jira_source(request: AddJiraSourceRequest):
    """
    Add a Jira project as a knowledge source.

    The API token is encrypted and stored. Only a credential_id reference
    is kept in the source config. The token is never returned by GET endpoints.

    After adding, call POST /{source_id}/index to index the project issues.
    """
    from src.sources.jira_indexer import PROJECT_KEY
    if not PROJECT_KEY.match(request.project_key.strip().upper()):
        raise HTTPException(status_code=422, detail="Project key must be letters/digits, e.g. BT or PROJ2")
    if not request.base_url.strip().lower().startswith("https://"):
        raise HTTPException(status_code=422, detail="Jira URL must start with https://")
    request.project_key = request.project_key.strip()

    source_id = f"jira-{uuid.uuid4().hex[:8]}"

    # Store credential encrypted (email:token format for basic auth)
    credential_id = credential_store.store(
        provider="jira",
        label=f"{request.base_url} / {request.project_key} token",
        plaintext_token=f"{request.email}:{request.token}",
    )

    name = request.name or f"{request.project_key} @ {request.base_url}"

    config = JiraSourceConfig(
        source_id=source_id,
        base_url=request.base_url.rstrip("/"),
        project_key=request.project_key.upper(),
        credential_id=credential_id,
    )

    source = source_registry.create_jira_source(config, name=name)

    logger.info(
        "jira_source_added",
        source_id=source_id,
        project=request.project_key,
        base_url=request.base_url,
    )

    return JiraSourceResponse(
        **source.model_dump(),
        base_url=config.base_url,
        project_key=config.project_key,
        issue_count=0,
        last_sync_issue_updated=None,
        credential_configured=True,
    )


@router.get("", response_model=List[JiraSourceResponse], summary="List Jira sources")
async def list_jira_sources():
    """Return all registered Jira sources. Tokens are never included."""
    return source_registry.list_jira_sources()


@router.get(
    "/{source_id}",
    response_model=JiraSourceResponse,
    summary="Get a Jira source",
)
async def get_jira_source(source_id: str):
    """Get a single Jira source by ID. Token is never returned."""
    source = source_registry.get_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")

    cfg = source_registry.get_jira_config(source_id)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Jira config not found: {source_id}")

    return JiraSourceResponse(
        **source.model_dump(),
        base_url=cfg.base_url,
        project_key=cfg.project_key,
        issue_count=cfg.issue_count,
        last_sync_issue_updated=cfg.last_sync_issue_updated,
        credential_configured=True,
    )


@router.put(
    "/{source_id}",
    response_model=JiraSourceResponse,
    summary="Update a Jira source",
)
async def update_jira_source(source_id: str, request: UpdateJiraSourceRequest):
    """
    Update configuration for a Jira source.

    Leave token blank (or omit) to keep the existing credential.
    Providing a new token replaces the stored credential securely.
    """
    source = source_registry.get_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")

    cfg = source_registry.get_jira_config(source_id)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Jira config not found: {source_id}")

    # Replace credential if a new token is provided
    if request.token and request.token.strip():
        email = request.email or ""
        credential_store.update(
            cfg.credential_id,
            plaintext_token=f"{email}:{request.token}",
        )
        logger.info("jira_credential_updated", source_id=source_id)

    source_registry.update_jira_config(
        source_id=source_id,
        base_url=request.base_url,
        project_key=request.project_key.upper() if request.project_key else None,
    )

    updated_source = source_registry.get_source(source_id)
    updated_cfg = source_registry.get_jira_config(source_id)

    return JiraSourceResponse(
        **updated_source.model_dump(),
        base_url=updated_cfg.base_url,
        project_key=updated_cfg.project_key,
        issue_count=updated_cfg.issue_count,
        last_sync_issue_updated=updated_cfg.last_sync_issue_updated,
        credential_configured=True,
    )


@router.delete("/{source_id}", summary="Delete a Jira source")
async def delete_jira_source(source_id: str):
    """
    Remove a Jira source, delete its credential, and remove
    its chunks from the vector store.
    """
    source = source_registry.get_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")

    if index_jobs.is_running(source_id):
        raise HTTPException(status_code=409, detail="Indexing is running; wait for it to finish, then delete")

    cfg = source_registry.get_jira_config(source_id)

    # Remove chunks first; if that fails, keep the source so deletion can be retried.
    try:
        from src.retrieval.vector_store import vector_store
        from src.ingestion.ingestion_pipeline import ingestion_pipeline
        vector_store.delete_by_source_id(source_id)
        ingestion_pipeline._refresh_bm25_index(strict=True)
        logger.info("jira_chunks_deleted", source_id=source_id)
    except Exception as exc:
        logger.error("chunk_deletion_failed", source_id=source_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Could not remove indexed chunks; the source was kept. Retry.")

    if cfg and credential_store.exists(cfg.credential_id):
        credential_store.delete(cfg.credential_id)
        logger.info("jira_credential_deleted", source_id=source_id)

    deleted = source_registry.delete_source(source_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")

    logger.info("jira_source_deleted", source_id=source_id)
    return {"message": f"Source {source_id} deleted successfully"}


# ============================================================
# Indexing and sync (background jobs; poll GET /{source_id} for status)
# ============================================================

def _start_job(source_id: str, job, message: str) -> IndexJobResponse:
    source = source_registry.get_source(source_id)
    if not source or source.type.value != "jira":
        raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")
    if not index_jobs.start(source_id, job):
        raise HTTPException(status_code=409, detail="Indexing is already running for this source")
    return IndexJobResponse(source_id=source_id, status="started", message=message)


@router.post(
    "/{source_id}/index",
    response_model=IndexJobResponse,
    summary="Index a Jira project",
)
async def index_jira_source(source_id: str):
    """Start a full index in the background. Old chunks stay searchable until it succeeds."""
    from src.sources.jira_indexer import jira_indexer
    return _start_job(source_id, jira_indexer.index, "Indexing started")


@router.post(
    "/{source_id}/sync",
    response_model=IndexJobResponse,
    summary="Sync a Jira project",
)
async def sync_jira_source(source_id: str):
    """Refresh the project: re-fetches all issues and replaces the old chunks."""
    from src.sources.jira_indexer import jira_indexer
    return _start_job(source_id, jira_indexer.index, "Sync started")
