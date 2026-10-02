# ============================================================
# src/api/routes/sources/bitbucket.py - Bitbucket source management
#
# Phase 1: Full CRUD + test connection stub.
# Phase 3 will add real Bitbucket API connection testing and
# dynamic repo/branch discovery.
# Phase 4 will implement actual indexing.
# Phase 5 will implement incremental sync.
#
# Endpoints:
#   POST   /api/v1/sources/bitbucket/test          test connection (stub)
#   GET    /api/v1/sources/bitbucket/repos         list repos (stub, Phase 3)
#   GET    /api/v1/sources/bitbucket/branches      list branches (stub, Phase 3)
#   POST   /api/v1/sources/bitbucket               add source
#   GET    /api/v1/sources/bitbucket               list all
#   GET    /api/v1/sources/bitbucket/{source_id}   get one
#   PUT    /api/v1/sources/bitbucket/{source_id}   update
#   DELETE /api/v1/sources/bitbucket/{source_id}   remove
#   POST   /api/v1/sources/bitbucket/{source_id}/index  (stub, Phase 4)
#   POST   /api/v1/sources/bitbucket/{source_id}/sync   (stub, Phase 5)
# ============================================================

from typing import List, Optional
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.sources.credentials import credential_store
from src.sources.models import (
    AddBitbucketSourceRequest,
    BitbucketSourceConfig,
    BitbucketSourceResponse,
    IndexJobResponse,
    SourceStatus,
    TestBitbucketRequest,
    UpdateBitbucketSourceRequest,
)
from src.sources.jobs import index_jobs
from src.sources.registry import source_registry
from src.observability.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/sources/bitbucket", tags=["Sources - Bitbucket"])


# ============================================================
# Test connection
# ============================================================

class TestConnectionResponse(BaseModel):
    success: bool
    message: str
    workspace: Optional[str] = None
    repos_found: Optional[int] = None


@router.post("/test", response_model=TestConnectionResponse, summary="Test Bitbucket connection")
async def test_bitbucket_connection(request: TestBitbucketRequest):
    """
    Test a Bitbucket connection using provided credentials.
    Returns success/failure without storing any credentials.

    Phase 1: Basic HTTP test against Bitbucket API.
    Phase 3: Returns full repository list for dynamic dropdown.
    """
    try:
        import requests as http_requests
        from requests.auth import HTTPBasicAuth

        auth = HTTPBasicAuth(request.username, request.token)
        url = f"https://api.bitbucket.org/2.0/repositories/{request.workspace}"
        response = http_requests.get(url, auth=auth, timeout=10)

        if response.status_code == 200:
            data = response.json()
            repo_count = data.get("size", 0)
            return TestConnectionResponse(
                success=True,
                message="Connection successful",
                workspace=request.workspace,
                repos_found=repo_count,
            )
        elif response.status_code == 401:
            return TestConnectionResponse(
                success=False,
                message="Authentication failed: invalid username or token",
            )
        elif response.status_code == 404:
            return TestConnectionResponse(
                success=False,
                message=f"Workspace '{request.workspace}' not found",
            )
        else:
            return TestConnectionResponse(
                success=False,
                message=f"Bitbucket returned status {response.status_code}",
            )

    except Exception as exc:
        logger.warning("bitbucket_connection_test_failed", error=str(exc))
        return TestConnectionResponse(
            success=False,
            message=f"Connection error: {str(exc)}",
        )


# ============================================================
# Repository and branch discovery (Phase 3 stubs)
# ============================================================

@router.get("/repos", summary="List repositories for a workspace (Phase 3)")
async def list_repos(workspace: str, username: str, token: str):
    """
    Dynamically list Bitbucket repositories for a workspace.
    Phase 1 stub - full implementation in Phase 3.
    """
    return {
        "status": "not_implemented",
        "message": "Repository discovery will be available in Phase 3",
        "workspace": workspace,
    }


@router.get("/branches", summary="List branches for a repository (Phase 3)")
async def list_branches(workspace: str, repository: str, username: str, token: str):
    """
    Dynamically list branches for a Bitbucket repository.
    Phase 1 stub - full implementation in Phase 3.
    """
    return {
        "status": "not_implemented",
        "message": "Branch discovery will be available in Phase 3",
        "repository": f"{workspace}/{repository}",
    }


# ============================================================
# CRUD operations
# ============================================================

@router.post("", response_model=BitbucketSourceResponse, summary="Add a Bitbucket source")
async def add_bitbucket_source(request: AddBitbucketSourceRequest):
    """
    Add a Bitbucket repository as a knowledge source.

    The token is encrypted and stored. Only a credential_id reference
    is kept in the source config. The token is never returned by GET endpoints.

    After adding, call POST /{source_id}/index to index the repository.
    """
    # Generate a unique source ID
    source_id = f"bb-{uuid.uuid4().hex[:8]}"

    # Store the credential encrypted
    credential_id = credential_store.store(
        provider="bitbucket",
        label=f"{request.workspace}/{request.repository} token",
        plaintext_token=f"{request.username}:{request.token}",
    )

    # Build display name
    name = request.name or f"{request.workspace}/{request.repository}"

    # Create config (no token stored here)
    config = BitbucketSourceConfig(
        source_id=source_id,
        workspace=request.workspace,
        repository=request.repository,
        branch=request.branch,
        credential_id=credential_id,
    )

    source = source_registry.create_bitbucket_source(config, name=name)

    logger.info(
        "bitbucket_source_added",
        source_id=source_id,
        repo=f"{request.workspace}/{request.repository}",
    )

    return BitbucketSourceResponse(
        **source.model_dump(),
        workspace=request.workspace,
        repository=request.repository,
        branch=request.branch,
        last_commit=None,
        file_count=0,
        credential_configured=True,
    )


@router.get("", response_model=List[BitbucketSourceResponse], summary="List Bitbucket sources")
async def list_bitbucket_sources():
    """Return all registered Bitbucket sources. Tokens are never included."""
    return source_registry.list_bitbucket_sources()


@router.get(
    "/{source_id}",
    response_model=BitbucketSourceResponse,
    summary="Get a Bitbucket source",
)
async def get_bitbucket_source(source_id: str):
    """Get a single Bitbucket source by ID. Token is never returned."""
    source = source_registry.get_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")

    cfg = source_registry.get_bitbucket_config(source_id)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Bitbucket config not found: {source_id}")

    return BitbucketSourceResponse(
        **source.model_dump(),
        workspace=cfg.workspace,
        repository=cfg.repository,
        branch=cfg.branch,
        last_commit=cfg.last_commit,
        file_count=cfg.file_count,
        credential_configured=True,
    )


@router.put(
    "/{source_id}",
    response_model=BitbucketSourceResponse,
    summary="Update a Bitbucket source",
)
async def update_bitbucket_source(source_id: str, request: UpdateBitbucketSourceRequest):
    """
    Update configuration for a Bitbucket source.

    Leave token blank (or omit) to keep the existing credential.
    Providing a new token replaces the stored credential securely.
    """
    source = source_registry.get_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")

    cfg = source_registry.get_bitbucket_config(source_id)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Bitbucket config not found: {source_id}")

    # If a new token is provided, update the credential
    if request.token and request.token.strip():
        username = request.username or cfg.workspace  # fallback
        credential_store.update(
            cfg.credential_id,
            plaintext_token=f"{username}:{request.token}",
        )
        logger.info("bitbucket_credential_updated", source_id=source_id)

    # Update config fields
    source_registry.update_bitbucket_config(
        source_id=source_id,
        workspace=request.workspace,
        repository=request.repository,
        branch=request.branch,
    )

    # Re-fetch to return updated data
    updated_source = source_registry.get_source(source_id)
    updated_cfg = source_registry.get_bitbucket_config(source_id)

    return BitbucketSourceResponse(
        **updated_source.model_dump(),
        workspace=updated_cfg.workspace,
        repository=updated_cfg.repository,
        branch=updated_cfg.branch,
        last_commit=updated_cfg.last_commit,
        file_count=updated_cfg.file_count,
        credential_configured=True,
    )


@router.delete("/{source_id}", summary="Delete a Bitbucket source")
async def delete_bitbucket_source(source_id: str):
    """
    Remove a Bitbucket source, delete its credential, and remove
    its chunks from the vector store.
    """
    source = source_registry.get_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")

    if index_jobs.is_running(source_id):
        raise HTTPException(status_code=409, detail="Indexing is running; wait for it to finish, then delete")

    cfg = source_registry.get_bitbucket_config(source_id)

    # Remove chunks first; if that fails, keep the source so deletion can be retried.
    try:
        from src.retrieval.vector_store import vector_store
        from src.ingestion.ingestion_pipeline import ingestion_pipeline
        vector_store.delete_by_source_id(source_id)
        ingestion_pipeline._refresh_bm25_index(strict=True)
        logger.info("bitbucket_chunks_deleted", source_id=source_id)
    except Exception as exc:
        logger.error("chunk_deletion_failed", source_id=source_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Could not remove indexed chunks; the source was kept. Retry.")

    if cfg and credential_store.exists(cfg.credential_id):
        credential_store.delete(cfg.credential_id)
        logger.info("bitbucket_credential_deleted", source_id=source_id)

    deleted = source_registry.delete_source(source_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")

    logger.info("bitbucket_source_deleted", source_id=source_id)
    return {"message": f"Source {source_id} deleted successfully"}


# ============================================================
# Indexing and sync (background jobs; poll GET /{source_id} for status)
# ============================================================

def _start_job(source_id: str, job, message: str) -> IndexJobResponse:
    source = source_registry.get_source(source_id)
    if not source or source.type.value != "bitbucket":
        raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")
    if not index_jobs.start(source_id, job):
        raise HTTPException(status_code=409, detail="Indexing is already running for this source")
    return IndexJobResponse(source_id=source_id, status="started", message=message)


@router.post(
    "/{source_id}/index",
    response_model=IndexJobResponse,
    summary="Index a Bitbucket repository",
)
async def index_bitbucket_source(source_id: str):
    """Start a full index in the background. Old chunks stay searchable until it succeeds."""
    from src.sources.bitbucket_indexer import bitbucket_indexer
    return _start_job(source_id, bitbucket_indexer.index, "Indexing started")


@router.post(
    "/{source_id}/sync",
    response_model=IndexJobResponse,
    summary="Sync a Bitbucket repository",
)
async def sync_bitbucket_source(source_id: str):
    """Re-index only if the branch has new commits since the last index."""
    from src.sources.bitbucket_indexer import bitbucket_indexer
    return _start_job(source_id, bitbucket_indexer.sync, "Sync started")
