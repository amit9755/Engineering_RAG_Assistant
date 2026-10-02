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


def _normalize_target(server_url, workspace, repository):
    """
    Returns (server_url or None, workspace/project, repository). A pasted Bitbucket
    Server repository URL fills in a missing project key and repository.
    """
    from src.sources.bitbucket_indexer import parse_server_url
    workspace, repository = (workspace or "").strip(), (repository or "").strip()
    if not (server_url or "").strip():
        return None, workspace, repository
    try:
        base, project, repo = parse_server_url(server_url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return base, (workspace or project or ""), (repository or repo or "")


@router.post("/test", response_model=TestConnectionResponse, summary="Test Bitbucket connection")
async def test_bitbucket_connection(request: TestBitbucketRequest):
    """
    Test Bitbucket Cloud or Bitbucket Server / Data Center credentials without
    storing them. Uses the same client as indexing, so a passing test means
    indexing can reach the repository.
    """
    from starlette.concurrency import run_in_threadpool
    from src.sources.bitbucket_indexer import make_client
    from src.sources.jobs import IndexingError

    server_url, workspace, repository = _normalize_target(request.server_url, request.workspace,
                                                          request.repository)
    if not workspace:
        return TestConnectionResponse(success=False, message=(
            "Enter the project key (e.g. WSQAAUTO)" if server_url else "Enter the workspace"))
    try:
        client = make_client(request.username.strip(), request.token, server_url=server_url)
        repos = await run_in_threadpool(client.test_connection, workspace, repository or None)
        target = f"{workspace}/{repository}" if repository else workspace
        return TestConnectionResponse(success=True, message=f"Connection successful: {target}",
                                      workspace=workspace, repos_found=repos)
    except IndexingError as exc:
        return TestConnectionResponse(success=False, message=str(exc))
    except Exception as exc:
        logger.warning("bitbucket_connection_test_failed", error=str(exc))
        return TestConnectionResponse(success=False, message=f"Connection error: {exc.__class__.__name__}")


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

def _create_source(server_url, workspace, repository, branch, username, token, name=None):
    """Register one repository. Each source gets its own encrypted copy of the credential,
    so deleting one repository never breaks the others."""
    source_id = f"bb-{uuid.uuid4().hex[:8]}"
    credential_id = credential_store.store(
        provider="bitbucket",
        label=f"{workspace}/{repository} token",
        plaintext_token=f"{(username or '').strip()}:{token}",
    )
    config = BitbucketSourceConfig(source_id=source_id, workspace=workspace, repository=repository,
                                   branch=branch or "main", credential_id=credential_id, server_url=server_url)
    source = source_registry.create_bitbucket_source(config, name=name or f"{workspace}/{repository}")
    logger.info("bitbucket_source_added", source_id=source_id, repo=f"{workspace}/{repository}")
    return BitbucketSourceResponse(**source.model_dump(), workspace=workspace, repository=repository,
                                   branch=config.branch, last_commit=None, file_count=0,
                                   server_url=server_url, credential_configured=True)


@router.post("", response_model=BitbucketSourceResponse, summary="Add a Bitbucket source")
async def add_bitbucket_source(request: AddBitbucketSourceRequest):
    """
    Add a Bitbucket repository as a knowledge source.

    The token is encrypted and stored. Only a credential_id reference
    is kept in the source config. The token is never returned by GET endpoints.

    After adding, call POST /{source_id}/index to index the repository.
    """
    server_url, workspace, repository = _normalize_target(
        request.server_url, request.workspace, request.repository)
    if not workspace or not repository or not request.token:
        raise HTTPException(status_code=422, detail="Workspace / project key, repository and token are required")
    return _create_source(server_url, workspace, repository, request.branch, request.username,
                          request.token, request.name)


class DiscoverRequest(BaseModel):
    server_url: Optional[str] = None
    workspace: str = ""          # Cloud: workspace (required). Server: project key, or empty for all projects
    username: str = ""
    token: str


@router.post("/discover", summary="List repositories available to a token")
async def discover_repositories(request: DiscoverRequest):
    """Repositories in a workspace / project (Server: every visible repository if no project key is given)."""
    from starlette.concurrency import run_in_threadpool
    from src.sources.bitbucket_indexer import make_client
    from src.sources.jobs import IndexingError

    server_url, workspace, _ = _normalize_target(request.server_url, request.workspace, "")
    try:
        client = make_client(request.username.strip(), request.token, server_url=server_url)
        repos = await run_in_threadpool(client.list_repositories, workspace or None)
    except IndexingError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    existing = {(cfg.server_url, cfg.workspace.lower(), cfg.repository.lower())
                for cfg in (source_registry.get_bitbucket_config(s.id) for s in source_registry.list_sources())
                if cfg}
    for repo in repos:
        repo["already_added"] = (server_url, repo["workspace"].lower(), repo["slug"].lower()) in existing
    return {"repositories": repos}


class BulkRepository(BaseModel):
    workspace: str
    slug: str


class BulkAddRequest(BaseModel):
    server_url: Optional[str] = None
    username: str = ""
    token: str
    branch: str = "main"
    repositories: List[BulkRepository]


@router.post("/bulk", summary="Add many repositories and start indexing them")
async def bulk_add_repositories(request: BulkAddRequest):
    """Adds every listed repository not already added, then queues indexing (a few run at a time)."""
    from src.sources.bitbucket_indexer import bitbucket_indexer

    server_url, _, _ = _normalize_target(request.server_url, "x", "x")
    if not request.repositories:
        raise HTTPException(status_code=422, detail="Select at least one repository")
    if len(request.repositories) > 500:
        raise HTTPException(status_code=422, detail="Add at most 500 repositories at a time")
    existing = {(cfg.server_url, cfg.workspace.lower(), cfg.repository.lower())
                for cfg in (source_registry.get_bitbucket_config(s.id) for s in source_registry.list_sources())
                if cfg}
    added, skipped = [], []
    for repo in request.repositories:
        key = (server_url, repo.workspace.lower(), repo.slug.lower())
        if key in existing:
            skipped.append(f"{repo.workspace}/{repo.slug}")
            continue
        existing.add(key)
        source = _create_source(server_url, repo.workspace, repo.slug, request.branch,
                                request.username, request.token)
        index_jobs.start(source.id, bitbucket_indexer.index)
        added.append(source.id)
    logger.info("bitbucket_bulk_added", added=len(added), skipped=len(skipped))
    return {"added": len(added), "skipped": skipped, "source_ids": added}


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
        server_url=cfg.server_url,
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
        # Keep the saved username when only the token is being replaced.
        username = request.username
        if username is None:
            try:
                username = credential_store.retrieve(cfg.credential_id).split(":", 1)[0]
            except KeyError:
                username = ""
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
        server_url=updated_cfg.server_url,
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
