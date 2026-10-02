"""Confluence sources: each source is one space; every current page in it is indexed."""

import uuid
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src.sources.credentials import credential_store
from src.sources.jobs import IndexingError, index_jobs
from src.sources.models import (
    AddConfluenceSourceRequest, BulkAddConfluenceRequest, ConfluenceCredentials, ConfluenceSourceConfig,
    ConfluenceSourceResponse, IndexJobResponse, TestConfluenceRequest,
)
from src.sources.registry import source_registry
from src.observability.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/sources/confluence", tags=["Sources - Confluence"])


class TestConnectionResponse(BaseModel):
    success: bool
    message: str
    spaces_found: int = None


def _target(url: str, space_key: str = ""):
    """(base_url, space_key) from a base URL or a pasted space / page link."""
    from src.sources.confluence_indexer import parse_confluence_url, SPACE_KEY
    try:
        base, linked_space = parse_confluence_url(url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    space = (space_key or linked_space or "").strip()
    if space and not SPACE_KEY.match(space):
        raise HTTPException(status_code=422, detail=f"'{space}' is not a valid Confluence space key")
    return base, space


def _client(request: ConfluenceCredentials, base_url: str):
    from src.sources.confluence_indexer import ConfluenceClient
    return ConfluenceClient(base_url, request.username, request.token)


def _existing_spaces():
    return {(c.base_url, c.space_key.lower()) for c in
            (source_registry.get_confluence_config(s.id) for s in source_registry.list_sources()) if c}


def _create(base_url: str, space_key: str, username: str, token: str, name: str = None):
    source_id = f"conf-{uuid.uuid4().hex[:8]}"
    credential_id = credential_store.store(provider="confluence", label=f"{base_url} / {space_key} token",
                                           plaintext_token=f"{(username or '').strip()}:{token}")
    config = ConfluenceSourceConfig(source_id=source_id, base_url=base_url, space_key=space_key,
                                    credential_id=credential_id)
    source = source_registry.create_confluence_source(config, name=name or f"Confluence {space_key}")
    return ConfluenceSourceResponse(**source.model_dump(), base_url=base_url, space_key=space_key, page_count=0)


@router.post("/test", response_model=TestConnectionResponse, summary="Test Confluence connection")
async def test_confluence_connection(request: TestConfluenceRequest):
    try:
        base, space = _target(request.base_url, request.space_key or "")
    except HTTPException as exc:
        return TestConnectionResponse(success=False, message=exc.detail)
    try:
        info = await run_in_threadpool(_client(request, base).test_connection, space or None)
        return TestConnectionResponse(success=True, spaces_found=info["spaces_found"],
                                      message=f"Connection successful{': space ' + space if space else ''}")
    except IndexingError as exc:
        return TestConnectionResponse(success=False, message=str(exc))
    except Exception as exc:
        logger.warning("confluence_connection_test_failed", error=str(exc))
        return TestConnectionResponse(success=False, message=f"Connection error: {exc.__class__.__name__}")


@router.post("/discover", summary="List Confluence spaces visible to a token")
async def discover_spaces(request: ConfluenceCredentials):
    base, _ = _target(request.base_url)
    try:
        spaces = await run_in_threadpool(_client(request, base).list_spaces)
    except IndexingError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    existing = _existing_spaces()
    for space in spaces:
        space["already_added"] = (base, space["key"].lower()) in existing
    return {"base_url": base, "spaces": spaces}


@router.post("", response_model=ConfluenceSourceResponse, summary="Add a Confluence space and start indexing")
async def add_confluence_source(request: AddConfluenceSourceRequest):
    from src.sources.confluence_indexer import confluence_indexer
    base, space = _target(request.base_url, request.space_key)
    if not space or not request.token:
        raise HTTPException(status_code=422, detail="Space key (or a space link) and token are required")
    if (base, space.lower()) in _existing_spaces():
        raise HTTPException(status_code=409, detail=f"Space {space} is already added")
    source = _create(base, space, request.username, request.token, request.name)
    index_jobs.start(source.id, confluence_indexer.index)
    return source


@router.post("/bulk", summary="Add many Confluence spaces and start indexing them")
async def bulk_add_spaces(request: BulkAddConfluenceRequest):
    from src.sources.confluence_indexer import confluence_indexer
    base, _ = _target(request.base_url)
    keys = [str(k).strip() for k in request.space_keys if str(k).strip()]
    if not keys:
        raise HTTPException(status_code=422, detail="Select at least one space")
    if len(keys) > 200:
        raise HTTPException(status_code=422, detail="Add at most 200 spaces at a time")
    existing, added, skipped = _existing_spaces(), [], []
    for key in keys:
        _target(request.base_url, key)  # validates the key
        if (base, key.lower()) in existing:
            skipped.append(key)
            continue
        existing.add((base, key.lower()))
        source = _create(base, key, request.username, request.token)
        index_jobs.start(source.id, confluence_indexer.index)
        added.append(source.id)
    return {"added": len(added), "skipped": skipped, "source_ids": added}


@router.get("", response_model=List[ConfluenceSourceResponse], summary="List Confluence sources")
async def list_confluence_sources():
    return source_registry.list_confluence_sources()


@router.delete("/{source_id}", summary="Delete a Confluence source")
async def delete_confluence_source(source_id: str):
    source = source_registry.get_source(source_id)
    cfg = source_registry.get_confluence_config(source_id)
    if not source or not cfg:
        raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")
    if index_jobs.is_running(source_id):
        raise HTTPException(status_code=409, detail="Indexing is running; wait for it to finish, then delete")
    try:
        from src.retrieval.vector_store import vector_store
        from src.ingestion.ingestion_pipeline import ingestion_pipeline
        vector_store.delete_by_source_id(source_id)
        ingestion_pipeline._refresh_bm25_index(strict=True)
    except Exception as exc:
        logger.error("chunk_deletion_failed", source_id=source_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Could not remove indexed chunks; the source was kept. Retry.")
    if credential_store.exists(cfg.credential_id):
        credential_store.delete(cfg.credential_id)
    source_registry.delete_source(source_id)
    return {"message": f"Source {source_id} deleted successfully"}


def _start(source_id: str, message: str) -> IndexJobResponse:
    from src.sources.confluence_indexer import confluence_indexer
    if not source_registry.get_confluence_config(source_id):
        raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")
    if not index_jobs.start(source_id, confluence_indexer.index):
        raise HTTPException(status_code=409, detail="Indexing is already running for this source")
    return IndexJobResponse(source_id=source_id, status="started", message=message)


@router.post("/{source_id}/index", response_model=IndexJobResponse, summary="Index a Confluence space")
async def index_confluence_source(source_id: str):
    return _start(source_id, "Indexing started")


@router.post("/{source_id}/sync", response_model=IndexJobResponse, summary="Sync a Confluence space")
async def sync_confluence_source(source_id: str):
    """Re-fetches all pages; unchanged pages reuse their stored vectors, so this is fast."""
    return _start(source_id, "Sync started")
