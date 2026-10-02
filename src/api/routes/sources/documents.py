"""Document source upload, inspection, reindexing, and deletion."""

from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from src.api.routes.ingest import read_upload
from src.sources.documents import document_service
from src.sources.models import DocumentSourceResponse, IndexJobResponse
from src.observability.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/sources/documents", tags=["Sources - Documents"])


def source_operation(operation, *args):
    try:
        return operation(*args)
    except KeyError:
        raise HTTPException(status_code=404, detail="Document source not found")
    except FileNotFoundError:
        raise HTTPException(status_code=409, detail="Original file is missing. Upload the document again.")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception:
        logger.exception("document_source_operation_failed")
        raise HTTPException(status_code=500, detail="Document operation failed. The source record was retained; please retry.")


@router.post("", response_model=DocumentSourceResponse, status_code=201)
async def upload_document_source(file: UploadFile = File(...)):
    content = await read_upload(file)
    return await run_in_threadpool(source_operation, document_service.upload, content, file.filename)


@router.get("", response_model=List[DocumentSourceResponse])
def list_document_sources():
    return document_service.registry.list_document_sources()


# Legacy uploads predate the source registry: their chunks have no source_id
# and no retained original, so they can be listed and removed but not reindexed.
# These routes must be declared before /{source_id}.

@router.get("/legacy")
def list_legacy_documents():
    return document_service.vectors.list_legacy_files()


@router.delete("/legacy")
def delete_legacy_document(source_file: str):
    with document_service._lock:
        deleted = source_operation(document_service.vectors.delete_legacy_file, source_file)
        if not deleted:
            raise HTTPException(status_code=404, detail="No unregistered chunks found for that file")
        source_operation(document_service.refresh_search)
    return {"message": f"Removed {deleted} chunks", "chunks_deleted": deleted}


@router.get("/{source_id}", response_model=DocumentSourceResponse)
def get_document_source(source_id: str):
    return source_operation(document_service.get, source_id)


@router.delete("/{source_id}")
def delete_document_source(source_id: str):
    source_operation(document_service.delete, source_id)
    return {"message": "Document and its indexed chunks were deleted"}


@router.post("/{source_id}/reindex", response_model=IndexJobResponse)
def reindex_document_source(source_id: str):
    source = source_operation(document_service.reindex, source_id)
    return IndexJobResponse(source_id=source.id, status="success",
                            message="Document reindexed", chunks_added=source.chunk_count)
