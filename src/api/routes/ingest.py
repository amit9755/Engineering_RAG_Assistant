# ============================================================
# src/api/routes/ingest.py - Document Ingestion Endpoints
#
# Learning Note:
#   These endpoints allow users to upload documents via the UI.
#   We use FastAPI's UploadFile for multipart form data uploads.
#   Files go through: upload -> temp storage -> chunking ->
#   embedding -> vector store indexing -> BM25 index update.
# ============================================================

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from src.observability.logger import get_logger
from starlette.concurrency import run_in_threadpool
from src.sources.documents import ALLOWED_EXTENSIONS, MAX_FILE_BYTES

logger = get_logger(__name__)

router = APIRouter()

MAX_FILE_SIZE_MB = 50


class IngestResponse(BaseModel):
    status: str
    source: str
    chunks_added: int
    message: Optional[str] = None
    source_id: Optional[str] = None


class IngestDirectoryRequest(BaseModel):
    directory_path: str


async def read_upload(file: UploadFile) -> bytes:
    """Validate uploads consistently, reading at most one byte beyond the size limit."""
    from pathlib import Path

    if Path(file.filename or "").suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload PDF, DOCX, TXT, or MD.")
    content = await file.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum is 50 MB.")
    if not content:
        raise HTTPException(status_code=422, detail="The uploaded file is empty")
    return content


@router.post("/ingest/file", response_model=IngestResponse, summary="Upload and ingest a document")
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload a document (PDF, TXT, DOCX, MD) and ingest it into the RAG system.

    The file will be:
    1. Validated (type and size check)
    2. Split into chunks
    3. Embedded and stored in the vector database
    4. Indexed in BM25 for keyword search

    After ingestion, the document is immediately queryable.
    """
    content = await read_upload(file)
    size_mb = len(content) / (1024 * 1024)

    logger.info("file_upload_received", filename=file.filename, size_mb=round(size_mb, 2))

    try:
        from src.ingestion.ingestion_pipeline import ingestion_pipeline
        result = await run_in_threadpool(ingestion_pipeline.ingest_bytes, content, file.filename)

        if result["status"] == "error":
            raise HTTPException(status_code=422, detail=result.get("message", "Ingestion failed"))

        return IngestResponse(
            status="success",
            source=result["source"],
            chunks_added=result["chunks_added"],
            source_id=result.get("source_id"),
        )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("ingest_endpoint_error", filename=file.filename, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/ingest/files", summary="Upload multiple documents at once")
async def ingest_multiple_files(files: List[UploadFile] = File(...)):
    """Ingest multiple documents in one request."""
    results = []
    for file in files:
        try:
            suffix = __import__("pathlib").Path(file.filename).suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                results.append({"file": file.filename, "status": "skipped", "reason": "unsupported type"})
                continue

            content = await read_upload(file)
            from src.ingestion.ingestion_pipeline import ingestion_pipeline
            result = await run_in_threadpool(ingestion_pipeline.ingest_bytes, content, file.filename)
            results.append({"file": file.filename, **result})
        except Exception as exc:
            results.append({"file": file.filename, "status": "error", "message": str(exc)})

    return {"results": results, "total_files": len(files)}


@router.post("/ingest/gcs", summary="Ingest a document from Google Cloud Storage")
async def ingest_from_gcs(gcs_uri: str):
    """
    Ingest a document from GCS bucket.
    URI format: gs://bucket-name/path/to/document.pdf

    Learning Note:
        In production, this endpoint is often triggered automatically
        via a Cloud Function or Pub/Sub subscription when a new file
        is uploaded to the GCS bucket.
    """
    if not gcs_uri.startswith("gs://"):
        raise HTTPException(status_code=400, detail="GCS URI must start with gs://")

    try:
        from src.ingestion.ingestion_pipeline import ingestion_pipeline
        result = ingestion_pipeline.ingest_gcs_file(gcs_uri)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/ingest/stats", summary="Get ingestion statistics")
async def get_ingestion_stats():
    """Return current knowledge base statistics."""
    try:
        from src.ingestion.ingestion_pipeline import ingestion_pipeline
        return ingestion_pipeline.get_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/ingest/reset", summary="Clear all ingested documents (DANGEROUS)")
async def reset_knowledge_base():
    """
    Delete all documents from the vector store and reset BM25 index.
    WARNING: This is irreversible. Use only for testing/reset.
    """
    try:
        from src.retrieval.vector_store import vector_store
        vector_store.delete_collection()
        logger.warning("knowledge_base_reset")
        return {"status": "success", "message": "All documents deleted from knowledge base"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
