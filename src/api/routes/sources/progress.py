"""Live progress of running indexing jobs (polled by the UI while a source is indexing)."""

from fastapi import APIRouter

from src.sources.jobs import index_jobs

router = APIRouter(prefix="/sources", tags=["Sources - Progress"])


@router.get("/progress", summary="Progress of running indexing jobs")
def indexing_progress():
    """{source_id: {stage, done, total, unit, percent, elapsed_seconds}} for each running job."""
    return index_jobs.progress()
