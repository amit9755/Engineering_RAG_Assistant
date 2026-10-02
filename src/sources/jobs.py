"""Background indexing jobs for Bitbucket and Jira sources.

Indexing a repository or project can take minutes (download + embedding),
so the API starts a job and returns immediately; the UI polls the source
status (indexing -> ready / error). One job per source at a time.
"""

import threading
from datetime import datetime, timezone

from src.sources.models import SourceStatus
from src.sources.registry import source_registry
from src.observability.logger import get_logger

logger = get_logger(__name__)


class IndexingError(Exception):
    """An expected failure whose message is safe and useful to show the user."""


class IndexJobRunner:
    def __init__(self, registry=None):
        self.registry = registry if registry is not None else source_registry
        self._running = set()
        self._lock = threading.Lock()

    def is_running(self, source_id: str) -> bool:
        with self._lock:
            return source_id in self._running

    def start(self, source_id: str, job, background: bool = True) -> bool:
        """
        Run job(source_id) -> chunk_count. Returns False if a job for this
        source is already running. background=False runs inline (tests).
        """
        with self._lock:
            if source_id in self._running:
                return False
            self._running.add(source_id)
        self.registry.update_source_status(source_id, SourceStatus.INDEXING)
        if background:
            threading.Thread(target=self._run, args=(source_id, job),
                             name=f"index-{source_id}", daemon=True).start()
        else:
            self._run(source_id, job)
        return True

    def _run(self, source_id: str, job) -> None:
        try:
            chunk_count = job(source_id)
            self.registry.update_source_status(
                source_id, SourceStatus.READY, chunk_count=chunk_count,
                last_sync=datetime.now(timezone.utc),
            )
        except IndexingError as exc:
            logger.warning("index_job_failed", source_id=source_id, error=str(exc))
            self.registry.update_source_status(source_id, SourceStatus.ERROR, error_message=str(exc))
        except Exception as exc:
            logger.exception("index_job_crashed", source_id=source_id)
            self.registry.update_source_status(
                source_id, SourceStatus.ERROR,
                error_message=f"Indexing failed unexpectedly ({type(exc).__name__}). "
                              "Previously indexed content was kept; retry indexing.",
            )
        finally:
            with self._lock:
                self._running.discard(source_id)

    def recover_interrupted(self) -> None:
        """Jobs do not survive a restart; mark sources left mid-index as failed."""
        for source in self.registry.list_sources():
            if source.status in (SourceStatus.INDEXING, SourceStatus.SYNCING):
                self.registry.update_source_status(
                    source.id, SourceStatus.ERROR,
                    error_message="Indexing was interrupted by a server restart. Index again.",
                )


index_jobs = IndexJobRunner()
