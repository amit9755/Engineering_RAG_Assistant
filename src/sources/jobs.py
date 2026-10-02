"""Background indexing jobs for Bitbucket and Jira sources.

Indexing a repository or project can take minutes (download + embedding),
so the API starts a job and returns immediately; the UI polls the source
status (indexing -> ready / error). One job per source at a time.
"""

import os
import threading
import time
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
        self._progress = {}   # source_id -> current stage of a running job
        # Indexing is CPU heavy: run a few jobs at a time and queue the rest.
        self._slots = threading.Semaphore(max(1, int(os.environ.get("INDEX_PARALLEL_JOBS", "2"))))

    def report(self, source_id: str, stage: str, done: int = None, total: int = None, unit: str = "") -> None:
        """Record what a running job is doing, for the UI and the server log."""
        with self._lock:
            entry = self._progress.get(source_id)
            if entry is None:
                return
            changed = entry["stage"] != stage
            # Log stage changes and every 10% within a stage, so long steps stay visible in the server log.
            decile = int(10 * done / total) if (done is not None and total) else None
            log = changed or (decile is not None and decile != entry.get("logged_decile"))
            entry.update(stage=stage, done=done, total=total, unit=unit,
                         logged_decile=decile if log else entry.get("logged_decile"))
        if log:
            percent = f" ({round(100 * done / total)}%)" if decile is not None else ""
            logger.info("index_progress", source_id=source_id,
                        stage=f"{stage}{'' if done is None else f' {done}/{total or '?'} {unit}'}{percent}")

    def progress(self) -> dict:
        """Snapshot of running jobs with elapsed seconds and percent (when the total is known)."""
        now = time.time()
        with self._lock:
            out = {}
            for source_id, entry in self._progress.items():
                item = {k: v for k, v in entry.items() if k not in ("started", "logged_decile")}
                item["elapsed_seconds"] = int(now - entry["started"])
                item["percent"] = (round(100 * entry["done"] / entry["total"])
                                   if entry.get("total") and entry.get("done") is not None else None)
                out[source_id] = item
            return out

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
            self._progress[source_id] = {"stage": "Queued (waiting for other indexing jobs)", "done": None,
                                         "total": None, "unit": "", "started": time.time()}
        self.registry.update_source_status(source_id, SourceStatus.INDEXING)
        if background:
            threading.Thread(target=self._run, args=(source_id, job),
                             name=f"index-{source_id}", daemon=True).start()
        else:
            self._run(source_id, job)
        return True

    def _run(self, source_id: str, job) -> None:
        self._slots.acquire()
        try:
            self.report(source_id, "Starting")
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
            self._slots.release()
            with self._lock:
                self._running.discard(source_id)
                self._progress.pop(source_id, None)

    def recover_interrupted(self) -> None:
        """Jobs do not survive a restart; mark sources left mid-index as failed."""
        for source in self.registry.list_sources():
            if source.status in (SourceStatus.INDEXING, SourceStatus.SYNCING):
                self.registry.update_source_status(
                    source.id, SourceStatus.ERROR,
                    error_message="Indexing was interrupted by a server restart. Index again.",
                )


index_jobs = IndexJobRunner()


def report_progress(source_id: str, stage: str, done: int = None, total: int = None, unit: str = "") -> None:
    index_jobs.report(source_id, stage, done, total, unit)
