# ============================================================
# src/sources/registry.py - Source registry backed by SQLite
#
# Stores all source records (documents, Bitbucket repos, Jira projects)
# in a local SQLite database at ./data/sources.db.
#
# Tables:
#   sources          - top-level source record (id, type, name, status, ...)
#   document_configs - document-specific config
#   bitbucket_configs- bitbucket-specific config
#   jira_configs     - jira-specific config
#
# The registry is the single source of truth for what has been
# configured in the system. The retrieval layer uses source_ids
# from the registry to filter results.
# ============================================================

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.sources.models import (
    BitbucketSourceConfig,
    BitbucketSourceResponse,
    DocumentSourceConfig,
    DocumentSourceResponse,
    JiraSourceConfig,
    JiraSourceResponse,
    Source,
    SourceResponse,
    SourceStatus,
    SourceType,
)
from src.observability.logger import get_logger

logger = get_logger(__name__)

_DATA_DIR = Path("./data")
_REGISTRY_DB = _DATA_DIR / "sources.db"


class SourceRegistry:
    """
    SQLite-backed registry of all knowledge sources.

    Provides CRUD operations for sources and their provider-specific
    configuration. All methods are synchronous (SQLite is fast enough
    for the expected volume of source records).
    """

    def __init__(self, db_path: Path = _REGISTRY_DB):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ----------------------------------------------------------
    # Schema initialisation
    # ----------------------------------------------------------

    def _init_db(self) -> None:
        """Create all tables if they do not exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sources (
                    id          TEXT PRIMARY KEY,
                    type        TEXT NOT NULL,
                    name        TEXT NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'pending',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    last_sync   TEXT,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS document_configs (
                    source_id   TEXT PRIMARY KEY REFERENCES sources(id) ON DELETE CASCADE,
                    filename    TEXT NOT NULL,
                    file_type   TEXT NOT NULL,
                    size_bytes  INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS bitbucket_configs (
                    source_id       TEXT PRIMARY KEY REFERENCES sources(id) ON DELETE CASCADE,
                    workspace       TEXT NOT NULL,
                    repository      TEXT NOT NULL,
                    branch          TEXT NOT NULL DEFAULT 'main',
                    credential_id   TEXT NOT NULL,
                    last_commit     TEXT,
                    file_count      INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS jira_configs (
                    source_id               TEXT PRIMARY KEY REFERENCES sources(id) ON DELETE CASCADE,
                    base_url                TEXT NOT NULL,
                    project_key             TEXT NOT NULL,
                    credential_id           TEXT NOT NULL,
                    issue_count             INTEGER NOT NULL DEFAULT 0,
                    last_sync_issue_updated TEXT
                );
            """)
            # Migration: Bitbucket Server / Data Center support (NULL = Bitbucket Cloud).
            columns = {r["name"] for r in conn.execute("PRAGMA table_info(bitbucket_configs)")}
            if "server_url" not in columns:
                conn.execute("ALTER TABLE bitbucket_configs ADD COLUMN server_url TEXT")
        logger.info("source_registry_initialized", path=str(self._db_path))

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat()

    def _row_to_source(self, row: sqlite3.Row) -> Source:
        return Source(
            id=row["id"],
            type=SourceType(row["type"]),
            name=row["name"],
            status=SourceStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_sync=datetime.fromisoformat(row["last_sync"]) if row["last_sync"] else None,
            chunk_count=row["chunk_count"],
            error_message=row["error_message"],
        )

    # ----------------------------------------------------------
    # Generic source operations
    # ----------------------------------------------------------

    def list_sources(self, source_type: Optional[SourceType] = None) -> List[Source]:
        """List all sources, optionally filtered by type."""
        with self._connect() as conn:
            if source_type:
                rows = conn.execute(
                    "SELECT * FROM sources WHERE type = ? ORDER BY created_at DESC",
                    (source_type.value,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM sources ORDER BY created_at DESC"
                ).fetchall()
        return [self._row_to_source(r) for r in rows]

    def get_source(self, source_id: str) -> Optional[Source]:
        """Get a source by ID. Returns None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sources WHERE id = ?", (source_id,)
            ).fetchone()
        return self._row_to_source(row) if row else None

    def update_source_status(
        self,
        source_id: str,
        status: SourceStatus,
        error_message: Optional[str] = None,
        chunk_count: Optional[int] = None,
        last_sync: Optional[datetime] = None,
    ) -> None:
        """Update the status and related fields of a source."""
        now = self._now()
        with self._connect() as conn:
            fields = ["status = ?", "updated_at = ?"]
            values: list = [status.value, now]

            if error_message is not None or status == SourceStatus.READY:
                fields.append("error_message = ?")
                values.append(error_message)
            if chunk_count is not None:
                fields.append("chunk_count = ?")
                values.append(chunk_count)
            if last_sync is not None:
                fields.append("last_sync = ?")
                values.append(last_sync.isoformat())

            values.append(source_id)
            conn.execute(
                f"UPDATE sources SET {', '.join(fields)} WHERE id = ?",
                values,
            )
        logger.info("source_status_updated", source_id=source_id, status=status.value)

    def delete_source(self, source_id: str) -> bool:
        """Delete a source and its config (cascades). Returns True if deleted."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("source_deleted", source_id=source_id)
        return deleted

    # ----------------------------------------------------------
    # Document source operations
    # ----------------------------------------------------------

    def create_document_source(self, config: DocumentSourceConfig, name: str) -> Source:
        """Register a new document source."""
        source = Source(
            id=config.source_id,
            type=SourceType.DOCUMENT,
            name=name,
            status=SourceStatus.PENDING,
        )
        now = self._now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sources (id, type, name, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source.id, source.type.value, source.name, source.status.value, now, now),
            )
            conn.execute(
                """
                INSERT INTO document_configs (source_id, filename, file_type, size_bytes)
                VALUES (?, ?, ?, ?)
                """,
                (config.source_id, config.filename, config.file_type, config.size_bytes),
            )

        logger.info("document_source_created", source_id=source.id, filename=config.filename)
        return source

    def get_document_config(self, source_id: str) -> Optional[DocumentSourceConfig]:
        """Get document-specific configuration."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM document_configs WHERE source_id = ?", (source_id,)
            ).fetchone()
        if not row:
            return None
        return DocumentSourceConfig(
            source_id=row["source_id"],
            filename=row["filename"],
            file_type=row["file_type"],
            size_bytes=row["size_bytes"],
        )

    def list_document_sources(self) -> List[DocumentSourceResponse]:
        """List all document sources with their configs."""
        sources = self.list_sources(SourceType.DOCUMENT)
        results = []
        for s in sources:
            cfg = self.get_document_config(s.id)
            if cfg:
                results.append(
                    DocumentSourceResponse(
                        **s.model_dump(),
                        filename=cfg.filename,
                        file_type=cfg.file_type,
                        size_bytes=cfg.size_bytes,
                    )
                )
        return results

    # ----------------------------------------------------------
    # Bitbucket source operations
    # ----------------------------------------------------------

    def create_bitbucket_source(
        self, config: BitbucketSourceConfig, name: str
    ) -> Source:
        """Register a new Bitbucket repository source."""
        source = Source(
            id=config.source_id,
            type=SourceType.BITBUCKET,
            name=name,
            status=SourceStatus.PENDING,
        )
        now = self._now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sources (id, type, name, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source.id, source.type.value, source.name, source.status.value, now, now),
            )
            conn.execute(
                """
                INSERT INTO bitbucket_configs
                    (source_id, workspace, repository, branch, credential_id, last_commit, file_count,
                     server_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    config.source_id,
                    config.workspace,
                    config.repository,
                    config.branch,
                    config.credential_id,
                    config.last_commit,
                    config.file_count,
                    config.server_url,
                ),
            )

        logger.info(
            "bitbucket_source_created",
            source_id=source.id,
            repo=f"{config.workspace}/{config.repository}",
        )
        return source

    def get_bitbucket_config(self, source_id: str) -> Optional[BitbucketSourceConfig]:
        """Get Bitbucket-specific configuration (no token)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM bitbucket_configs WHERE source_id = ?", (source_id,)
            ).fetchone()
        if not row:
            return None
        return BitbucketSourceConfig(
            source_id=row["source_id"],
            workspace=row["workspace"],
            repository=row["repository"],
            branch=row["branch"],
            credential_id=row["credential_id"],
            last_commit=row["last_commit"],
            file_count=row["file_count"],
            server_url=row["server_url"],
        )

    def update_bitbucket_config(
        self,
        source_id: str,
        workspace: Optional[str] = None,
        repository: Optional[str] = None,
        branch: Optional[str] = None,
        credential_id: Optional[str] = None,
        last_commit: Optional[str] = None,
        file_count: Optional[int] = None,
    ) -> None:
        """Partially update Bitbucket config fields."""
        fields = []
        values = []
        if workspace is not None:
            fields.append("workspace = ?")
            values.append(workspace)
        if repository is not None:
            fields.append("repository = ?")
            values.append(repository)
        if branch is not None:
            fields.append("branch = ?")
            values.append(branch)
        if credential_id is not None:
            fields.append("credential_id = ?")
            values.append(credential_id)
        if last_commit is not None:
            fields.append("last_commit = ?")
            values.append(last_commit)
        if file_count is not None:
            fields.append("file_count = ?")
            values.append(file_count)

        if not fields:
            return

        values.append(source_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE bitbucket_configs SET {', '.join(fields)} WHERE source_id = ?",
                values,
            )
        # Also update the parent source updated_at
        with self._connect() as conn:
            conn.execute(
                "UPDATE sources SET updated_at = ? WHERE id = ?",
                (self._now(), source_id),
            )

    def list_bitbucket_sources(self) -> List[BitbucketSourceResponse]:
        """List all Bitbucket sources with their configs."""
        sources = self.list_sources(SourceType.BITBUCKET)
        results = []
        for s in sources:
            cfg = self.get_bitbucket_config(s.id)
            if cfg:
                results.append(
                    BitbucketSourceResponse(
                        **s.model_dump(),
                        workspace=cfg.workspace,
                        repository=cfg.repository,
                        branch=cfg.branch,
                        last_commit=cfg.last_commit,
                        file_count=cfg.file_count,
                        server_url=cfg.server_url,
                        credential_configured=True,
                    )
                )
        return results

    # ----------------------------------------------------------
    # Jira source operations
    # ----------------------------------------------------------

    def create_jira_source(self, config: JiraSourceConfig, name: str) -> Source:
        """Register a new Jira project source."""
        source = Source(
            id=config.source_id,
            type=SourceType.JIRA,
            name=name,
            status=SourceStatus.PENDING,
        )
        now = self._now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sources (id, type, name, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source.id, source.type.value, source.name, source.status.value, now, now),
            )
            conn.execute(
                """
                INSERT INTO jira_configs
                    (source_id, base_url, project_key, credential_id, issue_count, last_sync_issue_updated)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    config.source_id,
                    config.base_url,
                    config.project_key,
                    config.credential_id,
                    config.issue_count,
                    config.last_sync_issue_updated.isoformat()
                    if config.last_sync_issue_updated else None,
                ),
            )

        logger.info(
            "jira_source_created",
            source_id=source.id,
            project=config.project_key,
        )
        return source

    def get_jira_config(self, source_id: str) -> Optional[JiraSourceConfig]:
        """Get Jira-specific configuration (no token)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jira_configs WHERE source_id = ?", (source_id,)
            ).fetchone()
        if not row:
            return None
        return JiraSourceConfig(
            source_id=row["source_id"],
            base_url=row["base_url"],
            project_key=row["project_key"],
            credential_id=row["credential_id"],
            issue_count=row["issue_count"],
            last_sync_issue_updated=(
                datetime.fromisoformat(row["last_sync_issue_updated"])
                if row["last_sync_issue_updated"] else None
            ),
        )

    def update_jira_config(
        self,
        source_id: str,
        base_url: Optional[str] = None,
        project_key: Optional[str] = None,
        credential_id: Optional[str] = None,
        issue_count: Optional[int] = None,
        last_sync_issue_updated: Optional[datetime] = None,
    ) -> None:
        """Partially update Jira config fields."""
        fields = []
        values = []
        if base_url is not None:
            fields.append("base_url = ?")
            values.append(base_url)
        if project_key is not None:
            fields.append("project_key = ?")
            values.append(project_key)
        if credential_id is not None:
            fields.append("credential_id = ?")
            values.append(credential_id)
        if issue_count is not None:
            fields.append("issue_count = ?")
            values.append(issue_count)
        if last_sync_issue_updated is not None:
            fields.append("last_sync_issue_updated = ?")
            values.append(last_sync_issue_updated.isoformat())

        if not fields:
            return

        values.append(source_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE jira_configs SET {', '.join(fields)} WHERE source_id = ?",
                values,
            )
        with self._connect() as conn:
            conn.execute(
                "UPDATE sources SET updated_at = ? WHERE id = ?",
                (self._now(), source_id),
            )

    def list_jira_sources(self) -> List[JiraSourceResponse]:
        """List all Jira sources with their configs."""
        sources = self.list_sources(SourceType.JIRA)
        results = []
        for s in sources:
            cfg = self.get_jira_config(s.id)
            if cfg:
                results.append(
                    JiraSourceResponse(
                        **s.model_dump(),
                        base_url=cfg.base_url,
                        project_key=cfg.project_key,
                        issue_count=cfg.issue_count,
                        last_sync_issue_updated=cfg.last_sync_issue_updated,
                        credential_configured=True,
                    )
                )
        return results

    # ----------------------------------------------------------
    # Utility: get all source IDs of a type
    # ----------------------------------------------------------

    def get_all_source_ids(self, source_type: Optional[SourceType] = None) -> List[str]:
        """Return list of all source IDs, optionally filtered by type."""
        sources = self.list_sources(source_type)
        return [s.id for s in sources]


# ============================================================
# Singleton instance
# ============================================================

source_registry = SourceRegistry()
