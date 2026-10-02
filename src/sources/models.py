# ============================================================
# src/sources/models.py - Pydantic data models for all source types
#
# Separates source configuration (workspace, repo, branch) from
# credentials (tokens). Credentials are stored by reference only
# via credential_id. Tokens never appear in source records.
# ============================================================

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


# ============================================================
# Enumerations
# ============================================================

class SourceType(str, Enum):
    DOCUMENT = "document"
    BITBUCKET = "bitbucket"
    JIRA = "jira"


class SourceStatus(str, Enum):
    PENDING = "pending"
    INDEXING = "indexing"
    READY = "ready"
    ERROR = "error"
    SYNCING = "syncing"


# ============================================================
# Core Source record (provider-agnostic)
# ============================================================

class Source(BaseModel):
    """Top-level source record stored in the registry."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: SourceType
    name: str = Field(description="Human-readable display name")
    status: SourceStatus = SourceStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_sync: Optional[datetime] = None
    chunk_count: int = 0
    error_message: Optional[str] = None


# ============================================================
# Provider-specific configuration records
# ============================================================

class DocumentSourceConfig(BaseModel):
    """Configuration for an uploaded document source."""
    source_id: str
    filename: str
    file_type: str        # "pdf", "docx", "txt", "md"
    size_bytes: int = 0


class BitbucketSourceConfig(BaseModel):
    """Configuration for a Bitbucket repository source.

    Note: credential_id is a reference to a Credential record.
    The actual token is NEVER stored here.
    """
    source_id: str
    workspace: str                      # Cloud: workspace slug. Server: project key
    repository: str
    branch: str = "main"
    credential_id: str
    last_commit: Optional[str] = None   # last indexed commit hash
    file_count: int = 0
    server_url: Optional[str] = None    # None = Bitbucket Cloud; else Bitbucket Server / Data Center base URL


class JiraSourceConfig(BaseModel):
    """Configuration for a Jira project source.

    Note: credential_id is a reference to a Credential record.
    """
    source_id: str
    base_url: str          # e.g. https://company.atlassian.net
    project_key: str
    credential_id: str
    issue_count: int = 0
    last_sync_issue_updated: Optional[datetime] = None


# ============================================================
# Credential record (encrypted token storage)
# ============================================================

class Credential(BaseModel):
    """Stores an encrypted credential reference.

    The plaintext token is NEVER stored here.
    encrypted_value contains the Fernet-encrypted bytes (base64).
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    provider: str          # "bitbucket" | "jira"
    label: str             # display hint, e.g. "my-workspace token"
    # encrypted_value is stored only in the DB layer, never exposed via API
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# API request/response models (never include raw tokens)
# ============================================================

class SourceResponse(BaseModel):
    """Generic source info returned by GET endpoints."""
    id: str
    type: SourceType
    name: str
    status: SourceStatus
    created_at: datetime
    updated_at: datetime
    last_sync: Optional[datetime]
    chunk_count: int
    error_message: Optional[str]


class DocumentSourceResponse(SourceResponse):
    filename: str
    file_type: str
    size_bytes: int


class BitbucketSourceResponse(SourceResponse):
    workspace: str
    repository: str
    branch: str
    last_commit: Optional[str]
    file_count: int
    server_url: Optional[str] = None
    credential_configured: bool = True  # always true; never returns token


class JiraSourceResponse(SourceResponse):
    base_url: str
    project_key: str
    issue_count: int
    last_sync_issue_updated: Optional[datetime]
    credential_configured: bool = True


# ============================================================
# Request models for creating/testing sources
# ============================================================

class AddBitbucketSourceRequest(BaseModel):
    """Request to add a Bitbucket source. Token provided once, then stored encrypted."""
    workspace: str = Field(description="Cloud: workspace slug. Server / Data Center: project key")
    repository: str = Field(description="Repository slug")
    branch: str = Field(default="main")
    username: str = Field(default="", description="Cloud: account email. Server: optional with an HTTP access token")
    token: str = Field(description="Cloud: API token / app password. Server: HTTP access token")
    name: Optional[str] = None  # display name; defaults to workspace/repository
    server_url: Optional[str] = Field(default=None, description="Bitbucket Server / Data Center URL; omit for Cloud")


class TestBitbucketRequest(BaseModel):
    workspace: str = ""          # may come from a pasted Bitbucket Server repository link
    username: str = ""
    token: str
    server_url: Optional[str] = None
    repository: Optional[str] = None


class AddJiraSourceRequest(BaseModel):
    base_url: str = Field(description="Jira base URL, e.g. https://company.atlassian.net")
    project_key: str = Field(description="Jira project key, e.g. BT")
    email: str = Field(description="Jira account email")
    token: str = Field(description="Jira API token")
    name: Optional[str] = None


class TestJiraRequest(BaseModel):
    base_url: str
    email: str
    token: str


class UpdateBitbucketSourceRequest(BaseModel):
    """Update a Bitbucket source. Leave token blank to keep existing credential."""
    workspace: Optional[str] = None
    repository: Optional[str] = None
    branch: Optional[str] = None
    username: Optional[str] = None
    token: Optional[str] = None   # blank = keep existing


class UpdateJiraSourceRequest(BaseModel):
    base_url: Optional[str] = None
    project_key: Optional[str] = None
    email: Optional[str] = None
    token: Optional[str] = None   # blank = keep existing


# ============================================================
# Indexing job response
# ============================================================

class IndexJobResponse(BaseModel):
    source_id: str
    status: str
    message: str
    chunks_added: Optional[int] = None
    files_processed: Optional[int] = None
    issues_processed: Optional[int] = None
