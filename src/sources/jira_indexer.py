"""Index the issues of a Jira project into the vector store.

Each issue becomes a text document (summary, key fields, description,
comments) split into chunks that all start with the issue key and
summary, so every chunk can be cited. Jira Cloud's /rest/api/3/search/jql
is used, with the older /rest/api/2/search as a fallback for Jira
Server / Data Center.
"""

import re

import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.sources.credentials import credential_store
from src.sources.jobs import IndexingError
from src.sources.registry import source_registry
from src.observability.logger import get_logger

logger = get_logger(__name__)

MAX_ISSUES = 5000
MAX_COMMENTS = 30
PAGE_SIZE = 100
PROJECT_KEY = re.compile(r"^[A-Z][A-Z0-9_]+$")
FIELDS = ("summary,description,status,issuetype,priority,assignee,reporter,created,updated,"
          "labels,components,fixVersions,resolution,parent,comment")

# Sized to fit the embedding model's 256-token window (see bitbucket_indexer).
_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)


def adf_to_text(node) -> str:
    """Flatten Atlassian Document Format (Jira Cloud rich text) to plain text."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node  # Jira Server / API v2 returns wiki-markup strings
    if isinstance(node, list):
        return "".join(adf_to_text(child) for child in node)
    kind = node.get("type")
    attrs = node.get("attrs") or {}
    if kind == "text":
        return node.get("text", "")
    if kind == "hardBreak":
        return "\n"
    if kind in ("mention", "emoji"):
        return attrs.get("text") or attrs.get("shortName", "")
    if kind == "inlineCard":
        return attrs.get("url", "")
    inner = adf_to_text(node.get("content", []))
    if kind == "listItem":
        return "- " + inner.strip() + "\n"
    if kind == "codeBlock":
        return "```\n" + inner + "\n```\n"
    if kind in ("paragraph", "heading", "blockquote", "rule", "tableRow"):
        return inner + "\n"
    if kind == "tableCell" or kind == "tableHeader":
        return inner.strip() + " | "
    return inner


def _name(value, key="displayName"):
    return (value or {}).get(key) or ""


def issue_to_text(issue: dict, base_url: str) -> str:
    f = issue.get("fields") or {}
    key = issue["key"]
    lines = [
        f"Jira issue {key}: {f.get('summary') or ''}",
        f"Type: {_name(f.get('issuetype'), 'name')} | Status: {_name(f.get('status'), 'name')}"
        f" | Priority: {_name(f.get('priority'), 'name') or 'None'}"
        f" | Resolution: {_name(f.get('resolution'), 'name') or 'Unresolved'}",
        f"Assignee: {_name(f.get('assignee')) or 'Unassigned'} | Reporter: {_name(f.get('reporter'))}",
        f"Created: {(f.get('created') or '')[:10]} | Updated: {(f.get('updated') or '')[:10]}",
    ]
    if f.get("parent"):
        lines.append(f"Parent: {f['parent'].get('key')} {((f['parent'].get('fields') or {}).get('summary') or '')}")
    if f.get("labels"):
        lines.append("Labels: " + ", ".join(f["labels"]))
    if f.get("components"):
        lines.append("Components: " + ", ".join(c.get("name", "") for c in f["components"]))
    if f.get("fixVersions"):
        lines.append("Fix versions: " + ", ".join(v.get("name", "") for v in f["fixVersions"]))
    lines.append(f"URL: {base_url}/browse/{key}")
    description = adf_to_text(f.get("description")).strip()
    lines += ["", "Description:", description or "(no description)"]
    comments = ((f.get("comment") or {}).get("comments") or [])[-MAX_COMMENTS:]
    if comments:
        lines += ["", "Comments:"]
        for comment in comments:
            body = adf_to_text(comment.get("body")).strip()
            lines.append(f"- {_name(comment.get('author'))} ({(comment.get('created') or '')[:10]}): {body}")
    return "\n".join(lines)


class JiraClient:
    def __init__(self, base_url: str, email: str, token: str, session=None):
        self.base_url = base_url.rstrip("/")
        self.auth = (email, token)
        self.http = session or requests.Session()

    def _get(self, path, params):
        try:
            response = self.http.get(f"{self.base_url}{path}", params=params, auth=self.auth,
                                     headers={"Accept": "application/json"}, timeout=60)
        except requests.RequestException as exc:
            raise IndexingError(f"Could not reach Jira at {self.base_url}: {exc.__class__.__name__}") from exc
        if response.status_code == 401:
            raise IndexingError("Jira rejected the saved credentials. Delete and re-add the project with a valid API token.")
        if response.status_code == 403:
            raise IndexingError("The Jira account lacks permission to browse this project.")
        return response

    def search_issues(self, project_key: str):
        jql = f'project = "{project_key}" ORDER BY updated DESC'
        issues, token = [], None
        while len(issues) < MAX_ISSUES:
            params = {"jql": jql, "fields": FIELDS, "maxResults": PAGE_SIZE}
            if token:
                params["nextPageToken"] = token
            response = self._get("/rest/api/3/search/jql", params)
            if response.status_code in (404, 405, 410) and not issues:
                return self._search_issues_v2(jql)
            self._raise_for_search(response, project_key)
            data = response.json()
            issues += data.get("issues", [])
            token = data.get("nextPageToken")
            if not token or data.get("isLast"):
                break
        return issues[:MAX_ISSUES]

    def _search_issues_v2(self, jql):
        """Jira Server / Data Center fallback (offset pagination)."""
        issues = []
        while len(issues) < MAX_ISSUES:
            response = self._get("/rest/api/2/search",
                                 {"jql": jql, "fields": FIELDS, "maxResults": PAGE_SIZE, "startAt": len(issues)})
            self._raise_for_search(response, None)
            data = response.json()
            page = data.get("issues", [])
            issues += page
            if not page or len(issues) >= data.get("total", 0):
                break
        return issues[:MAX_ISSUES]

    @staticmethod
    def _raise_for_search(response, project_key):
        if response.status_code == 400:
            raise IndexingError(f"Jira could not search project '{project_key}'. "
                                "Check the project key and that your account can browse it.")
        if response.status_code != 200:
            raise IndexingError(f"Jira search failed (HTTP {response.status_code}).")


class JiraIndexer:
    def __init__(self, registry=None, credentials=None, vector_store=None, refresh=None, client_factory=None):
        self.registry = registry if registry is not None else source_registry
        self.credentials = credentials if credentials is not None else credential_store
        self._vector_store = vector_store
        self._refresh = refresh
        self.client_factory = client_factory or JiraClient

    @property
    def vectors(self):
        if self._vector_store is None:
            from src.retrieval.vector_store import vector_store
            self._vector_store = vector_store
        return self._vector_store

    def refresh_search(self):
        if self._refresh is not None:
            self._refresh()
        else:
            from src.ingestion.ingestion_pipeline import ingestion_pipeline
            ingestion_pipeline._refresh_bm25_index(strict=True)

    def index(self, source_id: str) -> int:
        """Full index of the project's issues. Returns the number of chunks stored."""
        source = self.registry.get_source(source_id)
        cfg = self.registry.get_jira_config(source_id)
        if not source or not cfg:
            raise IndexingError("Jira source not found.")
        if not PROJECT_KEY.match(cfg.project_key):
            raise IndexingError(f"'{cfg.project_key}' is not a valid Jira project key.")
        try:
            email, token = self.credentials.retrieve(cfg.credential_id).split(":", 1)
        except (KeyError, ValueError) as exc:
            raise IndexingError("Saved Jira credentials are missing. Delete and re-add the project.") from exc

        issues = self.client_factory(cfg.base_url, email, token).search_issues(cfg.project_key)
        if not issues:
            raise IndexingError(f"No issues found in project {cfg.project_key} (or the account cannot see them).")

        documents = []
        for issue in issues:
            text = issue_to_text(issue, cfg.base_url)
            summary = (issue.get("fields") or {}).get("summary") or ""
            header = f"Jira issue {issue['key']}: {summary}\n"
            chunks = _splitter.split_text(text)
            for index, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk if index == 0 else header + chunk,
                    metadata={
                        "source_id": source_id,
                        "source_type": "jira",
                        "source_name": source.name,
                        "source_file": issue["key"],
                        "issue_key": issue["key"],
                        "project_key": cfg.project_key,
                        "status": _name((issue.get("fields") or {}).get("status"), "name"),
                        "url": f"{cfg.base_url}/browse/{issue['key']}",
                        "chunk_index": index,
                        "total_chunks": len(chunks),
                        "chunk_id": f"{source_id}:{issue['key']}:{index}",
                    },
                ))

        self.vectors.replace_source_documents(source_id, documents)
        self.refresh_search()
        latest = max(((i.get("fields") or {}).get("updated") or "") for i in issues)
        self.registry.update_jira_config(source_id, issue_count=len(issues),
                                         last_sync_issue_updated=_parse_jira_time(latest))
        logger.info("jira_index_done", source_id=source_id, issues=len(issues), chunks=len(documents))
        return len(documents)


def _parse_jira_time(value):
    from datetime import datetime
    if not value:
        return None
    try:
        # Jira format: 2024-05-01T10:20:30.000+0000
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f%z")
    except ValueError:
        return None


jira_indexer = JiraIndexer()
