"""Index a Bitbucket Cloud repository branch into the vector store.

Flow: resolve branch -> latest commit, download the commit as one zip
archive, keep readable source/text files, split them into chunks that
carry their repository and file path, and atomically replace the
source's previous chunks. A file-tree overview is indexed too, so broad
questions ("what does this repo contain?") have something to retrieve.
"""

import io
import tempfile
import zipfile
from pathlib import PurePosixPath
from urllib.parse import quote

import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.sources.credentials import credential_store
from src.sources.jobs import IndexingError, report_progress
from src.sources.registry import source_registry
from src.observability.logger import get_logger

logger = get_logger(__name__)

API_BASE = "https://api.bitbucket.org/2.0"
WEB_BASE = "https://bitbucket.org"
MAX_ARCHIVE_BYTES = 300 * 1024 * 1024
MAX_FILE_BYTES = 400 * 1024
MAX_FILES = 5000
MAX_COMMITS = 200          # recent commit history indexed per branch
MAX_LISTED_REPOS = 2000    # repository browser limit
COMMITS_PER_CHUNK = 12

TEXT_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue", ".svelte",
    ".java", ".kt", ".kts", ".scala", ".groovy", ".go", ".rs", ".rb", ".php",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".swift", ".m", ".dart", ".lua",
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".sql", ".graphql", ".proto",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".xml", ".gradle",
    ".properties", ".tf", ".tfvars", ".hcl",
    ".md", ".rst", ".txt", ".adoc",
}
TEXT_FILENAMES = {
    "dockerfile", "makefile", "procfile", "jenkinsfile", "gemfile", "rakefile",
    "readme", "license", "changelog", ".gitignore", ".dockerignore", ".env.example",
    ".env.sample", ".env.template",
}
SKIP_DIRS = {
    ".git", "node_modules", "bower_components", "vendor", "dist", "build", "out",
    "target", "bin", "obj", "coverage", "__pycache__", ".venv", "venv", "env",
    ".next", ".nuxt", ".gradle", ".idea", ".vscode", ".terraform", ".pytest_cache",
}
SKIP_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "pipfile.lock",
    "composer.lock", "gemfile.lock", "cargo.lock", "go.sum",
}
# Never index likely secrets, even if a repository contains them.
SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".jks", ".keystore"}
SECRET_NAMES = {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", ".npmrc", ".pypirc", ".netrc"}

LANGUAGES = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript",
    ".tsx": "typescript", ".java": "java", ".go": "go", ".rs": "rust", ".rb": "ruby",
    ".php": "php", ".cs": "csharp", ".kt": "kotlin", ".swift": "swift", ".sql": "sql",
    ".md": "markdown", ".html": "html", ".css": "css", ".yaml": "yaml", ".yml": "yaml",
    ".json": "json", ".sh": "shell", ".tf": "terraform",
}

# all-MiniLM-L6-v2 embeds only the first 256 tokens of a chunk; ~900 characters
# of code (plus the repository/file header) stays within that, so nothing is cut off.
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=100,
    separators=["\nclass ", "\ndef ", "\nasync def ", "\nfunction ", "\nexport ",
                "\npublic ", "\nprivate ", "\n\n", "\n", " ", ""],
)


def is_indexable(path: str, size: int) -> bool:
    parts = PurePosixPath(path).parts
    if not parts or any(p.lower() in SKIP_DIRS for p in parts[:-1]):
        return False
    name = parts[-1].lower()
    suffix = PurePosixPath(name).suffix
    if name in SKIP_FILENAMES or name.endswith((".min.js", ".min.css", ".map")):
        return False
    if suffix in SECRET_SUFFIXES or name in SECRET_NAMES:
        return False
    if name == ".env" or (name.startswith(".env.") and name not in TEXT_FILENAMES):
        return False
    if size == 0 or size > MAX_FILE_BYTES:
        return False
    return suffix in TEXT_EXTENSIONS or name in TEXT_FILENAMES


class BitbucketClient:
    """Bitbucket Cloud (bitbucket.org): workspace / repository, email + API token."""

    def __init__(self, username: str, token: str, session=None):
        self.auth = (username, token)
        self.http = session or requests.Session()

    def test_connection(self, workspace, repository=None) -> int:
        """Raise IndexingError with a readable message, else return the number of visible repositories."""
        response = self._get(f"{API_BASE}/repositories/{workspace}")
        if response.status_code == 404:
            raise IndexingError(f"Workspace '{workspace}' not found")
        _raise_for_status(response)
        if repository:
            self.get_repository(workspace, repository)
        return response.json().get("size", 0)

    def list_repositories(self, workspace=None):
        """Repositories in a workspace: [{workspace, slug, name}]."""
        if not workspace:
            raise IndexingError("Enter the workspace to list its repositories.")
        url, params, repos = f"{API_BASE}/repositories/{workspace}", {"pagelen": 100}, []
        while url and len(repos) < MAX_LISTED_REPOS:
            response = self._get(url, params=params)
            if response.status_code == 404:
                raise IndexingError(f"Workspace '{workspace}' not found")
            _raise_for_status(response)
            data = response.json()
            repos += [{"workspace": workspace, "slug": r["slug"], "name": r.get("name") or r["slug"]}
                      for r in data.get("values", [])]
            url, params = data.get("next"), None
        return repos

    def file_url(self, workspace, repository, commit, path=None):
        base = f"{WEB_BASE}/{workspace}/{repository}"
        return f"{base}/src/{commit}/{path}" if path else base

    def _get(self, url, **kwargs):
        _check_network(url)
        try:
            response = self.http.get(url, auth=self.auth, timeout=kwargs.pop("timeout", 30), **kwargs)
        except requests.RequestException as exc:
            raise IndexingError(f"Could not reach Bitbucket: {exc.__class__.__name__}") from exc
        if response.status_code == 401:
            raise IndexingError("Bitbucket rejected the saved credentials. "
                                "Delete and re-add the repository with a valid token.")
        if response.status_code == 403:
            raise IndexingError("The Bitbucket token lacks permission to read this repository "
                                "(it needs repository read access).")
        return response

    def get_repository(self, workspace, repository):
        response = self._get(f"{API_BASE}/repositories/{workspace}/{repository}")
        if response.status_code == 404:
            raise IndexingError(f"Repository {workspace}/{repository} was not found or is not visible to this token.")
        _raise_for_status(response)
        return response.json()

    def get_branch_commit(self, workspace, repository, branch):
        response = self._get(f"{API_BASE}/repositories/{workspace}/{repository}/refs/branches/{quote(branch, safe='')}")
        if response.status_code == 404:
            return None
        _raise_for_status(response)
        return response.json()["target"]["hash"]

    def get_commits(self, workspace, repository, commit, limit=MAX_COMMITS):
        """Newest-first history reachable from commit: [{id, author, date, message}]."""
        url = f"{API_BASE}/repositories/{workspace}/{repository}/commits/{commit}"
        commits, params = [], {"pagelen": 100}
        while url and len(commits) < limit:
            response = self._get(url, params=params)
            _raise_for_status(response)
            data = response.json()
            for c in data.get("values", []):
                raw_author = (c.get("author") or {}).get("raw", "")   # "Name <email>"
                commits.append({"id": c.get("hash", ""), "author": raw_author.split("<")[0].strip() or raw_author,
                                "date": (c.get("date") or "")[:10], "message": c.get("message", "")})
            url, params = data.get("next"), None
        return commits[:limit]

    def download_archive(self, workspace, repository, commit, on_progress=None) -> bytes:
        url = f"{WEB_BASE}/{workspace}/{repository}/get/{commit}.zip"
        response = self._get(url, stream=True, timeout=120)
        if response.status_code != 200:
            raise IndexingError(f"Downloading the repository archive failed (HTTP {response.status_code}).")
        return _read_limited(response, on_progress)


class BitbucketServerClient:
    """
    Bitbucket Server / Data Center (self-hosted, e.g. https://bitbucket.company.com):
    project key / repository, REST API 1.0, HTTP access token sent as a Bearer token.
    If the token is rejected and a username was given, Basic auth is tried instead
    (older servers accept username + token or password that way).
    """

    def __init__(self, username: str, token: str, server_url: str, session=None):
        self.base = server_url.rstrip("/")
        self.username, self.token = username, token
        self.http = session or requests.Session()

    def _api(self, project, repository=None, suffix=""):
        url = f"{self.base}/rest/api/1.0/projects/{quote(project, safe='')}"
        if repository:
            url += f"/repos/{quote(repository, safe='')}"
        return url + suffix

    def _get(self, url, accept="application/json", **kwargs):
        _check_network(url)
        timeout = kwargs.pop("timeout", 30)
        try:
            response = self.http.get(url, headers={"Authorization": f"Bearer {self.token}", "Accept": accept},
                                     timeout=timeout, **kwargs)
            if response.status_code == 401 and self.username:
                response = self.http.get(url, auth=(self.username, self.token), headers={"Accept": accept},
                                         timeout=timeout, **kwargs)
        except requests.exceptions.SSLError as exc:
            raise IndexingError(f"SSL certificate check failed for {self.base}. On a company network, install "
                                "pip-system-certs in the app's Python environment so it trusts the company "
                                "certificate.") from exc
        except requests.RequestException as exc:
            raise IndexingError(f"Could not reach {self.base}: {exc.__class__.__name__}. "
                                "Check the URL and that you are on the company network or VPN.") from exc
        if response.status_code == 401:
            raise IndexingError("Bitbucket Server rejected the token. Use an HTTP access token "
                                "(Profile > Manage account > HTTP access tokens) with Repository read permission.")
        if response.status_code == 403:
            raise IndexingError("The token lacks permission to read this project or repository "
                                "(it needs Repository read).")
        return response

    def test_connection(self, project, repository=None) -> int:
        response = self._get(self._api(project, suffix="/repos"), params={"limit": 100})
        if response.status_code == 404:
            raise IndexingError(f"Project '{project}' not found on {self.base} (use the project KEY, e.g. WSQAAUTO).")
        _raise_for_status(response)
        if repository:
            self.get_repository(project, repository)
        return response.json().get("size", 0)

    def list_repositories(self, project=None):
        """Repositories in a project, or every repository the token can see: [{workspace, slug, name}]."""
        url = self._api(project, suffix="/repos") if project else f"{self.base}/rest/api/1.0/repos"
        repos, start = [], 0
        while len(repos) < MAX_LISTED_REPOS:
            response = self._get(url, params={"limit": 100, "start": start})
            if response.status_code == 404:
                raise IndexingError(f"Project '{project}' not found on {self.base} (use the project KEY).")
            _raise_for_status(response)
            data = response.json()
            repos += [{"workspace": (r.get("project") or {}).get("key") or project, "slug": r["slug"],
                       "name": r.get("name") or r["slug"]} for r in data.get("values", [])]
            if data.get("isLastPage", True):
                break
            start = data.get("nextPageStart", start + 100)
        return repos[:MAX_LISTED_REPOS]

    def get_repository(self, project, repository):
        response = self._get(self._api(project, repository))
        if response.status_code == 404:
            raise IndexingError(f"Repository {project}/{repository} was not found or is not visible to this token.")
        _raise_for_status(response)
        repo = response.json()
        default = self._default_branch(project, repository)
        repo["mainbranch"] = {"name": default} if default else None
        return repo

    def _default_branch(self, project, repository):
        for suffix in ("/default-branch", "/branches/default"):  # newer, then older servers
            response = self._get(self._api(project, repository, suffix))
            if response.status_code == 200:
                return response.json().get("displayId")
        return None

    def get_branch_commit(self, project, repository, branch):
        response = self._get(self._api(project, repository, "/branches"),
                             params={"filterText": branch, "limit": 100})
        if response.status_code == 404:
            raise IndexingError(f"Repository {project}/{repository} was not found or is not visible to this token.")
        _raise_for_status(response)
        for ref in response.json().get("values", []):
            if ref.get("displayId") == branch or ref.get("id") == f"refs/heads/{branch}":
                return ref.get("latestCommit")
        return None

    def get_commits(self, project, repository, commit, limit=MAX_COMMITS):
        """Newest-first history reachable from commit: [{id, author, date, message}]."""
        from datetime import datetime, timezone
        commits, start = [], 0
        while len(commits) < limit:
            response = self._get(self._api(project, repository, "/commits"),
                                 params={"until": commit, "limit": 100, "start": start})
            _raise_for_status(response)
            data = response.json()
            for c in data.get("values", []):
                stamp = c.get("authorTimestamp")
                author = c.get("author") or {}
                commits.append({
                    "id": c.get("id", ""),
                    "author": author.get("displayName") or author.get("name", ""),
                    "date": datetime.fromtimestamp(stamp / 1000, timezone.utc).strftime("%Y-%m-%d") if stamp else "",
                    "message": c.get("message", ""),
                })
            if data.get("isLastPage", True):
                break
            start = data.get("nextPageStart", start + 100)
        return commits[:limit]

    def download_archive(self, project, repository, commit, on_progress=None) -> bytes:
        # prefix gives entries the same "<folder>/" layout as Bitbucket Cloud archives.
        # The archive is binary: asking for JSON makes the server answer 406 Not Acceptable.
        response = self._get(self._api(project, repository, "/archive"), accept="*/*",
                             params={"at": commit, "format": "zip", "prefix": f"{repository}/"},
                             stream=True, timeout=300)
        if response.status_code != 200:
            raise IndexingError(f"Downloading the repository archive failed (HTTP {response.status_code}).")
        return _read_limited(response, on_progress)

    def file_url(self, project, repository, commit, path=None):
        base = f"{self.base}/projects/{project}/repos/{repository}/browse"
        return f"{base}/{path}?at={commit}" if path else base


def _raise_for_status(response):
    """Unexpected HTTP status as a readable IndexingError (shown in the UI) instead of a server error."""
    if response.status_code >= 400:
        detail = ""
        try:
            data = response.json()
            errors = data.get("errors") or [data.get("error") or {}]
            detail = (errors[0] or {}).get("message", "") if isinstance(errors, list) else ""
        except Exception:
            pass
        raise IndexingError(f"Bitbucket returned HTTP {response.status_code}{': ' + detail if detail else ''}.")


def _check_network(url):
    from src import network_policy
    try:
        network_policy.check_url(url, "a Bitbucket request")
    except network_policy.ExternalNetworkBlocked as exc:
        raise IndexingError(str(exc)) from exc


def _read_limited(response, on_progress=None) -> bytes:
    """Read a streamed download, enforcing the size limit. on_progress(bytes_so_far, bytes_total_or_None)."""
    try:
        expected = int(response.headers.get("Content-Length") or 0) or None
    except (AttributeError, TypeError, ValueError):
        expected = None
    with tempfile.SpooledTemporaryFile(max_size=50 * 1024 * 1024) as buffer:
        total = 0
        for block in response.iter_content(1024 * 1024):
            total += len(block)
            if on_progress:
                on_progress(total, expected)
            if total > MAX_ARCHIVE_BYTES:
                raise IndexingError("Repository archive exceeds 300 MB; it is too large to index locally.")
            buffer.write(block)
        buffer.seek(0)
        return buffer.read()


def make_client(username: str, token: str, server_url: str = None):
    """Bitbucket Cloud client, or a Bitbucket Server / Data Center client when server_url is set."""
    if server_url:
        return BitbucketServerClient(username, token, server_url)
    return BitbucketClient(username, token)


def parse_server_url(url: str):
    """
    Accept a server base URL or any repository page URL and return
    (base_url, project_key or None, repository or None), e.g.
    https://bitbucket.sw.nxp.com/projects/WSQAAUTO/repos/wireless_centralized_server_gen2/browse
    -> ("https://bitbucket.sw.nxp.com", "WSQAAUTO", "wireless_centralized_server_gen2").
    """
    import re
    url = (url or "").strip()
    if not re.match(r"^https://[^/\s]+", url, re.I):
        raise ValueError("Bitbucket Server URL must start with https://")
    match = re.match(r"^(https://.+?)/(?:projects|users)/([^/]+)/repos/([^/?#]+)", url, re.I)
    if match:
        return match.group(1).rstrip("/"), match.group(2), match.group(3)
    match = re.match(r"^(https://.+?)/projects/([^/?#]+)", url, re.I)
    if match:
        return match.group(1).rstrip("/"), match.group(2), None
    return re.split(r"[?#]", url)[0].rstrip("/"), None, None


def resolve_branch(client, cfg):
    """Return (branch, commit). Falls back to the repository's default branch if the configured one is missing."""
    commit = client.get_branch_commit(cfg.workspace, cfg.repository, cfg.branch)
    if commit:
        return cfg.branch, commit
    repo = client.get_repository(cfg.workspace, cfg.repository)
    default = (repo.get("mainbranch") or {}).get("name")
    if not default:
        raise IndexingError(f"Branch '{cfg.branch}' does not exist and the repository has no default branch.")
    commit = client.get_branch_commit(cfg.workspace, cfg.repository, default)
    if not commit:
        raise IndexingError(f"Branch '{cfg.branch}' does not exist.")
    logger.warning("bitbucket_branch_fallback", configured=cfg.branch, default=default)
    return default, commit


def build_documents(archive: bytes, source, cfg, branch, commit, file_url=None):
    """Turn a repository zip archive into chunk Documents. Returns (documents, file_count)."""
    file_url = file_url or BitbucketClient("", "").file_url
    repo_name = f"{cfg.workspace}/{cfg.repository}"
    documents, paths = [], []
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            # Archive entries are prefixed with one folder ("<workspace>-<repo>-<hash>/" on
            # Cloud, "<repo>/" on Server via the prefix parameter).
            path = "/".join(info.filename.split("/")[1:])
            if not is_indexable(path, info.file_size):
                continue
            if len(paths) >= MAX_FILES:
                logger.warning("bitbucket_file_limit_reached", limit=MAX_FILES)
                break
            raw = zf.read(info)
            if b"\x00" in raw[:8192]:
                continue  # binary content despite a text extension
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if not text.strip():
                continue
            paths.append(path)
            suffix = PurePosixPath(path).suffix.lower()
            header = f"Repository: {repo_name} (branch {branch})\nFile: {path}\n\n"
            chunks = _splitter.split_text(text)
            for index, chunk in enumerate(chunks):
                documents.append(_document(header + chunk, source, cfg, branch, commit, path, index, len(chunks),
                                           LANGUAGES.get(suffix, suffix.lstrip(".") or "text"),
                                           file_url(cfg.workspace, cfg.repository, commit, path)))

    if not paths:
        raise IndexingError("No readable source files were found in this branch.")

    tree = "\n".join(sorted(paths))
    overview = (f"Repository: {repo_name} (branch {branch}, commit {commit[:12]})\n"
                f"Repository overview: {len(paths)} indexed files. File tree:\n")
    tree_chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=0,
                                                 separators=["\n"]).split_text(tree)
    for index, chunk in enumerate(tree_chunks):
        documents.append(_document(overview + chunk, source, cfg, branch, commit,
                                   "(file tree)", index, len(tree_chunks), "text",
                                   file_url(cfg.workspace, cfg.repository, commit)))
    return documents, len(paths)


def build_commit_documents(commits, source, cfg, branch, url):
    """
    Recent commit history as chunks of COMMITS_PER_CHUNK, newest first. Each chunk
    says where it sits in the history so "last 5 commits" retrieves the first one.
    """
    repo_name = f"{cfg.workspace}/{cfg.repository}"
    documents = []
    groups = [commits[i:i + COMMITS_PER_CHUNK] for i in range(0, len(commits), COMMITS_PER_CHUNK)]
    for index, group in enumerate(groups):
        first = index * COMMITS_PER_CHUNK + 1
        title = ("Latest (most recent) commits" if index == 0
                 else f"Older commits, numbers {first} to {first + len(group) - 1} counting from the newest")
        lines = [f"Repository: {repo_name} (branch {branch})",
                 f"Commit history, newest first. {title}. Git log of recent commits:"]
        for n, c in enumerate(group, first):
            message = " ".join((c.get("message") or "").split())[:300]
            lines.append(f"{n}. {c['id'][:10]} | {c.get('date', '')} | {c.get('author', '')} | {message}")
        documents.append(_document("\n".join(lines), source, cfg, branch, commits[0]["id"],
                                   "(commit history)", index, len(groups), "git", url))
    return documents


def _document(text, source, cfg, branch, commit, path, index, total, language, url):
    repo_name = f"{cfg.workspace}/{cfg.repository}"
    return Document(page_content=text, metadata={
        "source_id": source.id,
        "source_type": "bitbucket",
        "source_name": source.name,
        "source_file": f"{repo_name}/{path}",
        "file_path": path,
        "repository": repo_name,
        "branch": branch,
        "commit": commit,
        "language": language,
        "chunk_index": index,
        "total_chunks": total,
        "chunk_id": f"{source.id}:{path}:{index}",
        "url": url,
    })


class BitbucketIndexer:
    def __init__(self, registry=None, credentials=None, vector_store=None, refresh=None, client_factory=None):
        self.registry = registry if registry is not None else source_registry
        self.credentials = credentials if credentials is not None else credential_store
        self._vector_store = vector_store
        self._refresh = refresh
        self.client_factory = client_factory or make_client

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

    def _client(self, cfg):
        try:
            username, token = self.credentials.retrieve(cfg.credential_id).split(":", 1)
        except (KeyError, ValueError) as exc:
            raise IndexingError("Saved Bitbucket credentials are missing. Delete and re-add the repository.") from exc
        return self.client_factory(username, token, server_url=cfg.server_url)

    def sync(self, source_id: str) -> int:
        """Re-index only if the branch has new commits; otherwise keep the existing chunks."""
        source = self.registry.get_source(source_id)
        cfg = self.registry.get_bitbucket_config(source_id)
        if not source or not cfg:
            raise IndexingError("Bitbucket source not found.")
        report_progress(source_id, "Checking for new commits")
        branch, commit = resolve_branch(self._client(cfg), cfg)
        # Repositories indexed before commit history existed get a full index once.
        if (commit == cfg.last_commit and branch == cfg.branch and source.chunk_count
                and self.vectors.has_commit_history(source_id)):
            logger.info("bitbucket_sync_up_to_date", source_id=source_id, commit=commit[:12])
            return source.chunk_count
        return self.index(source_id)

    def index(self, source_id: str) -> int:
        """Full index of the configured branch. Returns the number of chunks stored."""
        source = self.registry.get_source(source_id)
        cfg = self.registry.get_bitbucket_config(source_id)
        if not source or not cfg:
            raise IndexingError("Bitbucket source not found.")
        client = self._client(cfg)
        report_progress(source_id, "Checking branch")
        branch, commit = resolve_branch(client, cfg)
        logger.info("bitbucket_index_start", source_id=source_id, branch=branch, commit=commit[:12])

        mb = 1024 * 1024
        report_progress(source_id, "Downloading repository", 0, None, "MB")
        archive = client.download_archive(
            cfg.workspace, cfg.repository, commit,
            on_progress=lambda got, size: report_progress(
                source_id, "Downloading repository", round(got / mb, 1), round(size / mb, 1) if size else None, "MB"))

        report_progress(source_id, "Reading and splitting files")
        documents, file_count = build_documents(archive, source, cfg, branch, commit,
                                               getattr(client, "file_url", None))
        logger.info("bitbucket_files_read", source_id=source_id, files=file_count, chunks=len(documents))

        # Commit history is optional: a failure here must not lose the file index.
        report_progress(source_id, "Fetching commit history")
        try:
            commits = client.get_commits(cfg.workspace, cfg.repository, commit)
            if commits:
                url_of = getattr(client, "file_url", None) or BitbucketClient("", "").file_url
                documents += build_commit_documents(commits, source, cfg, branch,
                                                    url_of(cfg.workspace, cfg.repository, commit))
            logger.info("bitbucket_commits_read", source_id=source_id, commits=len(commits))
        except Exception as exc:
            logger.warning("bitbucket_commit_history_skipped", source_id=source_id, error=str(exc)[:200])

        # Embedding runs on the CPU and is the slow step for large repositories.
        report_progress(source_id, "Embedding chunks", 0, len(documents), "chunks")
        self.vectors.replace_source_documents(
            source_id, documents,
            on_progress=lambda done, total: report_progress(source_id, "Embedding chunks", done, total, "chunks"))
        report_progress(source_id, "Updating keyword search")
        self.refresh_search()
        self.registry.update_bitbucket_config(source_id, branch=branch, last_commit=commit, file_count=file_count)
        logger.info("bitbucket_index_done", source_id=source_id, files=file_count, chunks=len(documents))
        return len(documents)


bitbucket_indexer = BitbucketIndexer()
