"""Index every current page of a Confluence space (Cloud or Server / Data Center).

Pages are fetched with their storage-format HTML, converted to readable text
(headings, lists, tables and code kept legible), split into chunks that start
with the page title and its place in the page tree, and atomically replace the
space's previous chunks. Unchanged pages reuse their stored vectors.
"""

import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urlparse, unquote

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.sources.atlassian import AtlassianSession, is_cloud
from src.sources.credentials import credential_store
from src.sources.jobs import IndexingError, report_progress
from src.sources.registry import source_registry
from src.observability.logger import get_logger

logger = get_logger(__name__)

MAX_PAGES = 5000
PAGE_SIZE = 50
SPACE_KEY = re.compile(r"^~?[A-Za-z0-9_]+$")

# Sized to fit the embedding model's 256-token window (see bitbucket_indexer).
_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)


def parse_confluence_url(url: str):
    """
    Accept a Confluence base URL or a space / page link and return (base_url, space_key or None):
      https://confluence.company.com/display/WSQ/Home          -> (https://confluence.company.com, WSQ)
      https://confluence.company.com/spaces/WSQ/pages/123/X    -> (https://confluence.company.com, WSQ)
      https://acme.atlassian.net/wiki/spaces/ENG/overview      -> (https://acme.atlassian.net/wiki, ENG)
    """
    url = (url or "").strip()
    if not re.match(r"^https://[^/\s]+", url, re.I):
        raise ValueError("Confluence URL must start with https://")
    parsed = urlparse(url)
    path = parsed.path
    space = None
    match = re.search(r"/(?:display|spaces)/([^/?#]+)", path)
    if match:
        space = unquote(match.group(1))
        path = path[:match.start()]
    else:
        params = dict(p.split("=", 1) for p in parsed.query.split("&") if "=" in p)
        space = params.get("spaceKey")
        path = re.split(r"/(?:pages|rest|plugins)/", path)[0]
    base = f"{parsed.scheme}://{parsed.netloc}{path}".rstrip("/")
    if is_cloud(base) and not base.endswith("/wiki"):
        base += "/wiki"   # Confluence Cloud lives under /wiki
    return base, space


class _HtmlToText(HTMLParser):
    """Confluence storage-format XHTML to plain text with readable structure."""

    BLOCK = {"p", "div", "br", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote", "ul", "ol", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("style", "script") or (tag == "ac:parameter"):
            self.skip += 1
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in ("td", "th"):
            self.parts.append(" | ")
        elif tag == "pre" or (tag == "ac:plain-text-body"):
            self.parts.append("\n```\n")
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("style", "script") or (tag == "ac:parameter"):
            self.skip = max(0, self.skip - 1)
        elif tag == "pre" or tag == "ac:plain-text-body":
            self.parts.append("\n```\n")
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

    def text(self):
        text = unescape("".join(self.parts))
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


def html_to_text(html: str) -> str:
    parser = _HtmlToText()
    # CDATA (code macros) is not reported as data by HTMLParser; unwrap it first.
    parser.feed(re.sub(r"<!\[CDATA\[(.*?)\]\]>", lambda m: m.group(1).replace("<", "&lt;"), html or "", flags=re.S))
    parser.close()
    return parser.text()


class ConfluenceClient:
    def __init__(self, base_url: str, username: str, token: str, session=None):
        self.base_url = base_url.rstrip("/")
        self.api = AtlassianSession(self.base_url, username, token, "Confluence", session)

    def test_connection(self, space_key: str = None) -> dict:
        response = self.api.get("/rest/api/space", {"limit": 1})
        if response.status_code == 404:
            raise IndexingError(f"No Confluence REST API at {self.base_url}. Check the URL "
                                "(Confluence Cloud URLs end with /wiki).")
        if response.status_code != 200:
            raise IndexingError(f"Confluence returned HTTP {response.status_code} for the connection test.")
        if space_key:
            space = self.api.get(f"/rest/api/space/{space_key}")
            if space.status_code == 404:
                raise IndexingError(f"Space '{space_key}' was not found or is not visible to this account.")
        return {"spaces_found": response.json().get("size")}

    def list_spaces(self, limit: int = 1000) -> list:
        spaces, start = [], 0
        while len(spaces) < limit:
            response = self.api.get("/rest/api/space", {"limit": 100, "start": start})
            if response.status_code != 200:
                raise IndexingError(f"Listing Confluence spaces failed (HTTP {response.status_code}).")
            data = response.json()
            batch = data.get("results", [])
            spaces += [{"key": s["key"], "name": s.get("name") or s["key"], "type": s.get("type", "")}
                       for s in batch]
            if len(batch) < 100 or not (data.get("_links") or {}).get("next"):
                break
            start += len(batch)
        return spaces[:limit]

    def fetch_pages(self, space_key: str, on_page=None) -> list:
        pages, start = [], 0
        while len(pages) < MAX_PAGES:
            response = self.api.get("/rest/api/content", {
                "spaceKey": space_key, "type": "page", "status": "current", "limit": PAGE_SIZE,
                "start": start, "expand": "body.storage,version,ancestors"})
            if response.status_code == 404:
                raise IndexingError(f"Space '{space_key}' was not found or is not visible to this account.")
            if response.status_code != 200:
                raise IndexingError(f"Fetching Confluence pages failed (HTTP {response.status_code}).")
            data = response.json()
            batch = data.get("results", [])
            pages += batch
            if on_page:
                on_page(len(pages))
            if len(batch) < PAGE_SIZE or not (data.get("_links") or {}).get("next"):
                break
            start += len(batch)
        return pages[:MAX_PAGES]


def build_page_documents(pages: list, source, space_key: str, base_url: str) -> list:
    documents = []
    for page in pages:
        text = html_to_text(((page.get("body") or {}).get("storage") or {}).get("value", ""))
        if not text:
            continue
        title = page.get("title") or f"Page {page.get('id')}"
        ancestors = [a.get("title", "") for a in page.get("ancestors") or []]
        path = " > ".join(ancestors + [title])
        version = page.get("version") or {}
        webui = (page.get("_links") or {}).get("webui", "")
        url = f"{base_url}{webui}" if webui else base_url
        header = (f"Confluence page: {title} (space {space_key})\nPath: {path}\n"
                  f"Last updated: {(version.get('when') or '')[:10]} by "
                  f"{(version.get('by') or {}).get('displayName', '')}\n\n")
        chunks = _splitter.split_text(text)
        for index, chunk in enumerate(chunks):
            documents.append(Document(page_content=header + chunk, metadata={
                "source_id": source.id, "source_type": "confluence", "source_name": source.name,
                "source_file": f"{space_key}/{title}", "page_id": str(page.get("id", "")), "page_title": title,
                "space_key": space_key, "depth": len(ancestors), "url": url, "chunk_index": index,
                "total_chunks": len(chunks), "chunk_id": f"{source.id}:{page.get('id')}:{index}",
            }))
    return documents


class ConfluenceIndexer:
    def __init__(self, registry=None, credentials=None, vector_store=None, refresh=None, client_factory=None):
        self.registry = registry if registry is not None else source_registry
        self.credentials = credentials if credentials is not None else credential_store
        self._vector_store = vector_store
        self._refresh = refresh
        self.client_factory = client_factory or ConfluenceClient

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
        """Index every current page of the space. Returns the number of chunks stored."""
        source = self.registry.get_source(source_id)
        cfg = self.registry.get_confluence_config(source_id)
        if not source or not cfg:
            raise IndexingError("Confluence source not found.")
        try:
            username, token = self.credentials.retrieve(cfg.credential_id).split(":", 1)
        except (KeyError, ValueError) as exc:
            raise IndexingError("Saved Confluence credentials are missing. Delete and re-add the space.") from exc

        client = self.client_factory(cfg.base_url, username, token)
        report_progress(source_id, "Fetching pages", 0, None, "pages")
        pages = client.fetch_pages(cfg.space_key,
                                   on_page=lambda n: report_progress(source_id, "Fetching pages", n, None, "pages"))
        report_progress(source_id, "Converting pages")
        documents = build_page_documents(pages, source, cfg.space_key, cfg.base_url)
        if not documents:
            raise IndexingError(f"No readable pages found in space {cfg.space_key}.")

        report_progress(source_id, "Embedding chunks", 0, len(documents), "chunks")
        self.vectors.replace_source_documents(
            source_id, documents,
            on_progress=lambda done, total: report_progress(source_id, "Embedding chunks", done, total, "chunks"))
        report_progress(source_id, "Updating keyword search")
        self.refresh_search()
        self.registry.update_confluence_config(source_id, page_count=len(pages))
        logger.info("confluence_index_done", source_id=source_id, pages=len(pages), chunks=len(documents))
        return len(documents)


confluence_indexer = ConfluenceIndexer()
