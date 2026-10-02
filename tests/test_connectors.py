"""Bitbucket / Jira indexing, source-filtered retrieval, and legacy uploads (offline)."""

import io
import zipfile
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.retrieval.source_filter import SourceFilter
from src.sources.bitbucket_indexer import BitbucketIndexer, build_documents, is_indexable, resolve_branch
from src.sources.jira_indexer import JiraClient, JiraIndexer, adf_to_text, issue_to_text
from src.sources.jobs import IndexingError, IndexJobRunner
from src.sources.models import BitbucketSourceConfig, JiraSourceConfig, SourceStatus
from src.sources.registry import SourceRegistry
from tests.test_sources import TestEmbeddings


@pytest.fixture
def vectors(tmp_path):
    from langchain_community.vectorstores import Chroma
    with patch("src.ingestion.embedder.EmbeddingModel._load_model"):
        from src.retrieval.vector_store import VectorStore
    store = object.__new__(VectorStore)
    store._store = Chroma(collection_name="connector-tests", embedding_function=TestEmbeddings(),
                          persist_directory=str(tmp_path / "vectors"))
    return store


@pytest.fixture
def registry(tmp_path):
    return SourceRegistry(tmp_path / "sources.db")


class FakeCredentials:
    def retrieve(self, credential_id):
        return "me@example.com:secret-token"


def repo_zip(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for path, content in files.items():
            zf.writestr(f"ws-repo-abc123/{path}", content)
    return buffer.getvalue()


def add_repo(registry, branch="main", source_id="bb-1"):
    registry.create_bitbucket_source(
        BitbucketSourceConfig(source_id=source_id, workspace="ws", repository="repo",
                              branch=branch, credential_id="cred"), name="ws/repo")
    return source_id


class FakeBitbucket:
    def __init__(self, files, branches=None, default="main"):
        self.files, self.branches, self.default = files, branches or {"main": "c" * 40}, default
        self.downloads = 0

    def __call__(self, username, token):
        assert (username, token) == ("me@example.com", "secret-token")
        return self

    def get_branch_commit(self, workspace, repository, branch):
        return self.branches.get(branch)

    def get_repository(self, workspace, repository):
        return {"mainbranch": {"name": self.default}}

    def download_archive(self, workspace, repository, commit):
        self.downloads += 1
        return repo_zip(self.files)


# ---------------------------------------------------------------- filtering

def test_source_filter_matches_registered_and_legacy_chunks():
    f = SourceFilter.from_request(["bb-1"], ["tmp1.txt"])
    assert f.matches({"source_id": "bb-1"})
    assert not f.matches({"source_id": "doc-2"})
    assert f.matches({"source_file": "tmp1.txt"})
    # A registered chunk is never matched by file name alone.
    assert not f.matches({"source_id": "doc-2", "source_file": "tmp1.txt"})
    assert f.chroma_where() == {"$or": [{"source_id": {"$in": ["bb-1"]}},
                                        {"source_file": {"$in": ["tmp1.txt"]}}]}
    assert SourceFilter.from_request(None, None) is None
    assert SourceFilter.from_request([], []).is_empty


def test_vector_and_keyword_search_respect_selected_sources(vectors):
    from src.retrieval.hybrid_retriever import BM25Retriever
    docs = [Document(page_content="deploy pipeline uses cloud run", metadata={"source_id": "bb-1", "chunk_id": "a"}),
            Document(page_content="deploy checklist for releases", metadata={"source_id": "doc-1", "chunk_id": "b"}),
            Document(page_content="deploy notes from old upload", metadata={"source_file": "tmp1.txt", "chunk_id": "c"})]
    vectors._store.add_documents(docs, ids=["a", "b", "c"])
    only_repo = SourceFilter(["bb-1"], [])
    assert {d.metadata["chunk_id"] for d in vectors.similarity_search_with_filter("deploy", only_repo, k=5)} == {"a"}
    legacy = SourceFilter([], ["tmp1.txt"])
    assert {d.metadata["chunk_id"] for d in vectors.similarity_search_with_filter("deploy", legacy, k=5)} == {"c"}
    assert vectors.similarity_search_with_filter("deploy", SourceFilter(), k=5) == []

    bm25 = BM25Retriever(docs)
    assert [d.metadata["chunk_id"] for d, _ in bm25.search("deploy", source_filter=only_repo)] == ["a"]
    assert len(bm25.search("deploy")) == 3


def test_legacy_files_are_listed_and_removed_without_touching_registered_chunks(vectors):
    vectors._store.add_documents([
        Document(page_content="old text one", metadata={"source_file": "tmp1.txt", "chunk_index": 0}),
        Document(page_content="old text two", metadata={"source_file": "tmp1.txt", "chunk_index": 1}),
        Document(page_content="managed", metadata={"source_file": "tmp1.txt", "source_id": "doc-1"}),
    ], ids=["l1", "l2", "m1"])
    assert vectors.list_legacy_files() == [{"source_file": "tmp1.txt", "chunk_count": 2, "preview": "old text one"}]
    assert vectors.delete_legacy_file("tmp1.txt") == 2
    assert vectors._store._collection.get()["ids"] == ["m1"]
    assert vectors.list_legacy_files() == []


# ---------------------------------------------------------------- bitbucket

@pytest.mark.parametrize("path,expected", [
    ("src/app.py", True), ("Dockerfile", True), ("README.md", True), (".env.example", True),
    (".env", False), (".env.production", False), ("keys/server.pem", False), ("id_rsa", False),
    ("node_modules/x/index.js", False), ("package-lock.json", False), ("static/app.min.js", False),
    ("logo.png", False), ("dist/bundle.js", False),
])
def test_repository_file_selection(path, expected):
    assert is_indexable(path, 100) is expected


def test_build_documents_adds_paths_metadata_and_file_tree(registry):
    add_repo(registry)
    archive = repo_zip({"src/app.py": "def main():\n    return 'hi'\n", "README.md": "# Dhanvi\nSocial automation",
                        ".env": "SECRET=1", "img.png": "\x89PNG\x00\x00", "empty.txt": "   "})
    docs, file_count = build_documents(archive, registry.get_source("bb-1"),
                                       registry.get_bitbucket_config("bb-1"), "main", "c" * 40)
    assert file_count == 2
    paths = {d.metadata["file_path"] for d in docs}
    assert paths == {"src/app.py", "README.md", "(file tree)"}
    app = next(d for d in docs if d.metadata["file_path"] == "src/app.py")
    assert app.page_content.startswith("Repository: ws/repo (branch main)\nFile: src/app.py")
    assert app.metadata["source_type"] == "bitbucket" and app.metadata["source_id"] == "bb-1"
    assert app.metadata["source_file"] == "ws/repo/src/app.py"
    tree = next(d for d in docs if d.metadata["file_path"] == "(file tree)")
    assert "README.md\nsrc/app.py" in tree.page_content
    assert all(v is not None for d in docs for v in d.metadata.values())
    assert not any("SECRET" in d.page_content for d in docs)


def test_bitbucket_index_replaces_chunks_and_sync_skips_unchanged_commit(registry, vectors):
    add_repo(registry)
    fake = FakeBitbucket({"src/app.py": "print('v1')\n"})
    indexer = BitbucketIndexer(registry=registry, credentials=FakeCredentials(), vector_store=vectors,
                               refresh=Mock(), client_factory=fake)
    jobs = IndexJobRunner(registry)
    jobs.start("bb-1", indexer.index, background=False)
    source, cfg = registry.get_source("bb-1"), registry.get_bitbucket_config("bb-1")
    assert source.status == SourceStatus.READY and source.chunk_count == 2
    assert cfg.last_commit == "c" * 40 and cfg.file_count == 1

    jobs.start("bb-1", indexer.sync, background=False)
    assert fake.downloads == 1  # same commit: nothing re-downloaded

    fake.branches["main"] = "d" * 40
    fake.files = {"src/app.py": "print('v2')\n", "src/util.py": "x = 1\n"}
    jobs.start("bb-1", indexer.sync, background=False)
    texts = vectors._store._collection.get()["documents"]
    assert fake.downloads == 2 and len(texts) == 3
    assert not any("v1" in t for t in texts)


def test_missing_branch_falls_back_to_default_branch(registry):
    add_repo(registry, branch="develop")
    fake = FakeBitbucket({}, branches={"master": "e" * 40}, default="master")
    assert resolve_branch(fake, registry.get_bitbucket_config("bb-1")) == ("master", "e" * 40)


def test_failed_index_keeps_previous_chunks_and_reports_error(registry, vectors):
    add_repo(registry)
    fake = FakeBitbucket({"a.py": "a = 1\n"})
    indexer = BitbucketIndexer(registry=registry, credentials=FakeCredentials(), vector_store=vectors,
                               refresh=Mock(), client_factory=fake)
    jobs = IndexJobRunner(registry)
    jobs.start("bb-1", indexer.index, background=False)
    fake.files = {"logo.png": "binary"}
    jobs.start("bb-1", indexer.index, background=False)
    source = registry.get_source("bb-1")
    assert source.status == SourceStatus.ERROR
    assert "No readable source files" in source.error_message
    assert vectors._store._collection.count() == 2


def test_job_runner_rejects_concurrent_jobs_and_recovers_after_restart(registry):
    add_repo(registry)
    jobs = IndexJobRunner(registry)
    jobs._running.add("bb-1")
    assert jobs.start("bb-1", lambda _: 1, background=False) is False
    registry.update_source_status("bb-1", SourceStatus.INDEXING)
    IndexJobRunner(registry).recover_interrupted()
    assert registry.get_source("bb-1").status == SourceStatus.ERROR


# ---------------------------------------------------------------- jira

ISSUE = {
    "key": "BT-7",
    "fields": {
        "summary": "Login fails on Safari",
        "status": {"name": "In Progress"}, "issuetype": {"name": "Bug"}, "priority": {"name": "High"},
        "assignee": {"displayName": "Asha"}, "reporter": {"displayName": "Ravi"},
        "created": "2026-09-01T10:00:00.000+0000", "updated": "2026-09-03T12:00:00.000+0000",
        "labels": ["auth"],
        "description": {"type": "doc", "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "Cookie is "},
                                              {"type": "text", "text": "dropped"}]},
            {"type": "bulletList", "content": [{"type": "listItem", "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": "Safari 17"}]}]}]},
        ]},
        "comment": {"comments": [{"author": {"displayName": "Asha"}, "created": "2026-09-02T00:00:00.000+0000",
                                  "body": "Fixed SameSite flag"}]},
    },
}


def test_issue_text_flattens_rich_text_and_comments():
    assert adf_to_text(ISSUE["fields"]["description"]) == "Cookie is dropped\n- Safari 17\n"
    text = issue_to_text(ISSUE, "https://acme.atlassian.net")
    assert text.startswith("Jira issue BT-7: Login fails on Safari")
    assert "Status: In Progress" in text and "Assignee: Asha" in text
    assert "https://acme.atlassian.net/browse/BT-7" in text
    assert "- Asha (2026-09-02): Fixed SameSite flag" in text


def test_jira_index_stores_issue_chunks(registry, vectors):
    registry.create_jira_source(JiraSourceConfig(source_id="jira-1", base_url="https://acme.atlassian.net",
                                                 project_key="BT", credential_id="cred"), name="BT")
    client = Mock()
    client.search_issues.return_value = [ISSUE]
    indexer = JiraIndexer(registry=registry, credentials=FakeCredentials(), vector_store=vectors,
                          refresh=Mock(), client_factory=Mock(return_value=client))
    IndexJobRunner(registry).start("jira-1", indexer.index, background=False)
    assert registry.get_source("jira-1").status == SourceStatus.READY
    assert registry.get_jira_config("jira-1").issue_count == 1
    meta = vectors._store._collection.get()["metadatas"][0]
    assert meta["source_type"] == "jira" and meta["source_file"] == "BT-7"


def response(status, payload=None):
    r = Mock(status_code=status)
    r.json.return_value = payload or {}
    return r


def test_jira_client_paginates_and_falls_back_to_server_api():
    session = Mock()
    session.get.side_effect = [
        response(200, {"issues": [ISSUE], "nextPageToken": "p2"}),
        response(200, {"issues": [ISSUE], "isLast": True}),
    ]
    assert len(JiraClient("https://acme.atlassian.net", "e", "t", session).search_issues("BT")) == 2
    assert session.get.call_args.kwargs["params"]["nextPageToken"] == "p2"

    server = Mock()
    server.get.side_effect = [response(404), response(200, {"issues": [ISSUE], "total": 1})]
    assert len(JiraClient("https://jira.local", "e", "t", server).search_issues("BT")) == 1
    assert server.get.call_args.args[0] == "https://jira.local/rest/api/2/search"


def test_jira_auth_failure_is_a_readable_error():
    session = Mock()
    session.get.return_value = response(401)
    with pytest.raises(IndexingError, match="rejected the saved credentials"):
        JiraClient("https://acme.atlassian.net", "e", "t", session).search_issues("BT")


# ---------------------------------------------------------------- API

@pytest.fixture
def api(registry, vectors, monkeypatch):
    from src.api.main import app
    import src.api.routes.sources.bitbucket as bb_routes
    import src.api.routes.sources.documents as doc_routes
    from src.sources.documents import DocumentSourceService
    monkeypatch.setattr(bb_routes, "source_registry", registry)
    monkeypatch.setattr(bb_routes, "index_jobs", IndexJobRunner(registry))
    monkeypatch.setattr(doc_routes, "document_service",
                        DocumentSourceService(registry=registry, vector_store=vectors, refresh=Mock()))
    with TestClient(app) as client:
        yield client


def test_index_endpoint_starts_job_and_rejects_unknown_source(api, registry, monkeypatch):
    import src.sources.bitbucket_indexer as module
    add_repo(registry)
    started = []
    monkeypatch.setattr(module.bitbucket_indexer, "index", lambda sid: started.append(sid) or 3)
    r = api.post("/api/v1/sources/bitbucket/bb-1/index")
    assert r.status_code == 200 and r.json()["status"] == "started"
    assert api.post("/api/v1/sources/bitbucket/nope/index").status_code == 404


def test_legacy_document_endpoints(api, vectors):
    vectors._store.add_documents([Document(page_content="old", metadata={"source_file": "tmp1.txt"})], ids=["l1"])
    assert api.get("/api/v1/sources/documents/legacy").json()[0]["source_file"] == "tmp1.txt"
    assert api.delete("/api/v1/sources/documents/legacy", params={"source_file": "tmp1.txt"}).json()["chunks_deleted"] == 1
    assert api.delete("/api/v1/sources/documents/legacy", params={"source_file": "tmp1.txt"}).status_code == 404


def test_query_forwards_selected_sources_to_pipeline(api, monkeypatch):
    from src.graph import pipeline
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"answer": "ok", "sources": [], "true_data_chunks": [], "noisy_data_chunks": [],
                "eval_metrics": {}, "pipeline_steps": [], "query_intent": "factual", "rewritten_query": "q",
                "hallucination_score": 0.0, "model_used": "m", "latency_ms": 1,
                "input_safe": True, "output_safe": True}
    monkeypatch.setattr(pipeline.rag_pipeline, "run", fake_run)
    api.post("/api/v1/query", json={"question": "what is in the repo?", "source_ids": ["bb-1"]})
    assert captured["source_filter"] == SourceFilter(["bb-1"], [])
    api.post("/api/v1/query", json={"question": "what is in the repo?"})
    assert captured["source_filter"] is None


def test_multi_question_messages_are_split_for_retrieval():
    from src.retrieval.hybrid_retriever import split_questions
    message = ("Explain the architecture of Dhanvi.\nHow does WhatsApp ingestion work? "
               "Which files implement Supabase integration?")
    assert split_questions(message) == ["Explain the architecture of Dhanvi.",
                                        "How does WhatsApp ingestion work?",
                                        "Which files implement Supabase integration?"]
    assert split_questions("What is RAG?") == ["What is RAG?"]


def test_keyword_tokenizer_splits_paths_and_camel_case():
    from src.retrieval.hybrid_retriever import tokenize
    assert tokenize("File: lib/instagram/publish.ts publishToInstagram()") == [
        "file", "lib", "instagram", "publish", "ts", "publish", "to", "instagram"]


def test_keyword_index_is_built_on_first_search_after_restart(monkeypatch):
    with patch("src.retrieval.hybrid_retriever.SemanticReRanker._load_model"):
        from src.retrieval.hybrid_retriever import HybridRetriever
    import src.ingestion.ingestion_pipeline as pipeline_module
    retriever = object.__new__(HybridRetriever)
    retriever._bm25_ready = False
    refresh = Mock()
    monkeypatch.setattr(pipeline_module.ingestion_pipeline, "_refresh_bm25_index", refresh)
    retriever._ensure_bm25()
    retriever._ensure_bm25()
    refresh.assert_called_once()
