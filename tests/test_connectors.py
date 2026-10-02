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


def repo_zip(files, prefix="ws-repo-abc123/"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for path, content in files.items():
            zf.writestr(f"{prefix}{path}", content)
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

    def __call__(self, username, token, server_url=None):
        assert (username, token) == ("me@example.com", "secret-token")
        self.server_url = server_url
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


# ---------------------------------------------------------------- no-context answers

def test_generation_skips_llm_and_lists_sources_when_nothing_is_retrieved(monkeypatch):
    from src.graph import nodes
    monkeypatch.setattr(nodes, "knowledge_catalog", lambda f=None: ["- Bitbucket repository: ws/repo (10 chunks)"])
    llm = Mock()
    monkeypatch.setattr("src.gateway.llm_gateway.llm_gateway.complete", llm)
    out = nodes.generation_node({"original_query": "how can you help me", "true_data_chunks": [],
                                 "pipeline_steps": []})
    llm.assert_not_called()
    assert "ws/repo" in out["llm_response"] and "Sources I can search" in out["llm_response"]
    off_topic = nodes.no_context_answer(["- Bitbucket repository: ws/repo (10 chunks)"], "capital of France?")
    assert "won't guess" in off_topic
    assert "Sources I can search" in nodes.no_context_answer(["- x"], "whats knowledge you have")
    empty = nodes.no_context_answer([])
    assert "nothing to search yet" in empty


def test_knowledge_catalog_respects_selected_sources(registry, vectors, monkeypatch):
    from src.graph.nodes import knowledge_catalog
    import src.sources.registry as registry_module
    import src.retrieval.vector_store as vector_module
    add_repo(registry)
    registry.update_source_status("bb-1", SourceStatus.READY, chunk_count=5)
    add_repo(registry, source_id="bb-2")  # never indexed: not listed
    monkeypatch.setattr(registry_module, "source_registry", registry)
    monkeypatch.setattr(vector_module, "vector_store", vectors)
    assert knowledge_catalog() == ["- Bitbucket repository: ws/repo (5 chunks)"]
    assert knowledge_catalog(SourceFilter([], [])) == []


def test_stream_answers_without_llm_when_nothing_is_retrieved(api, monkeypatch):
    import json as _json
    from src.retrieval.hybrid_retriever import hybrid_retriever
    from src.gateway.llm_gateway import llm_gateway
    from src.graph import nodes
    monkeypatch.setattr(hybrid_retriever, "retrieve", lambda *a, **k: ([], []))
    monkeypatch.setattr(nodes, "knowledge_catalog", lambda f=None: [])
    called = []

    async def fake_stream(*a, **k):
        called.append(True)
        yield "invented"
    monkeypatch.setattr(llm_gateway, "astream", fake_stream)
    body = api.post("/api/v1/query/stream", json={"question": "what knowledge do you have"}).text
    events = [_json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    assert not called
    assert "nothing to search yet" in events[0]["token"]
    assert events[-1]["sources"] == []


# ---------------------------------------------------------------- bitbucket server / data center

from src.sources.bitbucket_indexer import BitbucketServerClient, make_client, parse_server_url

SERVER = "https://bitbucket.example.com"


def test_parse_server_url_accepts_base_or_repository_page():
    assert parse_server_url("https://bitbucket.example.com/projects/WSQAAUTO/repos/my_repo/browse") == (
        SERVER, "WSQAAUTO", "my_repo")
    assert parse_server_url("https://bitbucket.example.com/") == (SERVER, None, None)
    assert parse_server_url("https://host/bitbucket/projects/ABC") == ("https://host/bitbucket", "ABC", None)
    with pytest.raises(ValueError):
        parse_server_url("bitbucket.example.com")
    assert isinstance(make_client("", "t", server_url=SERVER), BitbucketServerClient)


class FakeServerSession:
    """Answers Bitbucket Server REST 1.0 calls for project WSQ, repository app."""

    def __init__(self, archive=b"", token="good"):
        self.archive, self.token, self.calls = archive, token, []

    def get(self, url, headers=None, auth=None, params=None, timeout=None, stream=False):
        self.calls.append((url, params, headers, auth))
        bearer = (headers or {}).get("Authorization") == f"Bearer {self.token}"
        if not bearer:
            return response(401)
        api = f"{SERVER}/rest/api/1.0/projects/WSQ"
        routes = {
            f"{api}/repos": {"size": 3, "values": []},
            f"{api}/repos/app": {"slug": "app"},
            f"{api}/repos/app/default-branch": {"displayId": "master"},
            f"{api}/repos/app/branches": {"values": [
                {"id": "refs/heads/master", "displayId": "master", "latestCommit": "f" * 40},
                {"id": "refs/heads/main-old", "displayId": "main-old", "latestCommit": "0" * 40}]},
        }
        if url == f"{api}/repos/app/archive":
            r = response(200)
            r.iter_content = lambda size: [self.archive]
            return r
        return response(200, routes[url]) if url in routes else response(404)


def test_server_client_tests_connection_and_resolves_default_branch(registry):
    session = FakeServerSession()
    client = BitbucketServerClient("", "good", SERVER + "/", session)
    assert client.test_connection("WSQ", "app") == 3
    assert session.calls[0][2]["Authorization"] == "Bearer good"
    assert client.get_branch_commit("WSQ", "app", "main") is None  # filterText match must be exact
    add_repo(registry)
    registry.update_bitbucket_config("bb-1", workspace="WSQ", repository="app")
    assert resolve_branch(client, registry.get_bitbucket_config("bb-1")) == ("master", "f" * 40)


def test_server_client_reports_readable_errors():
    with pytest.raises(IndexingError, match="HTTP access token"):
        BitbucketServerClient("", "bad", SERVER, FakeServerSession()).test_connection("WSQ")
    with pytest.raises(IndexingError, match="project KEY"):
        BitbucketServerClient("", "good", SERVER, FakeServerSession()).test_connection("NOPE")


def test_server_repository_is_indexed_with_server_links(registry, vectors):
    archive = repo_zip({"src/main.py": "def run():\n    return 1\n"}, prefix="app/")
    session = FakeServerSession(archive=archive)
    registry.create_bitbucket_source(
        BitbucketSourceConfig(source_id="bb-s", workspace="WSQ", repository="app", branch="master",
                              credential_id="cred", server_url=SERVER), name="WSQ/app")
    indexer = BitbucketIndexer(registry=registry, credentials=FakeCredentials(), vector_store=vectors,
                               refresh=Mock(),
                               client_factory=lambda u, t, server_url=None: BitbucketServerClient(
                                   u, t.replace("secret-token", "good"), server_url, session))
    IndexJobRunner(registry).start("bb-s", indexer.index, background=False)
    source = registry.get_source("bb-s")
    assert source.status == SourceStatus.READY, source.error_message
    archive_call = next(c for c in session.calls if c[0].endswith("/archive"))
    assert archive_call[1] == {"at": "f" * 40, "format": "zip", "prefix": "app/"}
    meta = next(m for m in vectors._store._collection.get()["metadatas"] if m["file_path"] == "src/main.py")
    assert meta["source_file"] == "WSQ/app/src/main.py"
    assert meta["url"] == f"{SERVER}/projects/WSQ/repos/app/browse/src/main.py?at={'f' * 40}"
    assert registry.get_bitbucket_config("bb-s").server_url == SERVER


def test_add_server_repository_from_pasted_url(api, registry, monkeypatch):
    import src.api.routes.sources.bitbucket as bb_routes
    monkeypatch.setattr(bb_routes.credential_store, "store", lambda **kw: "cred-x")
    r = api.post("/api/v1/sources/bitbucket", json={
        "server_url": f"{SERVER}/projects/WSQAAUTO/repos/wireless_centralized_server_gen2/browse",
        "workspace": "", "repository": "", "token": "t"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["server_url"], body["workspace"], body["repository"]) == (
        SERVER, "WSQAAUTO", "wireless_centralized_server_gen2")
    assert api.post("/api/v1/sources/bitbucket", json={
        "server_url": "http://insecure", "workspace": "A", "repository": "b", "token": "t"}).status_code == 422


def test_no_context_answer_points_out_unindexed_sources(registry, monkeypatch):
    from src.graph import nodes
    import src.sources.registry as registry_module
    add_repo(registry)  # added, never indexed
    monkeypatch.setattr(registry_module, "source_registry", registry)
    assert nodes.unindexed_sources() == ["ws/repo"]
    assert nodes.unindexed_sources(SourceFilter(["other"], [])) == []
    answer = nodes.no_context_answer(["- Document: a.pdf (6 chunks)"], "what does this repo do?", ["ws/repo"])
    assert "Not searchable yet:** ws/repo" in answer and "click **Index**" in answer
