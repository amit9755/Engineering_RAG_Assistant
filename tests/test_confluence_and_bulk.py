"""Confluence spaces, Bitbucket bulk add / browse, and the indexing queue (offline)."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from src.sources.confluence_indexer import (ConfluenceClient, ConfluenceIndexer, html_to_text,
                                            parse_confluence_url)
from src.sources.jobs import IndexingError, IndexJobRunner
from src.sources.models import ConfluenceSourceConfig, SourceStatus
from src.sources.registry import SourceRegistry
from tests.test_connectors import (FakeCredentials, FakeServerSession, SERVER, registry, response,  # noqa: F401
                                   vectors)


@pytest.mark.parametrize("url,expected", [
    ("https://confluence.example.com/display/WSQ/Home+Page", ("https://confluence.example.com", "WSQ")),
    ("https://confluence.example.com/spaces/WSQ/pages/123/Title", ("https://confluence.example.com", "WSQ")),
    ("https://confluence.example.com/pages/viewpage.action?pageId=1&spaceKey=ABC",
     ("https://confluence.example.com", "ABC")),
    ("https://host.example.com/confluence/display/K/X", ("https://host.example.com/confluence", "K")),
    ("https://acme.atlassian.net/wiki/spaces/ENG/overview", ("https://acme.atlassian.net/wiki", "ENG")),
    ("https://acme.atlassian.net", ("https://acme.atlassian.net/wiki", None)),
    ("https://confluence.example.com/", ("https://confluence.example.com", None)),
])
def test_parse_confluence_url(url, expected):
    assert parse_confluence_url(url) == expected


def test_html_to_text_keeps_structure():
    html = ("<h2>Setup</h2><p>Run the <strong>server</strong>.</p><ul><li>one</li><li>two</li></ul>"
            "<table><tr><th>Key</th><th>Value</th></tr><tr><td>port</td><td>8080</td></tr></table>"
            "<ac:structured-macro ac:name=\"code\"><ac:parameter ac:name=\"language\">bash</ac:parameter>"
            "<ac:plain-text-body><![CDATA[npm start && echo <done>]]></ac:plain-text-body></ac:structured-macro>")
    text = html_to_text(html)
    assert "## Setup" in text and "Run the server." in text
    assert "- one\n- two" in text
    assert "| Key | Value" in text and "| port | 8080" in text
    assert "npm start && echo <done>" in text and "bash" not in text


def page(i, title, depth=0, body="<p>Body text</p>"):
    return {"id": str(i), "title": title, "body": {"storage": {"value": body}},
            "ancestors": [{"title": "Home"}] * depth, "version": {"when": "2026-09-01T00:00:00", "by": {
                "displayName": "Amit"}}, "_links": {"webui": f"/display/WSQ/{title}"}}


def test_confluence_server_client_pages_with_bearer_token():
    session = Mock()
    first = [page(i, f"P{i}") for i in range(50)]
    session.get.side_effect = [response(200, {"results": first, "_links": {"next": "/x"}}),
                               response(200, {"results": [page(50, "Last")], "_links": {}})]
    pages = ConfluenceClient("https://confluence.example.com", "", "pat", session).fetch_pages("WSQ")
    assert len(pages) == 51
    call = session.get.call_args_list[1]
    assert call.kwargs["headers"]["Authorization"] == "Bearer pat"
    assert call.kwargs["params"]["start"] == 50 and call.kwargs["params"]["spaceKey"] == "WSQ"
    rejected = Mock()
    rejected.get.return_value = response(401)
    with pytest.raises(IndexingError, match="Personal Access Token"):
        ConfluenceClient("https://confluence.example.com", "", "bad", rejected).test_connection()


def test_confluence_space_is_indexed(registry, vectors):  # noqa: F811
    registry.create_confluence_source(ConfluenceSourceConfig(
        source_id="conf-1", base_url="https://confluence.example.com", space_key="WSQ", credential_id="c"),
        name="Confluence WSQ")
    client = Mock()
    client.fetch_pages.return_value = [page(1, "Home"), page(2, "Setup guide", 1, "<p>Install with npm</p>"),
                                       page(3, "Empty", 1, "")]
    indexer = ConfluenceIndexer(registry=registry, credentials=FakeCredentials(), vector_store=vectors,
                                refresh=Mock(), client_factory=Mock(return_value=client))
    IndexJobRunner(registry).start("conf-1", indexer.index, background=False)
    assert registry.get_source("conf-1").status == SourceStatus.READY
    assert registry.get_confluence_config("conf-1").page_count == 3
    data = vectors._store._collection.get(include=["documents", "metadatas"])
    setup = next(d for d, m in zip(data["documents"], data["metadatas"]) if m["page_title"] == "Setup guide")
    assert setup.startswith("Confluence page: Setup guide (space WSQ)\nPath: Home > Setup guide")
    meta = next(m for m in data["metadatas"] if m["page_title"] == "Setup guide")
    assert meta["source_type"] == "confluence" and meta["source_file"] == "WSQ/Setup guide"
    assert meta["url"] == "https://confluence.example.com/display/WSQ/Setup guide"
    overview = vectors.get_overview_chunks()
    assert overview and overview[0].metadata["page_title"] == "Home"


def test_job_queue_limits_parallel_jobs(registry):  # noqa: F811
    import threading
    import time
    from tests.test_connectors import add_repo
    for i in range(4):
        add_repo(registry, source_id=f"bb-{i}")
    jobs = IndexJobRunner(registry)
    jobs._slots = threading.Semaphore(2)
    running, peak, lock = [0], [0], threading.Lock()

    def job(_):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.2)
        with lock:
            running[0] -= 1
        return 1
    for i in range(4):
        jobs.start(f"bb-{i}", job)
    assert "Queued" in "".join(p["stage"] for p in jobs.progress().values())
    for _ in range(50):
        if not jobs.progress():
            break
        time.sleep(0.05)
    assert peak[0] == 2 and all(registry.get_source(f"bb-{i}").status == SourceStatus.READY for i in range(4))


@pytest.fixture
def api(registry, monkeypatch):  # noqa: F811
    from src.api.main import app
    import src.api.routes.sources.bitbucket as bb_routes
    import src.api.routes.sources.confluence as conf_routes
    jobs = IndexJobRunner(registry)
    monkeypatch.setattr(jobs, "start", Mock(return_value=True))
    for module in (bb_routes, conf_routes):
        monkeypatch.setattr(module, "source_registry", registry)
        monkeypatch.setattr(module, "index_jobs", jobs)
        monkeypatch.setattr(module.credential_store, "store", lambda **kw: "cred")
    with TestClient(app) as client:
        client.jobs = jobs
        yield client


def test_bitbucket_browse_and_bulk_add(api, registry, monkeypatch):  # noqa: F811
    import src.sources.bitbucket_indexer as bb
    session = FakeServerSession()
    session_routes = {f"{SERVER}/rest/api/1.0/repos": {"isLastPage": True, "values": [
        {"slug": "app", "name": "App", "project": {"key": "WSQ"}},
        {"slug": "tools", "name": "Tools", "project": {"key": "OTHER"}}]},
        f"{SERVER}/rest/api/1.0/projects/WSQ/repos": {"isLastPage": True, "values": [
            {"slug": "app", "name": "App", "project": {"key": "WSQ"}}]}}
    real_get = session.get

    def get(url, **kw):
        if url in session_routes:
            return response(200, session_routes[url])
        return real_get(url, **kw)
    monkeypatch.setattr(bb, "make_client", lambda u, t, server_url=None: bb.BitbucketServerClient(
        u, "good", server_url, Mock(get=get)))
    found = api.post("/api/v1/sources/bitbucket/discover", json={"server_url": SERVER, "token": "t"}).json()
    assert [(r["workspace"], r["slug"]) for r in found["repositories"]] == [("WSQ", "app"), ("OTHER", "tools")]
    assert api.post("/api/v1/sources/bitbucket/discover", json={
        "server_url": SERVER, "workspace": "WSQ", "token": "t"}).json()["repositories"][0]["slug"] == "app"

    body = {"server_url": SERVER, "token": "t", "repositories": [
        {"workspace": "WSQ", "slug": "app"}, {"workspace": "OTHER", "slug": "tools"}]}
    assert api.post("/api/v1/sources/bitbucket/bulk", json=body).json()["added"] == 2
    again = api.post("/api/v1/sources/bitbucket/bulk", json=body).json()
    assert again["added"] == 0 and again["skipped"] == ["WSQ/app", "OTHER/tools"]
    assert api.jobs.start.call_count == 2
    names = sorted(s.name for s in registry.list_sources())
    assert names == ["OTHER/tools", "WSQ/app"]


def test_confluence_add_from_link_and_bulk(api, registry, monkeypatch):  # noqa: F811
    r = api.post("/api/v1/sources/confluence", json={
        "base_url": "https://confluence.example.com/display/WSQ/Home", "token": "t"})
    assert r.status_code == 200, r.text
    assert (r.json()["base_url"], r.json()["space_key"]) == ("https://confluence.example.com", "WSQ")
    assert api.post("/api/v1/sources/confluence", json={
        "base_url": "https://confluence.example.com/display/WSQ/Home", "token": "t"}).status_code == 409
    bulk = api.post("/api/v1/sources/confluence/bulk", json={
        "base_url": "https://confluence.example.com", "token": "t", "space_keys": ["WSQ", "ENG", "QA"]}).json()
    assert bulk["added"] == 2 and bulk["skipped"] == ["WSQ"]
    assert len(api.get("/api/v1/sources/confluence").json()) == 3
    assert api.post("/api/v1/sources/confluence", json={
        "base_url": "https://x.example.com", "space_key": "bad key!", "token": "t"}).status_code == 422


def test_unexpected_bitbucket_status_is_a_readable_error():
    from src.sources.bitbucket_indexer import BitbucketClient
    session = Mock()
    session.get.return_value = response(400, {"error": {"message": "Bad request"}})
    with pytest.raises(IndexingError, match="HTTP 400: Bad request"):
        BitbucketClient("x", "bad", session).list_repositories("ws")


def html_response(status, title, url="https://jira.example.com/rest/api/2/search"):
    r = Mock(status_code=status, url=url, text=f"<html><head><title>{title}</title></head><body>x</body></html>",
             headers={"Content-Type": "text/html;charset=UTF-8"})
    r.json.side_effect = ValueError("Expecting value")
    return r


def test_jira_server_searches_with_api_v2_directly():
    from src.sources.jira_indexer import JiraClient
    from tests.test_connectors import ISSUE
    session = Mock()
    session.get.return_value = response(200, {"issues": [ISSUE], "total": 1})
    pages = []
    issues = JiraClient("https://jira.sw.example.com", "", "pat", session).search_issues("WSQ", on_page=pages.append)
    assert len(issues) == 1 and pages == [1]
    assert session.get.call_args_list[0].args[0] == "https://jira.sw.example.com/rest/api/2/search"


def test_web_page_instead_of_data_is_a_readable_error():
    from src.sources.jira_indexer import JiraClient
    session = Mock()
    session.get.return_value = html_response(200, "Log in - NXP SSO", "https://sso.example.com/login")
    with pytest.raises(IndexingError, match="web page instead of data.*Log in - NXP SSO.*login page"):
        JiraClient("https://jira.sw.example.com", "", "pat", session).search_issues("WSQ")
    other = Mock()
    other.get.return_value = html_response(200, "Dashboard")
    with pytest.raises(IndexingError, match="Dashboard.*base address"):
        ConfluenceClient("https://confluence.example.com", "", "pat", other).fetch_pages("WSQ")
