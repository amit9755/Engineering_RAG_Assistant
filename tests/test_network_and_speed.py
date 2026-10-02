"""ALLOW_EXTERNAL_NETWORK policy and embedding reuse on re-index (offline)."""

from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from src import network_policy
from src.config import settings
from src.sources.jobs import IndexingError
from tests.test_connectors import vectors  # noqa: F401  (fixture)


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(settings, "allow_external_network", False)
    monkeypatch.setattr(settings, "internal_domains", "nxp.com")


@pytest.mark.parametrize("host,internal", [
    ("bitbucket.sw.nxp.com", True), ("nxp.com", True), ("localhost", True), ("127.0.0.1", True),
    ("10.12.0.5", True), ("192.168.1.20", True), ("intranet", True),
    ("bitbucket.org", False), ("api.bitbucket.org", False), ("huggingface.co", False),
    ("raw.githubusercontent.com", False), ("evilnxp.com", False), ("nxp.com.attacker.io", False),
    ("8.8.8.8", False),
])
def test_internal_hosts(offline, host, internal):
    assert network_policy.is_internal_host(host) is internal


def test_check_url_blocks_only_when_external_disabled(offline, monkeypatch):
    network_policy.check_url("https://bitbucket.sw.nxp.com/rest/api/1.0/projects")
    with pytest.raises(network_policy.ExternalNetworkBlocked, match="api.bitbucket.org is outside"):
        network_policy.check_url("https://api.bitbucket.org/2.0/repositories/x")
    monkeypatch.setattr(settings, "allow_external_network", True)
    network_policy.check_url("https://api.bitbucket.org/2.0/repositories/x")


def test_bitbucket_cloud_and_jira_cloud_are_blocked_offline(offline):
    from src.sources.bitbucket_indexer import BitbucketClient, BitbucketServerClient
    from src.sources.jira_indexer import JiraClient
    session = Mock()
    with pytest.raises(IndexingError, match="outside the company network"):
        BitbucketClient("u", "t", session).test_connection("ws")
    with pytest.raises(IndexingError, match="outside the company network"):
        JiraClient("https://acme.atlassian.net", "e", "t", session).search_issues("BT")
    session.get.assert_not_called()
    # an internal Bitbucket Server is still reachable
    session.get.return_value = Mock(status_code=200, json=lambda: {"size": 1})
    assert BitbucketServerClient("", "t", "https://bitbucket.sw.nxp.com", session).test_connection("WSQ") == 1


def test_llm_gateway_uses_only_local_model_offline(offline, monkeypatch):
    from src.gateway.llm_gateway import llm_gateway
    monkeypatch.setenv("GEMINI_API_KEY", "set-but-external")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
    assert llm_gateway._build_model_string() == "ollama/llama3.2"
    monkeypatch.setenv("OLLAMA_API_BASE", "https://ollama.example.org")
    with pytest.raises(network_policy.ExternalNetworkBlocked):
        llm_gateway._build_model_string()


def test_apply_sets_offline_switches(offline, monkeypatch):
    for name in ("HF_HUB_OFFLINE", "LITELLM_LOCAL_MODEL_COST_MAP", "ANONYMIZED_TELEMETRY"):
        monkeypatch.delenv(name, raising=False)
    network_policy.apply()
    import os
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
    assert os.environ["ANONYMIZED_TELEMETRY"] == "False"


def test_reindex_reuses_vectors_of_unchanged_chunks(vectors):  # noqa: F811
    embedder = vectors._store.embeddings
    calls = []
    real = embedder.embed_documents
    embedder.embed_documents = lambda texts: calls.append(len(texts)) or real(texts)

    def docs(texts):
        return [Document(page_content=t, metadata={"source_id": "bb-1", "chunk_index": i})
                for i, t in enumerate(texts)]
    progress = []
    vectors.replace_source_documents("bb-1", docs(["a", "b", "c"]), on_progress=lambda d, t: progress.append((d, t)))
    assert sum(calls) == 3 and progress[-1] == (3, 3)
    calls.clear(); progress.clear()
    vectors.replace_source_documents("bb-1", docs(["a", "b", "changed"]), on_progress=lambda d, t: progress.append((d, t)))
    assert sum(calls) == 1               # only the changed chunk is embedded again
    assert progress[0] == (2, 3) and progress[-1] == (3, 3)
    stored = vectors._store._collection.get(where={"source_id": "bb-1"}, include=["documents"])["documents"]
    assert sorted(stored) == ["a", "b", "changed"]
    hits = vectors._store.similarity_search("a", k=3)  # reused vectors still search correctly
    assert {d.page_content for d in hits} == {"a", "b", "changed"}
