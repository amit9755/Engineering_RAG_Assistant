"""Explain a named file block by block, follow-ups, and screenshot search refinements (offline)."""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.graph.image_flow import extract_search_terms, find_code_matches, format_locate_answer, is_weak_term
from src.graph.nodes import file_for_question, find_file_reference, is_file_explain_request, whole_file
from tests.test_connectors import vectors  # noqa: F401


def chunk(path, text, i, source="bb-1"):
    return Document(page_content=f"Repository: WSQ/app (branch main)\nFile: {path}\n\n{text}", metadata={
        "source_id": source, "source_type": "bitbucket", "file_path": path, "chunk_index": i,
        "source_file": f"WSQ/app/{path}", "url": f"https://bb.example.com/{path}", "chunk_id": f"{source}:{path}:{i}"})


@pytest.fixture
def repo(vectors, monkeypatch):  # noqa: F811
    import src.retrieval.vector_store as vector_module
    docs = [
        chunk("FE/src/constants/constants.ts", "export const API = '/api';\nexport const STATUS = {", 0),
        chunk("FE/src/constants/constants.ts", "export const STATUS = {\n  3: 'ACTIVE IN USE IN OFFLINE MODE',\n};", 1),
        chunk("FE/src/app/testbed/testbed-list.component.html",
              "<h2>Monitor and manage your test bed infrastructure</h2>\n<span>Overall Health</span>\n"
              "<div>Total Testbeds</div><small>Last checked: {{ last || 'Never' }}</small>", 0),
        chunk("BE/routes/a/route.ts", "export default a", 0),
        chunk("BE/routes/b/route.ts", "export default b", 0),
        chunk("BE/services/scheduler.js", "// Never blocks", 0),
    ]
    vectors._store.add_documents(docs, ids=[d.metadata["chunk_id"] for d in docs])
    monkeypatch.setattr(vector_module, "vector_store", vectors)
    return vectors


def test_file_named_by_path_or_unique_name(repo):
    assert find_file_reference("FE/src/constants/constants.ts - can you explain this file code")[2] == \
        "FE/src/constants/constants.ts"
    assert find_file_reference("explain constants.ts please")[2] == "FE/src/constants/constants.ts"
    assert find_file_reference("explain route.ts") is None            # two files share that name
    assert find_file_reference("explain b/route.ts")[2] == "BE/routes/b/route.ts"


def test_follow_up_uses_the_file_from_the_previous_message(repo):
    history = [{"role": "user", "content": "FE/src/constants/constants.ts - can you explain this file code"},
               {"role": "assistant", "content": "It holds constants."},
               {"role": "user", "content": "each code block i mean explain"}]
    assert file_for_question("each code block i mean explain", history)[2] == "FE/src/constants/constants.ts"
    assert file_for_question("how does login work", history) is None    # not a follow-up about the file
    assert is_file_explain_request("each code block i mean explain")
    assert not is_file_explain_request("write a unit test for constants.ts")   # that's code mode


def test_whole_file_is_reassembled_without_overlap(repo):
    text, truncated = whole_file("bb-1", "FE/src/constants/constants.ts")
    assert text == "export const API = '/api';\nexport const STATUS = {\n  3: 'ACTIVE IN USE IN OFFLINE MODE',\n};"
    assert not truncated and "Repository:" not in text


def test_explain_file_streams_whole_file_to_code_model(repo, monkeypatch):
    from src.api.main import app
    from src.gateway.llm_gateway import llm_gateway
    calls = {}

    async def stream(messages, temperature=0.1, max_tokens=1024, model=None, stop=None):
        calls.update(messages=messages, model=model)
        yield "### `export const API`\nThe API base path."
    monkeypatch.setattr(llm_gateway, "astream", stream)
    monkeypatch.setattr("dotenv.dotenv_values", lambda *a, **k: {})
    monkeypatch.delenv("OLLAMA_CODE_MODEL", raising=False)
    history = [{"role": "user", "content": "FE/src/constants/constants.ts - can you explain this file code"},
               {"role": "assistant", "content": "It holds constants."},
               {"role": "user", "content": "each code block i mean explain"}]
    with TestClient(app) as client:
        text = client.post("/api/v1/query/stream", json={"question": "each code block i mean explain",
                                                         "conversation_history": history}).text
    events = [json.loads(l[6:]) for l in text.splitlines() if l.startswith("data: ")]
    assert calls["model"] == "ollama/qwen2.5-coder:7b"
    assert "ACTIVE IN USE IN OFFLINE MODE" in calls["messages"][0]["content"]     # the whole file is in the prompt
    assert "block by block" in calls["messages"][0]["content"]
    assert events[0]["status"].startswith("Explaining FE/src/constants/constants.ts")
    assert events[-1]["sources"] == ["WSQ/app/FE/src/constants/constants.ts"]


def test_screenshot_search_handles_casing_misreads_and_noise(repo):
    transcription = """TestBed
Monitor and manage your tested bed infrastructure
OVERALL HEALTH
TOTAL TESTBEDS
Last checked: Never"""
    terms = extract_search_terms(transcription)
    assert "Monitor and manage" in terms and is_weak_term("Never") and not is_weak_term("OVERALL HEALTH")
    matches = find_code_matches(terms, None)
    files = [m["file"].split("/", 2)[-1] for m in matches]
    assert files[0] == "FE/src/app/testbed/testbed-list.component.html"   # found despite UPPERCASE + misread
    assert "BE/services/scheduler.js" not in files                          # matched only "Never": dropped
    page = matches[0]
    assert {"OVERALL HEALTH", "TOTAL TESTBEDS", "Monitor and manage"} <= set(page["terms"])
    answer = format_locate_answer(matches, ["WSQ/app/BE/routes/a/route.ts"], transcription)
    assert "**Related by meaning**" in answer and "`BE/routes/a/route.ts`" in answer


def test_code_names_are_found_word_for_word_definition_first(repo):
    from unittest.mock import Mock
    from src.retrieval.hybrid_retriever import HybridRetriever, BM25Retriever, code_names
    repo._store.add_documents([
        chunk("FE/src/app/services/device.service.ts", "getDetails() { return this.http.get(fetchDeviceDetails); }", 0),
        chunk("FE/src/constants/constants.ts", "export const fetchDeviceDetails = `${base}/devices/details`;", 2),
        chunk("FE/src/app/x.ts", "const fetchDeviceDetailsOld = 1", 0),
    ], ids=["svc", "def", "other"])
    assert code_names("explain the fetchDeviceDetails and fetchDevicesAnalytics functions") == [
        "fetchDeviceDetails", "fetchDevicesAnalytics"]
    retriever = object.__new__(HybridRetriever)
    retriever.vector_store, retriever.bm25, retriever._bm25_ready = repo, BM25Retriever(), True
    retriever.reranker = Mock()
    retriever.reranker.rerank.return_value = []      # meaning-based search finds nothing
    true, _ = retriever.retrieve("Can you explain the code in fetchDeviceDetails?")
    files = [c.document.metadata["file_path"] for c in true]
    assert files[0] == "FE/src/constants/constants.ts"          # the definition comes first
    assert "FE/src/app/services/device.service.ts" in files     # then where it is used


def test_answers_stop_before_inventing_a_user_turn(monkeypatch):
    from src.graph import nodes
    from src.gateway.llm_gateway import llm_gateway
    seen = {}
    monkeypatch.setattr(nodes, "knowledge_catalog", lambda f=None: [])
    monkeypatch.setattr(llm_gateway, "complete", lambda messages, **k: seen.update(k) or "answer")
    nodes.generation_node({"original_query": "how does publish work?", "true_data_chunks": [{"content": "x"}],
                           "assembled_context": "x", "pipeline_steps": []})
    assert "\nUser:" in seen["stop"]


@pytest.mark.parametrize("question,review,arch", [
    ("tell me whts you want changes here", True, False), ("what would you change in constants.ts", True, False),
    ("review it", True, False), ("can you draw arch of this project", False, True),
    ("show the architecture diagram", False, True), ("how is the repository structured?", False, True),
    ("how does login work", False, False), ("explain this file", False, False),
])
def test_review_and_architecture_detection(question, review, arch):
    from src.graph.nodes import is_architecture_request, is_review_request
    assert is_review_request(question) is review and is_architecture_request(question) is arch


def test_review_follow_up_reuses_previous_file(repo):
    history = [{"role": "user", "content": "FE/src/constants/constants.ts explain this file"},
               {"role": "assistant", "content": "..."},
               {"role": "user", "content": "tell me whts you want changes here"}]
    assert file_for_question("tell me whts you want changes here", history)[2] == "FE/src/constants/constants.ts"


def test_repo_layout_is_exact(repo):
    from src.graph.nodes import repo_layout
    layout, repos = repo_layout()
    assert repos == ["WSQ/app"] and layout.startswith("Repository WSQ/app (5 files):")
    assert "FE/ (2 files): src/ (2)" in layout and "BE/ (3 files): routes/ (2), services/ (1)" in layout


def test_review_and_architecture_stream_to_code_model(repo, monkeypatch):
    from src.api.main import app
    from src.gateway.llm_gateway import llm_gateway
    prompts = []

    async def stream(messages, temperature=0.1, max_tokens=1024, model=None, stop=None):
        prompts.append(messages[0]["content"])
        yield "ok"
    monkeypatch.setattr(llm_gateway, "astream", stream)
    monkeypatch.setattr("dotenv.dotenv_values", lambda *a, **k: {})
    history = [{"role": "user", "content": "FE/src/constants/constants.ts explain this file"},
               {"role": "assistant", "content": "..."},
               {"role": "user", "content": "tell me whts you want changes here"}]
    with TestClient(app) as client:
        review = client.post("/api/v1/query/stream", json={"question": "tell me whts you want changes here",
                                                           "conversation_history": history}).text
        arch = client.post("/api/v1/query/stream", json={"question": "can you draw arch of this project"}).text
    assert "Reviewing FE/src/constants/constants.ts" in review
    assert "at most 5 improvements" in prompts[0] and "ACTIVE IN USE IN OFFLINE MODE" in prompts[0]
    assert "Drawing the architecture" in arch
    assert "Repository WSQ/app (5 files)" in prompts[1] and "flowchart LR" in prompts[1]
