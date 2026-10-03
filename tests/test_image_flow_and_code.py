"""Screenshot -> code location, and code suggestion mode (offline; models mocked)."""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.graph.image_flow import extract_search_terms, find_code_matches, format_locate_answer, is_locate_question
from src.graph.nodes import code_context_chunks, is_code_request
from src.retrieval.source_filter import SourceFilter
from tests.test_connectors import vectors  # noqa: F401
from tests.test_vision import PNG

TRANSCRIPTION = """Reservation Details
Reservation in progress - auto-trigger disabled
Testbed: WCS-TB-07
Error: overall_status could not be computed
at computeOverallStatus (BE/controllers/testBedController.js:310)
Image shows: a web page with a reservation panel"""


def test_search_terms_from_a_screenshot():
    terms = extract_search_terms(TRANSCRIPTION)
    assert terms[:3] == ["testBedController.js", "overall_status", "computeOverallStatus"]
    assert "Reservation in progress - auto-trigger disabled" in terms
    assert "overall_status could not be computed" in terms     # message part after "Error:"
    assert all("Image shows" not in t for t in terms)


def code_doc(path, text, i=0, source="bb-1", kind="bitbucket"):
    meta = {"source_id": source, "source_type": kind, "file_path": path, "chunk_index": i,
            "source_file": f"WSQ/app/{path}", "url": f"https://bitbucket.example.com/{path}",
            "chunk_id": f"{source}:{path}:{i}"}
    return Document(page_content=f"Repository: WSQ/app\nFile: {path}\n\n{text}", metadata=meta)


@pytest.fixture
def code_index(vectors, monkeypatch):  # noqa: F811
    import src.retrieval.vector_store as vector_module
    docs = [
        code_doc("BE/controllers/testBedController.js", "function computeOverallStatus(tb) {\n  // overall_status\n}"),
        code_doc("FE/src/app/reservation/panel.html", '<span>Reservation in progress - auto-trigger disabled</span>'),
        code_doc("tests/testBed.test.js", "it('computes overall_status', () => computeOverallStatus())"),
        code_doc("README.md", "General readme"),
    ]
    vectors._store.add_documents(docs, ids=[d.metadata["chunk_id"] for d in docs])
    monkeypatch.setattr(vector_module, "vector_store", vectors)
    return vectors


def test_exact_matches_rank_files_by_matched_terms(code_index):
    matches = find_code_matches(extract_search_terms(TRANSCRIPTION), SourceFilter(["bb-1"], []))
    files = [m["file"].split("/", 2)[-1] for m in matches]
    assert files[0] == "BE/controllers/testBedController.js"           # 3 terms
    assert "FE/src/app/reservation/panel.html" in files
    assert files.index("tests/testBed.test.js") > files.index("BE/controllers/testBedController.js")
    assert "README.md" not in files

    code_index._store.add_documents([code_doc("docs/guide.md", "computeOverallStatus overall_status testBedController.js "
                                              "Reservation in progress - auto-trigger disabled")], ids=["doc-md"])
    ranked = [m["file"].split("/", 2)[-1] for m in find_code_matches(extract_search_terms(TRANSCRIPTION), None)]
    assert ranked[-1] == "docs/guide.md"   # documentation after code, even with more matched terms
    controller = matches[0]
    assert "computeOverallStatus" in controller["terms"] and "computeOverallStatus" in controller["line"]
    assert find_code_matches(["computeOverallStatus"], SourceFilter(["other"], [])) == []   # respects selection


def test_locate_answer_lists_files_and_transcription(code_index):
    matches = find_code_matches(extract_search_terms(TRANSCRIPTION), None)
    answer = format_locate_answer(matches, [], TRANSCRIPTION)
    assert answer.startswith("### Where this appears in the code")
    assert "[BE/controllers/testBedController.js](https://bitbucket.example.com/" in answer
    assert "**Text read from the image:**" in answer and "WCS-TB-07" in answer
    none = format_locate_answer([], ["WSQ/app/src/x.ts"], "Hello")
    assert "word-for-word" in none and "`src/x.ts`" in none
    assert is_locate_question("can you find where is the code related?")
    assert is_locate_question("which file implements this page")
    assert not is_locate_question("what does this error mean?")


def test_where_is_the_code_is_answered_without_a_long_generation(code_index, monkeypatch):
    from src.api.main import app
    from src.gateway.llm_gateway import llm_gateway
    from src.retrieval.hybrid_retriever import hybrid_retriever
    from src.graph import nodes
    import src.graph.image_flow as flow
    monkeypatch.setattr(flow, "read_image", lambda *a, **k: TRANSCRIPTION)
    monkeypatch.setattr(hybrid_retriever, "retrieve", lambda *a, **k: ([], []))
    monkeypatch.setattr(nodes, "knowledge_catalog", lambda f=None: [])

    async def must_not_generate(*a, **k):
        raise AssertionError("no generation needed for 'where is the code'")
        yield  # pragma: no cover
    monkeypatch.setattr(llm_gateway, "astream", must_not_generate)
    monkeypatch.setattr("dotenv.dotenv_values", lambda *a, **k: {})
    with TestClient(app) as client:
        text = client.post("/api/v1/query/stream", json={
            "question": "can you find where is the code related to this?", "images": [PNG]}).text
    events = [json.loads(l[6:]) for l in text.splitlines() if l.startswith("data: ")]
    answer = "".join(e.get("token", "") for e in events)
    assert "testBedController.js" in answer and "panel.html" in answer
    assert events[-1]["sources"][0] == "WSQ/app/BE/controllers/testBedController.js"


@pytest.mark.parametrize("question,expected", [
    ("suggest the code to add retry with backoff in publishInstagram", True),
    ("write a unit test for computeOverallStatus", True),
    ("how do I implement pagination for the RC-Kernel mapping?", True),
    ("refactor the reservation controller to use async/await", True),
    ("give me code for exporting the table to CSV", True),
    ("how does the reservation flow work?", False),
    ("explain this project", False),
    ("tell me last 5 commits", False),
    ("all jira created by me last month", False),
])
def test_code_request_detection(question, expected):
    assert is_code_request(question) is expected


def test_code_context_expands_best_files(code_index):
    for i in range(1, 12):
        code_index._store.add_documents([code_doc("BE/controllers/testBedController.js", f"// part {i}", i)],
                                        ids=[f"bb-1:ctrl:{i}"])
    hit = {"content": "x", "source": "WSQ/app/BE/controllers/testBedController.js", "source_type": "bitbucket",
           "score": 0.9, "metadata": {"source_id": "bb-1", "file_path": "BE/controllers/testBedController.js",
                                      "chunk_index": 6}}
    doc_hit = {"content": "a doc", "source": "guide.pdf", "source_type": "document", "score": 0.5, "metadata": {}}
    out = code_context_chunks([hit, doc_hit], per_file=5)
    parts = [c["content"].rsplit("// part ", 1)[-1] for c in out[:5]]
    assert parts == ["4", "5", "6", "7", "8"]                  # consecutive chunks around the hit
    assert out[-1]["content"] == "a doc"


def test_code_mode_uses_code_model_and_falls_back_with_hint(code_index, monkeypatch):
    from src.api.main import app
    from src.gateway.llm_gateway import llm_gateway
    from src.retrieval.hybrid_retriever import hybrid_retriever, ScoredChunk
    from src.graph import nodes
    chunk = ScoredChunk(code_doc("BE/controllers/testBedController.js", "function computeOverallStatus() {}"),
                        0.9, True, "reranked")
    monkeypatch.setattr(hybrid_retriever, "retrieve", lambda *a, **k: ([chunk], []))
    monkeypatch.setattr(nodes, "knowledge_catalog", lambda f=None: [])
    monkeypatch.setattr("dotenv.dotenv_values", lambda *a, **k: {})
    monkeypatch.delenv("OLLAMA_CODE_MODEL", raising=False)
    calls = []

    async def stream(messages, temperature=0.1, max_tokens=1024, model=None, stop=None):
        calls.append(model)
        if model == "ollama/qwen2.5-coder:7b":
            raise RuntimeError("model not found")
        yield "```js\nfunction computeOverallStatus() { /* new */ }\n```"
    monkeypatch.setattr(llm_gateway, "astream", stream)
    with TestClient(app) as client:
        text = client.post("/api/v1/query/stream", json={
            "question": "write a unit test for computeOverallStatus"}).text
    body = "".join(json.loads(l[6:]).get("token", "") for l in text.splitlines() if l.startswith("data: "))
    assert calls == ["ollama/qwen2.5-coder:7b", None]
    assert "```js" in body and "ollama pull qwen2.5-coder:7b" in body
    assert '"status": "Writing code with qwen2.5-coder:7b..."' in text
