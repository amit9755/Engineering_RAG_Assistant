"""Saved chats, overview retrieval for broad questions, and follow-up suggestions (offline)."""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.chats.store import ChatStore
from src.retrieval.source_filter import SourceFilter
from tests.test_connectors import vectors  # noqa: F401  (fixture)


def msgs(question, answer="an answer"):
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer, "sources": ["a.py"]}]


def test_chat_store_saves_lists_and_keeps_only_newest(tmp_path):
    store = ChatStore(tmp_path / "chats.db", max_chats=3)
    for i in range(5):
        store.save(f"sess-{i}", msgs(f"question {i}"), "admin1")
    assert [c["id"] for c in store.list("admin1")] == ["sess-4", "sess-3", "sess-2"]
    chat = store.get("sess-4", "admin1")
    assert chat["title"] == "question 4"
    assert chat["messages"][1] == {"role": "assistant", "content": "an answer", "sources": ["a.py"]}
    store.save("sess-2", msgs("question 2", "updated"), "admin1")  # re-saving moves it to the top
    assert store.list("admin1")[0]["id"] == "sess-2"
    assert store.get("sess-2", "admin1")["messages"][1]["content"] == "updated"
    assert store.delete("sess-3", "admin1") and store.get("sess-3", "admin1") is None
    with pytest.raises(ValueError):
        store.save("../etc", msgs("x"), "admin1")
    with pytest.raises(ValueError):
        store.save("sess-x", [{"role": "system", "content": "ignored"}], "admin1")


def test_each_user_sees_only_their_own_chats(tmp_path):
    store = ChatStore(tmp_path / "chats.db", max_chats=2)
    store.save("sess-a", msgs("admin1 question"), "admin1")
    store.save("sess-b", msgs("admin2 question"), "admin2")
    for i in range(3):  # admin2 filling their 2-chat limit never deletes admin1's chats
        store.save(f"sess-b{i}", msgs(f"admin2 q{i}"), "admin2")
    assert [c["id"] for c in store.list("admin1")] == ["sess-a"]
    assert len(store.list("admin2")) == 2
    assert store.get("sess-a", "admin2") is None and not store.delete("sess-a", "admin2")
    with pytest.raises(PermissionError):
        store.save("sess-a", msgs("overwrite attempt"), "admin2")
    assert store.get("sess-a", "admin1")["title"] == "admin1 question"


def test_overview_chunks_prefer_readme_docs_and_file_tree(vectors):  # noqa: F811
    def chunk(source_id, path, index, kind="bitbucket", text=None):
        meta = {"source_id": source_id, "source_type": kind, "file_path": path, "chunk_index": index,
                "source_file": path, "chunk_id": f"{source_id}:{path}:{index}"}
        return Document(page_content=text or f"{path} part {index}", metadata=meta)
    docs = [chunk("bb-1", "src/app.py", 0), chunk("bb-1", "README.md", 0), chunk("bb-1", "README.md", 1),
            chunk("bb-1", "docs/architecture.md", 0), chunk("bb-1", "(file tree)", 0),
            chunk("bb-1", "lib/README.md", 0), chunk("bb-1", "README.md", 5),
            chunk("doc-1", "", 0, kind="document", text="Resume opening"), chunk("doc-1", "", 3, kind="document"),
            chunk("bb-2", "README.md", 0)]
    vectors._store.add_documents(docs, ids=[d.metadata["chunk_id"] for d in docs])
    picked = vectors.get_overview_chunks(SourceFilter(["bb-1", "doc-1"], []))
    paths = [(d.metadata["source_id"], d.metadata["file_path"], d.metadata["chunk_index"]) for d in picked]
    assert paths[:4] == [("bb-1", "README.md", 0), ("bb-1", "README.md", 1), ("bb-1", "lib/README.md", 0),
                         ("bb-1", "docs/architecture.md", 0)]
    assert ("doc-1", "", 0) in paths and all(p[0] != "bb-2" for p in paths)
    assert not any(p[1] == "src/app.py" for p in paths)


def test_broad_questions_get_overview_even_when_reranker_rejects_everything(vectors):  # noqa: F811
    with patch("src.retrieval.hybrid_retriever.SemanticReRanker._load_model"):
        from src.retrieval.hybrid_retriever import HybridRetriever, BM25Retriever, is_overview_question
    readme = Document(page_content="# Project X\nWhat it does", metadata={
        "source_id": "bb-1", "source_type": "bitbucket", "file_path": "README.md", "chunk_index": 0,
        "chunk_id": "bb-1:README.md:0"})
    vectors._store.add_documents([readme], ids=["r"])
    retriever = object.__new__(HybridRetriever)
    retriever.vector_store, retriever.bm25, retriever._bm25_ready = vectors, BM25Retriever(), True
    retriever.reranker = Mock()
    retriever.reranker.rerank.return_value = []
    true, _ = retriever.retrieve("okay explain this project", source_filter=SourceFilter(["bb-1"], []))
    assert [c.document.metadata["file_path"] for c in true] == ["README.md"]
    for q in ["what does this repository do?", "give me an overview", "describe it", "tell me about the repo"]:
        assert is_overview_question(q), q
    for q in ["how does login work", "explain the auth flow", "what does authController do"]:
        assert not is_overview_question(q), q


def test_followup_suggestions_are_cleaned_and_fall_back(monkeypatch):
    from src.graph import nodes
    from src.gateway.llm_gateway import llm_gateway
    catalog = ["- Bitbucket repository: WSQ/app (10 chunks)", "- Document: cv.pdf (3 chunks)"]
    monkeypatch.setattr(llm_gateway, "complete", lambda *a, **k:
                        "1. How is the token refreshed?\n- Which file defines the routes?\nSure! Here you go\n"
                        "\"Where is logging configured?\"\nHow is the token refreshed?")
    assert nodes.suggest_followups("how does auth work?", "answer", ["WSQ/app/auth.js"], catalog) == [
        "How is the token refreshed?", "Which file defines the routes?", "Where is logging configured?"]
    assert nodes.suggest_followups("hi", "Hi!", [], catalog) == [
        "Explain this project", "What are the latest 5 commits?", "Which files handle configuration?"]

    def boom(*a, **k):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(llm_gateway, "complete", boom)
    assert nodes.suggest_followups("q?", "a", ["x.py"], ["- Document: cv.pdf (3 chunks)"]) == [
        "Summarize the uploaded documents"]
