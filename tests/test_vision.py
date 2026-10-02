"""Questions with attached images: validation, Ollama message format, endpoint behaviour (model mocked)."""

import base64
import json

import pytest
from fastapi.testclient import TestClient

from src.graph.nodes import build_vision_messages, decode_images

PNG = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 100).decode()


def test_decode_images_validates_type_count_and_size():
    assert decode_images([PNG]) == [PNG.split(",", 1)[1]]
    with pytest.raises(ValueError, match="PNG, JPEG"):
        decode_images(["data:text/html;base64,PGI+"])
    with pytest.raises(ValueError, match="at most 3"):
        decode_images([PNG] * 4)
    with pytest.raises(ValueError, match="not valid"):
        decode_images(["data:image/png;base64,abc"])
    big = "data:image/jpeg;base64," + base64.b64encode(b"0" * (6 * 1024 * 1024 + 1)).decode()
    with pytest.raises(ValueError, match="under 6 MB"):
        decode_images([big])


def test_vision_messages_put_images_on_the_user_message():
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "answer"},
               {"role": "user", "content": "what is this error?"}]
    messages = build_vision_messages("what is this error?", "[1] (code) a.py\nx", ["- repo"], history, ["QUJD"])
    assert messages[0]["role"] == "system" and "attached image" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "what is this error?", "images": ["QUJD"]}
    assert [m["content"] for m in messages[1:-1]] == ["earlier", "answer"]   # current question not duplicated
    assert all("images" not in m for m in messages[:-1])


@pytest.fixture
def client(monkeypatch):
    from src.api.main import app
    from src.retrieval.hybrid_retriever import hybrid_retriever
    from src.graph import nodes
    monkeypatch.setattr(hybrid_retriever, "retrieve", lambda *a, **k: ([], []))
    monkeypatch.setattr(nodes, "knowledge_catalog", lambda f=None: [])
    with TestClient(app) as client:
        yield client


def stream_events(client, body):
    text = client.post("/api/v1/query/stream", json=body).text
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def test_image_is_read_once_then_answered_by_text_model(client, monkeypatch):
    from src.gateway.llm_gateway import llm_gateway
    from src.retrieval.hybrid_retriever import hybrid_retriever
    reads, searched, answered = [], [], {}
    monkeypatch.setattr(llm_gateway, "complete_via_stream", lambda messages, **k: reads.append((messages, k)) or
                        "Error 190: Invalid OAuth access token\nat publishInstagram (lib/instagram/publish.ts:142)\n"
                        "Image shows: an error dialog")
    monkeypatch.setattr(hybrid_retriever, "retrieve", lambda q, **k: searched.append(q) or ([], []))

    async def fake_stream(messages, temperature=0.1, max_tokens=1024, model=None):
        answered.update(messages=messages, model=model)
        yield "The token expired."
    monkeypatch.setattr(llm_gateway, "astream", fake_stream)
    monkeypatch.setattr("dotenv.dotenv_values", lambda *a, **k: {})
    monkeypatch.delenv("OLLAMA_VISION_MODEL", raising=False)
    events = stream_events(client, {"question": "what is this error?", "images": [PNG]})
    assert [e["status"] for e in events if "status" in e][0].startswith("Reading the image")
    assert len(reads) == 1 and reads[0][1]["model"] == "ollama_chat/gemma3:4b"
    assert reads[0][0][0]["images"] == [PNG.split(",", 1)[1]] and reads[0][1]["max_tokens"] <= 400
    assert answered["model"] is None                       # answered by the (faster) text model
    assert "publishInstagram" in answered["messages"][0]["content"] and "images" not in answered["messages"][-1]
    assert "\n" not in searched[0] and "publishInstagram" in searched[0]   # one-line search query
    assert "".join(e.get("token", "") for e in events) == "The token expired."


def test_missing_vision_model_gives_install_hint(client, monkeypatch):
    from src.gateway.llm_gateway import llm_gateway

    def missing(*a, **k):
        raise RuntimeError('OllamaException - {"error":"model \\"gemma3:4b\\" not found, try pulling it first"}')
    monkeypatch.setattr(llm_gateway, "complete_via_stream", missing)
    monkeypatch.setattr("dotenv.dotenv_values", lambda *a, **k: {})
    monkeypatch.delenv("OLLAMA_VISION_MODEL", raising=False)
    body = client.post("/api/v1/query/stream", json={"question": "describe", "images": [PNG]}).text
    assert "ollama pull gemma3:4b" in body


def test_bad_image_is_rejected(client):
    body = client.post("/api/v1/query/stream", json={"question": "x", "images": ["data:text/html;base64,PGI+"]}).text
    assert "PNG, JPEG, WebP or GIF" in body
    r = client.post("/api/v1/query", json={"question": "x", "images": ["nope"]})
    assert r.status_code == 422
