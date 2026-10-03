"""Questions with attached images: validation, Ollama message format, endpoint behaviour (model mocked)."""

import base64
import json

import pytest
from fastapi.testclient import TestClient

from src.graph.nodes import decode_images

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
    import src.graph.image_flow as flow
    monkeypatch.setattr(flow, "read_image", lambda images, model: reads.append((images, model)) or
                        "Error 190: Invalid OAuth access token\nat publishInstagram (lib/instagram/publish.ts:142)\n"
                        "Image shows: an error dialog")
    monkeypatch.setattr(hybrid_retriever, "retrieve", lambda q, **k: searched.append(q) or ([], []))

    async def fake_stream(messages, temperature=0.1, max_tokens=1024, model=None, stop=None):
        answered.update(messages=messages, model=model)
        yield "The token expired."
    monkeypatch.setattr(llm_gateway, "astream", fake_stream)
    monkeypatch.setattr("dotenv.dotenv_values", lambda *a, **k: {})
    monkeypatch.delenv("OLLAMA_VISION_MODEL", raising=False)
    events = stream_events(client, {"question": "what is this error?", "images": [PNG]})
    assert [e["status"] for e in events if "status" in e][0].startswith("Reading the image")
    assert reads == [([PNG.split(",", 1)[1]], "ollama_chat/gemma3:4b")]   # read once
    assert answered["model"] is None                       # answered by the (faster) text model
    assert "publishInstagram" in answered["messages"][0]["content"] and "images" not in answered["messages"][-1]
    assert "\n" not in searched[0] and "publishInstagram" in searched[0]   # one-line search query
    assert "".join(e.get("token", "") for e in events) == "The token expired."


def test_missing_vision_model_gives_install_hint(client, monkeypatch):
    from src.gateway.llm_gateway import llm_gateway

    from unittest.mock import Mock
    reply = Mock(status_code=404, text="")
    reply.json.return_value = {"error": 'model "gemma3:4b" not found, try pulling it first'}
    monkeypatch.setattr("requests.post", lambda *a, **k: reply)
    monkeypatch.setattr("dotenv.dotenv_values", lambda *a, **k: {})
    monkeypatch.delenv("OLLAMA_VISION_MODEL", raising=False)
    body = client.post("/api/v1/query/stream", json={"question": "describe", "images": [PNG]}).text
    assert "ollama pull gemma3:4b" in body


def test_bad_image_is_rejected(client):
    body = client.post("/api/v1/query/stream", json={"question": "x", "images": ["data:text/html;base64,PGI+"]}).text
    assert "PNG, JPEG, WebP or GIF" in body
    r = client.post("/api/v1/query", json={"question": "x", "images": ["nope"]})
    assert r.status_code == 422


@pytest.mark.parametrize("status,error,expected", [
    (404, 'model "gemma3:4b" not found, try pulling it first', "ollama pull gemma3:4b"),
    (500, "unknown model architecture: 'gemma3'", "Update Ollama"),
    (500, "model requires more system memory (5.1 GiB) than is available (3.2 GiB)", "Not enough memory"),
    (500, "something else broke", "HTTP 500): something else broke"),
])
def test_ollama_image_errors_are_explained(monkeypatch, status, error, expected):
    from unittest.mock import Mock
    from src.graph.image_flow import ImageModelError, read_image
    reply = Mock(status_code=status, text=error)
    reply.json.return_value = {"error": error}
    monkeypatch.setattr("requests.post", lambda *a, **k: reply)
    with pytest.raises(ImageModelError, match=expected.replace("(", r"\(").replace(")", r"\)")):
        read_image(["QUJD"], "ollama_chat/gemma3:4b")


def test_read_image_calls_ollama_chat_directly(monkeypatch):
    from unittest.mock import Mock
    from src.graph.image_flow import read_image
    calls = {}
    reply = Mock(status_code=200)
    reply.json.return_value = {"message": {"content": " Error 190\nImage shows: a dialog "}}
    monkeypatch.setattr("requests.post", lambda url, json, timeout: calls.update(url=url, body=json) or reply)
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
    monkeypatch.setattr("dotenv.dotenv_values", lambda *a, **k: {})
    assert read_image(["QUJD"], "ollama_chat/gemma3:4b") == "Error 190\nImage shows: a dialog"
    assert calls["url"] == "http://localhost:11434/api/chat"
    assert calls["body"]["model"] == "gemma3:4b" and calls["body"]["messages"][0]["images"] == ["QUJD"]
    assert calls["body"]["stream"] is False


def test_ollama_not_running_is_explained(monkeypatch):
    import requests
    from src.graph.image_flow import ImageModelError, read_image

    def refuse(*a, **k):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr("requests.post", refuse)
    with pytest.raises(ImageModelError, match="Make sure Ollama is running"):
        read_image(["QUJD"], "ollama_chat/gemma3:4b")
