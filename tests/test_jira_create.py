"""Create Jira tickets from chat: intent, draft, metadata, payload, errors (Jira and model mocked)."""

import json
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from src.sources import jira_create
from src.sources.jobs import IndexingError
from src.sources.models import JiraSourceConfig
from tests.test_connectors import FakeCredentials, registry, response  # noqa: F401


@pytest.mark.parametrize("question,expected", [
    ("create jira ticket with this context", True), ("raise a bug for this error", True),
    ("create a story for the retry feature", True), ("log a task to clean up the scheduler", True),
    ("all jira created by me last month", False), ("write code to create a jira ticket", False),
    ("create a function for retry", False), ("how does the reservation flow work?", False),
])
def test_create_intent(question, expected):
    assert jira_create.is_create_request(question) is expected


def test_draft_uses_model_json_and_adds_related_sources(monkeypatch):
    from src.gateway.llm_gateway import llm_gateway
    monkeypatch.setattr(llm_gateway, "complete", lambda *a, **k: 'Sure:\n{"issue_type": "Bug", "summary": '
                        '"Instagram publish fails with Error 190", "description": "Token expired.", '
                        '"priority": "High", "labels": ["instagram", "oauth token"]}')
    history = [{"role": "user", "content": "what is this error?"},
               {"role": "assistant", "content": "Error 190 means the token expired."},
               {"role": "user", "content": "create jira ticket with this context"}]
    draft = jira_create.draft_ticket("create jira ticket with this context", history, ["WSQ/app/lib/errors.ts"])
    assert draft["issue_type"] == "Bug" and draft["priority"] == "High"
    assert draft["summary"] == "Instagram publish fails with Error 190"
    assert draft["labels"] == ["instagram", "oauth-token"]
    assert draft["description"].startswith("Token expired.") and "- WSQ/app/lib/errors.ts" in draft["description"]


def test_draft_falls_back_without_the_model(monkeypatch):
    from src.gateway.llm_gateway import llm_gateway
    monkeypatch.setattr(llm_gateway, "complete", Mock(side_effect=RuntimeError("ollama down")))
    history = [{"role": "assistant", "content": "The publish step fails with an exception."}]
    draft = jira_create.draft_ticket("create a ticket for this", history)
    assert draft["issue_type"] == "Bug" and draft["description"] == "The publish step fails with an exception."


@pytest.fixture
def jira(registry, monkeypatch):  # noqa: F811
    import src.sources.registry as registry_module
    import src.sources.credentials as credentials_module
    registry.create_jira_source(JiraSourceConfig(source_id="jira-1", base_url="https://jira.example.com",
                                                 project_key="WSQ", credential_id="c"), name="WSQ")
    monkeypatch.setattr(registry_module, "source_registry", registry)
    monkeypatch.setattr(credentials_module, "credential_store", FakeCredentials())
    session = Mock()
    monkeypatch.setattr("src.sources.atlassian.requests.Session", lambda: session)
    return session


TYPES = {"values": [{"id": "3", "name": "Task"}, {"id": "5", "name": "Sub-task", "subtask": True},
                    {"id": "10", "name": "Epic"}, {"id": "1", "name": "Bug"}, {"id": "7", "name": "Story"}]}
FIELDS = {"values": [
    {"fieldId": "summary", "name": "Summary", "required": True, "schema": {"type": "string"}},
    {"fieldId": "priority", "name": "Priority", "required": False, "schema": {"type": "priority"},
     "allowedValues": [{"id": "2", "name": "High"}, {"id": "3", "name": "Medium"}]},
    {"fieldId": "customfield_100", "name": "Team", "required": True, "schema": {"type": "option"},
     "allowedValues": [{"id": "501", "value": "WSQA Automation"}]},
    {"fieldId": "components", "name": "Component/s", "required": True, "schema": {"type": "array", "items": "component"},
     "allowedValues": [{"id": "900", "name": "Backend"}]},
    {"fieldId": "customfield_200", "name": "Optional text", "required": False, "schema": {"type": "string"}},
    {"fieldId": "customfield_300", "name": "Has default", "required": True, "hasDefaultValue": True,
     "schema": {"type": "string"}},
]}


def test_issue_types_and_required_fields(jira):
    jira.get.side_effect = [response(200, TYPES)]
    assert [t["name"] for t in jira_create.issue_types("jira-1")] == ["Bug", "Task", "Story", "Epic"]
    jira.get.side_effect = [response(200, FIELDS)]
    specs = {s["id"]: s for s in jira_create.create_fields("jira-1", "1")}
    assert set(specs) == {"priority", "customfield_100", "components"}       # optional / defaulted ones hidden
    assert specs["customfield_100"]["options"] == [{"id": "501", "name": "WSQA Automation"}]
    assert specs["components"]["type"] == "multi-option" and specs["priority"]["type"] == "option"


def test_issue_types_fall_back_for_older_jira(jira):
    jira.get.side_effect = [response(404), response(200, {"projects": [{"issuetypes": [{"id": "1", "name": "Bug"}]}]})]
    assert jira_create.issue_types("jira-1") == [{"id": "1", "name": "Bug"}]


def test_create_issue_sends_fields_and_returns_link(jira):
    jira.get.side_effect = [response(200, FIELDS)]
    jira.post.return_value = response(201, {"id": "1", "key": "WSQ-42"})
    result = jira_create.create_issue("jira-1", "1", " Publish fails ", "Token expired.", ["instagram"],
                                      {"priority": "2", "customfield_100": "501", "components": ["900"], "x": ""},
                                      created_by="admin2")
    assert result == {"key": "WSQ-42", "url": "https://jira.example.com/browse/WSQ-42"}
    call = jira.post.call_args
    assert call.args[0] == "https://jira.example.com/rest/api/2/issue"
    assert call.kwargs["headers"]["Authorization"] == "Bearer secret-token"
    assert call.kwargs["json"]["fields"] == {
        "project": {"key": "WSQ"}, "issuetype": {"id": "1"}, "summary": "Publish fails",
        "description": "Token expired.", "labels": ["instagram"], "priority": {"id": "2"},
        "customfield_100": {"id": "501"}, "components": [{"id": "900"}]}


def test_jira_validation_errors_are_shown(jira):
    jira.get.side_effect = [response(200, FIELDS)]
    jira.post.return_value = response(400, {"errorMessages": [], "errors": {"customfield_100": "Team is required."}})
    with pytest.raises(IndexingError, match="Jira did not create the ticket: Team: Team is required."):
        jira_create.create_issue("jira-1", "1", "Summary", "", [], {})


def test_chat_returns_a_draft_action_and_never_creates(jira, monkeypatch):
    from src.api.main import app
    from src.gateway.llm_gateway import llm_gateway
    monkeypatch.setattr(llm_gateway, "complete", lambda *a, **k: '{"issue_type": "Task", "summary": "Clean up"}')
    with TestClient(app) as client:
        text = client.post("/api/v1/query/stream", json={"question": "create jira ticket with this context"}).text
        events = [json.loads(l[6:]) for l in text.splitlines() if l.startswith("data: ")]
        action = next(e["action"] for e in events if "action" in e)
        assert action["type"] == "jira_draft" and action["draft"]["summary"] == "Clean up"
        assert "Create in Jira" in "".join(e.get("token", "") for e in events)
        jira.post.assert_not_called()
        jira.get.side_effect = [response(200, FIELDS)]
        jira.post.return_value = response(201, {"key": "WSQ-7"})
        r = client.post("/api/v1/jira/jira-1/issues", json={"issue_type_id": "3", "summary": "Clean up",
                                                            "fields": {"customfield_100": "501"}})
        assert r.status_code == 200 and r.json()["key"] == "WSQ-7"


@pytest.mark.real_auth
def test_ticket_routes_need_sign_in_but_not_admin(tmp_path, monkeypatch):
    from src.api.main import app
    from src.auth import store as store_module
    from src.auth.store import UserStore
    users = UserStore(tmp_path / "users.db")
    monkeypatch.setattr(users, "ensure_initial_admins", lambda *a, **k: [])
    monkeypatch.setattr(store_module, "user_store", users)
    users.create_user("viewer", "viewer-pass-1", role="user")
    with TestClient(app) as client:
        assert client.get("/api/v1/jira/projects").status_code == 401
        client.post("/api/v1/auth/login", json={"username": "viewer", "password": "viewer-pass-1"})
        assert client.get("/api/v1/jira/projects").status_code == 200
